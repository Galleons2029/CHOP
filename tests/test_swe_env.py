import asyncio
import json
import subprocess

import pytest

from envs.swe import codeact
from envs.swe.editor import Editor
from envs.swe.sandbox import LocalSandbox, model_patch
from envs.toy import EOS, ToyTokenizer


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def f():\n    return 1\n")
    for cmd in ("git init -q", "git add -A", "git -c user.email=a@b -c user.name=a commit -qm init"):
        subprocess.run(cmd, shell=True, cwd=tmp_path, check=True)
    return LocalSandbox(tmp_path)


def test_editor_semantics(repo):
    ed = Editor(repo)
    assert "return 1" in ed("view", "pkg/core.py")
    assert "Multiple" not in ed("str_replace", "pkg/core.py", old_str="return 1", new_str="return 2")
    assert repo.read("pkg/core.py") == "def f():\n    return 2\n"
    assert ed("str_replace", "pkg/core.py", old_str="missing", new_str="x").startswith("ERROR")
    assert "already exists" in ed("create", "pkg/core.py", file_text="")
    ed("insert", "pkg/core.py", insert_line=0, new_str="# header")
    assert repo.read("pkg/core.py").startswith("# header")
    ed("undo_edit", "pkg/core.py")
    ed("undo_edit", "pkg/core.py")
    assert repo.read("pkg/core.py") == "def f():\n    return 1\n"
    assert "files and directories" in ed("view", "pkg")


def test_parse_tool_calls():
    txt = 'I will look.<tool_call>{"name": "execute_bash", "arguments": {"command": "ls"}}</tool_call>'
    assert codeact.parse_tool_calls(txt) == [("execute_bash", {"command": "ls"})]
    assert codeact.parse_tool_calls("<tool_call>{bad json}</tool_call>") == []


class ToyRenderer:
    def __init__(self):
        self.tok = ToyTokenizer()
        self.eos_id = EOS

    def initial(self, messages, tools):
        return self.tok.encode(messages[-1]["content"][:40])

    def observation(self, text):
        return self.tok.encode("<obs>" + text[:30] + "</obs>")

    def decode(self, ids):
        return self.tok.decode(ids)


def _scripted(turns):
    tok = ToyTokenizer()
    queue = [tok.encode(t) + [EOS] for t in turns]
    seen, pending = [], []

    async def generate(context, max_tokens):
        """Continues the current scripted turn across chunk boundaries, like a real server would."""
        seen.append((list(context), max_tokens))
        if not pending:
            pending.extend(queue.pop(0) if queue else [EOS])
        out = pending[:max_tokens]
        del pending[:max_tokens]
        return out

    return generate, seen


def _call(name, **args):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def test_episode_edits_masks_and_patch(repo):
    gen, _ = _scripted([_call("execute_bash", command="cat pkg/core.py"),
                        _call("str_replace_editor", command="str_replace", path="pkg/core.py",
                              old_str="return 1", new_str="return 2"),
                        _call("finish", message="done")])
    res = asyncio.run(codeact.run_episode(gen, ToyRenderer(), repo, "f should return 2", base_commit=None))
    assert res.finished and res.turns == 3 and not res.timed_out
    assert "-    return 1" in res.patch and "+    return 2" in res.patch
    assert len(res.response_ids) == len(res.response_mask)
    assert 0 in res.response_mask and res.response_mask[0] == 1
    assert [p for _, p in res.edit_events] == ["pkg/core.py"]


def test_turn_cap_and_generation_follows_rolling_plan(repo):
    gen, seen = _scripted([_call("execute_bash", command="echo hi")] * 10)
    res = asyncio.run(codeact.run_episode(gen, ToyRenderer(), repo, "x", base_commit=None, max_turns=4,
                                          window=160, chunk=32, reserve=8, budget=400))
    assert res.turns == 4 and not res.finished
    assert all(len(ctx) + mt + 8 <= 160 for ctx, mt in seen)   # every request fits the window + reserve


def test_patch_empty_without_edits(repo):
    assert model_patch(repo) == ""


def test_prepare_swe_row_budgets_pi_and_plan():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prep_swe", "scripts/prepare_swe.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    tok = ToyTokenizer()
    row = {"instance_id": "o__r-1", "repo": "o/r", "base_commit": "c", "problem_statement": "bug",
           "patch": "diff --git a/pkg/core.py b/pkg/core.py\n@@ -1 +1 @@\n-x\n+" + "y" * 500 + "\n",
           "test_patch": "", "FAIL_TO_PASS": '["t::a"]', "PASS_TO_PASS": "[]"}
    render = lambda msgs: tok.encode(msgs[-1]["content"][:20])  # noqa: E731
    out = prep.build_row(row, render, render, tok.encode, tok.decode,
                         ["gold_patch", "test_manifest", "next_file", "call_graph"], max_tokens=64, call_graph="g")
    ex = out["extra_info"]
    assert set(ex["pi_prefix_ids"]) == {"gold_patch", "test_manifest", "call_graph"}
    assert len(ex["pi_prefix_ids"]["gold_patch"]) <= 64 + 40                 # truncated to the budget + markers
    assert ex["plan_prompt_len"] >= len(ex["teacher_prompt_ids"]) + max(len(v) for v in ex["pi_prefix_ids"].values())
    assert ex["instance"]["instance_id"] == "o__r-1"


def test_train_env_helpers():
    from envs.swe import train_env

    assert train_env.infer_python("python_requires='>=3.8'") == "3.8"
    classifiers = "Programming Language :: Python :: 3.10\nProgramming Language :: Python :: 3.11"
    assert train_env.infer_python("", classifiers) == "3.11"
    assert train_env.infer_python() == train_env.DEFAULT_PYTHON
    log = "PASSED t.py::a\nFAILED t.py::b - x\nPASSED t.py::c\n"
    before = train_env.parse_pytest_ra(log)
    after = {"t.py::a": "PASSED", "t.py::b": "PASSED", "t.py::c": "FAILED", "t.py::d": "PASSED"}
    m = train_env.derive_manifest(before, after)
    assert m == {"FAIL_TO_PASS": ["t.py::b", "t.py::d"], "PASS_TO_PASS": ["t.py::a"]}
    assert "git checkout abc" in train_env.dockerfile("o/r", "abc", "3.9")
