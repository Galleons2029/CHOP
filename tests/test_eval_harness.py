import asyncio
import subprocess

from eval.backends import HFBackend
from eval.math_eval import evaluate, generate_capped


def test_capped_generation_follows_plan_and_stops():
    from chop.tiny import tiny_qwen3

    m = tiny_qwen3(vocab_size=128)
    calls = []

    class Spy(HFBackend):
        def generate_batch(self, contexts, max_tokens, temperature, top_p, seeds):
            calls.extend(zip([len(c) for c in contexts], max_tokens, strict=True))
            return super().generate_batch(contexts, max_tokens, temperature, top_p, seeds)

    outs = generate_capped(Spy(m, eos_id=None), [[5, 6, 7], [8, 9]], k=2, cap=40, window=24, chunk=8,
                           temperature=1.0, top_p=1.0, eos_id=None)
    assert [len(s) for p in outs for s in p] == [40, 40, 40, 40]
    assert all(ctx + mt <= 24 for ctx, mt in calls)


def test_math_evaluate_toy():
    from chop.tiny import tiny_qwen3
    from envs.toy import ToyArithmetic, ToyTokenizer

    tok, task = ToyTokenizer(), ToyArithmetic(n_eval=3)
    task.eval_instances = lambda split="eval": ToyArithmetic(n_eval=3)._eval
    records, summary = evaluate(HFBackend(tiny_qwen3(vocab_size=128), tok.eos_token_id), task, tok, ["eval"], k=2,
                                cap=16, window=64, chunk=8, temperature=1.0, top_p=1.0, seed=0)
    assert len(records) == 6 and 0.0 <= summary["eval"]["avg@2"] <= 100.0


def test_swe_eval_run_all_with_local_sandbox(tmp_path):
    from envs.swe.sandbox import LocalSandbox
    from eval.swe_eval import run_all
    from tests.test_swe_env import ToyRenderer, _call, _scripted

    (tmp_path / "a.py").write_text("x = 1\n")
    for cmd in ("git init -q", "git add -A", "git -c user.email=a@b -c user.name=a commit -qm i"):
        subprocess.run(cmd, shell=True, cwd=tmp_path, check=True)
    gen, _ = _scripted([_call("str_replace_editor", command="str_replace", path="a.py", old_str="x = 1",
                              new_str="x = 2"), _call("finish")])
    eps = asyncio.run(run_all([{"instance_id": "i1", "problem_statement": "p"}], lambda _i: gen, ToyRenderer(),
                              lambda _i: LocalSandbox(tmp_path), max_turns=5, time_limit=60, window=4096,
                              chunk=512, budget=2048))
    assert eps[0]["finished"] and "+x = 2" in eps[0]["model_patch"]
