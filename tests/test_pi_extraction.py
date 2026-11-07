import textwrap

from envs.swe import categories, leakage
from pi_extraction import sources
from pi_extraction.call_graph import build_call_graph, referenced_modules
from pi_extraction.diffs import hunks, is_test_path, modified_files

PATCH = textwrap.dedent("""\
    diff --git a/pkg/core.py b/pkg/core.py
    --- a/pkg/core.py
    +++ b/pkg/core.py
    @@ -10,3 +10,4 @@ def f():
    -    return 1
    +    return 2
    diff --git a/pkg/util.py b/pkg/util.py
    --- a/pkg/util.py
    +++ b/pkg/util.py
    @@ -3 +3 @@
    -x = 1
    +x = 2
    diff --git a/tests/test_core.py b/tests/test_core.py
    --- a/tests/test_core.py
    +++ b/tests/test_core.py
    @@ -1 +1 @@
    -a
    +b
    """)


def test_diff_parsing():
    assert modified_files(PATCH) == ["pkg/core.py", "pkg/util.py", "tests/test_core.py"]
    hs = hunks(PATCH)
    assert hs[0].path == "pkg/core.py" and hs[0].old_range == (10, 13)
    assert hs[1].old_range == (3, 4)
    assert is_test_path("tests/test_core.py") and is_test_path("pkg/conftest.py") and not is_test_path("pkg/core.py")


def test_gold_patch_excludes_tests():
    g = sources.gold_patch(PATCH)
    assert "pkg/core.py" in g and "pkg/util.py" in g and "tests/test_core.py" not in g


def test_test_manifest_names_only():
    m = sources.test_manifest('["t::a", "t::b"]', ["t::c"])
    assert m.splitlines() == ["FAIL_TO_PASS:", "  t::a", "  t::b", "PASS_TO_PASS:", "  t::c"]


def test_next_file_follows_edit_log():
    files = {"pkg/core.py": '"""Core module."""\n', "pkg/util.py": "x = 1\n"}
    assert sources.next_file(PATCH, [], files.get) == "pkg/core.py\nCore module."
    assert sources.next_file(PATCH, ["pkg/core.py"], files.get) == "pkg/util.py"
    assert "all gold-modified" in sources.next_file(PATCH, ["pkg/core.py", "pkg/util.py"], files.get)


def test_call_graph_from_issue_module(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("from pkg.util import helper\n\ndef run():\n    return helper(1)\n\n"
                                              "class A:\n    def m(self):\n        return run()\n")
    (tmp_path / "pkg" / "util.py").write_text("def helper(x):\n    return inner(x)\n\ndef inner(x):\n    return x\n")
    (tmp_path / "tests" / "test_core.py").write_text("def test_run():\n    run()\n")
    assert referenced_modules("bug in pkg/core.py when calling run", ["pkg/core.py", "pkg/util.py"]) == ["pkg/core.py"]
    g = build_call_graph(tmp_path, "bug in pkg/core.py when calling run")
    assert g.splitlines()[0] == "modules: pkg/core.py"
    assert "pkg/core.py:run (line 3) -> pkg/util.py:helper" in g
    assert "pkg/core.py:A.m (line 7) -> pkg/core.py:run" in g
    assert "pkg/util.py:helper (line 1) -> pkg/util.py:inner" in g        # reached at depth 1
    assert "tests/" not in g


def test_leakage_rules():
    verified = [{"instance_id": "v1", "repo": "o/r", "base_commit": "c1", "patch": PATCH}]
    train = [
        {"instance_id": "t1", "repo": "o/r", "base_commit": "c1", "patch": ""},
        {"instance_id": "t2", "repo": "x/y", "base_commit": "c2", "patch": PATCH},
        {"instance_id": "t3", "repo": "o/r", "base_commit": "c3", "patch": ""},
        {"instance_id": "t4", "repo": "x/y", "base_commit": "c4",
         "patch": "diff --git a/setup.cfg b/setup.cfg\n"},
    ]
    kept, rep = leakage.apply(train, verified)
    assert kept == ["t3", "t4"] and rep["removed"] == 2
    kept_s, _ = leakage.apply(train, verified, strict=True)
    assert kept_s == ["t4"]


def test_categories_are_deterministic():
    row = {"patch": PATCH, "FAIL_TO_PASS": "[]", "problem_statement": "crash"}
    assert categories.categorize(row) == "multi_file_refactor"
    assert categories.categorize({**row, "problem_statement": "wrong warning message"}) == "documentation_edge_case"
