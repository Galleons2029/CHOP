"""p1-p3 and the math answer: pure functions of the instance (and, for p3, the agent's edit log)."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable

from pi_extraction.diffs import is_test_path, modified_files, split_by_file


def gold_patch(patch: str) -> str:
    """p1: the gold diff restricted to the (non-test) files it modifies."""
    per_file = split_by_file(patch)
    return "".join(body for path, body in per_file.items() if not is_test_path(path))


def _as_list(x) -> list[str]:
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = [x]
    return list(x or [])


def test_manifest(fail_to_pass, pass_to_pass, max_pass_to_pass: int | None = None) -> str:
    """p2: test-function names with their expected status (no test source, no patch)."""
    f2p, p2p = _as_list(fail_to_pass), _as_list(pass_to_pass)
    if max_pass_to_pass is not None:
        p2p = p2p[:max_pass_to_pass]
    lines = ["FAIL_TO_PASS:"] + [f"  {t}" for t in f2p] + ["PASS_TO_PASS:"] + [f"  {t}" for t in p2p]
    return "\n".join(lines)


def top_level_docstring(source: str) -> str:
    try:
        return (ast.get_docstring(ast.parse(source)) or "").strip()
    except SyntaxError:
        return ""


def next_file(patch: str, edited: Iterable[str], read_file: Callable[[str], str | None] | None = None,
              docstring_chars: int = 400) -> str:
    """p3: the next gold-modified (non-test) file the agent has not edited yet, with its docstring.

    Recomputed at every agent step from the gold diff and the agent's edit log.
    """
    done = set(edited)
    for path in modified_files(patch):
        if is_test_path(path) or path in done:
            continue
        doc = ""
        if read_file is not None:
            src = read_file(path)
            doc = top_level_docstring(src or "")[:docstring_chars]
        return path + (f"\n{doc}" if doc else "")
    return "(all gold-modified files have been edited)"


def final_answer(answer: str) -> str:
    return str(answer).strip()
