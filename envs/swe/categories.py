"""Issue-category assignment for the per-category breakdown (tab:swebench_perinstance).

The paper does not specify how instances were assigned to its four categories, so this is a
documented heuristic, applied to public fields only (gold patch, problem statement, test lists).
Rules are evaluated in order:
  documentation_edge_case  : the gold patch touches documentation files, or the issue is about messages,
                             warnings, docstrings, formatting or error text
  multi_file_refactor      : the gold patch modifies >= 2 non-test source files
  test_driven_implementation: >= 3 FAIL_TO_PASS tests, or the issue requests new behaviour
  single_file_bug_fix      : everything else
Counts from this rule are written to data/manifests/categories.csv. They will not reproduce the
paper's (211, 86, 124, 79) split.
"""

from __future__ import annotations

import json
import re

from pi_extraction.diffs import is_test_path, modified_files

CATEGORIES = ("test_driven_implementation", "multi_file_refactor", "single_file_bug_fix", "documentation_edge_case")
_DOC_RE = re.compile(r"\b(docstring|documentation|typo|warning message|error message|repr|formatting|deprecat)", re.I)
_FEATURE_RE = re.compile(r"\b(add support|should support|feature request|implement|would be nice|allow .* to)\b", re.I)


def categorize(row: dict) -> str:
    files = [f for f in modified_files(row["patch"]) if not is_test_path(f)]
    f2p = row.get("FAIL_TO_PASS") or []
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if any(f.startswith("doc") or f.endswith((".rst", ".md", ".txt")) for f in files) or _DOC_RE.search(
            row.get("problem_statement", "")):
        return "documentation_edge_case"
    if len(files) >= 2:
        return "multi_file_refactor"
    if len(f2p) >= 3 or _FEATURE_RE.search(row.get("problem_statement", "")):
        return "test_driven_implementation"
    return "single_file_bug_fix"
