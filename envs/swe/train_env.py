"""Best-effort execution environments for the SWE-bench *training* split (App. C.6, rule 3).

SWE-bench publishes environment specifications and images only for its evaluation repositories. The
35 training repositories need reconstructed environments: to obtain the p2 test manifests, which are
empty in the training split, and to compute outcome rewards during CHOP training. This module
generates one Dockerfile per (repo, base commit), using a Python version inferred from the package
metadata and an editable install with test extras. It then derives FAIL_TO_PASS / PASS_TO_PASS by
running the instance's test files before and after the gold patch.

Reconstruction does not always succeed. ``scripts/build_train_envs.py`` records the build and test
status of every instance, and only instances with a working environment enter training (the
--train-image-map of prepare_swe.py). Report that coverage with any result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pi_extraction.diffs import modified_files

DEFAULT_PYTHON = "3.9"


def infer_python(setup_text: str = "", pyproject_text: str = "") -> str:
    text = setup_text + "\n" + pyproject_text
    m = re.search(r"python_requires\s*=\s*['\"]>=\s*3\.(\d+)", text) or re.search(
        r"requires-python\s*=\s*['\"]>=\s*3\.(\d+)", text)
    if m:
        return f"3.{max(int(m.group(1)), 7)}"
    versions = [int(v) for v in re.findall(r"Programming Language :: Python :: 3\.(\d+)", text)]
    return f"3.{max(versions)}" if versions else DEFAULT_PYTHON


def dockerfile(repo: str, commit: str, python: str) -> str:
    url = f"https://github.com/{repo}.git"
    return "\n".join([
        f"FROM python:{python}-slim",
        "RUN apt-get update && apt-get install -y --no-install-recommends git build-essential "
        "&& rm -rf /var/lib/apt/lists/*",
        f"RUN git clone {url} /testbed && cd /testbed && git checkout {commit}",
        "WORKDIR /testbed",
        "RUN pip install --no-cache-dir -U pip setuptools wheel pytest",
        "RUN (pip install --no-cache-dir -e '.[test]' || pip install --no-cache-dir -e '.[tests]' || "
        "pip install --no-cache-dir -e '.[dev]' || pip install --no-cache-dir -e .) && "
        "(for f in requirements*.txt test*-requirements.txt; do "
        "[ -f \"$f\" ] && pip install --no-cache-dir -r \"$f\" || true; done)",
        "RUN git config --global user.email chop@local && git config --global user.name chop",
        "",
    ])


def test_files(test_patch: str) -> list[str]:
    return [f for f in modified_files(test_patch) if f.endswith(".py")]


_RESULT_RE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+(\S+)", re.M)


def parse_pytest_ra(log: str) -> dict[str, str]:
    """Test id -> status from ``pytest -rA`` short test summary lines."""
    return {tid: status for status, tid in _RESULT_RE.findall(log)}


def derive_manifest(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    """FAIL_TO_PASS: fails (or missing) before the gold patch and passes after; PASS_TO_PASS: passes in both."""
    f2p = sorted(t for t, s in after.items() if s == "PASSED" and before.get(t) != "PASSED")
    p2p = sorted(t for t, s in after.items() if s == "PASSED" and before.get(t) == "PASSED")
    return {"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}


def test_command(files: Iterable[str]) -> str:
    return "python -m pytest -rA -p no:cacheprovider " + " ".join(files)
