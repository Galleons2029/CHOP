"""Training-set leakage filters against SWE-bench Verified (App. C.6).

Rules, each reported separately so the removed counts can be audited:
  commit      same base commit as a Verified instance
  path        the gold patch modifies a (non-generic) file path that some Verified gold patch modifies
  path_any    as ``path`` but counting generic files (setup.cfg, setup.py, docs/conf.py, ...)
  same_repo   same repository as a Verified instance (the paper's "stricter" filter)

The default filter is commit + path; the strict filter adds same_repo.  On the public
princeton-nlp/SWE-bench train split the repositories are disjoint from Verified, so same_repo removes
nothing there; see data/manifests/leakage_report.json for the counts actually obtained.
"""

from __future__ import annotations

from collections import Counter

from pi_extraction.diffs import modified_files

GENERIC_FILES = frozenset({
    "setup.py", "setup.cfg", "pyproject.toml", "tox.ini", "README.md", "README.rst", "CHANGELOG.md",
    "CHANGES.rst", ".gitignore", "requirements.txt", "Makefile", "docs/conf.py", "conftest.py", "MANIFEST.in",
})


def flags(train: list[dict], verified: list[dict]) -> dict[str, dict[str, bool]]:
    v_commits = {r["base_commit"] for r in verified}
    v_repos = {r["repo"] for r in verified}
    v_paths: set[str] = set()
    for r in verified:
        v_paths |= set(modified_files(r["patch"]))
    out = {}
    for r in train:
        files = set(modified_files(r["patch"]))
        shared = files & v_paths
        out[r["instance_id"]] = {
            "commit": r["base_commit"] in v_commits,
            "path": bool(shared - GENERIC_FILES),
            "path_any": bool(shared),
            "same_repo": r["repo"] in v_repos,
        }
    return out


def apply(train: list[dict], verified: list[dict], strict: bool = False) -> tuple[list[str], dict]:
    """Return (kept instance ids, report)."""
    f = flags(train, verified)
    rules = ("commit", "path", "same_repo") if strict else ("commit", "path")
    kept = [iid for iid, fl in f.items() if not any(fl[k] for k in rules)]
    counts = Counter(k for fl in f.values() for k, v in fl.items() if v)
    report = {"train_instances": len(train), "verified_instances": len(verified), "rule_hits": dict(counts),
              "rules_applied": list(rules), "kept": len(kept), "removed": len(train) - len(kept)}
    return kept, report
