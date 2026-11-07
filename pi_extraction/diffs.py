"""Unified-diff helpers shared by the PI extractors, the leakage filter and adapter alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.M)

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


@dataclass(frozen=True)
class Hunk:
    path: str
    old_start: int
    old_len: int

    @property
    def old_range(self) -> tuple[int, int]:
        return self.old_start, self.old_start + max(self.old_len, 1)


def modified_files(patch: str) -> list[str]:
    """Paths modified by a patch, in diff order (b-side path)."""
    return [b for _, b in _FILE_RE.findall(patch)]


def split_by_file(patch: str) -> dict[str, str]:
    parts = re.split(r"(?=^diff --git )", patch, flags=re.M)
    out = {}
    for p in parts:
        m = _FILE_RE.match(p)
        if m:
            out[m.group(2)] = p
    return out


def hunks(patch: str) -> list[Hunk]:
    out = []
    for path, body in split_by_file(patch).items():
        for m in _HUNK_RE.finditer(body):
            out.append(Hunk(path, int(m.group(1)), int(m.group(2) or 1)))
    return out


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path))
