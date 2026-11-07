"""Per-component overhead profile (tab:compute): wall-clock of each CHOP phase relative to plain OPD."""

from __future__ import annotations

import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.totals: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

    def relative_to(self, reference: str) -> dict[str, float]:
        ref = self.totals.get(reference, 0.0)
        return {k: (100.0 * v / ref if ref else float("nan")) for k, v in self.totals.items() if k != reference}
