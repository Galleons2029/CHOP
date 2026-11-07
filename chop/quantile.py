"""Streaming quantile thresholds over the preceding W eligible states (Sec. 4.1).

The paper defines tau_rho(t) as the lower-tail 30th percentile of the preceding W = 8,192
eligible states and kappa_cut as the upper-tail 70th percentile over the same window.  With
30K-token trajectories a window is shorter than one trajectory, so the threshold genuinely
depends on t.  :meth:`StreamingQuantile.thresholds_for_stream` computes, for every state of a
newly streamed block, the quantile of the W states preceding it; the value is refreshed every
``refresh`` states (block-exact), which bounds cost at N/refresh sorts of length W.

Stream order is the caller's responsibility; the distributed trainer all-gathers eligible states
and uses global batch order so that every rank sees identical thresholds.
"""

from __future__ import annotations

import torch


class StreamingQuantile:
    def __init__(self, window: int = 8192, refresh: int = 64, min_history: int = 256):
        if window <= 0 or refresh <= 0:
            raise ValueError("window and refresh must be positive")
        self.window = window
        self.refresh = refresh
        self.min_history = min_history
        self._buf = torch.empty(0, dtype=torch.float32)

    # ------------------------------------------------------------------ state
    @property
    def history(self) -> torch.Tensor:
        """Buffered values in stream order (oldest first), at most ``window`` of them."""
        return self._buf

    def __len__(self) -> int:
        return self._buf.numel()

    def update(self, values: torch.Tensor) -> None:
        """Append eligible-state values (stream order) and keep the last ``window``."""
        v = values.detach().float().flatten().cpu()
        self._buf = torch.cat([self._buf, v])[-self.window:]

    def initialize(self, values: torch.Tensor) -> None:
        """Pre-fill from the calibration corpus so that training starts without a warm-up."""
        self._buf = torch.empty(0, dtype=torch.float32)
        self.update(values)

    def state_dict(self) -> dict:
        return {"buf": self._buf.clone(), "window": self.window, "refresh": self.refresh,
                "min_history": self.min_history}

    def load_state_dict(self, state: dict) -> None:
        self.window, self.refresh, self.min_history = state["window"], state["refresh"], state["min_history"]
        self._buf = state["buf"].clone().float()

    # ------------------------------------------------------------- thresholds
    def threshold(self, q: float) -> float | None:
        """Quantile of the current buffer, or None while in warm-up."""
        if len(self) < self.min_history:
            return None
        return float(torch.quantile(self._buf, q))

    def thresholds_for_stream(self, values: torch.Tensor, q: float) -> torch.Tensor:
        """Per-state thresholds for a block of new values (does not mutate the buffer).

        Entry i is the q-quantile of the ``window`` values preceding the start of i's refresh
        block in the stream (buffer followed by ``values``).  NaN marks warm-up states whose
        history is shorter than ``min_history``.
        """
        v = values.detach().float().flatten().cpu()
        stream = torch.cat([self._buf, v])
        offset = self._buf.numel()
        out = torch.full((v.numel(),), float("nan"))
        for start in range(0, v.numel(), self.refresh):
            end_hist = offset + start
            hist = stream[max(0, end_hist - self.window):end_hist]
            if hist.numel() >= self.min_history:
                out[start:start + self.refresh] = torch.quantile(hist, q)
        return out.to(values.device)
