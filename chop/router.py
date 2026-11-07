"""Two-gate supervision router (Sec. 4.1, Algorithm 1).

Gate 1 (granularity): rho_t > tau_rho(t) -> token branch (trusted), else segment branch.
Gate 2 (PI):          kappa_t >= kappa_cut -> PI-conditional teacher.

Crossing them gives the four cells used in the paper:
  A = trusted, no PI     B = trusted, PI     C = untrusted, no PI     D = untrusted, PI
The cells label the loss selected at a state; they are not separate training branches.
All ablation routes of Tables ``tab:ablation`` and ``tab:stress_controls_full`` are modes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from chop.config import RouterConfig
from chop.quantile import StreamingQuantile


class Cell(IntEnum):
    IGNORE = -1
    A = 0
    B = 1
    C = 2
    D = 3


@dataclass
class Route:
    cells: torch.Tensor        # long [N], Cell values, -1 for ineligible states
    trusted: torch.Tensor      # bool [N]
    pi_gate: torch.Tensor      # bool [N]
    tau_rho: torch.Tensor      # float [N], NaN during warm-up / for ineligible states
    kappa_cut: torch.Tensor    # float [N]
    in_stratum: torch.Tensor   # bool [N], True unless a rho stratum restriction is configured

    @property
    def eligible(self) -> torch.Tensor:
        return self.cells >= 0

    @property
    def segment(self) -> torch.Tensor:
        return self.eligible & ~self.trusted

    def fractions(self) -> dict[str, float]:
        n = max(int(self.eligible.sum()), 1)
        out = {c.name: float((self.cells == int(c)).sum()) / n for c in (Cell.A, Cell.B, Cell.C, Cell.D)}
        out["segment"] = out["C"] + out["D"]
        out["pi"] = out["B"] + out["D"]
        return out


class CellRouter:
    def __init__(self, cfg: RouterConfig, seed: int = 0):
        self.cfg = cfg
        self.rho_q = StreamingQuantile(cfg.window, cfg.refresh, cfg.min_history)
        self.kappa_q = StreamingQuantile(cfg.window, cfg.refresh, cfg.min_history)
        self.oracle_q = StreamingQuantile(cfg.window, cfg.refresh, cfg.min_history)
        self.gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ helpers
    def _bernoulli(self, n: int, p: float) -> torch.Tensor:
        return torch.rand(n, generator=self.gen) < p

    def _trusted(self, rho, depth_frac, oracle_rho, frozen_trusted) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        n = rho.numel()
        nan = torch.full((n,), float("nan"))
        if c.rho_gate == "quantile":
            tau = self.rho_q.thresholds_for_stream(rho, c.rho_quantile)
            return ~(rho <= tau), tau          # warm-up (NaN) -> trusted
        if c.rho_gate == "static":
            return rho > c.static_rho, torch.full((n,), c.static_rho)
        if c.rho_gate == "none":
            return torch.ones(n, dtype=torch.bool), nan
        if c.rho_gate == "all_segment":
            return torch.zeros(n, dtype=torch.bool), nan
        if c.rho_gate == "random_quota":
            return ~self._bernoulli(n, c.rho_quantile), nan
        if c.rho_gate == "depth_decile":
            if depth_frac is None:
                raise ValueError("rho_gate='depth_decile' needs depth_frac (position / trajectory length)")
            return depth_frac < 1.0 - c.rho_quantile, nan
        if c.rho_gate == "oracle":
            if oracle_rho is None:
                raise ValueError("rho_gate='oracle' needs oracle_rho (reference reliability)")
            tau = self.oracle_q.thresholds_for_stream(oracle_rho, c.rho_quantile)
            return ~(oracle_rho <= tau), tau
        if c.rho_gate == "frozen":
            if frozen_trusted is None:
                raise ValueError("rho_gate='frozen' needs frozen_trusted from a frozen OPD checkpoint")
            return frozen_trusted.bool(), nan
        raise ValueError(f"unknown rho_gate {c.rho_gate!r}")

    def _pi_gate(self, kappa) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        n = kappa.numel()
        nan = torch.full((n,), float("nan"))
        if c.kappa_gate == "none":
            return torch.zeros(n, dtype=torch.bool), nan
        if c.kappa_gate == "always":
            return torch.ones(n, dtype=torch.bool), nan
        if c.kappa_gate == "random":
            return self._bernoulli(n, c.random_pi_rate), nan
        if c.kappa_gate == "static":
            return kappa >= c.static_kappa, torch.full((n,), c.static_kappa)
        if c.kappa_gate in ("quantile", "random_rate_matched"):
            cut = self.kappa_q.thresholds_for_stream(kappa, c.kappa_quantile)
            gate = kappa >= cut                 # NaN comparison is False: no PI during warm-up
            if c.kappa_gate == "random_rate_matched":
                # true PI at random states, activation count matched exactly to the kappa gate
                gate = gate[torch.randperm(n, generator=self.gen)]
            return gate, cut
        raise ValueError(f"unknown kappa_gate {c.kappa_gate!r}")

    # --------------------------------------------------------------------- API
    def route(
        self,
        rho: torch.Tensor,
        kappa: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
        depth_frac: torch.Tensor | None = None,
        oracle_rho: torch.Tensor | None = None,
        frozen_trusted: torch.Tensor | None = None,
    ) -> Route:
        """Route a block of states given in stream order (1-D tensors of equal length)."""
        rho = rho.detach().float().flatten().cpu()
        n = rho.numel()
        elig = torch.ones(n, dtype=torch.bool) if eligible is None else eligible.flatten().bool().cpu()
        kap = None if kappa is None else kappa.detach().float().flatten().cpu()

        def sub(x):
            return None if x is None else x.flatten().cpu()[elig]

        trusted_e, tau_e = self._trusted(rho[elig], sub(depth_frac), sub(oracle_rho), sub(frozen_trusted))
        if kap is None:
            m = int(elig.sum())
            gate_e, cut_e = torch.zeros(m, dtype=torch.bool), torch.full((m,), float("nan"))
        else:
            gate_e, cut_e = self._pi_gate(kap[elig])
        if self.cfg.three_cell:
            gate_e = gate_e & ~trusted_e
        if self.cfg.cell_a_uses_pi and kap is not None:
            gate_e = gate_e | trusted_e

        stratum_e = torch.ones_like(trusted_e)
        if self.cfg.stratum is not None:
            lo, hi = self.cfg.stratum
            r = rho[elig]
            t_lo = self.rho_q.thresholds_for_stream(r, lo) if lo > 0 else torch.full_like(r, -float("inf"))
            t_hi = self.rho_q.thresholds_for_stream(r, hi) if hi < 1 else torch.full_like(r, float("inf"))
            stratum_e = (r >= t_lo) & (r < t_hi)

        trusted = torch.ones(n, dtype=torch.bool)
        gate = torch.zeros(n, dtype=torch.bool)
        tau = torch.full((n,), float("nan"))
        cut = torch.full((n,), float("nan"))
        in_stratum = torch.zeros(n, dtype=torch.bool)
        trusted[elig], gate[elig], tau[elig], cut[elig], in_stratum[elig] = trusted_e, gate_e, tau_e, cut_e, stratum_e
        cells = 2 * (~trusted).long() + gate.long()
        cells[~elig] = int(Cell.IGNORE)
        return Route(cells, trusted, gate, tau, cut, in_stratum)

    def update(self, rho: torch.Tensor, kappa: torch.Tensor | None = None,
               eligible: torch.Tensor | None = None, oracle_rho: torch.Tensor | None = None) -> None:
        """Push this step's eligible states into the streaming buffers (end of Algorithm 1)."""
        rho = rho.detach().flatten().cpu()
        elig = torch.ones_like(rho, dtype=torch.bool) if eligible is None else eligible.flatten().bool().cpu()
        self.rho_q.update(rho[elig])
        if kappa is not None:
            self.kappa_q.update(kappa.detach().flatten().cpu()[elig])
        if oracle_rho is not None:
            self.oracle_q.update(oracle_rho.detach().flatten().cpu()[elig])

    def state_dict(self) -> dict:
        return {"rho": self.rho_q.state_dict(), "kappa": self.kappa_q.state_dict(),
                "oracle": self.oracle_q.state_dict(), "gen": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.rho_q.load_state_dict(state["rho"])
        self.kappa_q.load_state_dict(state["kappa"])
        self.oracle_q.load_state_dict(state["oracle"])
        self.gen.set_state(state["gen"])
