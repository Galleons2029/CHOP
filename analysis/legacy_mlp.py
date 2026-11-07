"""Legacy criticality predictor (App. criticality_predictor): a 2-layer MLP over the student's last hidden
state regressing the joint-TV kappa, used only as a latency ablation (tab:phi_refresh)."""

from __future__ import annotations

import torch


class KappaMLP(torch.nn.Module):
    def __init__(self, d: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float()).squeeze(-1)


def fit(h: torch.Tensor, kappa: torch.Tensor, epochs: int = 50, lr: float = 1e-3, seed: int = 0) -> KappaMLP:
    torch.manual_seed(seed)
    m = KappaMLP(h.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m(h), kappa.float())
        loss.backward()
        opt.step()
    return m


def route_stability(pred: torch.Tensor, kappa: torch.Tensor, quantile: float = 0.7) -> float:
    """Agreement of the predicted PI gate with the joint-TV gate at the same upper-tail quantile."""
    return float(((pred >= torch.quantile(pred, quantile)) == (kappa >= torch.quantile(kappa, quantile))).float()
                 .mean())
