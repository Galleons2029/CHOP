"""Reset-rollout intervention (App. reset_rollout): is the OPD gradient informative at low-rho prefixes?

For each selected state the prefix is taken in one of three ways:
  original             the student's own prefix
  teacher_continuation a teacher-generated prefix of the same depth
  random               shuffled student fragments from other instances, of the same depth

At that prefix we measure the sampled-token OPD gradient norm on the parameter subset. We also measure
the KL reduction KL(p||q) before minus after one SGD step on the same prefix; the model is restored
afterwards. The Pearson correlation between the two, over the states of a condition, is the reported
statistic (bootstrap CI over 5,000 resamples).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

from analysis.gradients import last_logprobs, param_subset


def kl_at(student, teacher, context: list[int]) -> float:
    with torch.no_grad():
        logp = last_logprobs(student, context)
        logq = last_logprobs(teacher, context)
        return float((logp.exp() * (logp - logq)).sum())


def grad_norm_and_kl_reduction(student, teacher, context: list[int], lr: float = 1e-3, subset: str = "last_block",
                               generator: torch.Generator | None = None) -> tuple[float, float]:
    params = param_subset(student, subset)
    logp = last_logprobs(student, context)
    with torch.no_grad():
        logq = last_logprobs(teacher, context)
        y = torch.multinomial(logp.exp().detach(), 1, generator=generator).item()
    loss = (logp[y] - logq[y]).detach() * logp[y]
    grads = torch.autograd.grad(loss, params)
    norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))
    before = kl_at(student, teacher, context)
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, g in zip(params, grads, strict=True):
            p -= lr * g
    after = kl_at(student, teacher, context)
    with torch.no_grad():
        for p, b in zip(params, backup, strict=True):
            p.copy_(b)
    return norm, before - after


def pearson_ci(x, y, n_resamples: int = 5000, seed: int = 0) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    r = float(sps.pearsonr(x, y).statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    boots = [sps.pearsonr(x[i], y[i]).statistic for i in idx if np.std(x[i]) > 0 and np.std(y[i]) > 0]
    return {"r": r, "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "n": int(len(x))}


def random_prefix(fragments: list[list[int]], depth: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    while len(out) < depth:
        out += fragments[int(rng.integers(len(fragments)))]
    return out[:depth]


def run(student, teacher, conditions: dict[str, list[list[int]]], lr: float = 1e-3, out: str | Path | None = None,
        seed: int = 0) -> dict:
    """conditions: name -> list of contexts. Returns per-condition points and Pearson r with CI."""
    g = torch.Generator().manual_seed(seed)
    res = {"conditions": {}, "pearson": {}}
    for name, contexts in conditions.items():
        pts = [grad_norm_and_kl_reduction(student, teacher, c, lr, generator=g) for c in contexts]
        gn, kr = [p[0] for p in pts], [p[1] for p in pts]
        res["conditions"][name] = {"grad_norm": gn, "kl_reduction": kr}
        res["pearson"][name] = pearson_ci(gn, kr, seed=seed) if len(pts) > 2 else {"r": float("nan")}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res))
    return res
