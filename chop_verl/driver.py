"""Driver-side CHOP logic (single process, no verl imports): routing, anchors, advantages, sharding.

Routing runs on the driver over the whole batch in a fixed stream order (trajectory order of the
batch), so the streaming quantiles are identical to a single-process run and no distributed
synchronisation of the router is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from chop.config import MethodConfig
from chop.monitors import CollapseGuard
from chop.router import CellRouter, Route
from chop.segment import advantages, select_anchor_windows
from chop.truncation import rolling_chunks
from chop_verl.compute import AnchorSpec, TrajPayload


def build_payload(uid: str, prompt_ids: list[int], response_ids: list[int], response_mask: list[int], reward: float,
                  extra: dict, cfg: MethodConfig, native_context: int, rollout_chunk: int) -> TrajPayload:
    """TrajPayload from one rollout row; ``extra`` is the dataset's extra_info (see data/prepare_*)."""
    t_prompt = list(extra.get("teacher_prompt_ids") or prompt_ids)
    pi_prefix = {k: list(v) for k, v in (extra.get("pi_prefix_ids") or {}).items() if k in cfg.pi.sources}
    per_chunk = {k: [list(x) for x in v] for k, v in (extra.get("pi_prefix_ids_per_chunk") or {}).items()
                 if k in cfg.pi.sources}
    missing = [s for s in cfg.pi.sources if s not in pi_prefix and s not in per_chunk]
    if missing:
        raise KeyError(f"instance {uid} lacks PI prefixes for {missing}")
    longest = max([len(v) for v in pi_prefix.values()] + [len(x) for v in per_chunk.values() for x in v] + [0])
    needed = max(len(prompt_ids), len(t_prompt) + longest)
    # the rollout used max(prompt, plan_prompt_len); both sides must use the identical plan
    plan_len = max(len(prompt_ids), int(extra.get("plan_prompt_len", 0))) if "plan_prompt_len" in extra else needed
    if needed > plan_len:
        raise ValueError(f"instance {uid}: teacher context ({needed}) exceeds the planned prompt length ({plan_len})")
    chunks = rolling_chunks(plan_len, max(len(response_ids), 1), native_context, rollout_chunk,
                            reserve=cfg.segment.segment_length)
    return TrajPayload(uid, list(prompt_ids), t_prompt, list(response_ids), list(response_mask), float(reward),
                       pi_prefix, chunks, per_chunk)


def shard(payloads: list[TrajPayload], world: int) -> list[list[TrajPayload]]:
    """Greedy longest-first assignment balancing total tokens across ranks."""
    order = sorted(range(len(payloads)), key=lambda i: -len(payloads[i].response))
    loads = [0] * world
    out: list[list[TrajPayload]] = [[] for _ in range(world)]
    for i in order:
        r = loads.index(min(loads))
        out[r].append(payloads[i])
        loads[r] += len(payloads[i].response) + len(payloads[i].s_prompt)
    return out


@dataclass
class RoutedBatch:
    routes: dict[str, Route]                 # per uid, over eligible states
    specs: dict[str, list[AnchorSpec]]       # per uid
    fractions: dict[str, float]


def route_batch(router: CellRouter, payloads: list[TrajPayload], scores: dict[str, dict], cfg: MethodConfig,
                generator: torch.Generator | None = None) -> RoutedBatch:
    """Route all eligible states of the batch (stream order = payload order) and choose anchors."""
    rho = torch.cat([scores[p.uid]["rho"] for p in payloads])
    kap = None
    if scores[payloads[0].uid]["kappa"] is not None:
        kap = torch.cat([scores[p.uid]["kappa"] for p in payloads])
    depth = torch.cat([_depth(p) for p in payloads])
    full = router.route(rho, kap, depth_frac=depth)
    router.update(rho, kap)
    routes, specs, i = {}, {}, 0
    use_segments = cfg.router.rho_gate != "none" and cfg.name not in ("reopold", "grpo")
    for p in payloads:
        n = int(p.eligible.sum())
        r = Route(full.cells[i:i + n], full.trusted[i:i + n], full.pi_gate[i:i + n], full.tau_rho[i:i + n],
                  full.kappa_cut[i:i + n], full.in_stratum[i:i + n])
        routes[p.uid] = r
        specs[p.uid] = anchor_specs(p, r, cfg, generator) if use_segments else []
        i += n
    return RoutedBatch(routes, specs, full.fractions())


def _depth(p: TrajPayload) -> torch.Tensor:
    pos = torch.nonzero(p.eligible).flatten().float()
    return pos / max(len(p.response), 1)


def anchor_specs(p: TrajPayload, r: Route, cfg: MethodConfig, generator: torch.Generator | None = None
                 ) -> list[AnchorSpec]:
    s = cfg.segment
    elig_pos = torch.nonzero(p.eligible).flatten()
    seg_mask = torch.zeros(len(p.response), dtype=torch.bool)
    seg_mask[elig_pos[r.segment & r.in_stratum]] = True
    pos, w = select_anchor_windows(seg_mask, s.segment_length, s.max_windows_per_traj, generator)
    state_of_pos = {int(q): i for i, q in enumerate(elig_pos.tolist())}
    return [AnchorSpec(int(a), bool(r.pi_gate[state_of_pos[int(a)]]), float(x))
            for a, x in zip(pos.tolist(), w.tolist(), strict=True)]


def revert_segments(routed: RoutedBatch) -> None:
    """Collapse fallback: Cells C/D revert to Cell A (token KL, no PI) and anchors are dropped."""
    for uid, r in routed.routes.items():
        seg = r.segment.clone()
        r.trusted[seg] = True
        r.pi_gate[seg] = False
        r.cells[seg] = 0
        routed.specs[uid] = []


def batch_advantages(costs: dict[str, torch.Tensor], cfg: MethodConfig) -> dict[str, torch.Tensor]:
    """Centre/normalise per-sample costs over the whole batch, then split back per uid."""
    if not costs:
        return {}
    uids = list(costs)
    cat = torch.cat([costs[u] for u in uids])
    adv = advantages(cat, cfg.segment.estimator)
    out, i = {}, 0
    for u in uids:
        n = costs[u].shape[0]
        out[u] = adv[i:i + n]
        i += n
    return out


def rl_extras(payloads: list[TrajPayload], scores: dict[str, dict], group_of: dict[str, str], cfg: MethodConfig,
              step: int) -> dict[str, dict]:
    """Per-trajectory extras for REOPOLD (entropy threshold, step) and GRPO (group advantage)."""
    if cfg.name == "reopold":
        h = torch.cat([scores[p.uid]["student_entropy"] for p in payloads])
        tau = float(torch.quantile(h.float(), 1 - cfg.reopold_entropy_top)) if h.numel() else math.inf
        return {p.uid: {"step": step, "entropy_tau": tau} for p in payloads}
    if cfg.name == "grpo":
        groups: dict[str, list[TrajPayload]] = {}
        for p in payloads:
            groups.setdefault(group_of[p.uid], []).append(p)
        out = {}
        for members in groups.values():
            r = torch.tensor([m.reward for m in members])
            adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            out.update({m.uid: {"advantage": float(a)} for m, a in zip(members, adv.tolist(), strict=True)})
        return out
    return {}


def observe_collapse(guard: CollapseGuard, ranks: list[float], counts: list[int]) -> bool:
    valid = [(r, c) for r, c in zip(ranks, counts, strict=True) if c and not math.isnan(r)]
    if not valid:
        return guard.active
    mean_rank = sum(r * c for r, c in valid) / sum(c for _, c in valid)
    return guard.observe(mean_rank)


def chop_step(wg, router: CellRouter, guard: CollapseGuard, payloads: list[TrajPayload], cfg: MethodConfig,
              step: int, world: int, need_pi: bool, group_of: dict[str, str] | None = None,
              generator: torch.Generator | None = None) -> dict[str, float]:
    """One CHOP update over a rollout batch.

    ``wg`` exposes chop_score / chop_sample / chop_costs / chop_update with verl's DP_COMPUTE calling
    convention (every argument is a list with one entry per rank, results are per-rank lists): the verl
    worker group or :class:`chop_verl.local.LocalWorkers`.
    """
    shards = shard(payloads, world)
    score_out = wg.chop_score(shards, [need_pi] * world)
    scores = {d["uid"]: d for outs in score_out for d in outs}
    routed = route_batch(router, payloads, scores, cfg, generator)
    metrics = {f"chop/cell_{k}": v for k, v in routed.fractions.items()}
    adv: dict[str, torch.Tensor] = {}
    if any(routed.specs.values()):
        wg.chop_sample([{p.uid: routed.specs[p.uid] for p in sh} for sh in shards], [step] * world)
        cost_out = wg.chop_costs([[p.uid for p in sh] for sh in shards], [step] * world)
        ranks = [c["effective_rank"] for c in cost_out]
        counts = [len(c["per_uid"]) for c in cost_out]
        if observe_collapse(guard, ranks, counts):
            revert_segments(routed)
            metrics["chop/collapse_fallback"] = 1.0
        else:
            costs = {u: d["costs"] for c in cost_out for u, d in c["per_uid"].items()}
            adv = batch_advantages(costs, cfg)
            dist = torch.cat([d["dist"] for c in cost_out for d in c["per_uid"].values()])
            metrics["chop/segment_distance"] = float(dist.mean())
        metrics["chop/anchors"] = float(sum(len(s) for s in routed.specs.values()))
        metrics["chop/embedding_rank"] = float(sum(r * c for r, c in zip(ranks, counts, strict=True) if c)
                                               / max(sum(counts), 1))
    extras = rl_extras(payloads, scores, group_of or {p.uid: p.uid for p in payloads}, cfg, step)
    normalizer = sum(int(p.eligible.sum()) for p in payloads)
    upd = wg.chop_update([{p.uid: routed.routes[p.uid] for p in sh} for sh in shards],
                         [{p.uid: adv[p.uid] for p in sh if p.uid in adv} for sh in shards],
                         [normalizer] * world,
                         [{p.uid: extras[p.uid] for p in sh if p.uid in extras} for sh in shards])
    metrics["chop/loss"] = sum(u["loss"] for u in upd)
    metrics["actor/grad_norm"] = upd[0]["grad_norm"]
    metrics["actor/lr"] = upd[0]["lr"]
    metrics["chop/token_states"] = sum(u["token_states"] for u in upd)
    metrics["chop/reward"] = sum(p.reward for p in payloads) / max(len(payloads), 1)
    metrics["chop/response_length"] = sum(len(p.response) for p in payloads) / max(len(payloads), 1)
    return metrics
