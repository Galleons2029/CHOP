"""Single-process emulation of the CHOP worker group (tests; single-GPU runs without Ray).

It follows the DP_COMPUTE calling convention of :mod:`chop_verl.worker` exactly — every method takes
per-rank lists — so :func:`chop_verl.driver.chop_step` can be exercised end to end without verl.
Gradients from all emulated ranks accumulate into the one model before a single optimizer step,
which is what FSDP's gradient reduction produces across real ranks.
"""

from __future__ import annotations

import torch

from chop.config import MethodConfig
from chop.segment import effective_rank
from chop_verl import compute as C


class LocalWorkers:
    def __init__(self, models: C.ChopModels, cfg: MethodConfig, optimizer: torch.optim.Optimizer,
                 scheduler=None, grad_clip: float | None = 1.0, seed: int = 0):
        self.m, self.cfg, self.opt, self.sched, self.grad_clip = models, cfg, optimizer, scheduler, grad_clip
        self.cache: dict[str, dict] = {}
        self.seed = seed

    def _gen(self, step: int, phase: int) -> torch.Generator:
        return torch.Generator().manual_seed(hash((self.seed, step, phase)) % (2 ** 63))

    def chop_score(self, shards, need_pi):
        self.cache = {}
        out = []
        for payloads, np_ in zip(shards, need_pi, strict=True):
            rank_out = []
            for p in payloads:
                st, rho, tgt = C.score_traj(self.m, p, self.cfg, np_)
                self.cache[p.uid] = {"payload": p, "targets": tgt, "segs": []}
                rank_out.append({"uid": p.uid, "rho": rho, "entropy": st.entropy, "tv": st.tv, "kappa": st.kappa,
                                 "student_entropy": st.student_entropy})
            out.append(rank_out)
        return out

    def chop_sample(self, specs, steps):
        out = []
        for sp, step in zip(specs, steps, strict=True):
            g = self._gen(step, 1)
            n = 0
            for uid, s in sp.items():
                self.cache[uid]["segs"] = C.sample_segments(self.m, self.cache[uid]["payload"], s, self.cfg, g)
                n += len(s)
            out.append({"anchors": n})
        return out

    def chop_costs(self, uids, steps):
        out = []
        for us, step in zip(uids, steps, strict=True):
            g = self._gen(step, 2)
            per, embs = {}, []
            for uid in us:
                segs = self.cache[uid]["segs"]
                if segs:
                    costs, dist, _, s_emb = C.segment_costs(self.m, self.cache[uid]["payload"], segs, self.cfg, g)
                    per[uid] = {"costs": costs, "dist": dist}
                    embs.append(s_emb.reshape(-1, s_emb.shape[-1]))
            out.append({"per_uid": per, "effective_rank": effective_rank(torch.cat(embs)) if embs else float("nan")})
        return out

    def chop_update(self, routes, adv, normalizer, extra):
        self.opt.zero_grad()
        outs = []
        for r, a, n, e in zip(routes, adv, normalizer, extra, strict=True):
            m = {"loss": 0.0, "token_states": 0.0, "segment_anchors": 0.0}
            for uid, route in r.items():
                c = self.cache[uid]
                loss, mm = C.update_traj(self.m, c["payload"], route, c["segs"], a.get(uid), c["targets"], self.cfg,
                                         n, extra=e.get(uid))
                m["loss"] += loss
                for k in ("token_states", "segment_anchors"):
                    m[k] += mm.get(k, 0.0)
            outs.append(m)
        params = [p for g in self.opt.param_groups for p in g["params"]]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.grad_clip or float("inf")))
        self.opt.step()
        lr = self.opt.param_groups[0]["lr"]
        if self.sched is not None:
            self.sched.step()
        self.cache = {}
        for o in outs:
            o.update({"grad_norm": grad_norm, "lr": lr})
        return outs
