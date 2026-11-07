"""verl worker with the CHOP phases (FSDP2 student via verl's engine, replicated frozen teacher).

The driver (:mod:`chop_verl.trainer`) sends each rank its share of trajectories with
``Dispatch.DP_COMPUTE`` (one payload list per rank).  Per-trajectory state (sparse teacher targets,
sampled segments) stays cached on the rank between phases of the same step, keyed by uid.

Requirements: FSDP/FSDP2 strategy, Ulysses sequence parallel size 1 (each trajectory is processed
whole by one rank), and a teacher that fits replicated on every GPU (bf16 8B/14B teachers do).
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from chop.config import MethodConfig
from chop.segment import effective_rank
from chop_verl import compute as C
from chop_verl.lockstep import pad_to_global_max

try:  # verl is optional for the rest of the repository
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.engine_workers import ActorRolloutRefWorker
except ImportError:  # pragma: no cover - exercised only without verl installed
    ActorRolloutRefWorker = object

    class Dispatch:  # type: ignore[no-redef]
        DP_COMPUTE = ONE_TO_ALL = None

    def register(**_kw):  # type: ignore[no-redef]
        return lambda f: f


def method_config(raw, method_file: str | None = None) -> MethodConfig:
    """MethodConfig from the ``actor_rollout_ref.chop.method`` node, or from a resolved experiment file
    (``actor_rollout_ref.chop.method_file``, written by scripts/reproduce.py), which takes precedence."""
    base = OmegaConf.structured(MethodConfig)
    if method_file:
        exp = OmegaConf.load(method_file)
        return OmegaConf.to_object(OmegaConf.merge(base, exp.method if "method" in exp else exp))
    return OmegaConf.to_object(OmegaConf.merge(base, OmegaConf.create(OmegaConf.to_container(raw, resolve=True))))


class ChopActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ setup
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        chop = self.config.chop
        self.chop_method = method_config(chop.method, chop.get("method_file"))
        if int(self.config.actor.get("ulysses_sequence_parallel_size", 1)) != 1:
            raise ValueError("CHOP processes whole trajectories per rank: set ulysses_sequence_parallel_size=1")
        self.chop_teacher = self._load_teacher(chop)
        student = self.actor.engine.module
        tok = self._tokenizer_ids(chop)
        self.chop_models = C.ChopModels(student, self.chop_teacher, list(self.chop_method.pi.sources),
                                        attention=chop.get("attention", "flex"),
                                        block_size=int(chop.get("block_size", 128)), pad_id=tok["pad"],
                                        eos_id=tok["eos"], lockstep=True,
                                        logit_chunk=int(chop.get("logit_chunk", 1024)))
        self.chop_seed = int(chop.get("seed", 0))

    def _tokenizer_ids(self, chop) -> dict:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.config.model.path, trust_remote_code=True)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return {"pad": int(pad), "eos": int(tok.eos_token_id)}

    def _load_teacher(self, chop):
        from transformers import AutoModelForCausalLM

        dev = torch.device("cuda", torch.cuda.current_device())
        teacher = AutoModelForCausalLM.from_pretrained(chop.teacher_path, torch_dtype=torch.bfloat16,
                                                       attn_implementation=chop.get("teacher_attn", "sdpa"))
        sources = list(self.chop_method.pi.sources)
        if sources and self.chop_method.pi.interface == "lora_prefix":
            from peft import PeftModel

            root = chop.adapter_dir
            teacher = PeftModel.from_pretrained(teacher, f"{root}/{sources[0]}", adapter_name=sources[0])
            for s in sources[1:]:
                teacher.load_adapter(f"{root}/{s}", adapter_name=s)
            if self.chop_method.pi.content == "random_adapter":
                from chop.pi.controls import randomize_lora_

                g = torch.Generator().manual_seed(int(chop.get("seed", 0)))
                for s in sources:
                    randomize_lora_(teacher, s, g)
        teacher.eval().requires_grad_(False)
        return teacher.to(dev)

    def _generator(self, step: int, phase: int) -> torch.Generator:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        g = torch.Generator(device=self.chop_models.device)
        g.manual_seed(hash((self.chop_seed, step, phase, rank)) % (2 ** 63))
        return g

    def _pad(self, n_local: int, backward: bool = False) -> None:
        engine = self.actor.engine
        pad_to_global_max(n_local, lambda: C.dummy_forward(engine.module, backward),
                          group=engine.get_data_parallel_group(), device=self.chop_models.device)

    # ----------------------------------------------------------------- phases
    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def chop_score(self, payloads: list[C.TrajPayload], need_pi: bool) -> list[dict]:
        """Phase B: per-state entropy, TV, rho, kappa (returned) and sparse targets (cached)."""
        self._cache = {}
        out = []
        with self.actor.engine.eval_mode():
            for p in payloads:
                st, rho, tgt = C.score_traj(self.chop_models, p, self.chop_method, need_pi)
                self._cache[p.uid] = {"payload": p, "targets": tgt.to("cpu"), "segs": []}
                out.append({"uid": p.uid, "rho": rho.cpu(), "entropy": st.entropy.cpu(), "tv": st.tv.cpu(),
                            "kappa": None if st.kappa is None else st.kappa.cpu(),
                            "student_entropy": st.student_entropy.cpu()})
            self._pad(C.student_forwards("score", payloads, self.chop_method))
        return out

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def chop_sample(self, specs: dict[str, list[C.AnchorSpec]], step: int) -> dict:
        """Phase C: tree-decoded student segments and teacher anchors for this rank's anchors."""
        g = self._generator(step, 1)
        segs_all = []
        with self.actor.engine.eval_mode():
            for uid, sp in specs.items():
                segs = C.sample_segments(self.chop_models, self._cache[uid]["payload"], sp, self.chop_method, g)
                self._cache[uid]["segs"] = segs
                segs_all.append(segs)
            self._pad(C.student_forwards("sample", [], self.chop_method, segs_all))
        return {"anchors": sum(len(s) for s in segs_all)}

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def chop_costs(self, uids: list[str], step: int) -> dict:
        """Phase D: per-sample transport costs for the cached segments (advantages are batch-normalised
        on the driver)."""
        g = self._generator(step, 2)
        out, embs, segs_all = {}, [], []
        with self.actor.engine.eval_mode():
            for uid in uids:
                segs = self._cache[uid]["segs"]
                if not segs:
                    continue
                costs, dist, _, s_emb = C.segment_costs(self.chop_models, self._cache[uid]["payload"], segs,
                                                        self.chop_method, g)
                out[uid] = {"costs": costs.cpu(), "dist": dist.cpu()}
                embs.append(s_emb.reshape(-1, s_emb.shape[-1]).cpu())
                segs_all.append(segs)
            self._pad(C.student_forwards("costs", [], self.chop_method, segs_all))
        rank = effective_rank(torch.cat(embs)) if embs else float("nan")
        return {"per_uid": out, "effective_rank": rank}

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def chop_update(self, routes: dict, adv: dict, normalizer: int, extra: dict | None = None) -> dict:
        """Phase E: gradient accumulation over this rank's trajectories, then one optimizer step."""
        engine = self.actor.engine
        metrics = {"loss": 0.0, "token_states": 0.0, "segment_anchors": 0.0}
        payloads = [self._cache[u]["payload"] for u in routes]
        with engine.train_mode():
            engine.optimizer_zero_grad()
            for uid, route in routes.items():
                c = self._cache[uid]
                loss, m = C.update_traj(self.chop_models, c["payload"], route, c["segs"], adv.get(uid),
                                        c["targets"], self.chop_method, normalizer, backward=True,
                                        extra=(extra or {}).get(uid))
                metrics["loss"] += loss
                for k, v in m.items():
                    metrics[k] += v
            self._pad(C.student_forwards("update", payloads, self.chop_method), backward=True)
            grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()
        self._cache = {}
        metrics.update({"grad_norm": float(grad_norm), "lr": float(lr)})
        return metrics
