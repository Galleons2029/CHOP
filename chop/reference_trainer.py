"""Single-process reference trainer for CHOP and its baselines (HF models, CPU or one GPU).

It follows Algorithm 1 in five phases that the distributed verl recipe mirrors:

  A. rollout        student samples a trajectory per prompt; outcome reward R(tau)
  B. score + route  no-grad student / teacher passes per rolling chunk -> H, TV, rho, kappa;
                    the two-gate router assigns Cells A-D from streaming quantiles
  C. anchors        routed states are sparsified into L_a windows; K teacher anchor segments and
                    K'-1 student segments are sampled per anchor (the realised continuation is
                    the K'-th student segment)
  D. embeddings     one no-grad tree-packed forward embeds all segments with the current student;
                    SW costs -> REINFORCE advantages normalised over the batch
  E. update         one tree-packed forward with gradients per chunk; Eq. (chop_lite); AdamW

Run ``python -m chop.reference_trainer --config configs/smoke/tiny_math.yaml`` for the CPU smoke
test.  Large-scale runs use ``chop_verl``; this loop is the correctness reference for it.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from chop import divergences as D
from chop.config import ExperimentConfig, load_config, to_yaml
from chop.losses import (
    LossOutput,
    SegmentTerms,
    StateBatch,
    chop_loss,
    grpo_advantages,
    grpo_loss,
    reopold_loss,
    segment_advantages,
)
from chop.monitors import CollapseGuard
from chop.pi.composition import PIConditionalTeacher
from chop.pi.controls import apply_content_control, randomize_lora_
from chop.pi.serialize import format_pi_block
from chop.reliability import reliability
from chop.rollout import generate_response, sample_segment
from chop.router import CellRouter, Route
from chop.segment import build_tree, effective_rank, realised_segment, select_anchor_windows
from chop.truncation import Chunk, chunk_of, rolling_chunks

TOKEN_METHODS = {"chop", "opd", "gkd", "topk_lsm", "opsd", "li2026"}


@dataclass
class Trajectory:
    inst: object
    s_prompt: list[int]
    t_prompt: list[int]
    response: list[int]
    reward: float
    pi_prefix: dict[str, list[int]]
    chunks: list[Chunk]
    group: int = 0


@dataclass
class Scored:
    student_logits: torch.Tensor          # [R, V] detached
    teacher_base: torch.Tensor            # [R, V]
    teacher_pi: torch.Tensor | None       # [R, V]
    entropy: torch.Tensor                 # [R]
    tv: torch.Tensor                      # [R]
    rho: torch.Tensor                     # [R]
    kappa: torch.Tensor | None            # [R]


@dataclass
class Anchor:
    pos: int                              # response position t of the anchor state s_t
    weight: float
    gate: bool
    chunk: Chunk
    student_segs: list[list[int]] = field(default_factory=list)   # K'-1 sampled
    teacher_segs: list[list[int]] = field(default_factory=list)   # K anchors


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


class ReferenceTrainer:
    def __init__(self, cfg: ExperimentConfig, task=None, student=None, teacher=None, tokenizer=None,
                 teacher_tokenizer=None):
        self.cfg = cfg
        m = cfg.method
        seed = cfg.train.seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.task = task or self._build_task()
        if student is None:
            student, teacher, tokenizer, teacher_tokenizer = self._build_models()
        self.student, self.tok = student, tokenizer
        self.t_tok = teacher_tokenizer or tokenizer
        self.sources = list(m.pi.sources)
        self.teacher = PIConditionalTeacher(teacher, pad_id=self.tok.pad_token_id, interface=m.pi.interface)
        if m.pi.content == "random_adapter":
            for s in self.sources:
                randomize_lora_(teacher, s, self.gen)
        self.frozen = copy.deepcopy(student).eval() if m.segment.encoder == "frozen_student" else None
        self.router = CellRouter(m.router, seed=seed)
        self.guard = CollapseGuard(m.segment.rank_fallback_threshold)
        o = cfg.optim
        self.opt = torch.optim.AdamW(self.student.parameters(), lr=o.lr, betas=tuple(o.betas),
                                     weight_decay=o.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: _cosine_warmup(s, o.warmup_steps, cfg.train.steps))
        self.out = Path(cfg.output_dir) / cfg.name / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(to_yaml(cfg))
        self.log_path = self.out / "metrics.jsonl"
        self.step = 0

    # ------------------------------------------------------------------ setup
    def _build_task(self):
        d = self.cfg.data
        if d.task == "toy":
            from envs.toy import ToyArithmetic

            return ToyArithmetic(seed=self.cfg.train.seed)
        if d.task == "math":
            from envs.math import MathTask

            return MathTask(d.train, tuple(d.eval), d.train_limit)
        raise ValueError(f"reference trainer supports toy and math tasks, not {d.task!r}")

    def _build_models(self):
        mc, m = self.cfg.model, self.cfg.method
        if mc.tiny:
            from chop.tiny import tiny_qwen3
            from envs.toy import ToyTokenizer

            tok = ToyTokenizer()
            student = tiny_qwen3(vocab_size=tok.vocab_size, seed=self.cfg.train.seed,
                                 max_position_embeddings=4 * mc.native_context)
            teacher = tiny_qwen3(vocab_size=tok.vocab_size, seed=10_000 + self.cfg.train.seed,
                                 max_position_embeddings=4 * mc.native_context)
            student.train()
            if m.pi.sources:
                import peft

                lcfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                                       init_lora_weights=False)
                teacher = peft.get_peft_model(teacher, lcfg, adapter_name=m.pi.sources[0])
                for s in m.pi.sources[1:]:
                    teacher.add_adapter(s, lcfg)
            teacher.eval()
            return student, teacher, tok, tok
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, mc.dtype)
        tok = AutoTokenizer.from_pretrained(mc.student)
        t_tok = AutoTokenizer.from_pretrained(mc.teacher)
        student = AutoModelForCausalLM.from_pretrained(mc.student, torch_dtype=dtype)
        teacher = AutoModelForCausalLM.from_pretrained(mc.teacher, torch_dtype=dtype)
        if m.pi.sources and m.pi.interface == "lora_prefix":
            from peft import PeftModel

            if not m.pi.adapter_dir:
                raise ValueError("PI sources with the LoRA interface need method.pi.adapter_dir")
            root = Path(m.pi.adapter_dir)
            teacher = PeftModel.from_pretrained(teacher, root / m.pi.sources[0], adapter_name=m.pi.sources[0])
            for s in m.pi.sources[1:]:
                teacher.load_adapter(root / s, adapter_name=s)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return student, teacher, tok, t_tok

    def _special_mask(self, tokens: list[int]) -> torch.Tensor:
        special = {self.tok.eos_token_id, self.tok.pad_token_id, getattr(self.tok, "bos_token_id", None)}
        special |= set(getattr(self.tok, "all_special_ids", []) or [])
        return torch.tensor([x in special for x in tokens], dtype=torch.bool)

    # -------------------------------------------------------------- prompts
    def _prompt_ids(self, inst) -> tuple[list[int], list[int]]:
        text = self.task.prompt(inst)
        if self.cfg.model.tiny:
            ids = self.tok.encode(text)
            return ids, ids
        t_ids = self.t_tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                               tokenize=True)
        if self.cfg.data.prompt_template == "teacher_aligned":
            rendered = self.t_tok.apply_chat_template([{"role": "user", "content": text}],
                                                      add_generation_prompt=True, tokenize=False)
            return self.tok.encode(rendered, add_special_tokens=False), list(t_ids)
        return self.tok.encode(text, add_special_tokens=False), list(t_ids)

    def _pi_prefixes(self, insts) -> list[dict[str, list[int]]]:
        m = self.cfg.method.pi
        if not self.sources:
            return [{} for _ in insts]
        blocks = {s: [self.task.pi_blocks(i)[s] for i in insts] for s in self.sources}
        repos = [getattr(i, "meta", {}).get("repo", i.id) for i in insts]
        out = [{} for _ in insts]
        for s in self.sources:
            content = apply_content_control(m.content, blocks[s], repos, self.gen) if len(insts) > 1 else blocks[s]
            for j, c in enumerate(content):
                text = format_pi_block(s, c, lambda x: self.t_tok.encode(x, add_special_tokens=False),
                                       self.t_tok.decode, m.max_tokens_per_source)
                out[j][s] = self.t_tok.encode(text, add_special_tokens=False)
        return out

    def _plan(self, s_prompt, t_prompt, pi_prefix, response_len: int) -> list[Chunk]:
        prefix = max((len(v) for v in pi_prefix.values()), default=0)
        plen = max(len(s_prompt), len(t_prompt) + prefix)
        return rolling_chunks(plen, max(response_len, 1), self.cfg.model.native_context, self.cfg.train.rollout_chunk,
                              reserve=self.cfg.method.segment.segment_length)

    # --------------------------------------------------------------- phase A
    def rollout(self, insts, group_size: int = 1) -> list[Trajectory]:
        t = self.cfg.train
        prefixes = self._pi_prefixes(insts)
        trajs = []
        self.student.eval()
        for gi, (inst, pre) in enumerate(zip(insts, prefixes, strict=True)):
            s_prompt, t_prompt = self._prompt_ids(inst)
            plan_len = max(len(s_prompt), len(t_prompt) + max((len(v) for v in pre.values()), default=0))
            for _ in range(group_size):
                resp = generate_response(self.student, s_prompt, t.max_response_tokens, t.temperature, t.top_p,
                                         self.tok.eos_token_id, self.cfg.model.native_context, t.rollout_chunk,
                                         reserve=self.cfg.method.segment.segment_length + plan_len - len(s_prompt),
                                         generator=self.gen)
                reward = self.task.reward(inst, self.tok.decode(resp))
                trajs.append(Trajectory(inst, s_prompt, t_prompt, resp, reward, pre,
                                        self._plan(s_prompt, t_prompt, pre, len(resp)), gi))
        self.student.train()
        return trajs

    # --------------------------------------------------------------- phase B
    @torch.no_grad()
    def score(self, tr: Trajectory, need_pi: bool) -> Scored:
        s_logits, base, comp = [], [], []
        for c in tr.chunks:
            ids = tr.s_prompt + tr.response[c.ctx_start:c.gen_end]
            first = len(tr.s_prompt) + (c.gen_start - c.ctx_start)
            lg = self.student(input_ids=torch.tensor([ids])).logits[0].float()
            s_logits.append(lg[first - 1:first - 1 + (c.gen_end - c.gen_start)])
            b, cp = self.teacher.score(tr.t_prompt + tr.response[c.ctx_start:c.gen_start],
                                       tr.response[c.gen_start:c.gen_end], tr.pi_prefix,
                                       self.sources if need_pi else [])
            base.append(b)
            if cp is not None:
                comp.append(cp)
        sl, bl = torch.cat(s_logits), torch.cat(base)
        cl = torch.cat(comp) if comp else None
        h, tv = D.reliability_inputs(sl, bl)
        rho = reliability(h, tv, self.cfg.method.reliability)
        kappa = D.total_variation(bl, cl) if cl is not None else None
        return Scored(sl, bl, cl, h, tv, rho, kappa)

    def route(self, trajs: list[Trajectory], scored: list[Scored]) -> list[Route]:
        rho = torch.cat([s.rho for s in scored])
        kappa = torch.cat([s.kappa for s in scored]) if scored[0].kappa is not None else None
        depth = torch.cat([torch.arange(len(t.response)) / max(len(t.response), 1) for t in trajs])
        route = self.router.route(rho, kappa, depth_frac=depth)
        self.router.update(rho, kappa)
        out, i = [], 0
        for t in trajs:
            n = len(t.response)
            out.append(Route(route.cells[i:i + n], route.trusted[i:i + n], route.pi_gate[i:i + n],
                             route.tau_rho[i:i + n], route.kappa_cut[i:i + n], route.in_stratum[i:i + n]))
            i += n
        return out

    # --------------------------------------------------------------- phase C
    def anchors(self, tr: Trajectory, route: Route) -> list[Anchor]:
        s = self.cfg.method.segment
        mask = route.segment & route.in_stratum
        pos, w = select_anchor_windows(mask, s.segment_length, s.max_windows_per_traj, self.gen)
        out: list[Anchor] = []
        prev: Anchor | None = None
        for p, weight in zip(pos.tolist(), w.tolist(), strict=True):
            c = chunk_of(p, tr.chunks)
            a = Anchor(p, weight, bool(route.pi_gate[p]), c)
            s_ctx = tr.s_prompt + tr.response[c.ctx_start:p]
            a.student_segs = [sample_segment(self.student, s_ctx, s.segment_length, s.student_temperature, 1.0,
                                             self.tok.eos_token_id, self.gen)
                              for _ in range(s.num_student_segments - 1)]
            reuse = (s.anchor_cache and prev is not None and prev.gate == a.gate and prev.chunk == c
                     and p - prev.pos <= s.segment_length)
            if reuse:
                a.teacher_segs = prev.teacher_segs
            elif s.anchor_source == "student":
                a.teacher_segs = [sample_segment(self.student, s_ctx, s.segment_length, s.teacher_temperature, 1.0,
                                                 self.tok.eos_token_id, self.gen) for _ in range(s.num_anchors)]
            else:
                toks, lens = self.teacher.sample(tr.t_prompt + tr.response[c.ctx_start:p], tr.pi_prefix,
                                                 self.sources if a.gate else [], s.num_anchors, s.segment_length,
                                                 s.teacher_temperature, self.t_tok.eos_token_id, self.gen)
                a.teacher_segs = [toks[k, :int(lens[k])].tolist() for k in range(s.num_anchors)]
            out.append(a)
            prev = a
        return out

    # ------------------------------------------------------------ phases D/E
    def _packed(self, model, tr: Trajectory, c: Chunk, anchors: list[Anchor], with_teacher: bool, grad: bool):
        main = tr.s_prompt + tr.response[c.ctx_start:c.gen_end]
        branches, owners = [], []
        for ai, a in enumerate(anchors):
            idx = len(tr.s_prompt) + (a.pos - c.ctx_start)
            for seg in a.student_segs:
                branches.append((idx, torch.tensor(seg)))
                owners.append((ai, "s"))
            if with_teacher:
                for seg in a.teacher_segs:
                    branches.append((idx, torch.tensor(seg)))
                    owners.append((ai, "t"))
        lay = build_tree(torch.tensor(main), branches)
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            o = model(input_ids=lay.input_ids[None], attention_mask=lay.additive_mask(),
                      position_ids=lay.position_ids[None], output_hidden_states=True)
        return lay, owners, o.logits[0], o.hidden_states[-1][0]

    @torch.no_grad()
    def _segment_only_embed(self, segs: list[list[int]]) -> torch.Tensor:
        embs = []
        for seg in segs:
            o = self.student(input_ids=torch.tensor([seg]), output_hidden_states=True)
            embs.append(o.hidden_states[-1][0].mean(0))
        return torch.stack(embs)

    def embed_anchors(self, tr: Trajectory, anchors: list[Anchor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Stop-gradient embeddings: teacher [n, K, d], student [n, K', d] (realised first)."""
        s = self.cfg.method.segment
        L = s.segment_length
        teach, stud = [], []
        for c in tr.chunks:
            group = [a for a in anchors if a.chunk == c]
            if not group:
                continue
            if s.encoder == "segment_only":
                for a in group:
                    real = tr.response[a.pos:min(a.pos + L, c.gen_end)]
                    stud.append(self._segment_only_embed([real] + a.student_segs))
                    teach.append(self._segment_only_embed(a.teacher_segs))
                continue
            model = self.frozen if s.encoder == "frozen_student" else self.student
            lay, owners, logits, hidden = self._packed(model, tr, c, group, with_teacher=True, grad=False)
            emb = lay.branch_mean_hidden(hidden)
            main = lay.input_ids[:lay.main_len]
            for ai, a in enumerate(group):
                idx = len(tr.s_prompt) + (a.pos - c.ctx_start)
                _, e_real = realised_segment(main, logits, hidden, idx, L)
                s_e = [e_real] + [emb[b] for b, (o, k) in enumerate(owners) if o == ai and k == "s"]
                t_e = [emb[b] for b, (o, k) in enumerate(owners) if o == ai and k == "t"]
                stud.append(torch.stack(s_e))
                teach.append(torch.stack(t_e))
        return torch.stack(teach).float(), torch.stack(stud).float()

    def _grad_pass(self, tr: Trajectory, anchors: list[Anchor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Student logits with gradient for every response position, and segment log-probs [n, K', L]."""
        L = self.cfg.method.segment.segment_length
        logits_all, seg_lp, seg_mask = [], [], []
        for c in tr.chunks:
            group = [a for a in anchors if a.chunk == c]
            lay, owners, logits, hidden = self._packed(self.student, tr, c, group, with_teacher=False, grad=True)
            first = len(tr.s_prompt) + (c.gen_start - c.ctx_start)
            logits_all.append(logits[first - 1:first - 1 + (c.gen_end - c.gen_start)])
            if not group:
                continue
            blp, bmask = lay.branch_token_logprobs(logits)
            main = lay.input_ids[:lay.main_len]
            for ai, a in enumerate(group):
                idx = len(tr.s_prompt) + (a.pos - c.ctx_start)
                lp_real, _ = realised_segment(main, logits, hidden, idx, L)
                rows = [torch.nn.functional.pad(lp_real, (0, L - lp_real.numel()))]
                masks = [torch.arange(L) < lp_real.numel()]
                for b, (o, _k) in enumerate(owners):
                    if o == ai:
                        rows.append(torch.nn.functional.pad(blp[b], (0, L - blp.shape[1])))
                        masks.append(torch.nn.functional.pad(bmask[b], (0, L - bmask.shape[1])))
                seg_lp.append(torch.stack(rows))
                seg_mask.append(torch.stack(masks))
        lp = torch.stack(seg_lp) if seg_lp else torch.empty(0, 0, L)
        mk = torch.stack(seg_mask) if seg_mask else torch.empty(0, 0, L, dtype=torch.bool)
        return torch.cat(logits_all), lp, mk

    # ------------------------------------------------------------------ steps
    def _chop_step(self, trajs: list[Trajectory]) -> LossOutput:
        m = self.cfg.method
        need_pi = bool(self.sources) and m.router.kappa_gate != "none"
        scored = [self.score(t, need_pi) for t in trajs]
        routes = self.route(trajs, scored)
        use_segments = m.router.rho_gate != "none"
        anchors = [self.anchors(t, r) if use_segments else [] for t, r in zip(trajs, routes, strict=True)]
        adv_all, metrics = None, {}
        if any(anchors):
            embs = [self.embed_anchors(t, a) for t, a in zip(trajs, anchors, strict=True) if a]
            teach = torch.cat([e[0] for e in embs])
            stud = torch.cat([e[1] for e in embs])
            rank = effective_rank(stud.reshape(-1, stud.shape[-1]))
            metrics["embedding_rank"] = rank
            if self.guard.observe(rank):
                metrics["collapse_fallback"] = 1.0
                for r in routes:  # C/D revert to A: token KL without PI
                    seg = r.segment.clone()
                    r.trusted[seg] = True
                    r.pi_gate[seg] = False
                    r.cells[seg] = 0
                anchors = [[] for _ in trajs]
            else:
                dist, adv_all = segment_advantages(teach, stud, m, self.gen)
                metrics["segment_distance"] = float(dist.mean())
        n_total = sum(len(t.response) for t in trajs)
        total_loss, cursor = 0.0, 0
        agg: dict[str, float] = {}
        for t, sc, r, a in zip(trajs, scored, routes, anchors, strict=True):
            logits, lp, mk = self._grad_pass(t, a)
            seg = None
            if a:
                seg = SegmentTerms(lp, mk, torch.tensor([x.weight for x in a]),
                                   adv=adv_all[cursor:cursor + len(a)])
                cursor += len(a)
            states = StateBatch(logits, torch.tensor(t.response), sc.teacher_base, sc.teacher_pi,
                                torch.full((len(t.response),), t.reward), self._special_mask(t.response))
            out = chop_loss(states, r, seg, m, self.gen, normalizer=n_total)
            out.loss.backward()
            total_loss += float(out.loss.detach())
            for k, v in out.metrics.items():
                if k != "loss" and not math.isnan(v):
                    agg[k] = agg.get(k, 0.0) + v * len(t.response) / n_total
        agg.update(metrics)
        return LossOutput(torch.tensor(total_loss), agg)

    def _reopold_step(self, trajs: list[Trajectory]) -> LossOutput:
        scored = [self.score(t, False) for t in trajs]
        logits = torch.cat([self._grad_pass(t, [])[0] for t in trajs])
        states = StateBatch(logits, torch.tensor([x for t in trajs for x in t.response]),
                            torch.cat([s.teacher_base for s in scored]))
        out = reopold_loss(states, self.cfg.method, self.step)
        out.loss.backward()
        return out

    def _grpo_step(self, trajs: list[Trajectory]) -> LossOutput:
        rewards = torch.tensor([t.reward for t in trajs])
        adv = grpo_advantages(rewards, torch.tensor([t.group for t in trajs]))
        logits = torch.cat([self._grad_pass(t, [])[0] for t in trajs])
        tok_adv = torch.cat([adv[i].expand(len(t.response)) for i, t in enumerate(trajs)])
        out = grpo_loss(logits, torch.tensor([x for t in trajs for x in t.response]), tok_adv)
        out.loss.backward()
        return out

    def sft_warmup(self) -> None:
        """Off-policy cold start of the Li et al. (2026) recipe: SFT on teacher rollouts."""
        m, t = self.cfg.method, self.cfg.train
        insts = random.Random(t.seed).sample(self.task.train_instances(), m.sft_warmup_prompts)
        base = self.teacher.model
        data = []
        for inst in insts:
            s_prompt, t_prompt = self._prompt_ids(inst)
            with (base.disable_adapter() if hasattr(base, "peft_config") else contextlib.nullcontext()):
                resp = generate_response(base, t_prompt, t.max_response_tokens, t.temperature, t.top_p,
                                         self.t_tok.eos_token_id, self.cfg.model.native_context, t.rollout_chunk,
                                         generator=self.gen)
            data.append((s_prompt, resp))
        for _ in range(m.sft_warmup_epochs):
            for s_prompt, resp in data:
                ids = torch.tensor([s_prompt + resp])
                logits = self.student(input_ids=ids).logits[0, len(s_prompt) - 1:-1]
                loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(resp))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self._log({"phase": "sft_warmup", "examples": len(data)})

    def train_step(self) -> dict[str, float]:
        m, t = self.cfg.method, self.cfg.train
        t0 = time.time()
        group = m.grpo_group_size if m.name == "grpo" else 1
        pool = self.task.train_instances()
        insts = [pool[int(i)] for i in torch.randint(len(pool), (max(t.batch_size // group, 1),), generator=self.gen)]
        trajs = self.rollout(insts, group)
        self.opt.zero_grad()
        if m.name in TOKEN_METHODS:
            out = self._chop_step(trajs)
        elif m.name == "reopold":
            out = self._reopold_step(trajs)
        elif m.name == "grpo":
            out = self._grpo_step(trajs)
        else:
            raise ValueError(f"unknown method {m.name!r}")
        if self.cfg.optim.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.optim.grad_clip)
        self.opt.step()
        self.sched.step()
        rec = {"step": self.step, "loss": float(out.loss.detach()), "reward": sum(x.reward for x in trajs) / len(trajs),
               "response_len": sum(len(x.response) for x in trajs) / len(trajs), "sec": time.time() - t0,
               **out.metrics}
        self._log(rec)
        self.step += 1
        return rec

    @torch.no_grad()
    def evaluate(self, split: str, samples: int | None = None, limit: int | None = None) -> float:
        """avg@k (percent) on an evaluation split."""
        t = self.cfg.train
        k = samples or self.cfg.data.eval_samples
        insts = self.task.eval_instances(split)[:limit]
        self.student.eval()
        correct = 0.0
        for inst in insts:
            s_prompt, _ = self._prompt_ids(inst)
            for _ in range(k):
                resp = generate_response(self.student, s_prompt, t.max_response_tokens, t.temperature, t.top_p,
                                         self.tok.eos_token_id, self.cfg.model.native_context, t.rollout_chunk,
                                         generator=self.gen)
                correct += self.task.reward(inst, self.tok.decode(resp))
        self.student.train()
        score = 100.0 * correct / max(len(insts) * k, 1)
        self._log({"eval": split, f"avg@{k}": score, "step": self.step})
        return score

    def _log(self, rec: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def run(self) -> None:
        if self.cfg.method.sft_warmup_prompts:
            self.sft_warmup()
        for _ in range(self.cfg.train.steps):
            self.train_step()
        torch.save({"router": self.router.state_dict(), "step": self.step}, self.out / "router.pt")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. method.segment.num_anchors=4")
    ap.add_argument("--eval", nargs="*", default=None, help="evaluate these splits after training")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    trainer = ReferenceTrainer(cfg)
    trainer.run()
    for split in args.eval or []:
        print(split, trainer.evaluate(split))
    print(f"wrote {trainer.log_path}")


if __name__ == "__main__":
    main()
