# Design notes: decisions the paper leaves open

This repository reimplements CHOP from the paper text. Where the paper does not pin down an
implementation detail, we chose a default and made it configurable. Each row names the config
field that controls it.

| Topic | What the paper says | Default here | Config |
|---|---|---|---|
| Token-branch estimator | "token KL" (Eq. chop_lite); App. A defines Top-K OPD on the *student* support with k = 64 | Truncated reverse KL on the student top-64 support, renormalised | `method.token.{estimator,topk,support}` |
| Top-K LSM baseline | Local support matching (Revisiting OPD) | *Teacher* top-32 support, renormalised; optional special-token masking. We keep the shared decoding (T = 0.7, top-p = 0.95) rather than LSM's own top-p = 0.9 | `configs/method/topk_lsm.yaml` |
| Eligible states | "defined student and teacher next-token distribution after the warm-up window" | Student-generated tokens only; prompt, tool observations, padding and post-EOS positions are excluded | trainer |
| Streaming quantiles | 30th (ρ, lower tail) / 70th (κ̂, upper tail) percentile of the preceding W = 8,192 eligible states | Exact quantile over a ring buffer, recomputed every 64 streamed states. Stream order = global batch order after all-gather, so every rank gets the same thresholds. Before `min_history` states the router is in warm-up: token branch, no PI. The buffer can be pre-filled from the calibration corpus | `method.router.{window,refresh,min_history}` |
| (λ1, λ2) calibration | "calibrated once on 1024 states and then frozen" | Quantile routing is invariant to monotone transforms, so routing depends only on λ2/λ1. We fit that ratio by maximising Spearman correlation with ρ_ref, then fit the overall scale by least squares so reported ρ values sit on an absolute scale | `chop.reliability.calibrate_lambdas` |
| Anchor sparsification | Segment loss "at each untrusted state"; +24.4 % anchor overhead | At most one anchor per L_a = 64 window (its first routed state), weighted by the number of routed states in the window; at most 32 windows per trajectory, subsampled uniformly with Horvitz–Thompson reweighting | `method.segment.max_windows_per_traj` |
| Student segments | K' = 8 | The realised continuation ŷ_{t:t+L_a} is one of the K' samples; the other K'−1 are sampled from the student | trainer |
| Segment embedding | "mean-pooled through the current student encoder with stop-gradient embedding" | Prefix-conditioned: mean of the student's last-layer hidden states over the segment tokens, taken from the tree-packed forward and detached | `method.segment.encoder` |
| Per-sample transport cost | "mean baseline subtracts the mean cost across K' student samples" | c_k' = (1/L) Σ_ℓ (proj_ℓ(u_k') − matched teacher order statistic)², where the matching is the monotone 1-D coupling. The MMD analogue is exact (tested by enumeration). For SW, the dependence of the coupling on the other samples is ignored, which is a credit-assignment approximation | `chop.segment.sliced_w2` |
| Gradient estimator | LOO baseline + per-batch normalisation | LOO-centred costs, divided by their standard deviation over the whole batch (no re-centring) | `method.segment.estimator` |
| Segment coefficient | Eq. chop_lite has none | 1.0 | `method.segment.coef` |
| Loss normalisation | not stated | Sum over eligible states (plus anchor-weighted segment terms), divided by the number of eligible states | `chop.losses.chop_loss` |
| PI interface | a dedicated `[PI]` prefix plus the source adapter | Plain-text markers `[PI source=…] … [/PI]` placed before the prompt; the audit sentinel is `[NO_PI]`. No tokenizer resize | `chop.pi.serialize` |
| PI composition | Eq. 14: additive logits | composite = base + Σ_i (logits_i − base), with each variant carrying its own PI prefix. The variants run as one left-padded PEFT mixed-adapter batch; sampling composes the variants' logits at every decoding step, using a KV cache | `chop.pi.composition` |
| κ̂ when no PI source exists | — | The PI gate is closed (math PI-free variants and all no-PI baselines) | router |
| Rolling left truncation | Native 32,768 window; 60K cumulative cap | Chunked re-prefill: before chunk [g0, g1) the context is prompt + response[c:g0] with c = max(0, g1 − (W − \|prompt\|)). Scoring replays the same chunks | `chop.truncation` |
| Collapse fallback | Effective rank < 200 → C/D revert to A and the encoder is frozen until recovery | Recovery requires every rank in the trailing window to be ≥ 200 | `chop.monitors.CollapseGuard` |
| α_t gradient space | cosine of parameter-space gradient expectations | Final transformer block + LM head (configurable), documented as a diagnostic approximation | `analysis/alpha.py` |
| Math PI adapter data | only "the ground-truth final answer" | (problem + `[PI]` answer → the teacher's verified-correct solution) pairs | `adapters/` |
| SWE expert action a* | "aligning the teacher's emitted edit operations against the gold patch" | A teacher edit is a target when its (file, line range) overlaps a gold hunk; otherwise it is dropped | `adapters/` |

## Known paper-side inconsistencies (to be resolved by the authors, not by code)

* **Software versions.** App. K lists vLLM 0.7.3 with PyTorch 2.8. vLLM 0.7.3 cannot load Qwen3 and pins
  torch 2.5.1. This repository uses verl (FSDP2) with vLLM ≥ 0.8.5, not DeepSpeed.
* **Tokenizer.** DeepSeek-R1-0528-Qwen3-8B ships the DeepSeek-R1-0528 tokenizer configuration, not that of
  Qwen3-8B-Base. `scripts/check_tokenizers.py` finds 0 id mismatches over the shared vocabulary, so OPD over
  ordinary tokens is well defined. The special tokens differ, however: the teacher's EOS is
  `<｜end▁of▁sentence｜>` while the student's is `<|endoftext|>`, and the chat templates differ. The teacher
  therefore sees the prompt rendered with its own template, and end-of-sequence positions are exactly where
  special-token mismatch (Revisiting OPD) can penalise correct outputs. Qwen3-4B-Instruct-2507 and
  Qwen3-8B-Base share an identical vocabulary (151,669 tokens) and differ only in chat template and EOS.
* **"Evaluation-time PI".** The student never receives PI. In this code, the deployable and oracle PI
  protocols only select which sources the *teacher* may condition on during training. The only
  evaluation that uses PI is the teacher-alone audit.
* **Leakage filter counts.** On the public releases, the princeton-nlp/SWE-bench train split has
  19,008 instances from 35 repositories, and none of those repositories appear among SWE-bench
  Verified's 12. `scripts/build_manifests.py` therefore obtains:
  * 0 base-commit overlaps;
  * 0 non-generic path overlaps (the only shared path is `setup.cfg`, touched by 42 train instances);
  * 0 same-repository instances.

  The paper's default filter removing 27 instances and stricter filter removing 1,823 more cannot
  be reproduced from these splits (`data/manifests/leakage_report.json`).
* **Issue categories.** The paper does not state how the 500 Verified instances were assigned to
  its four categories (211/86/124/79). `envs/swe/categories.py` is a documented heuristic, and its
  counts differ from the paper's.
* **Unavailable checkpoints.** `Qwen3-14B-Distill` (12-pair table) has no public checkpoint. Its config
  requires a user-provided path.

## Distributed implementation (chop_verl)

The verl recipe keeps verl for data loading, vLLM rollouts (AgentLoops), rewards, checkpointing, FSDP→vLLM
weight synchronisation and validation. It replaces PPO's advantage and update path with the CHOP step in
`chop_verl.driver.chop_step`.

| Topic | Decision |
|---|---|
| Where the teacher lives | A frozen bf16 copy is replicated on every trainer GPU, inside the actor worker, with its PEFT PI adapters. Exact full-vocabulary H(q), TV(p, q) and κ̂ are then computed where both distributions exist. No vocabulary-sized tensor is sent between processes. Teachers that do not fit replicated (70B) would need tensor parallelism and are not supported by this recipe. |
| Student access under FSDP2 | The student is only called through its root forward. `logits_to_keep` restricts the LM head to the rows that are needed, and a hook on the final norm captures hidden states. Qwen3-4B ties its embeddings, so its root FSDP unit owns the head. |
| Teacher targets | `chop.engine.SparseTargets`: teacher logits on the student top-64 support, the teacher top-k support, and the teacher log-prob of the realised token. These are computed at scoring time and cached on the rank until the update. Losses computed from them equal the dense losses (tested). |
| Segment sampling | Multi-anchor tree decoding (`chop.engine.tree_decode`). One prefill per chunk and per teacher variant; each step appends one token per branch under a tree mask. Composite PI anchors (|P| > 1) are sampled exactly by composing the variants' logits at every step. vLLM is not involved. |
| Tree-packed forwards | FlexAttention BlockMask on GPU (`attention: flex`), or a dense additive mask (`attention: dense`, sdpa). FlexAttention cannot run on macOS CPU, so it is exercised only by `scripts/smoke_gpu.sh`. |
| Routing | On the driver, over the whole batch in batch order. The streaming quantiles are therefore identical to a single-process run. |
| Rank alignment | Every rank issues the same number of student forward/backward passes per phase, padding with zero-gradient dummy passes (`chop_verl.lockstep`). The teacher is replicated and issues no collectives. |
| Sequence parallelism | Not supported (`ulysses_sequence_parallel_size = 1`): each trajectory is processed whole by one rank. |
| Local verification | `chop_verl.local.LocalWorkers` emulates the worker group in one process. Tests check that the scalable path reproduces the dense reference trainer's statistics, embeddings and gradients, and that the summed gradient does not depend on the number of ranks. |
