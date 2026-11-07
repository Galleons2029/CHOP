"""Typed configuration for CHOP and its baselines.

Every experiment YAML under ``configs/`` is merged onto :class:`ExperimentConfig`.
A YAML file may list parent files under ``_base_`` (paths relative to itself); parents are
merged first, in order, so a table-level config only states what differs from the recipe in
``configs/base``.  Defaults below reproduce Table ``tab:hparams`` of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class ReliabilityConfig:
    # calibrated: exp(-l1*H - l2*TV)            (Eq. rho, lambdas fitted once on 1024 states)
    # free:       exp(-H/H_max - TV/TV_max)       (Eq. rho_free, outcome-independent)
    # entropy_only / tv_only: single-term ablations of tab:reliability_purity
    # reference:  routing on a supplied rho_ref (teacher self-rollout success rate)
    variant: str = "calibrated"
    lambda_entropy: float = 1.0
    lambda_tv: float = 1.0
    entropy_max: float = 1.0
    tv_max: float = 1.0


@dataclass
class RouterConfig:
    window: int = 8192               # W: preceding eligible states used by both quantiles
    rho_quantile: float = 0.30       # lower tail: rho <= tau_rho(t) -> segment branch
    kappa_quantile: float = 0.70     # upper tail: kappa >= kappa_cut -> PI-conditional teacher
    refresh: int = 64                # thresholds are recomputed every `refresh` streamed states
    min_history: int = 256           # below this many buffered states the router is in warm-up
    # rho gate: quantile | static | none (token everywhere) | all_segment | random_quota |
    #           depth_decile | oracle | frozen
    rho_gate: str = "quantile"
    static_rho: float = 0.5
    # kappa gate: quantile | static | none (never PI) | always (uniform PI) | random |
    #             random_rate_matched
    kappa_gate: str = "quantile"
    static_kappa: float = 0.5
    random_pi_rate: float = 0.28
    cell_a_uses_pi: bool = False     # ablation (h): trusted states always use the PI teacher
    three_cell: bool = False         # ablation (e): Cell B treated as Cell A
    # rho-decile conditional intervention: the method's branch is applied only to states whose
    # rho falls in [lo, hi) (streaming quantile levels); other states fall back to standard OPD.
    stratum: list[float] | None = None


@dataclass
class TokenLossConfig:
    estimator: str = "topk"          # sampled | full | topk | gkd
    topk: int = 64
    support: str = "student"         # top-k support: student (App. A, CHOP default) | teacher (Top-K LSM)
    gkd_beta: float = 1.0            # GKD generalised JSD; 1.0 = reverse KL
    mask_special_tokens: bool = False  # Top-K LSM special-token masking


@dataclass
class SegmentConfig:
    num_anchors: int = 8             # K teacher anchor segments
    num_student_segments: int = 8    # K' student segments (the realised continuation is one)
    segment_length: int = 64         # L_a
    num_projections: int = 32        # L
    distance: str = "sw"             # sw | mmd
    estimator: str = "loo_norm"      # vanilla | mean | loo | loo_norm
    anchor_source: str = "teacher"   # teacher | student (tab:anisotropy "student anchors")
    encoder: str = "student"         # student | frozen_student | segment_only | external
    external_encoder: str | None = None
    max_windows_per_traj: int = 32
    anchor_cache: bool = False
    teacher_temperature: float = 0.7
    student_temperature: float = 0.7
    coef: float = 1.0
    rank_fallback_threshold: float = 200.0


@dataclass
class PIConfig:
    sources: list[str] = field(default_factory=list)
    interface: str = "lora_prefix"   # lora_prefix | prompt (ablation g)
    content: str = "true"            # true | shuffled | wrong_repo | sentinel | random_adapter
    max_tokens_per_source: int = 2048
    adapter_dir: str | None = None


@dataclass
class MethodConfig:
    # chop | opd | gkd | topk_lsm | opsd | reopold | grpo  (token baselines are router presets of chop)
    name: str = "chop"
    beta_outcome: float = 0.1        # beta in Eq. (chop_lite)
    token: TokenLossConfig = field(default_factory=TokenLossConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    pi: PIConfig = field(default_factory=PIConfig)
    # REOPOLD (Ko et al., relaxed OPD): mixture-based reward clipping at log(l/(1-l)), reward-based
    # token filtering before the switch step, entropy-guided (top-beta) token sampling after it
    reopold_lambda: float = 0.3
    reopold_switch_step: int = 50
    reopold_entropy_top: float = 0.2
    grpo_group_size: int = 8
    # Li et al. (2026) full recipe: off-policy cold start on teacher rollouts before OPD
    sft_warmup_prompts: int = 0
    sft_warmup_epochs: int = 1


@dataclass
class ModelConfig:
    student: str = "Qwen/Qwen3-8B-Base"
    teacher: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    native_context: int = 32768
    dtype: str = "bfloat16"
    tiny: bool = False               # build random tiny Qwen3 models (tests / smoke only)


@dataclass
class OptimConfig:
    lr: float = 5e-6
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    warmup_steps: int = 200
    schedule: str = "cosine"
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    steps: int = 250
    batch_size: int = 128            # sequences per update
    grad_accum: int = 1
    max_response_tokens: int = 30_000  # cumulative generated-token cap
    rollout_chunk: int = 4096        # re-prefill interval for rolling left truncation
    max_turns: int | None = None     # agentic tasks
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 0


@dataclass
class DataConfig:
    task: str = "math"               # math | swe | toy
    train: str = "dapo_math_17k"
    eval: list[str] = field(default_factory=lambda: ["aime24", "aime25", "amc23"])
    manifest: str | None = None
    eval_samples: int = 16           # avg@16
    train_limit: int | None = None
    prompt_template: str = "dapo"    # dapo | teacher_aligned (Li et al. recipe)


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    table: str | None = None         # paper table label this run reproduces
    row: str | None = None           # row name in that table
    reported: dict[str, Any] = field(default_factory=dict)  # values printed in the paper
    reference: str | None = None     # run whose score is subtracted for metrics named *_gain
    gap: list[float] | None = None   # [student, teacher] references for gap recovery
    notes: str | None = None         # what must be set or measured before this row can run
    method: MethodConfig = field(default_factory=MethodConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    gpus: int = 8
    est_hours_per_seed: float | None = None
    output_dir: str = "outputs"
    eval_only: bool = False
    kind: str = "math"               # math | swe  (selects chop_verl config and evaluation harness)
    eval: dict[str, Any] = field(default_factory=dict)  # evaluation settings (max_turns, harness, checkpoint, ...)


def _load_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic _base_ chain: {' -> '.join(map(str, seen + (path,)))}")
    raw = OmegaConf.load(path)
    bases = raw.pop("_base_", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged = OmegaConf.create()
    for base in bases:
        merged = OmegaConf.merge(merged, _load_with_bases(path.parent / base, seen + (path,)))
    return OmegaConf.merge(merged, raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config (with ``_base_`` inheritance) and dotlist overrides onto the schema."""
    cfg = OmegaConf.structured(ExperimentConfig)
    if path is not None:
        cfg = OmegaConf.merge(cfg, _load_with_bases(Path(path)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(cfg)


def to_yaml(cfg: ExperimentConfig) -> str:
    return OmegaConf.to_yaml(OmegaConf.structured(cfg))
