#!/usr/bin/env python3
"""Generate configs/{base,method,experiments} from one specification of the paper's tables.

Every experiment row records the paper's table label and row name, the values the paper reports (so
paper/make_tables.py can compare them with measured runs), and how the row differs from its method
preset. Re-run after editing this file:  python scripts/gen_experiment_configs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path("configs")

# ------------------------------------------------------------------------------ base recipes (tab:hparams)
BASE = {
    "math": {
        "kind": "math",
        "model": {"student": "Qwen/Qwen3-8B-Base", "teacher": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                  "native_context": 32768, "dtype": "bfloat16"},
        "optim": {"lr": 5.0e-6, "betas": [0.9, 0.95], "weight_decay": 0.1, "warmup_steps": 200, "schedule": "cosine"},
        "train": {"steps": 250, "batch_size": 128, "grad_accum": 1, "max_response_tokens": 30000,
                  "rollout_chunk": 4096, "temperature": 0.7, "top_p": 0.95},
        "data": {"task": "math", "train": "dapo_math_17k", "eval": ["aime24", "aime25", "amc23"], "eval_samples": 16},
        "seeds": [0, 1, 2], "gpus": 8, "est_hours_per_seed": 24 / 7,
        "gap": [38.4, 57.0],
    },
    "swe": {
        "kind": "swe",
        "model": {"student": "Qwen/Qwen3-4B-Instruct-2507", "teacher": "Qwen/Qwen3-8B-Base", "native_context": 32768,
                  "dtype": "bfloat16"},
        "optim": {"lr": 2.0e-6, "betas": [0.9, 0.95], "weight_decay": 0.1, "warmup_steps": 500, "schedule": "cosine"},
        "train": {"steps": 4500, "batch_size": 32, "grad_accum": 4, "max_response_tokens": 60000,
                  "rollout_chunk": 4096, "max_turns": 50, "temperature": 0.7, "top_p": 0.95},
        "data": {"task": "swe", "train": "swe_train_filtered", "eval": ["verified_500"],
                 "manifest": "data/manifests/swe_train_filtered.txt", "eval_samples": 1},
        "method": {"pi": {"sources": ["call_graph", "test_manifest"], "adapter_dir": "outputs/adapters/swe"}},
        "seeds": [0, 1, 2], "gpus": 8, "est_hours_per_seed": 36.0,
        "eval": {"harness": "openhands", "max_turns": 50, "time_limit_s": 600},
    },
}

# --------------------------------------------------------------------------------------- method presets
TOKEN_ONLY = {"router": {"rho_gate": "none", "kappa_gate": "none"}}
METHOD = {
    "chop": {"method": {"name": "chop"}},
    "opd": {"method": {"name": "opd", "token": {"estimator": "sampled"}, **TOKEN_ONLY}},
    "gkd": {"method": {"name": "gkd", "token": {"estimator": "gkd", "gkd_beta": 1.0}, **TOKEN_ONLY}},
    "topk_lsm": {"method": {"name": "topk_lsm", "token": {"estimator": "topk", "topk": 32, "support": "teacher",
                                                          "mask_special_tokens": True}, **TOKEN_ONLY}},
    "opsd": {"method": {"name": "opsd", "token": {"estimator": "sampled"},
                        "router": {"rho_gate": "none", "kappa_gate": "always"}}},
    "reopold": {"method": {"name": "reopold", "reopold_lambda": 0.3, "reopold_switch_step": 50,
                           "reopold_entropy_top": 0.2}},
    "grpo": {"method": {"name": "grpo", "grpo_group_size": 8}},
    "li2026": {"method": {"name": "li2026", "token": {"estimator": "sampled"}, "sft_warmup_prompts": 1000,
                          **TOKEN_ONLY},
               "data": {"prompt_template": "teacher_aligned", "train_limit": 5000},
               "notes": "Li et al. full recipe: SFT on teacher rollouts of 1K prompts, then sampled-token OPD on "
                        "5K teacher-aligned prompts."},
    "chop_token_only": {"method": {"name": "chop", "router": {"rho_gate": "none"}}},
    "chop_no_pi": {"method": {"name": "chop", "router": {"kappa_gate": "none"}, "pi": {"sources": []}}},
    "chop_student_anchors": {"method": {"name": "chop", "segment": {"anchor_source": "student"}}},
}

MATH_PI = {"method": {"pi": {"sources": ["final_answer"], "adapter_dir": "outputs/adapters/math"}}}
ORACLE = {"method": {"pi": {"sources": ["test_manifest", "call_graph", "next_file", "gold_patch"]}}}

# ---------------------------------------------------------------------------------- reported results
CAPS = [1, 3, 7, 15, 30, 45, 60]
MATH_FULL = {  # tab:math_full (AIME 2024 / AIME 2025 / AMC 2023)
    "opd": ([38.1, 42.4, 45.2, 43.8, 35.6, 31.2, 28.4], [35.4, 39.6, 42.2, 40.7, 33.1, 29.4, 26.7],
            [66.7, 71.5, 75.3, 73.6, 69.4, 63.8, 58.2]),
    "topk_lsm": ([40.2, 44.1, 47.8, 48.6, 47.2, 45.1, 42.8], [37.6, 41.4, 44.9, 45.0, 43.6, 41.5, 39.1],
                 [72.4, 77.6, 80.8, 81.3, 80.1, 77.5, 74.6]),
    "opsd": ([38.5, 42.8, 45.6, 44.3, 38.1, 34.7, 32.5], [35.8, 40.0, 42.7, 41.6, 35.4, 32.4, 30.2],
             [69.1, 73.4, 76.8, 75.1, 72.8, 68.2, 63.5]),
    "chop": ([39.6, 43.9, 47.4, 48.2, 48.3, 46.7, 44.9], [37.0, 41.0, 44.7, 44.6, 44.7, 43.0, 41.0],
             [71.6, 77.1, 80.3, 80.9, 81.5, 79.2, 76.8]),
}
MATH_MAIN_EXTRA = {"grpo": {7: 43.6, 15: 44.1, 30: 42.7, 60: 38.4}, "li2026": {7: 46.5, 15: 46.4, 30: 42.1, 60: 37.3}}
DECILES = {"D10": ([0.9, 1.0], 6.2, 5.9, 6.4), "D9": ([0.8, 0.9], 5.8, 5.4, 5.9), "D8": ([0.7, 0.8], 4.6, 4.3, 5.1),
           "D7": ([0.6, 0.7], 3.1, 2.8, 4.4), "D6": ([0.5, 0.6], 1.8, 1.5, 3.6), "D5": ([0.4, 0.5], 0.4, 0.6, 3.1),
           "D4": ([0.3, 0.4], -0.2, 0.1, 2.8), "D1-D3": ([0.0, 0.3], -0.8, -0.3, 2.4)}
ANISO = {"opd": (38.6, 1.1, 0.09), "topk_lsm": (41.1, 14.5, 0.18), "opsd": (39.7, 7.0, 0.12),
         "li2026": (43.0, 24.7, 0.22), "chop_token_only": (39.6, 6.5, 0.13), "chop_no_pi": (42.5, 22.0, 0.31),
         "chop_student_anchors": (43.2, 25.8, 0.34), "chop": (44.3, 31.7, 0.39)}


def swe(resolve, ci=None, avg_len=None):
    r = {"resolve": resolve}
    if ci:
        r["ci"] = ci
    if avg_len:
        r["avg_len_k"] = avg_len
    return r


def merge(*dicts):
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def rows() -> list[tuple[str, str, dict]]:
    """(table directory, file stem, config dict)."""
    out = []

    def add(table, name, base, preset, row, reported, extra=None, **kw):
        cfg = {"_base_": [f"../../base/{base}.yaml"] + ([f"../../method/{preset}.yaml"] if preset else []),
               "name": name, "table": table, "row": row, "reported": reported, **kw}
        out.append((table.replace("tab:", ""), name, merge(cfg, extra or {})))

    # --- math length sweep (tab:math_full, tab:math_main) -----------------------------------------------
    for m, (a24, a25, amc) in MATH_FULL.items():
        for i, cap in enumerate(CAPS):
            add("tab:math_full", f"{m}_{cap}k", "math", m, f"{m} @ {cap}K",
                {"aime24": a24[i], "aime25": a25[i], "amc23": amc[i]},
                merge({"train": {"max_response_tokens": cap * 1000}}, MATH_PI if m == "chop" else {}))
    for m, vals in MATH_MAIN_EXTRA.items():
        for cap, v in vals.items():
            add("tab:math_main", f"{m}_{cap}k", "math", m, f"{m} @ {cap}K", {"aime24": v},
                {"train": {"max_response_tokens": cap * 1000}})
    # --- rho-decile conditional interventions at 60K (tab:rho_decile) ----------------------------------
    for d, (stratum, topk, tok, chop) in DECILES.items():
        for m, v in (("topk_lsm", topk), ("chop_token_only", tok), ("chop", chop)):
            add("tab:rho_decile", f"decile_{d}_{m}", "math", m, f"{m} restricted to {d}", {"aime24_gain": v},
                merge({"train": {"max_response_tokens": 60000}, "method": {"router": {"stratum": stratum}}},
                      MATH_PI if m.startswith("chop") else {}), reference="opd_60k")
    # --- anisotropy probe (tab:anisotropy): 30K, 300 steps ---------------------------------------------
    for m, (aime, gap, alpha) in ANISO.items():
        add("tab:anisotropy", f"aniso_{m}", "math", m, m, {"aime24": aime, "gap_recovery": gap, "alpha_diag": alpha},
            merge({"train": {"max_response_tokens": 30000, "steps": 300}, "data": {"eval": ["aime24"]}},
                  MATH_PI if m in ("chop", "chop_student_anchors", "chop_token_only") else {}))
    # --- SWE-bench main table (tab:swebench_main) -------------------------------------------------------
    add("tab:swebench_main", "offtheshelf", "swe", None, "Off-the-shelf Qwen3-4B-Instruct-2507",
        swe(11.8, [9.1, 14.7], 35.6), eval_only=True)
    for m, row, rep in (("opd", "Standard OPD / GKD", swe(6.5, [4.5, 8.9], 42.7)),
                        ("topk_lsm", "Top-K LSM", swe(8.2, [6.1, 10.8], 39.1)),
                        ("reopold", "REOPOLD implementation", swe(9.3, [7.0, 12.1], 38.7)),
                        ("li2026", "Li et al. full recipe", swe(9.6, [7.3, 12.4], 40.1))):
        add("tab:swebench_main", m, "swe", m, row, rep, {"method": {"pi": {"sources": []}}})
    add("tab:swebench_main", "opsd", "swe", "opsd", "OPSD-style all-PI", swe(10.1, [7.8, 12.9], 37.9), ORACLE)
    add("tab:swebench_main", "chop_deployable", "swe", "chop", "CHOP-deployable (call graph + test manifest)",
        swe(13.2, [10.5, 16.1], 33.2))
    add("tab:swebench_main", "chop_oracle", "swe", "chop", "CHOP-oracle (non-deployable ceiling)",
        swe(14.2, [11.4, 17.3], 31.4), ORACLE)
    # --- ablations on the oracle configuration (tab:ablation) ------------------------------------------
    abl = [("b_no_kappa_gate", "(b) no kappa gate", 9.1, {"method": {"router": {"kappa_gate": "none"}}}),
           ("c_no_rho_gate", "(c) no rho gate (token KL everywhere)", 11.4, {"method": {"router": {"rho_gate": "none"}}}),
           ("d_mmd", "(d) MMD replaces SW", 12.4, {"method": {"segment": {"distance": "mmd"}}}),
           ("e_three_cell", "(e) 3-cell variant", 14.3, {"method": {"router": {"three_cell": True}}}),
           ("f_static_thresholds", "(f) static thresholds at 0.5", 13.2,
            {"method": {"router": {"rho_gate": "static", "kappa_gate": "static", "static_rho": 0.5,
                                   "static_kappa": 0.5}}}),
           ("g_prompt_pi", "(g) prompt-injected PI", 12.7, {"method": {"pi": {"interface": "prompt"}}}),
           ("h_cell_a_pi", "(h) Cell A uses the PI teacher", 14.0, {"method": {"router": {"cell_a_uses_pi": True}}}),
           ("i_frozen_encoder", "(i) frozen embedding encoder", 13.8, {"method": {"segment": {"encoder": "frozen_student"}}})]
    for name, row, v, ov in abl:
        add("tab:ablation", f"ablation_{name}", "swe", "chop", row, swe(v), merge(ORACLE, ov))
    # --- PI removal and content controls (tab:swebench_pi_main) -----------------------------------------
    pim = [("no_gold_patch", "w/o gold patch", 13.3, ["test_manifest", "call_graph", "next_file"]),
           ("no_next_file", "w/o oracle next file", 12.8, ["test_manifest", "call_graph", "gold_patch"]),
           ("test_manifest_only", "test manifest only", 11.2, ["test_manifest"])]
    for name, row, v, src in pim:
        add("tab:swebench_pi_main", f"pi_{name}", "swe", "chop", row, swe(v), {"method": {"pi": {"sources": src}}})
    add("tab:swebench_pi_main", "pi_strict_filter", "swe", "chop", "stricter training filter", swe(13.9),
        merge(ORACLE, {"data": {"manifest": "data/manifests/swe_train_strict.txt"}}),
        notes="On the public splits the strict filter removes no instance (data/manifests/leakage_report.json).")
    for name, row, v, content in (("shuffled", "shuffled PI within batch", 7.6, "shuffled"),
                                  ("wrong_repo", "wrong-repo PI", 8.1, "wrong_repo"),
                                  ("random_adapter", "random adapter", 7.0, "random_adapter")):
        add("tab:swebench_pi_main", f"pi_{name}", "swe", "chop", row, swe(v),
            merge(ORACLE, {"method": {"pi": {"content": content}}}))
    # --- PI gating controls (tab:pi_gating_full) -------------------------------------------------------
    gating = [("no_pi", "no PI (reliability-only control)", 9.5, {"method": {"router": {"kappa_gate": "none"},
                                                                              "pi": {"sources": []}}}),
              ("uniform", "all PI, uniform activation", 10.1, merge(ORACLE, {"method": {"router": {"kappa_gate": "always"}}})),
              ("random", "random PI at random states", 7.8, merge(ORACLE, {"method": {"router": {"kappa_gate": "random"}}})),
              ("rate_matched", "activation-matched random placement", 8.4,
               merge(ORACLE, {"method": {"router": {"kappa_gate": "random_rate_matched"}}})),
              ("call_graph_only", "kappa-gated: call graph only", 10.7, {"method": {"pi": {"sources": ["call_graph"]}}})]
    for name, row, v, ov in gating:
        add("tab:pi_gating_full", f"gating_{name}", "swe", "chop", row, swe(v), ov)
    # --- stress controls (tab:stress_controls_full, tab:compute_parity) --------------------------------
    for name, row, v, ov in (("frozen_route", "frozen route from OPD checkpoint", 11.7,
                              {"method": {"router": {"rho_gate": "frozen"}}}),
                             ("random_quota", "random route with cell-mass quota", 9.6,
                              {"method": {"router": {"rho_gate": "random_quota"}}}),
                             ("depth_decile", "fixed depth-decile schedule", 10.4,
                              {"method": {"router": {"rho_gate": "depth_decile"}}}),
                             ("oracle_route", "oracle route", 15.0, {"method": {"router": {"rho_gate": "oracle"},
                                                                                "reliability": {"variant": "reference"}}})):
        notes = {"frozen_route": "Needs per-state routes from a frozen OPD checkpoint (pass frozen_trusted).",
                 "oracle_route": "Needs rho_ref for every state (teacher self-rollouts; see analysis)."}.get(name)
        add("tab:stress_controls_full", f"stress_{name}", "swe", "chop", row, swe(v), merge(ORACLE, ov), notes=notes)
    for regime, (chop_v, opsd_v, topk_v) in {"teacher_token": (13.6, 11.3, 9.4), "wall_clock": (13.2, 10.8, 9.1)}.items():
        for m, v in (("chop", chop_v), ("opsd", opsd_v), ("topk_lsm", topk_v)):
            add("tab:compute_parity", f"parity_{regime}_{m}", "swe", m, f"{m}, {regime} matched", swe(v),
                ORACLE if m in ("chop", "opsd") else {"method": {"pi": {"sources": []}}},
                notes=f"Set train.steps so that {regime.replace('_', ' ')} matches the baselines (measure from "
                      "the student-updates-matched runs).")
    # --- loss / estimator / sweeps (tab:loss_choice, tab:sw_gradient, tab:anchor_sweep, tab:threshold_sweep)
    for name, row, v, ov in (("mmd_k32", "MMD (K=32)", 13.7, {"method": {"segment": {"distance": "mmd", "num_anchors": 32,
                                                                                     "num_student_segments": 32}}}),
                             ("sw_l8", "sliced W. (L=8)", 14.0, {"method": {"segment": {"num_projections": 8}}}),
                             ("sw_l64", "sliced W. (L=64)", 14.2, {"method": {"segment": {"num_projections": 64}}})):
        add("tab:loss_choice", f"loss_{name}", "swe", "chop", row, swe(v), merge(ORACLE, ov))
    for est, v in (("vanilla", 14.1), ("mean", 14.5), ("loo", 14.2)):
        add("tab:sw_gradient", f"estimator_{est}", "swe", "chop", f"{est} estimator", swe(v),
            merge(ORACLE, {"method": {"segment": {"estimator": est}}}))
    for k, v in ((4, 12.8), (16, 14.4), (32, 14.5), (64, 14.6)):
        add("tab:anchor_sweep", f"anchors_k{k}", "swe", "chop", f"K = {k}", swe(v),
            merge(ORACLE, {"method": {"segment": {"num_anchors": k, "num_student_segments": k}}}))
    for la, v in ((16, 12.4), (32, 13.6), (128, 14.3), (256, 14.1), (512, 13.7)):
        add("tab:anchor_sweep", f"seglen_{la}", "swe", "chop", f"L_a = {la}", swe(v),
            merge(ORACLE, {"method": {"segment": {"segment_length": la}}}))
    for q, v in ((60, 12.4), (65, 12.9), (75, 13.0), (80, 12.7)):
        add("tab:threshold_sweep", f"kappa_q{q}", "swe", "chop", f"PI-gate percentile {q}", swe(v),
            {"method": {"router": {"kappa_quantile": q / 100}}})
    # --- reliability score used for routing (tab:reliability_purity) -----------------------------------
    for var, v in (("free", 13.9), ("entropy_only", 13.2), ("tv_only", 12.2), ("reference", 14.3)):
        add("tab:reliability_purity", f"reliability_{var}", "swe", "chop", f"rho variant: {var}", swe(v),
            merge(ORACLE, {"method": {"reliability": {"variant": var}}}))
    # --- evaluation-only rows ----------------------------------------------------------------------------
    for model, vals in (("offtheshelf", (4.6, 8.9, 11.8)), ("opd", (4.4, 6.1, 6.5)), ("chop_deployable", (4.8, 9.6, 13.2))):
        for cap, v in zip((10, 25, 50), vals, strict=True):
            add("tab:turncap", f"turncap_{model}_{cap}", "swe", None, f"{model}, {cap}-turn cap", swe(v), eval_only=True,
                eval={"checkpoint_of": model, "max_turns": cap})
    for model, v in (("offtheshelf", 10.4), ("opd", 5.8), ("chop_deployable", 11.9)):
        add("app:second_harness", f"sweagent_{model}", "swe", None, f"{model} under SWE-agent v1.1.0", swe(v),
            eval_only=True, eval={"checkpoint_of": model, "harness": "swe_agent",
                                  "instances": "data/manifests/swe_agent_300_seed0.txt"})
    for name, row, v, pi in (("no_adapter", "teacher, no PI adapter", 19.4, None),
                             ("pi_removed", "LoRA-fitted teacher, PI-removed (sentinel)", 20.0, "sentinel"),
                             ("deployable", "LoRA-fitted teacher, deployable PI", 22.6, "swe_deployable"),
                             ("oracle", "LoRA-fitted teacher, oracle PI", 28.4, "swe_oracle")):
        add("tab:pi_removed_audit", f"teacher_audit_{name}", "swe", None, row, swe(v), eval_only=True,
            eval={"harness": "teacher_audit", "pi": pi})
    # --- student/teacher pairs (tab:alpha_pairs_full; SWE-bench rows) -----------------------------------
    pairs = [("qwen3_1.7b__qwen3_8b", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B", 0.06, 0.74, 3.8, 8.2, None),
             ("qwen3_4b_instruct__qwen3_8b_base", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B-Base", 0.13, 0.75,
              6.5, 13.2, None),
             ("r1_distill_1.5b__7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 0.05, 0.71, 2.9, 7.8, None),
             ("qwen3_8b__qwen3_14b", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", 0.27, 0.78, 9.1, 13.6, None),
             ("llama3_8b__llama3_70b", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B", 0.18, 0.74, 5.7,
              10.4, "A 70B teacher does not fit replicated per GPU; this recipe does not support it."),
             ("qwen3_4b_instruct__qwen3_14b", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-14B", 0.17, 0.77, 7.2, 11.8,
              None),
             ("r1_distill_7b__14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 0.25, 0.76, 8.4, 12.9, None),
             ("mixtral_8x7b__8x22b", "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x22B-v0.1", 0.33, 0.73, 10.8,
              12.2, "A 141B-parameter teacher does not fit replicated per GPU; this recipe does not support it."),
             ("qwen3_14b__qwen3_14b_distill", "Qwen/Qwen3-14B", "Qwen3-14B-Distill (no public checkpoint)", 0.45,
              0.74, 12.4, 13.8, "The teacher checkpoint is not public; supply model.teacher."),
             ("qwen3_8b__qwen3_8b_base", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base", 0.38, 0.76, 11.9, 13.4, None),
             ("llama3_70b__llama3_70b_instruct", "meta-llama/Meta-Llama-3-70B", "meta-llama/Meta-Llama-3-70B-Instruct",
              0.49, 0.79, 14.1, 14.9, "70B student and teacher; not supported by this recipe.")]
    for key, student, teacher, alpha_conv, auroc, opd_v, chop_v, notes in pairs:
        model = {"model": {"student": student, "teacher": teacher}}
        add("tab:alpha_pairs_full", f"pair_{key}_opd", "swe", "opd", f"{key}: standard OPD",
            {"resolve": opd_v, "alpha_conv": alpha_conv, "auroc": auroc},
            merge(model, {"method": {"pi": {"sources": []}}, "train": {"steps": 4000}}), notes=notes)
        add("tab:alpha_pairs_full", f"pair_{key}_chop", "swe", "chop", f"{key}: CHOP-deployable", {"resolve": chop_v},
            model, notes=notes)
    return out


def main() -> None:
    for sub in ("base", "method", "experiments"):
        shutil.rmtree(ROOT / sub, ignore_errors=True)
        (ROOT / sub).mkdir(parents=True)
    for name, cfg in BASE.items():
        (ROOT / "base" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name, cfg in METHOD.items():
        (ROOT / "method" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    n = 0
    for table, stem, cfg in rows():
        d = ROOT / "experiments" / table
        d.mkdir(parents=True, exist_ok=True)
        header = "# Generated by scripts/gen_experiment_configs.py - edit the generator, not this file.\n"
        (d / f"{stem}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        n += 1
    print(f"wrote {n} experiment configs")


if __name__ == "__main__":
    main()
