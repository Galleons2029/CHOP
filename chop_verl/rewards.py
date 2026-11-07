"""verl ``custom_reward_function`` entry points (binary outcome rewards R(tau) in {0, 1})."""

from __future__ import annotations


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None, **_kw) -> float:
    if data_source.startswith("swe"):
        # SWE-bench rewards come from the harness run inside the agent loop (extra_info["resolved"])
        return float(bool((extra_info or {}).get("resolved", False)))
    from envs.math.reward import is_correct

    return float(is_correct(solution_str, str(ground_truth)))
