"""Long-horizon math: DAPO-Math-17K training, AIME 2024/2025 and AMC 2023 evaluation (avg@16).

Hub ids and field names are listed in ``DATASETS``; verify them against the Hub before a run
(dataset cards occasionally rename columns).  The student prompt is the DAPO template; the teacher
(R1-0528-Qwen3-8B) sees the same text rendered with its own chat template.
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.base import Instance
from envs.math.reward import is_correct


@dataclass(frozen=True)
class DatasetSpec:
    hub_id: str
    split: str
    question: str
    answer: str
    config: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "dapo_math_17k": DatasetSpec("BytedTsinghua-SIA/DAPO-Math-17k", "train", "prompt", "reward_model"),
    "aime24": DatasetSpec("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "aime25": DatasetSpec("yentinglin/aime_2025", "train", "problem", "answer"),
    "amc23": DatasetSpec("math-ai/amc23", "test", "question", "answer"),
}

EXPECTED_SIZES = {"aime24": 30, "aime25": 30, "amc23": 40}

DAPO_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def _dapo_fields(row) -> tuple[str, str]:
    prompt = row["prompt"]
    text = prompt[-1]["content"] if isinstance(prompt, list) else str(prompt)
    # DAPO rows already contain the full template; recover the bare problem between the blank lines
    parts = text.split("\n\n")
    problem = parts[1] if len(parts) >= 3 else text
    gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else row["reward_model"]
    return problem, str(gt)


def load_instances(name: str, limit: int | None = None) -> list[Instance]:
    from datasets import load_dataset

    spec = DATASETS[name]
    ds = load_dataset(spec.hub_id, spec.config, split=spec.split)
    out, seen = [], set()
    for i, row in enumerate(ds):
        if name == "dapo_math_17k":
            q, a = _dapo_fields(row)
            if q in seen:            # the released file repeats problems; keep unique ones
                continue
            seen.add(q)
        else:
            q, a = str(row[spec.question]), str(row[spec.answer])
        out.append(Instance(id=f"{name}-{i}", question=q, answer=a))
        if limit is not None and len(out) >= limit:
            break
    if name in EXPECTED_SIZES and limit is None and len(out) != EXPECTED_SIZES[name]:
        raise ValueError(f"{name}: expected {EXPECTED_SIZES[name]} problems, got {len(out)}")
    return out


class MathTask:
    name = "math"

    def __init__(self, train: str = "dapo_math_17k", evals: tuple[str, ...] = ("aime24", "aime25", "amc23"),
                 train_limit: int | None = None):
        self.train_name, self.eval_names, self.train_limit = train, evals, train_limit
        self._cache: dict[str, list[Instance]] = {}

    def _get(self, name: str, limit: int | None = None) -> list[Instance]:
        if name not in self._cache:
            self._cache[name] = load_instances(name, limit)
        return self._cache[name]

    def train_instances(self) -> list[Instance]:
        return self._get(self.train_name, self.train_limit)

    def eval_instances(self, split: str) -> list[Instance]:
        return self._get(split)

    def prompt(self, inst: Instance) -> str:
        return DAPO_TEMPLATE.format(problem=inst.question)

    def reward(self, inst: Instance, response: str) -> float:
        return float(is_correct(response, inst.answer))

    def pi_blocks(self, inst: Instance) -> dict[str, str]:
        return {"final_answer": inst.answer}
