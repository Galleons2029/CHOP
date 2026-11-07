"""(state, PI block, expert action) triples for PI adapter training (App. C.2).

SWE-bench: the base teacher runs a 50-turn OpenHands trajectory on a training instance; a step becomes an
expert target a* when its edit overlaps a gold-patch hunk (same file, overlapping old-line range).  Each
target is paired with the deterministic PI block p_i(x) of every source (p3 is recomputed from the
edit log at that step).  States whose block is empty (PI-irrelevant) are kept with probability 0.4.

Math: (problem, [PI] final answer) -> the teacher's verified-correct solution.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

from pi_extraction import sources
from pi_extraction.diffs import Hunk, hunks


@dataclass
class Edit:
    path: str
    start: int          # first old-file line touched (1-based)
    end: int            # exclusive


@dataclass
class Step:
    state: str          # rendered context (or a reference to token ids) preceding the action
    action: str         # the teacher's emitted action text
    edits: list[Edit]


@dataclass
class Triple:
    instance_id: str
    source: str
    state: str
    pi: str
    action: str


def overlaps(edit: Edit, hunk: Hunk) -> bool:
    lo, hi = hunk.old_range
    return edit.path == hunk.path and edit.start < hi and lo < edit.end


def expert_steps(steps: Iterable[Step], gold_patch: str) -> list[tuple[int, Step]]:
    """Steps whose edits overlap the gold patch (index into the trajectory, step)."""
    gold = hunks(gold_patch)
    return [(i, s) for i, s in enumerate(steps) if any(overlaps(e, h) for e in s.edits for h in gold)]


def swe_triples(instance: dict, steps: list[Step], source_names: list[str], read_file: Callable[[str], str] | None,
                call_graph: str | None, rng: random.Random, keep_irrelevant: float = 0.4) -> list[Triple]:
    out = []
    edited: list[str] = []
    expert = dict(expert_steps(steps, instance["patch"]))
    for i, step in enumerate(steps):
        if i in expert:
            for src in source_names:
                block = {
                    "gold_patch": lambda: sources.gold_patch(instance["patch"]),
                    "test_manifest": lambda: sources.test_manifest(instance["FAIL_TO_PASS"], instance["PASS_TO_PASS"]),
                    "next_file": lambda: sources.next_file(instance["patch"], edited, read_file),
                    "call_graph": lambda: call_graph or "",
                }[src]()
                if not block.strip() and rng.random() > keep_irrelevant:
                    continue
                out.append(Triple(instance["instance_id"], src, step.state, block, step.action))
        edited.extend(e.path for e in step.edits if e.path not in edited)
    return out


def math_triples(instance_id: str, problem_prompt: str, answer: str, solutions: list[str],
                 is_correct: Callable[[str, str], bool]) -> list[Triple]:
    return [Triple(instance_id, "final_answer", problem_prompt, sources.final_answer(answer), s)
            for s in solutions if is_correct(s, answer)]


def write_jsonl(triples: Iterable[Triple], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for t in triples:
            f.write(json.dumps(asdict(t)) + "\n")
            n += 1
    return n
