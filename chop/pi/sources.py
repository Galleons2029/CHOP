"""PI source registry and evaluation protocols (Sec. 4.2; App. C.1).

A PI source is a deterministic map from an instance to an auxiliary text block that only the
teacher sees.  ``deployable`` means computable from what the held-out runtime exposes before
solving (no gold patch, no oracle answer).  Extractors live in ``pi_extraction``; this module
only fixes names, deployability and the protocol presets used by the experiment configs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PISource:
    name: str
    task: str
    deployable: bool
    description: str
    per_step: bool = False   # recomputed at every agent step (oracle next file)


SOURCES: dict[str, PISource] = {
    s.name: s
    for s in [
        PISource("gold_patch", "swe", False, "p1: gold unified diff restricted to the files it modifies"),
        PISource("test_manifest", "swe", True,
                 "p2: FAIL_TO_PASS and PASS_TO_PASS test-function names (names and pass/fail only)"),
        PISource("next_file", "swe", False,
                 "p3: path + top-level docstring of the next gold-modified file given the edit log",
                 per_step=True),
        PISource("call_graph", "swe", True,
                 "p4: symbol-level call graph reachable from the issue-referenced module"),
        PISource("final_answer", "math", False, "ground-truth final answer (only math PI source)"),
    ]
}

# Protocol presets.  Adapters for all sources are fitted on the training split; the protocol
# decides which sources the teacher may condition on.
PROTOCOLS: dict[str, list[str]] = {
    "none": [],
    "swe_deployable": ["call_graph", "test_manifest"],
    "swe_oracle": ["test_manifest", "call_graph", "next_file", "gold_patch"],
    "swe_call_graph_only": ["call_graph"],
    "swe_test_manifest_only": ["test_manifest"],
    "swe_no_patch": ["test_manifest", "call_graph", "next_file"],
    "swe_no_next_file": ["test_manifest", "call_graph", "gold_patch"],
    "math": ["final_answer"],
}


def protocol_sources(name_or_list: str | list[str]) -> list[PISource]:
    names = PROTOCOLS[name_or_list] if isinstance(name_or_list, str) else list(name_or_list)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise KeyError(f"unknown PI sources: {unknown}")
    return [SOURCES[n] for n in names]


def is_deployable(names: list[str]) -> bool:
    return all(SOURCES[n].deployable for n in names)
