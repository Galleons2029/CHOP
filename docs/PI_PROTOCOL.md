# Privileged information (PI) protocols

## Sources

| Source | Content | Deployable | Where it comes from |
|---|---|---|---|
| `gold_patch` (p1) | Gold unified diff, restricted to the non-test files it modifies | no | SWE-bench `patch` |
| `test_manifest` (p2) | FAIL_TO_PASS / PASS_TO_PASS test-function names only (no test source) | yes | SWE-bench fields (Verified). Training split: executed test patch in the reconstructed environment |
| `next_file` (p3) | The next gold-modified file not yet edited, plus its top-level docstring; recomputed per chunk from the edit log | no | Gold patch + agent edit log |
| `call_graph` (p4) | Symbol-level call graph reachable from the modules the issue references; test files excluded | yes | tree-sitter over the base commit |
| `final_answer` (math) | Ground-truth final answer | no | Dataset |

## What "deployable" means

"Deployable" means computable from what the held-out runtime exposes before solving, without the gold patch or an
oracle answer.

The student **never** receives PI. The protocol only selects which sources the *teacher* may condition on:

- `swe_deployable`: call graph + test manifest.
- `swe_oracle`: all four sources.
- The teacher-alone audit (`eval/teacher_audit.py`) is the only evaluation that gives PI to a model at test
  time.

## Interface

1. Each source has one LoRA adapter (rank 16, α = 32, dropout 0.05, on q/k/v/o/gate/up/down).
2. A PI-conditioned teacher call places the instance block `[PI source=<name>] … [/PI]` before the prompt and
   activates that source's adapter.
3. For a set P of sources, the composite teacher is `base + Σ_{i∈P} (logits_i − base)` (Eq. 14), where each
   `logits_i` comes from its own adapter and its own prefix.
4. The PI gate uses κ̂ = TV(base, composite).

## Controls

Set these with `method.pi.content`:

| Control | What the teacher receives |
|---|---|
| `shuffled` | The PI blocks of other instances in the batch (a derangement) |
| `wrong_repo` | A block from an instance of a different repository |
| `sentinel` | `[NO_PI]` in place of the content |
| `random_adapter` | LoRA weights redrawn at matched norm |

## Leakage filter

`scripts/build_manifests.py` applies these rules to the training split:

- The default filter removes training instances that share a base commit, or a non-generic modified path, with a
  Verified instance.
- The strict filter also removes instances from the same repository.

On the public releases the two splits share no repository and no base commit, so both filters keep all
19,008 training instances. `data/manifests/leakage_report.json` records every rule's hit count.
