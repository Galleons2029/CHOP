# CHOP: Counterfactual Hierarchical On-Policy Distillation

Official code for the paper **When Token Supervision Fails: Segment-Level On-Policy Distillation for Long-Horizon Agents** (Main of AACL).

> 🚧 **Code coming soon.** We are cleaning up the code for release. Watch or star this repo to get notified when it is available.

## Overview

On-policy distillation works well on short tasks but breaks down on long ones, such as fixing a bug across a whole repository. The teacher can still tell good attempts from bad ones, but its step-by-step corrections stop leading the student toward success. CHOP decides at each step what the student should imitate: the teacher's next token where that advice is reliable, and the teacher's behaviour over a short segment where it is not. It also gives the teacher privileged context (test outcomes, a reference answer) only at steps where that context changes what the teacher would do.

## Planned Release

- [ ] CHOP training framework (PyTorch + DeepSpeed)
- [ ] Privileged-information LoRA adapters (SWE-bench and math)
- [ ] Student checkpoints for all reported configurations and baselines
- [ ] Leakage-filter and strict-filter data manifests
- [ ] Evaluation wrappers for AIME 2024/2025, AMC 2023, and SWE-bench Verified

## License

Code: Apache-2.0. Data manifests: CC-BY-4.0.

## Citation

BibTeX will be added once the paper is published.
