"""CHOP: Counterfactual Hierarchical On-Policy Distillation.

The ``chop`` package is framework-agnostic PyTorch: divergences, the reliability proxy, the
streaming-quantile two-gate router, the segment (sliced-Wasserstein) branch, PI composition and
the loss assembly.  Distributed training lives in ``chop_verl``; ``chop.reference_trainer`` is a
single-process loop used for tests and as a correctness reference.
"""

__version__ = "0.1.0"
