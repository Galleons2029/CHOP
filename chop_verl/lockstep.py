"""Keep FSDP collectives aligned when ranks hold different amounts of CHOP work.

Every FSDP forward all-gathers parameters and every backward reduce-scatters gradients, so all ranks
must issue the same number of student forward/backward calls in each phase.  A rank computes how many
it will issue (:func:`chop_verl.compute.student_forwards`), the maximum is all-reduced, and ranks with
less work run zero-gradient dummy passes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist


def global_max(n: int, group=None, device=None) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return n
    t = torch.tensor([n], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return int(t.item())


def pad_to_global_max(n_local: int, dummy: Callable[[], None], group=None, device=None) -> int:
    """Run ``max_over_ranks(n_local) - n_local`` dummy passes; returns the number run."""
    n_max = global_max(n_local, group, device)
    for _ in range(n_max - n_local):
        dummy()
    return n_max - n_local
