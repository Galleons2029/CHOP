"""Parameter-subset gradients shared by the alpha, reset-rollout, SNR and gradient-alignment analyses.

The paper's alignment cosines are parameter-space quantities without a stated parameter subset.
By default we use the last decoder block plus the LM head (``subset='last_block'``); ``'all'`` uses every
trainable parameter. Report the subset with any number.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def param_subset(model, subset: str = "last_block") -> list[torch.nn.Parameter]:
    if subset == "all":
        return [p for p in model.parameters() if p.requires_grad]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = list(base.model.layers[-1].parameters()) + list(base.get_output_embeddings().parameters())
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def flat_grad(scalar: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float()
                      for g, p in zip(grads, params, strict=True)])


def last_logprobs(model, context: list[int]) -> torch.Tensor:
    """log pi(. | context) at the next position, with gradient."""
    return F.log_softmax(model(input_ids=torch.tensor([context])).logits[0, -1].float(), -1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))
