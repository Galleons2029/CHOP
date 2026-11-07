"""Train one PI adapter (App. C.2): rank-16 LoRA over a frozen teacher, CE on the action span only.

  python -m adapters.train_lora --base Qwen/Qwen3-8B-Base --triples data/triples/call_graph.jsonl \
      --source call_graph --out outputs/adapters/swe/call_graph

Defaults reproduce the paper: r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down projections; AdamW
(0.9, 0.95), weight decay 0.1, peak lr 1e-4, cosine schedule with 200 warmup steps, 3 epochs, 32 sequences
of up to 4096 tokens per step with 4 accumulation steps.  Multi-GPU: launch with torchrun / accelerate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from chop.pi.serialize import format_pi_block

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


def encode_triple(tok, t: dict, max_len: int = 4096) -> dict | None:
    """ids = [PI] block + state + action; labels mask everything but the action span."""
    prefix = tok.encode(format_pi_block(t["source"], t["pi"]), add_special_tokens=False)
    state = tok.encode(t["state"], add_special_tokens=False)
    action = tok.encode(t["action"], add_special_tokens=False)
    if tok.eos_token_id is not None:
        action = action + [tok.eos_token_id]
    ctx = prefix + state
    budget = max_len - len(action)
    if budget <= len(prefix):
        return None
    ctx = prefix + state[-(budget - len(prefix)):] if len(ctx) > budget else ctx
    return {"input_ids": ctx + action, "labels": [-100] * len(ctx) + action}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    labels = torch.full((len(batch), n), -100)
    mask = torch.zeros(len(batch), n, dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def add_lora(model, source: str, spec: LoraSpec | None = None, targets=TARGETS):
    import peft

    spec = spec or LoraSpec()

    cfg = peft.LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                          target_modules=[t for t in targets if any(n.endswith(t) for n, _ in model.named_modules())],
                          task_type="CAUSAL_LM")
    return peft.get_peft_model(model, cfg, adapter_name=source)


def train(model, tok, triples: list[dict], out_dir: str | None, epochs: int = 3, lr: float = 1e-4,
          batch_size: int = 32, grad_accum: int = 4, warmup: int = 200, max_len: int = 4096,
          micro_batch: int = 1, log=print) -> list[float]:
    """Plain PyTorch loop (single process); returns per-update losses."""
    data = [e for t in triples if (e := encode_triple(tok, t, max_len)) is not None]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    per_update = batch_size * grad_accum
    total = max(1, epochs * ((len(data) + per_update - 1) // per_update))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup else
        0.5 * (1 + torch.cos(torch.tensor(torch.pi * (s - warmup) / max(total - warmup, 1))).item()))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data)).tolist()
        for u in range(0, len(order), per_update):
            idx = order[u:u + per_update]
            opt.zero_grad()
            tot = 0.0
            for m in range(0, len(idx), micro_batch):
                b = collate([data[i] for i in idx[m:m + micro_batch]], pad)
                loss = model(**b).loss * (len(idx[m:m + micro_batch]) / len(idx))
                loss.backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(tot)
            log({"update": len(losses), "loss": tot, "lr": sched.get_last_lr()[0]})
    if out_dir:
        model.save_pretrained(out_dir)
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--triples", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = add_lora(model, args.source)
    triples = [json.loads(line) for line in open(args.triples)]
    triples = [t for t in triples if t["source"] == args.source]
    train(model, tok, triples, args.out, args.epochs, args.lr, max_len=args.max_len)


if __name__ == "__main__":
    main()
