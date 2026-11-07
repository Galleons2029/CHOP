"""Randomly initialised tiny Qwen3 models for tests and CPU smoke runs (no downloads)."""

from __future__ import annotations

import torch


def tiny_qwen3(
    vocab_size: int = 96,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    max_position_embeddings: int = 1024,
    seed: int = 0,
    attn_implementation: str = "eager",
):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    cfg._attn_implementation = attn_implementation
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model
