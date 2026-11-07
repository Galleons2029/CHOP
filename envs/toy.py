"""A character-level arithmetic task for CPU smoke tests with tiny random models.

It exercises every code path of the trainer (rewards, PI prefixes, rolling truncation, all four
cells) without downloads; it says nothing about CHOP's accuracy.
"""

from __future__ import annotations

import random

from envs.base import Instance

PAD, BOS, EOS = 0, 1, 2
_OFFSET = 3


class ToyTokenizer:
    """Printable ASCII (32..126) mapped to ids 3..97; vocabulary size 128."""

    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS
    vocab_size = 128

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [_OFFSET + ord(c) - 32 for c in text if 32 <= ord(c) <= 126]
        return ([BOS] if add_special_tokens else []) + ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i < _OFFSET:
                if not skip_special_tokens:
                    out.append(["<pad>", "<bos>", "<eos>"][i])
                continue
            if i - _OFFSET + 32 <= 126:
                out.append(chr(i - _OFFSET + 32))
        return "".join(out)


class ToyArithmetic:
    name = "toy"

    def __init__(self, n_train: int = 256, n_eval: int = 16, seed: int = 0, max_operand: int = 50):
        rng = random.Random(seed)

        def make(i: int, split: str) -> Instance:
            a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
            return Instance(id=f"{split}-{i}", question=f"{a}+{b}", answer=str(a + b))

        self._train = [make(i, "train") for i in range(n_train)]
        self._eval = [make(i, "eval") for i in range(n_eval)]

    def train_instances(self) -> list[Instance]:
        return self._train

    def eval_instances(self, split: str = "eval") -> list[Instance]:
        return self._eval

    def prompt(self, inst: Instance) -> str:
        return f"Q:{inst.question}=? A:"

    def reward(self, inst: Instance, response: str) -> float:
        digits = "".join(c for c in response.split(" ")[0] if c.isdigit())
        return float(digits == inst.answer)

    def pi_blocks(self, inst: Instance) -> dict[str, str]:
        return {"final_answer": inst.answer, "hint": inst.question.replace("+", " plus ")}
