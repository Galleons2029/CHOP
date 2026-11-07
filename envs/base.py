"""Task interface shared by the reference trainer and the verl recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Instance:
    id: str
    question: str
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str

    def train_instances(self) -> list[Instance]: ...

    def eval_instances(self, split: str) -> list[Instance]: ...

    def prompt(self, inst: Instance) -> str: ...

    def reward(self, inst: Instance, response: str) -> float: ...

    def pi_blocks(self, inst: Instance) -> dict[str, str]: ...


class TextTokenizer(Protocol):
    eos_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids, skip_special_tokens: bool = True) -> str: ...
