"""SWE-bench datasets (public Hugging Face releases)."""

from __future__ import annotations

VERIFIED = ("princeton-nlp/SWE-bench_Verified", "test")
TRAIN = ("princeton-nlp/SWE-bench", "train")


def load(which: str):
    from datasets import load_dataset

    hub, split = {"verified": VERIFIED, "train": TRAIN}[which]
    return load_dataset(hub, split=split)


def rows(which: str) -> list[dict]:
    return [dict(r) for r in load(which)]
