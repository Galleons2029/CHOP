"""Answer extraction and verification for the math benchmarks.

Responses are accepted in either the DAPO format (a final line ``Answer: X``) or LaTeX
``\\boxed{X}``.  Verification uses ``math-verify`` when installed and otherwise falls back to
normalised string comparison (reported in the run metadata so results are not mixed).
"""

from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"Answer\s*:\s*(.+)", re.IGNORECASE)


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def extract_answer(response: str) -> str | None:
    boxed = last_boxed(response)
    if boxed is not None:
        return boxed.strip()
    hits = _ANSWER_RE.findall(response)
    if hits:
        return hits[-1].strip().rstrip(".").strip("$ ")
    return None


def _normalise(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace("\\!", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def verifier_backend() -> str:
    try:
        import math_verify  # noqa: F401

        return "math_verify"
    except ImportError:
        return "string"


def is_correct(response: str, gold: str) -> bool:
    pred = extract_answer(response)
    if pred is None:
        return False
    if verifier_backend() == "math_verify":
        from math_verify import parse, verify

        gold_p = parse(f"${gold}$")
        pred_p = parse(f"${pred}$")
        if gold_p and pred_p:
            return bool(verify(gold_p, pred_p))
    return _normalise(pred) == _normalise(gold)
