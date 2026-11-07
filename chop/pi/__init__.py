"""Privileged-information (PI) sources, serialisation, multi-LoRA composition and controls."""

from chop.pi.composition import compose_logits, kappa_hat
from chop.pi.serialize import NO_PI_SENTINEL, PI_CLOSE, PI_OPEN, format_pi_block
from chop.pi.sources import PROTOCOLS, SOURCES, PISource, protocol_sources

__all__ = [
    "compose_logits",
    "kappa_hat",
    "NO_PI_SENTINEL",
    "PI_CLOSE",
    "PI_OPEN",
    "format_pi_block",
    "PROTOCOLS",
    "SOURCES",
    "PISource",
    "protocol_sources",
]
