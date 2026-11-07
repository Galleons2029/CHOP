"""Token-in / token-out generation backends for evaluation.

VLLMBackend        offline batched generation (math avg@16)
VLLMAsyncBackend   concurrent agent episodes (SWE-bench)
HFBackend          small models on CPU (tests / smoke)
HFCompositeBackend frozen teacher + PI adapters composed at the logit level (teacher-alone audit);
                   vLLM cannot compose several LoRA adapters with different prefixes, so this is slow.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Sequence

import torch


class VLLMBackend:
    def __init__(self, model: str, max_model_len: int = 32768, seed: int = 0, **engine_kwargs):
        from vllm import LLM

        self.llm = LLM(model=model, max_model_len=max_model_len, seed=seed, **engine_kwargs)

    def generate_batch(self, contexts: Sequence[list[int]], max_tokens: Sequence[int], temperature: float,
                       top_p: float, seeds: Sequence[int]) -> list[list[int]]:
        from vllm import SamplingParams

        params = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=int(m), seed=int(s))
                  for m, s in zip(max_tokens, seeds, strict=True)]
        outs = self.llm.generate([{"prompt_token_ids": list(c)} for c in contexts], params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


class VLLMAsyncBackend:
    def __init__(self, model: str, max_model_len: int = 32768, seed: int = 0, temperature: float = 0.7,
                 top_p: float = 0.95, **engine_kwargs):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(model=model, max_model_len=max_model_len, seed=seed, **engine_kwargs))
        self.temperature, self.top_p, self.seed = temperature, top_p, seed

    async def generate(self, context: list[int], max_tokens: int) -> list[int]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=int(max_tokens))
        final = None
        async for out in self.engine.generate({"prompt_token_ids": list(context)}, sp, uuid.uuid4().hex):
            final = out
        return list(final.outputs[0].token_ids) if final else []


class HFBackend:
    def __init__(self, model, eos_id: int | None, seed: int = 0):
        self.model, self.eos_id = model, eos_id
        self.gen = torch.Generator().manual_seed(seed)

    def generate_batch(self, contexts, max_tokens, temperature, top_p, seeds):
        from chop.rollout import sample_segment

        return [sample_segment(self.model, c, int(m), temperature, top_p, self.eos_id, self.gen)
                for c, m in zip(contexts, max_tokens, strict=True)]

    def async_generate(self, temperature: float = 0.7, top_p: float = 0.95) -> Callable:
        async def generate(context: list[int], max_tokens: int) -> list[int]:
            return self.generate_batch([context], [max_tokens], temperature, top_p, [0])[0]

        return generate


class HFCompositeBackend:
    """Teacher-alone audit: base teacher + the given PI adapters, each with its own instance prefix."""

    def __init__(self, teacher, pad_id: int, eos_id: int | None, temperature: float = 0.7, seed: int = 0):
        from chop.pi.composition import PIConditionalTeacher

        self.t = PIConditionalTeacher(teacher, pad_id=pad_id)
        self.eos_id, self.temperature = eos_id, temperature
        self.gen = torch.Generator().manual_seed(seed)

    def for_instance(self, pi_prefix: dict[str, list[int]], sources: list[str]) -> Callable:
        async def generate(context: list[int], max_tokens: int) -> list[int]:
            toks, lens = await asyncio.to_thread(self.t.sample, context, pi_prefix, sources, 1, max_tokens,
                                                 self.temperature, self.eos_id, self.gen)
            return toks[0, :int(lens[0])].tolist()

        return generate
