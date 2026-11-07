"""verl AgentLoops used by CHOP.

``chop_rolling_math``: single-turn generation to a cumulative token budget beyond the native window.
Generation proceeds in the chunks of :func:`chop.truncation.rolling_chunks`; before each chunk the
context is re-prefilled with the left-truncated prefix (token-in / token-out through the rollout
server), which is exactly the conditioning the CHOP scoring pass replays.  ``plan_prompt_len`` (from
the dataset's extra_info) is the longest teacher-side prompt + PI prefix, so the same plan fits the
teacher calls too.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from chop.truncation import rolling_chunks


@register("chop_rolling_math")
class RollingMathAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from chop_verl.worker import method_config

        chop = self.config.actor_rollout_ref.chop
        self.window = int(chop.native_context)
        self.chunk = int(chop.rollout_chunk)
        self.reserve = method_config(chop.method, chop.get("method_file")).segment.segment_length
        self.budget = int(self.rollout_config.response_length)

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.ct_build_initial_tokens(messages)
        extra = kwargs.get("extra_info") or {}
        plan_len = max(len(prompt_ids), int(extra.get("plan_prompt_len", 0)))
        eos = self.tokenizer.eos_token_id
        response: list[int] = []
        for c in rolling_chunks(plan_len, self.budget, self.window, self.chunk, self.reserve):
            params = dict(sampling_params, max_tokens=c.gen_end - c.gen_start)
            out = await self.server_manager.generate(request_id=uuid4().hex,
                                                     prompt_ids=prompt_ids + response[c.ctx_start:c.gen_start],
                                                     sampling_params=params, priority=int(priority))
            response.extend(out.token_ids)
            if not out.token_ids or out.token_ids[-1] == eos or len(out.token_ids) < c.gen_end - c.gen_start:
                break
        response = response[:self.budget]
        return AgentLoopOutput(prompt_ids=prompt_ids, response_ids=response, response_mask=[1] * len(response),
                               num_turns=2, metrics={}, extra_fields={"turn_scores": [], "tool_rewards": []})


@register("chop_openhands")
class OpenHandsAgentLoop(AgentLoopBase):
    """SWE-bench rollouts: CodeAct tools in the instance container, graded by the official harness.

    Per-sample inputs come from ``extra_info`` (scripts/prepare_swe.py): ``instance`` (SWE-bench fields),
    ``image`` (instance image) and ``plan_prompt_len``. The trajectory's reward R(tau) is the harness's
    ``resolved`` flag. When the oracle ``next_file`` source is configured, the loop also emits its
    per-chunk PI prefixes, computed from the edit log at each chunk start. These use the student
    tokenizer, which requires a tokenizer-compatible student/teacher pair (the Qwen3 pairs are).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from chop_verl.worker import method_config

        chop = self.config.actor_rollout_ref.chop
        method = method_config(chop.method, chop.get("method_file"))
        oh = chop.get("openhands", {})
        self.window, self.chunk = int(chop.native_context), int(chop.rollout_chunk)
        self.reserve = method.segment.segment_length
        self.budget = int(self.rollout_config.response_length)
        self.max_turns = int(oh.get("max_turns", 50))
        self.time_limit = float(oh.get("timeout_s", 600))
        self.network = bool(oh.get("internet", False))
        self.sources = list(method.pi.sources)

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        import asyncio

        from chop.pi.serialize import format_pi_block
        from envs.swe.codeact import ChatRenderer, run_episode
        from envs.swe.grading import grade, instance_image
        from envs.swe.sandbox import DockerSandbox
        from pi_extraction.sources import next_file

        extra = kwargs.get("extra_info") or {}
        inst = dict(extra["instance"])
        sb = await asyncio.to_thread(DockerSandbox, extra.get("image") or instance_image(inst["instance_id"]),
                                     "/testbed", self.network)

        async def generate(ctx: list[int], max_tokens: int) -> list[int]:
            out = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=ctx,
                                                     sampling_params=dict(sampling_params, max_tokens=max_tokens),
                                                     priority=int(priority))
            return list(out.token_ids)

        try:
            res = await run_episode(generate, ChatRenderer(self.tokenizer), sb, inst["problem_statement"],
                                    base_commit=inst["base_commit"], max_turns=self.max_turns,
                                    time_limit=self.time_limit, window=self.window, chunk=self.chunk,
                                    reserve=self.reserve, plan_len=int(extra.get("plan_prompt_len", 0)),
                                    budget=self.budget)
            per_chunk = {}
            if "next_file" in self.sources:
                plan = rolling_chunks(max(len(res.prompt_ids), int(extra.get("plan_prompt_len", 0))),
                                      max(len(res.response_ids), 1), self.window, self.chunk, self.reserve)
                blocks = []
                for c in plan:
                    edited = [p for pos, p in res.edit_events if pos <= c.gen_start]
                    text = await asyncio.to_thread(next_file, inst["patch"], edited, sb.read)
                    blocks.append(self.tokenizer.encode(format_pi_block("next_file", text), add_special_tokens=False))
                per_chunk["next_file"] = blocks
        finally:
            await asyncio.to_thread(sb.close)
        report = await asyncio.to_thread(grade, inst, res.patch, f"chop-{uuid4().hex[:8]}")
        return AgentLoopOutput(prompt_ids=res.prompt_ids, response_ids=res.response_ids[:self.budget],
                               response_mask=res.response_mask[:self.budget], reward_score=float(report["resolved"]),
                               num_turns=2 * res.turns + 1, metrics={},
                               extra_fields={"resolved": bool(report["resolved"]), "turns": res.turns,
                                             "timed_out": res.timed_out, "pi_prefix_ids_per_chunk": per_chunk,
                                             "turn_scores": [], "tool_rewards": []})
