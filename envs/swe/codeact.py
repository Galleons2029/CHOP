"""SWE-bench episodes with an OpenHands-compatible CodeAct tool set, driven token-in/token-out.

The student generates assistant turns through ``generate(context_ids, max_tokens)``. Tool calls use the
Hermes ``<tool_call>{json}</tool_call>`` format of the Qwen chat templates. Observations are appended
as chat-template tool turns, and their tokens are masked out of the loss (response_mask = 0).

Generation follows the rolling chunk plan of :mod:`chop.truncation`, so every generated token is
conditioned exactly as the CHOP scoring pass conditions it. Limits follow the reproducibility card:
50 assistant turns, 600 s wall-clock per instance (timeout -> fail), no retries.

The tool set mirrors OpenHands' CodeAct agent (``execute_bash``, ``str_replace_editor``, ``finish``).
The prompts are this repository's own. Headline numbers are evaluated with the official OpenHands
harness (``eval/swe_openhands.sh``), not with this loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from chop.truncation import rolling_chunks
from envs.swe.editor import Editor
from envs.swe.sandbox import Sandbox, model_patch

SYSTEM_PROMPT = (
    "You are a software engineering agent working in a Linux sandbox. You solve the user's task by calling "
    "tools, one call per turn, and you finish by calling `finish`. Explore the repository before editing, make "
    "minimal changes to non-test files, and verify your fix by running code. Do not modify existing tests."
)

USER_TEMPLATE = (
    "<uploaded_files>\n{workdir}\n</uploaded_files>\nA Python repository is available at {workdir}. Consider the "
    "following issue description:\n\n<issue_description>\n{problem}\n</issue_description>\n\nMake the changes to "
    "non-test files in {workdir} that resolve the issue. Changes to test files described in the issue have "
    "already been taken care of; do not modify the tests."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_bash",
        "description": "Run a bash command in the repository sandbox and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120)."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": "View, create and edit files: view | create | str_replace | insert | undo_edit.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
            "path": {"type": "string"}, "file_text": {"type": "string"}, "old_str": {"type": "string"},
            "new_str": {"type": "string"}, "insert_line": {"type": "integer"},
            "view_range": {"type": "array", "items": {"type": "integer"}}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "finish", "description": "Finish the task.",
        "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": []}}},
]

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    calls = []
    for m in _CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append((str(obj.get("name", "")), args if isinstance(args, dict) else {}))
    return calls


class Renderer(Protocol):
    eos_id: int

    def initial(self, messages: list[dict], tools: list[dict]) -> list[int]: ...

    def observation(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ChatRenderer:
    """Chat-template renderer. Observation tokens are the template delta that follows the assistant's EOS."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.eos_text = tokenizer.decode([self.eos_id])

    def initial(self, messages, tools):
        return list(self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True))

    def observation(self, text: str) -> list[int]:
        a = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "x"}]
        before = self.tok.apply_chat_template(a, tokenize=False)
        after = self.tok.apply_chat_template(a + [{"role": "tool", "content": text}], tokenize=False,
                                             add_generation_prompt=True)
        cut = before.rindex(self.eos_text) + len(self.eos_text)
        return self.tok.encode(after[cut:], add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


@dataclass
class EpisodeResult:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    patch: str = ""
    turns: int = 0
    finished: bool = False
    timed_out: bool = False
    edit_events: list[tuple[int, str]] = field(default_factory=list)   # (response position, edited path)


Generate = Callable[[list[int], int], Awaitable[list[int]]]


async def generate_planned(generate: Generate, prompt: list[int], response: list[int], plan_len: int, window: int,
                           chunk: int, reserve: int, budget: int, eos_id: int | None) -> list[int]:
    """Generate one assistant turn from position len(response), re-prefilling at chunk boundaries."""
    plan = rolling_chunks(plan_len, budget, window, chunk, reserve)
    out: list[int] = []
    while len(response) + len(out) < budget:
        p = len(response) + len(out)
        c = next(c for c in plan if c.gen_start <= p < c.gen_end)
        full = response + out
        got = await generate(prompt + full[c.ctx_start:p], c.gen_end - p)
        out.extend(got)
        if not got or (eos_id is not None and got[-1] == eos_id) or len(got) < c.gen_end - p:
            break
    return out[:budget - len(response)]


def execute(call: tuple[str, dict], sb: Sandbox, editor: Editor, obs_limit: int) -> str:
    name, args = call
    if name == "execute_bash":
        _, out = sb.exec(str(args.get("command", "")), int(args.get("timeout", 120)))
    elif name == "str_replace_editor":
        a = dict(args)
        out = editor(a.pop("command", "view"), a.pop("path", sb.workdir), **a)
    else:
        out = f"ERROR: unknown tool {name!r}. Available tools: execute_bash, str_replace_editor, finish."
    return out if len(out) <= obs_limit else out[:obs_limit] + "\n<output truncated>"


async def run_episode(generate: Generate, renderer: Renderer, sb: Sandbox, problem: str, *, base_commit: str | None,
                      max_turns: int = 50, time_limit: float = 600.0, window: int = 32768, chunk: int = 4096,
                      reserve: int = 64, plan_len: int | None = None, budget: int = 60000,
                      obs_limit: int = 10_000) -> EpisodeResult:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(workdir=sb.workdir, problem=problem)}]
    res = EpisodeResult(renderer.initial(messages, TOOLS))
    plan_len = max(plan_len or 0, len(res.prompt_ids))
    editor = Editor(sb)
    t0 = time.monotonic()
    for _ in range(max_turns):
        if time.monotonic() - t0 > time_limit:
            res.timed_out = True
            break
        gen = await generate_planned(generate, res.prompt_ids, res.response_ids, plan_len, window, chunk, reserve,
                                     budget, renderer.eos_id)
        res.response_ids += gen
        res.response_mask += [1] * len(gen)
        res.turns += 1
        calls = parse_tool_calls(renderer.decode(gen))
        if not calls or calls[0][0] == "finish" or len(res.response_ids) >= budget:
            res.finished = bool(calls) and calls[0][0] == "finish"
            break
        obs = await asyncio.to_thread(execute, calls[0], sb, editor, obs_limit)
        if calls[0][0] == "str_replace_editor" and calls[0][1].get("command") in ("create", "str_replace", "insert"):
            res.edit_events.append((len(res.response_ids), str(calls[0][1].get("path", ""))))
        obs_ids = renderer.observation(obs)
        room = budget - len(res.response_ids)
        res.response_ids += obs_ids[:room]
        res.response_mask += [0] * len(obs_ids[:room])
        if room <= len(obs_ids):
            break
    if time.monotonic() - t0 > time_limit:
        res.timed_out = True
    res.patch = "" if res.timed_out else await asyncio.to_thread(model_patch, sb, base_commit)
    return res
