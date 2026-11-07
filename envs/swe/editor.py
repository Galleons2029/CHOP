"""``str_replace_editor`` with OpenHands-compatible semantics (view / create / str_replace / insert /
undo_edit), operating through a :class:`envs.swe.sandbox.Sandbox`."""

from __future__ import annotations

from envs.swe.sandbox import Sandbox

SNIPPET_LINES = 4
MAX_VIEW_CHARS = 16_000


def _numbered(text: str, start: int = 1) -> str:
    return "\n".join(f"{i:6}\t{line}" for i, line in enumerate(text.split("\n"), start))


class Editor:
    def __init__(self, sb: Sandbox):
        self.sb = sb
        self.history: dict[str, list[str]] = {}

    def __call__(self, command: str, path: str, **kw) -> str:
        fn = getattr(self, f"_{command}", None)
        if fn is None:
            return f"ERROR: unknown command {command!r}; use view, create, str_replace, insert or undo_edit."
        try:
            return fn(path, **kw)
        except TypeError as e:
            return f"ERROR: invalid parameters for {command}: {e}"

    def _view(self, path: str, view_range: list[int] | None = None, **_):
        text = self.sb.read(path)
        if text is None:
            code, out = self.sb.exec(f"find {path} -maxdepth 2 -not -path '*/.*' | sort | head -500", 60)
            if code == 0 and out.strip():
                return f"Here's the files and directories up to 2 levels deep in {path}:\n{out}"
            return f"ERROR: The path {path} does not exist."
        start = 1
        if view_range:
            lines = text.split("\n")
            lo, hi = view_range[0], (len(lines) if view_range[1] == -1 else view_range[1])
            text, start = "\n".join(lines[lo - 1:hi]), lo
        body = _numbered(text, start)
        if len(body) > MAX_VIEW_CHARS:
            body = body[:MAX_VIEW_CHARS] + "\n<response clipped>"
        return f"Here's the result of running `cat -n` on {path}:\n{body}"

    def _create(self, path: str, file_text: str = "", **_):
        if self.sb.read(path) is not None:
            return f"ERROR: File already exists at: {path}. Cannot overwrite files using command `create`."
        self.sb.write(path, file_text)
        self.history.setdefault(path, []).append("")
        return f"File created successfully at: {path}"

    def _str_replace(self, path: str, old_str: str = "", new_str: str = "", **_):
        text = self.sb.read(path)
        if text is None:
            return f"ERROR: The path {path} does not exist."
        n = text.count(old_str)
        if n == 0:
            return f"ERROR: No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if n > 1:
            lines = [i + 1 for i, line in enumerate(text.split("\n")) if old_str.split("\n")[0] in line]
            return (f"ERROR: No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines "
                    f"{lines}. Please ensure it is unique.")
        self.history.setdefault(path, []).append(text)
        new = text.replace(old_str, new_str)
        self.sb.write(path, new)
        line = text.split(old_str)[0].count("\n")
        lines = new.split("\n")
        lo, hi = max(0, line - SNIPPET_LINES), line + SNIPPET_LINES + new_str.count("\n") + 1
        return (f"The file {path} has been edited. Here's the result of running `cat -n` on a snippet of {path}:\n"
                f"{_numbered(chr(10).join(lines[lo:hi]), lo + 1)}\nReview the changes and make sure they are as "
                "expected. Edit the file again if necessary.")

    def _insert(self, path: str, insert_line: int = 0, new_str: str = "", **_):
        text = self.sb.read(path)
        if text is None:
            return f"ERROR: The path {path} does not exist."
        lines = text.split("\n")
        if not 0 <= insert_line <= len(lines):
            return f"ERROR: Invalid `insert_line` parameter: {insert_line}. It should be within [0, {len(lines)}]."
        self.history.setdefault(path, []).append(text)
        new = "\n".join(lines[:insert_line] + new_str.split("\n") + lines[insert_line:])
        self.sb.write(path, new)
        n = new_str.count("\n") + 1
        return f"The file {path} has been edited. Inserted {n} line(s) after line {insert_line}."

    def _undo_edit(self, path: str, **_):
        if not self.history.get(path):
            return f"ERROR: No edit history found for {path}."
        self.sb.write(path, self.history[path].pop())
        return f"Last edit to {path} undone successfully."
