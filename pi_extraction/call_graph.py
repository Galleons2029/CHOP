"""p4: symbol-level call graph reachable from the issue-referenced module(s).

Extraction uses tree-sitter (Python grammar) over the repository at the base commit. Cross-module
calls are resolved by name against a repository-wide definition index. Test files are never parsed,
and the extraction window is limited to symbols reachable from the modules that the issue text
references (leakage rule 2 of App. C.6). The paper additionally consults a project-wide LSP server
(pylsp); here that step is an optional ``resolver`` hook.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pi_extraction.diffs import is_test_path


def _parser():
    import tree_sitter_python
    from tree_sitter import Language, Parser

    lang = Language(tree_sitter_python.language())
    try:
        return Parser(lang)
    except TypeError:  # tree-sitter < 0.22
        p = Parser()
        p.set_language(lang)
        return p


@dataclass
class Symbol:
    qualname: str          # module.path:Class.method
    path: str
    line: int
    calls: set[str] = field(default_factory=set)   # bare callee names


def _module_symbols(path: str, source: bytes, parser) -> list[Symbol]:
    tree = parser.parse(source)
    out: list[Symbol] = []

    def text(n):
        return source[n.start_byte:n.end_byte].decode("utf8", "replace")

    def callee(n) -> str | None:
        fn = n.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type == "identifier":
            return text(fn)
        if fn.type == "attribute":
            attr = fn.child_by_field_name("attribute")
            return text(attr) if attr is not None else None
        return None

    def walk_calls(n, sym: Symbol):
        stack = [n]
        while stack:
            c = stack.pop()
            if c.type == "call":
                name = callee(c)
                if name:
                    sym.calls.add(name)
            if c is not n and c.type in ("function_definition", "class_definition"):
                continue
            stack.extend(c.children)

    def visit(n, prefix: str):
        for c in n.children:
            node = c.child_by_field_name("definition") if c.type == "decorated_definition" else c
            if node is None or node.type not in ("function_definition", "class_definition"):
                continue
            name = text(node.child_by_field_name("name"))
            q = f"{prefix}.{name}" if prefix else name
            sym = Symbol(f"{path}:{q}", path, node.start_point[0] + 1)
            body = node.child_by_field_name("body")
            if body is not None:
                walk_calls(body, sym)
                out.append(sym)
                visit(body, q)

    visit(tree.root_node, "")
    return out


def referenced_modules(issue: str, repo_files: list[str]) -> list[str]:
    """Repository .py files named by the issue: explicit paths, dotted module names, or file stems."""
    py = [f for f in repo_files if f.endswith(".py") and not is_test_path(f)]
    hits: list[str] = []
    for f in py:
        dotted = f[:-3].replace("/", ".")
        stem = Path(f).stem
        if f in issue or dotted in issue or (
            stem not in ("__init__", "setup", "utils") and re.search(rf"\b{re.escape(stem)}\.py\b", issue)
        ):
            hits.append(f)
    return hits


def build_call_graph(repo_root: str | Path, issue: str, max_depth: int = 2, max_symbols: int = 200,
                     resolver: Callable[[str, str], list[str]] | None = None) -> str:
    """Serialise the call graph reachable (<= max_depth) from the issue-referenced modules."""
    root = Path(repo_root)
    files = [str(p.relative_to(root)) for p in root.rglob("*.py")]
    files = [f for f in files if not is_test_path(f)]
    parser = _parser()
    index: dict[str, list[Symbol]] = {}
    by_file: dict[str, list[Symbol]] = {}
    for f in files:
        try:
            syms = _module_symbols(f, (root / f).read_bytes(), parser)
        except OSError:
            continue
        by_file[f] = syms
        for s in syms:
            index.setdefault(s.qualname.rsplit(".", 1)[-1].split(":")[-1], []).append(s)
    seeds = referenced_modules(issue, files)
    seen: set[str] = set()
    queue: deque[tuple[Symbol, int]] = deque((s, 0) for f in seeds for s in by_file.get(f, []))
    lines: list[str] = []
    while queue and len(seen) < max_symbols:
        sym, d = queue.popleft()
        if sym.qualname in seen:
            continue
        seen.add(sym.qualname)
        targets = []
        for name in sorted(sym.calls):
            cands = resolver(sym.path, name) if resolver else [c.qualname for c in index.get(name, [])]
            targets.extend(cands[:3])
            if d + 1 <= max_depth:
                queue.extend((c, d + 1) for c in index.get(name, [])[:3])
        lines.append(f"{sym.qualname} (line {sym.line}) -> " + (", ".join(targets) if targets else "-"))
    header = "modules: " + (", ".join(seeds) if seeds else "(none referenced by the issue)")
    return header + "\n" + "\n".join(lines)
