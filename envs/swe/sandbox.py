"""Per-instance execution sandboxes for SWE-bench episodes.

DockerSandbox runs a SWE-bench instance image (repository at /testbed, environment preinstalled) with
networking disabled. LocalSandbox runs commands in a local directory and backs the unit tests.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Protocol


class Sandbox(Protocol):
    workdir: str

    def exec(self, cmd: str, timeout: int = 120) -> tuple[int, str]: ...

    def read(self, path: str) -> str | None: ...

    def write(self, path: str, content: str) -> None: ...

    def close(self) -> None: ...


def model_patch(sb: Sandbox, base_commit: str | None = None) -> str:
    """The agent's changes as a unified diff against the base commit (untracked files included)."""
    ref = base_commit or "HEAD"
    code, out = sb.exec(f"cd {sb.workdir} && git add -A >/dev/null 2>&1 && git diff --cached {ref} --no-color", 120)
    return out if code == 0 else ""


class LocalSandbox:
    def __init__(self, root: str | Path):
        self.workdir = str(Path(root).resolve())

    def _abs(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self.workdir) / p

    def exec(self, cmd: str, timeout: int = 120) -> tuple[int, str]:
        try:
            r = subprocess.run(["bash", "-lc", cmd], cwd=self.workdir, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {timeout} seconds."

    def read(self, path: str) -> str | None:
        p = self._abs(path)
        return p.read_text() if p.is_file() else None

    def write(self, path: str, content: str) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def close(self) -> None:
        pass


class DockerSandbox:
    """Docker container from a SWE-bench instance image; commands run via ``docker exec``."""

    def __init__(self, image: str, workdir: str = "/testbed", network: bool = False, mem_limit: str = "16g"):
        import docker

        self.client = docker.from_env()
        self.workdir = workdir
        self.container = self.client.containers.run(image, command="sleep infinity", detach=True, tty=False,
                                                    network_disabled=not network, mem_limit=mem_limit,
                                                    working_dir=workdir)

    def exec(self, cmd: str, timeout: int = 120) -> tuple[int, str]:
        wrapped = f"timeout {int(timeout)} bash -lc {shlex.quote(cmd)}"
        r = self.container.exec_run(["bash", "-lc", wrapped], workdir=self.workdir, demux=False)
        out = (r.output or b"").decode("utf8", "replace")
        if r.exit_code == 124:
            out += f"\nCommand timed out after {timeout} seconds."
        return r.exit_code, out

    def read(self, path: str) -> str | None:
        code, out = self.exec(f"cat {shlex.quote(path)}", 60)
        return out if code == 0 else None

    def write(self, path: str, content: str) -> None:
        import base64

        data = base64.b64encode(content.encode()).decode()
        self.exec(f"mkdir -p $(dirname {shlex.quote(path)}) && echo {data} | base64 -d > {shlex.quote(path)}", 60)

    def close(self) -> None:
        try:
            self.container.remove(force=True)
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
