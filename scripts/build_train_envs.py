#!/usr/bin/env python3
"""Build reconstructed environments for the SWE-bench training split and collect p2 test manifests.

  python scripts/build_train_envs.py --manifest data/manifests/swe_train_filtered.txt --out outputs/train_envs \
      --workers 8 [--limit N]

Writes outputs/train_envs/{images.json, tests.json, status.jsonl}; images.json and tests.json are the
--train-image-map / --train-tests inputs of scripts/prepare_swe.py.  Requires Docker and network access
(for cloning and installing); evaluation containers themselves run with networking disabled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from envs.swe import train_env
from envs.swe.data import rows
from envs.swe.sandbox import DockerSandbox


def _meta(repo: str, commit: str) -> tuple[str, str]:
    base = f"https://raw.githubusercontent.com/{repo}/{commit}"
    out = []
    for name in ("setup.py", "pyproject.toml"):
        r = subprocess.run(["curl", "-fsSL", f"{base}/{name}"], capture_output=True, text=True)
        out.append(r.stdout if r.returncode == 0 else "")
    return out[0], out[1]


def build_one(row: dict) -> dict:
    iid = row["instance_id"]
    tag = f"chop-train:{iid.lower().replace('__', '-')[:120]}"
    status = {"instance_id": iid, "image": tag}
    try:
        py = train_env.infer_python(*_meta(row["repo"], row["base_commit"]))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "Dockerfile").write_text(train_env.dockerfile(row["repo"], row["base_commit"], py))
            r = subprocess.run(["docker", "build", "-q", "-t", tag, tmp], capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return {**status, "built": False, "error": r.stderr[-2000:]}
        files = train_env.test_files(row["test_patch"])
        sb = DockerSandbox(tag)
        try:
            sb.write("/tmp/test.patch", row["test_patch"])
            sb.exec("git apply --whitespace=fix /tmp/test.patch", 120)
            _, before = sb.exec(train_env.test_command(files), 1800)
            sb.write("/tmp/gold.patch", row["patch"])
            sb.exec("git apply --whitespace=fix /tmp/gold.patch", 120)
            _, after = sb.exec(train_env.test_command(files), 1800)
        finally:
            sb.close()
        manifest = train_env.derive_manifest(train_env.parse_pytest_ra(before), train_env.parse_pytest_ra(after))
        return {**status, "built": True, "python": py, **manifest, "usable": bool(manifest["FAIL_TO_PASS"])}
    except Exception as e:  # noqa: BLE001 - record and continue
        return {**status, "built": False, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/swe_train_filtered.txt")
    ap.add_argument("--out", default="outputs/train_envs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    keep = set(Path(args.manifest).read_text().split())
    data = [r for r in rows("train") if r["instance_id"] in keep][:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images, tests = {}, {}
    with ThreadPoolExecutor(args.workers) as ex, (out / "status.jsonl").open("w") as log:
        for st in ex.map(build_one, data):
            log.write(json.dumps(st) + "\n")
            if st.get("usable"):
                images[st["instance_id"]] = st["image"]
                tests[st["instance_id"]] = {"FAIL_TO_PASS": st["FAIL_TO_PASS"], "PASS_TO_PASS": st["PASS_TO_PASS"]}
    (out / "images.json").write_text(json.dumps(images, indent=0))
    (out / "tests.json").write_text(json.dumps(tests))
    print(f"usable environments: {len(images)} / {len(data)}")


if __name__ == "__main__":
    main()
