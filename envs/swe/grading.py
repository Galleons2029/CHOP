"""SWE-bench grading through the official harness (swebench >= 4; version-tolerant imports).

``grade`` runs the instance's evaluation in a fresh container built from the instance image: it
applies the model patch with ``git apply --whitespace=fix`` semantics, applies the held-out test patch,
runs the FAIL_TO_PASS / PASS_TO_PASS tests and parses the log. It returns the harness report.
Training rewards (R(tau) in {0, 1}) and all evaluation numbers use this path.
"""

from __future__ import annotations


def _make_test_spec(instance: dict):
    try:
        from swebench.harness.utils import make_test_spec
    except ImportError:  # swebench 2.x-4.x
        from swebench.harness.test_spec.test_spec import make_test_spec
    return make_test_spec(instance)


def grade(instance: dict, patch: str, run_id: str, model_name: str = "chop", timeout: int = 1800) -> dict:
    import docker
    from swebench.harness.run_evaluation import run_instance

    if not patch.strip():
        return {"resolved": False, "empty_patch": True}
    spec = _make_test_spec(instance)
    pred = {"instance_id": instance["instance_id"], "model_name_or_path": model_name, "model_patch": patch}
    out = run_instance(spec, pred, docker.from_env(), run_id, timeout)
    report = out.get("report", out) if isinstance(out, dict) else {}
    entry = report.get(instance["instance_id"], report) if isinstance(report, dict) else {}
    return {"resolved": bool(entry.get("resolved", False)), "report": entry}


def instance_image(instance_id: str, namespace: str = "swebench", arch: str = "x86_64") -> str:
    """Name of the prebuilt SWE-bench instance image (Verified images are published on Docker Hub)."""
    return f"{namespace}/sweb.eval.{arch}.{instance_id.replace('__', '_1776_')}:latest"
