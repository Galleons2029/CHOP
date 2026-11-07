"""verl wiring that can be checked without GPUs: imports, registrations, config composition."""

import os

import pytest

pytest.importorskip("verl")
hydra = pytest.importorskip("hydra")


def test_modules_import_and_agent_loop_registers():
    from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

    import chop_verl.agent_loops  # noqa: F401
    import chop_verl.main  # noqa: F401
    import chop_verl.trainer  # noqa: F401
    from chop_verl.worker import ChopActorRolloutRefWorker

    assert "chop_rolling_math" in _agent_loop_registry
    for name in ("chop_score", "chop_sample", "chop_costs", "chop_update", "init_model"):
        assert callable(getattr(ChopActorRolloutRefWorker, name))


@pytest.mark.parametrize("name,path,lr,sources", [
    ("chop_math", "Qwen/Qwen3-8B-Base", 5e-6, []),
    ("chop_swe", "Qwen/Qwen3-4B-Instruct-2507", 2e-6, ["call_graph", "test_manifest"]),
])
def test_configs_compose_with_verl_defaults(name, path, lr, sources):
    from hydra import compose, initialize_config_dir

    from chop_verl.worker import method_config

    with initialize_config_dir(config_dir=os.path.abspath("chop_verl/config"), version_base=None):
        cfg = compose(config_name=name)
    arr = cfg.actor_rollout_ref
    assert arr.model.path == path and arr.actor.optim.lr == lr and list(arr.actor.optim.betas) == [0.9, 0.95]
    m = method_config(arr.chop.method)
    assert list(m.pi.sources) == sources and m.segment.num_anchors == 8 and m.router.window == 8192
