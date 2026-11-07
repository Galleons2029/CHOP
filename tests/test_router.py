import torch

from chop.config import RouterConfig
from chop.router import Cell, CellRouter


def _router(**kw):
    r = CellRouter(RouterConfig(window=4096, refresh=64, min_history=64, **kw), seed=0)
    g = torch.Generator().manual_seed(0)
    r.rho_q.initialize(torch.rand(4096, generator=g))
    r.kappa_q.initialize(torch.rand(4096, generator=g))
    return r


def test_quantile_route_fractions(gen):
    r = _router()
    rho, kap = torch.rand(20000, generator=gen), torch.rand(20000, generator=gen)
    route = r.route(rho, kap)
    f = route.fractions()
    assert abs(f["segment"] - 0.30) < 0.02
    assert abs(f["pi"] - 0.30) < 0.02
    # cell labels follow the two gates
    assert torch.equal(route.cells, 2 * (~route.trusted).long() + route.pi_gate.long())


def test_ineligible_states_are_ignored(gen):
    r = _router()
    rho, kap = torch.rand(100, generator=gen), torch.rand(100, generator=gen)
    elig = torch.rand(100, generator=gen) > 0.5
    route = r.route(rho, kap, eligible=elig)
    assert (route.cells[~elig] == int(Cell.IGNORE)).all()
    assert (route.cells[elig] >= 0).all()


def test_warmup_defaults_to_trusted_without_pi(gen):
    r = CellRouter(RouterConfig(min_history=10_000), seed=0)
    route = r.route(torch.rand(50, generator=gen), torch.rand(50, generator=gen))
    assert route.trusted.all() and not route.pi_gate.any()


def test_ablation_modes(gen):
    rho, kap = torch.rand(500, generator=gen), torch.rand(500, generator=gen)
    assert _router(rho_gate="none").route(rho, kap).trusted.all()
    assert not _router(rho_gate="all_segment").route(rho, kap).trusted.any()
    assert _router(kappa_gate="always").route(rho, kap).pi_gate.all()
    assert not _router(kappa_gate="none").route(rho, kap).pi_gate.any()
    three = _router(three_cell=True).route(rho, kap)
    assert not (three.cells == int(Cell.B)).any()
    ca = _router(cell_a_uses_pi=True).route(rho, kap)
    assert not (ca.cells == int(Cell.A)).any()
    static = _router(rho_gate="static", static_rho=0.5).route(rho, kap)
    assert torch.equal(static.trusted, rho > 0.5)


def test_rate_matched_random_pi_keeps_activation_count(gen):
    rho, kap = torch.rand(1000, generator=gen), torch.rand(1000, generator=gen)
    q = _router().route(rho, kap).pi_gate
    m = _router(kappa_gate="random_rate_matched").route(rho, kap).pi_gate
    assert int(q.sum()) == int(m.sum()) and not torch.equal(q, m)


def test_depth_and_oracle_routes(gen):
    rho = torch.rand(100, generator=gen)
    depth = torch.linspace(0, 0.99, 100)
    route = _router(rho_gate="depth_decile").route(rho, depth_frac=depth)
    assert torch.equal(route.trusted, depth < 0.7)
    r = _router(rho_gate="oracle")
    r.oracle_q.initialize(torch.rand(4096, generator=gen))
    oracle = torch.rand(100, generator=gen)
    assert abs(float((~r.route(rho, oracle_rho=oracle).trusted).float().mean()) - 0.3) < 0.15


def test_stratum_restriction(gen):
    r = _router(stratum=[0.0, 0.1])
    rho = torch.rand(5000, generator=gen)
    route = r.route(rho)
    assert abs(float(route.in_stratum.float().mean()) - 0.1) < 0.02
    assert (rho[route.in_stratum] < 0.15).all()


def test_update_and_state_roundtrip(gen):
    r = _router()
    rho, kap = torch.rand(10, generator=gen), torch.rand(10, generator=gen)
    r.update(rho, kap)
    assert torch.equal(r.rho_q.history[-10:], rho)
    s = r.state_dict()
    r2 = CellRouter(RouterConfig(), seed=5)
    r2.load_state_dict(s)
    assert torch.equal(r2.kappa_q.history, r.kappa_q.history)
