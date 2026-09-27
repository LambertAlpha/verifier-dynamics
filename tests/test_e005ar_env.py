"""E005a-R calibration environment and exact oracle (research/09_e005ar_design.md §4)."""

import numpy as np
import pytest

from vdyn.e005ar import env
from vdyn.e005ar import sampling as sm


@pytest.fixture(scope="module")
def base():
    return env.make_base(8, "ordinary", np.random.SeedSequence(11), "b8o")


def _ctx(base, con="shortcut", rho=0.4, alpha=-0.3, lam=0.0, d=64, spectrum="flat"):
    sV = env.s_for_alpha(base, con, rho, alpha)
    return env.Context.build(base, con, rho, sV, lam, d, spectrum, v_seed=5)


def test_base_tables_are_valid_probabilities(base):
    p = np.asarray(base["p"])
    assert p.shape == (env.PROMPTS, env.CLASSES)
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert 0.3 <= p[:, 0].min() and p[:, 0].max() <= 0.7
    for con in env.CONSTRUCTIONS:
        ev = env.ev_table(base, con, 0.7)
        assert ev.shape == (env.PROMPTS, env.CLASSES) and (ev >= 0).all() and (ev <= 1).all()
    F = env.fisher_beh(base)
    assert np.linalg.matrix_rank(F) == base["r"]


def test_alpha_target_is_exact_and_behavior_c_vanishes_without_dose(base):
    for alpha in (-0.6, -0.3, 0.0):
        for lam in (0.0, 0.7):
            o = env.oracle(_ctx(base, rho=0.0, alpha=alpha, lam=lam))
            assert o["alpha"] == pytest.approx(alpha, abs=1e-12)
            assert o["C2_beh"] == pytest.approx(0.0, abs=1e-14)
            nuis = lam * np.linalg.norm(env.h_surf(_ctx(base, lam=lam).lam_spec, _ctx(base).v))
            assert o["C2_full"] == pytest.approx(nuis**2, rel=1e-9, abs=1e-14)


def test_behavior_c_is_linear_in_rho_at_fixed_scale(base):
    ctx1 = env.Context.build(base, "partial", 1.0, 1.3, 0.0, 64, "flat", v_seed=1)
    c1 = np.sqrt(env.oracle(ctx1)["C2_beh"])
    for rho in (0.1, 0.5):
        c = np.sqrt(env.oracle(env.Context.build(base, "partial", rho, 1.3, 0.0, 64, "flat",
                                                 v_seed=1))["C2_beh"])  # fmt: skip
        assert c == pytest.approx(rho * c1, rel=1e-9)


def test_exact_gradients_equal_the_mean_rloo_contribution(base):
    ctx = _ctx(base, rho=0.6, alpha=0.0, lam=0.8, d=40)
    o = env.oracle(ctx)
    au = sm.audit(ctx, np.random.default_rng(3), R=1, N=8 * 30000, m=8, per_rollout=False)
    for key, h in (("xG", o["h_G"]), ("xV", o["h_G"] + o["h_e"])):
        x = au[key][0]
        z = (x.mean(0) - h) / (x.std(0, ddof=1) / np.sqrt(len(x)))
        assert np.max(np.abs(z)) < 4.5, key


def test_surface_bonus_gradient_formula():
    lam = env.spectrum("spiked", 40, 8, 2.0)
    v = np.random.default_rng(0).standard_normal(32)
    v /= np.linalg.norm(v)
    w = np.random.default_rng(1).standard_normal((400000, 32)) * np.sqrt(lam)
    mc = ((w @ v > 0)[:, None] * w).mean(0)
    assert np.allclose(mc, env.h_surf(lam, v), atol=4 * np.sqrt(lam.max() / 400000))


def test_spectra_levels_and_shapes():
    fbar = 3.0
    assert np.allclose(env.spectrum("bulk", 64, 8, fbar), 0.25 * fbar)
    assert np.allclose(env.spectrum("flat", 64, 8, fbar), fbar)
    s = env.spectrum("spiked", 1024, 16, fbar)
    assert s.shape == (1008,) and s.mean() == pytest.approx(fbar) and (np.diff(s) < 0).all()
    assert s[0] / s[-1] == pytest.approx(np.sqrt(1008), rel=1e-9)


def test_functional_map_is_the_probe_fisher_square_root(base):
    Phi = env.functional_map(base, "main", d=64)
    r = base["r"]
    assert Phi.shape == (64, env.PROMPTS * env.CLASSES)
    assert np.allclose(Phi[r:], 0)
    assert np.allclose(Phi[:r] @ Phi[:r].T, env.fisher_beh(base))
    sc = env.functional_map(base, "scale", d=64)
    assert np.allclose(sc, 10 * Phi)
    const = env.functional_map(base, "const", d=64)
    assert np.allclose(const[:, : Phi.shape[1]], Phi) and np.allclose(const[:, Phi.shape[1] :], 0)


def test_dose_solver_hits_the_anchor(base):
    sol = env.solve_dose(base, "shortcut", -0.3, 0.02, m=8, n_groups=60000,
                         seed=np.random.SeedSequence(4))  # fmt: skip
    assert sol is not None
    ctx = env.Context.build(base, "shortcut", sol["rho"], sol["sV"], 0.0, 64, "flat", v_seed=0)
    o = env.oracle(ctx)
    assert o["alpha"] == pytest.approx(-0.3, abs=1e-12)
    st = env.behavior_stats(ctx, m=8, n_groups=200000, seed=np.random.SeedSequence(99))
    assert st["tau_beh"] == pytest.approx(0.02, rel=0.08)


def test_panel_matched_sets_share_the_behavior_geometry():
    panel = env.build_panel("design", limit_bases=1, n_groups=20000)
    pts = panel["points"]
    by: dict[str, list] = {}
    for p in pts:
        by.setdefault(p["matched_dose"] or p["matched_null"], []).append(p)
    assert len({(p["d"], p["spectrum"]) for p in pts}) == 9
    for key, grp in by.items():
        if key.startswith("dose|"):
            c = [p["oracle"]["C2_beh"] for p in grp]
            assert np.allclose(c, c[0], rtol=1e-6), key
            assert np.allclose([p["oracle"]["alpha"] for p in grp], grp[0]["oracle"]["alpha"])
            assert np.allclose([p["oracle"]["A2"] for p in grp], grp[0]["oracle"]["A2"])
        else:
            assert all(p["oracle"]["C2_beh"] < 1e-14 for p in grp), key
    cases = {p["case"] for p in pts}
    assert cases == {"clean", "dose", "nuis", "partial"}


def test_exact_rho_solver_hits_a_common_behavior_c(base):
    target = 0.5 * env.c_beh_exact(base, "deletion", -0.3, 1.0)
    for con in env.CONSTRUCTIONS:
        rho = env.solve_rho_for_c(base, con, -0.3, target)
        if rho is not None:
            assert env.c_beh_exact(base, con, -0.3, rho) == pytest.approx(target, rel=1e-9)
    assert env.solve_rho_for_c(base, "deletion", -0.3, 2 * target / 0.5) is None
