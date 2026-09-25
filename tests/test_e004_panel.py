"""E004a design panel (registry E004a §2-§4): constructions, calibration, twins, seeds."""

import numpy as np
import pytest

from vdyn.e004 import panel as pn
from vdyn.e004 import toy


@pytest.fixture(scope="module")
def small_panel():
    return pn.design_panel(n_per=3)


def test_every_construction_is_present_with_fixed_dimension(small_panel):
    counts: dict[str, int] = {}
    for st in small_panel:
        counts[st.construction] = counts.get(st.construction, 0) + 1
        assert st.theta0.shape == (toy.D,) and st.lam.shape == (3,) and st.w.shape == (4,)
    assert counts == {c: 3 for c in pn.CONSTRUCTIONS}
    assert {st.mechanism for st in small_panel} == set(pn.MECHANISMS)


def test_calibration_hits_the_common_targets(small_panel):
    for st in small_panel:
        ex = toy.single(st, st.theta0)
        t = st.meta["targets"]
        assert ex["J_G"] == pytest.approx(t["J_G"], abs=1e-8), st.sid
        assert ex["FPR"] == pytest.approx(t["FPR"], abs=1e-8), st.sid
        assert ex["FNR"] == pytest.approx(t["FNR"], abs=1e-8), st.sid


def test_channels_follow_their_construction(small_panel):
    for st in small_panel:
        c, m = st.construction, st.meta
        if c == "X1":
            assert st.deleted.sum() in (1, 2) and st.v0 == 1.0
        if c == "X2":
            assert st.deleted.sum() in (1, 2) and 0.0 <= st.v0 <= 0.3
        if c == "YA1":
            assert st.event in ("and2", "and3") and st.trig.all() and m["S_E0"] <= 0.05 + 1e-9
        if c == "YA2":
            assert st.event == "and2" and 1 <= st.trig.sum() <= 3
        if c == "YB1":
            assert st.event == "single" and st.trig.all()
        if c == "YB2":
            assert st.event == "or2" and st.trig.all()
        if c in ("B1", "B2"):
            credited = st.beta > 0
            assert credited.all() if c == "B1" else 1 <= credited.sum() <= 3
        if c == "D1":
            assert np.any(st.p < st.rho)
        if c == "D2":
            strict = np.array(m["strict"], bool)
            assert strict.any() and np.all(st.p[strict] >= st.rho[strict])
            accept_correct = (1 - np.array(m["fn_channel"])) * st.p
            assert np.all(accept_correct[strict] < st.rho[strict])


def test_mechanism_share_of_the_false_positive_rate(small_panel):
    for st in small_panel:
        if st.construction in ("YB1", "YB2", "D1", "B1") and not st.meta.get("capped", False):
            chan = st.with_(fp=st.meta["fp_channel"], fn=st.meta["fn_channel"])
            fpr_c = toy.single(chan, st.theta0)["FPR"]
            want = st.meta["omega"] * st.meta["targets"]["FPR"]
            assert fpr_c == pytest.approx(want, rel=1e-6), st.sid


def test_twins_keep_the_initial_distribution(small_panel):
    for st in small_panel:
        un, can = pn.uncoupled_twin(st), pn.canonical_twin(st)
        assert np.all(un.lam == 0) and np.all(can.lam == 0)
        np.testing.assert_allclose(toy.row_probs(un, un.theta0), toy.row_probs(st, st.theta0),
                                   rtol=1e-10, atol=1e-14)  # fmt: skip
        np.testing.assert_allclose(can.fp, st.meta["fp_channel"])
        np.testing.assert_allclose(can.fn, st.meta["fn_channel"])


def test_panel_is_reproducible():
    a, b = pn.design_panel(n_per=1), pn.design_panel(n_per=1)
    for x, y in zip(a, b, strict=True):
        assert x.to_dict() == y.to_dict()
