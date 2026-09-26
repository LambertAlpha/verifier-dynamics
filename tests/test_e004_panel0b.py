"""E004a Stage 0b panel (Amendment 3 §3, §4, §8): 18 constructions, Axis B, YB cap, calibration."""

import numpy as np
import pytest

from vdyn.e004 import panel0b as pb
from vdyn.e004 import toy


@pytest.fixture(scope="module")
def small():
    return pb.design_panel(n_per=2, yb_cap=0.05)


def test_cells_and_mechanisms(small):
    assert len(pb.CELLS) == 24
    assert {c for c, _ in pb.CELLS} == set(pb.CONSTRUCTIONS) and len(pb.CONSTRUCTIONS) == 18
    for mech in pb.MECHANISMS:
        assert sum(pb.mechanism_of(c) == mech for c in pb.CONSTRUCTIONS) >= 3
    counts: dict[tuple[str, str], int] = {}
    for st in small:
        key = (st.construction, st.meta["intended_axis"])
        counts[key] = counts.get(key, 0) + 1
        assert st.theta0.shape == (toy.D,)
    assert counts == {cell: 2 for cell in pb.CELLS}


def test_calibration_hits_targets_and_records_the_actual_axis(small):
    for st in small:
        ex = toy.single(st, st.theta0)
        t = st.meta["targets"]
        for k in ("J_G", "FPR", "FNR"):
            assert ex[k] == pytest.approx(t[k], abs=1e-8), (st.sid, k)
        assert st.meta["axis_b"] == toy.axis_b(st)
        if st.meta["intended_axis"] == "INVERTED":
            assert st.meta["axis_b"] == "INVERTED"  # D and intended-inverted Y are guaranteed


def test_construction_channels(small):
    for st in small:
        c, m, intended = st.construction, st.meta, st.meta["intended_axis"]
        if c == "X2":
            assert st.v0 == 0.0 and st.deleted.sum() in (1, 2)
        if c == "X3":
            assert 0.3 <= st.v0 <= 0.7
        if c == "YA3":
            assert st.event == "thr23"
        if c == "YB2":
            hard = np.argsort(np.array(m["delta"]))[: int(st.trig.astype(bool).sum())]
            assert set(np.flatnonzero(st.trig > 0)) == set(hard)
        if c.startswith("YB"):
            assert m["S_E0"] <= 0.05 + 1e-12  # the YB cap
        if c.startswith("Y"):
            on = st.trig > 0
            fn_ch = np.array(m["fn_channel"])
            if intended == "ALIGNED":
                assert np.all((st.trig[on] >= 0.5) & (st.trig[on] <= 0.9)) and np.all(fn_ch == 0)
            else:
                assert np.all(st.trig[on] == 1.0) and np.all(
                    (fn_ch[on] >= 0.1) & (fn_ch[on] <= 0.4)
                )
        if c == "B2":
            assert st.credit_event == "z3"
        if c == "D2":
            assert 1 <= np.count_nonzero(st.rho) <= 2
        if c == "D3":
            assert st.hack_event == "z1"
        if c == "R3":
            np.testing.assert_allclose(np.array(m["fp_channel"]), np.array(m["fn_channel"]))


def test_reproducible():
    a, b = pb.design_panel(n_per=1, yb_cap=0.1), pb.design_panel(n_per=1, yb_cap=0.1)
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]
