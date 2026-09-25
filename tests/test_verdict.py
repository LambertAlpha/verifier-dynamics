"""E002 verdict rules (registry E002 §7-§8 as operationalized by Amendment 2 §2)."""

import pytest

from vdyn.e002 import verdict as vd

CELLS = [(16, 64), (64, 256), (256, 1024)]
PRIMARY = [(64, 256), (256, 1024)]
ENDPOINTS = ("c_index", "auroc")
ARMS = ("G1", "G1-oracle", "P1", "P2", "P3", "P4")


def _point(values: dict[str, float], overrides=None) -> dict:
    """Same value per arm at every cell and endpoint, with optional (cell, arm, endpoint) edits."""
    point = {(c, a, e): values[a] for c in CELLS for a in ARMS for e in ENDPOINTS}
    point.update(overrides or {})
    return point


def test_frontier_dominance_uses_cheaper_or_equal_cells():
    c_index = {(c, a): 0.5 for c in CELLS for a in ARMS}
    c_index[((64, 256), "G1")] = 0.8
    c_index[((16, 64), "P1")] = 0.8  # cheaper on both axes, ties G1 -> dominates (64, 256)
    c_index[((256, 1024), "G1")] = 0.9
    dom = vd.dominated_cells(CELLS, c_index)
    assert dom == {(16, 64): True, (64, 256): True, (256, 1024): False}
    c_index[((256, 1024), "P4")] = 0.9
    assert all(vd.dominated_cells(CELLS, c_index).values())


def _call(point, holm_reject, dominated):
    return vd.verdict(point, holm_reject, dominated, PRIMARY, ENDPOINTS)


def test_success_needs_a_holm_win_and_no_uniform_domination():
    point = _point({"G1": 0.9, "G1-oracle": 0.9, "P1": 0.8, "P2": 0.8, "P3": 0.8, "P4": 0.7})
    reject = {((64, 256), "c_index"): True}
    dom = {c: c != (256, 1024) for c in CELLS}
    assert _call(point, reject, dom)["category"] == "SUCCESS"


def test_point_win_without_holm_is_no_practical_advantage():
    point = _point({"G1": 0.9, "G1-oracle": 0.9, "P1": 0.8, "P2": 0.8, "P3": 0.8, "P4": 0.7})
    dom = {c: c != (256, 1024) for c in CELLS}
    assert _call(point, {}, dom)["category"] == "NO PRACTICAL ADVANTAGE"


def test_abandon_a_when_g1_beats_none_of_p1_p2_p4():
    point = _point({"G1": 0.6, "G1-oracle": 0.6, "P1": 0.7, "P2": 0.7, "P3": 0.5, "P4": 0.6})
    out = _call(point, {}, {c: False for c in CELLS})
    assert out["category"] == "ABANDON" and out["abandon"]["a"]


def test_beating_only_p4_blocks_rule_a():
    point = _point({"G1": 0.6, "G1-oracle": 0.6, "P1": 0.7, "P2": 0.7, "P3": 0.8, "P4": 0.4})
    out = _call(point, {}, {c: c != (16, 64) for c in CELLS})
    assert not out["abandon"]["a"] and out["category"] == "NO PRACTICAL ADVANTAGE"


def test_abandon_b_when_only_the_exact_fisher_wins():
    point = _point({"G1": 0.6, "G1-oracle": 0.95, "P1": 0.7, "P2": 0.5, "P3": 0.9, "P4": 0.4})
    out = _call(point, {}, {c: False for c in CELLS})
    assert out["abandon"]["b"] and out["category"] == "ABANDON"


def test_abandon_c_when_uniformly_dominated():
    point = _point({"G1": 0.8, "G1-oracle": 0.8, "P1": 0.7, "P2": 0.7, "P3": 0.9, "P4": 0.4})
    out = _call(point, {}, {c: True for c in CELLS})
    assert out["abandon"]["c"] and out["category"] == "ABANDON"


def test_categories_are_mutually_exclusive_by_construction():
    point = _point({"G1": 0.9, "G1-oracle": 0.9, "P1": 0.8, "P2": 0.8, "P3": 0.8, "P4": 0.7})
    with pytest.raises(AssertionError):
        # a Holm "win" while G1's point estimate is below a competitor is inconsistent input
        bad = dict(point)
        bad[((64, 256), "P3", "c_index")] = 0.95
        vd.verdict(bad, {((64, 256), "c_index"): True}, {c: False for c in CELLS}, PRIMARY,
                   ENDPOINTS)  # fmt: skip


def test_a_more_expensive_competitor_does_not_dominate():
    c_index = {(c, a): 0.5 for c in CELLS for a in ARMS}
    c_index[((16, 64), "G1")] = 0.7
    c_index[((64, 256), "P3")] = 0.75  # costs more on both axes
    assert vd.dominated_cells(CELLS, c_index)[(16, 64)] is False


def test_a_significant_win_is_not_success_when_uniformly_dominated():
    point = _point({"G1": 0.9, "G1-oracle": 0.9, "P1": 0.8, "P2": 0.8, "P3": 0.8, "P4": 0.7})
    out = _call(point, {((64, 256), "c_index"): True}, {c: True for c in CELLS})
    assert out["category"] == "ABANDON" and not out["success"] and out["abandon"]["c"]
