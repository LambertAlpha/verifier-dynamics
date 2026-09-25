"""E004a hard-pair criteria (registry E004a §8.7): Correction A and Correction B."""

import numpy as np

from vdyn.e004 import hardpairs as hp

H_IDX = {0.002: (0, 1, 2), 0.02: (0, 3, 4)}  # checkpoint positions of (0, h/2, h)


def _traj(jg, jv, fpr, fnr):
    return {"J_G": np.array(jg), "J_V": np.array(jv), "FPR": np.array(fpr), "FNR": np.array(fnr)}


def test_correction_a_requires_every_l2_feature_within_half_an_se():
    a = _traj([0.2, 0.2, 0.2, 0.21, 0.22], [0.3] * 5, [0.1] * 5, [0.05] * 5)
    b = {k: v.copy() for k, v in a.items()}
    assert hp.correction_a(a, b, H_IDX)["accept"]
    b["FNR"] = b["FNR"] + 0.02  # FNR is an L2 input too
    res = hp.correction_a(a, b, H_IDX)
    assert not res["accept"] and res["worst"].startswith("FNR")


def test_l0_and_l1_matching():
    v = {"J_G": 0.2, "J_V": 0.3, "FPR": 0.1, "FNR": 0.05}
    assert hp.l0_match(v, dict(v))["accept"]
    assert not hp.l0_match(v, {**v, "J_G": 0.3})["accept"]
    g = {"A": 0.5, "alpha": -0.1, "C": 0.2}
    assert hp.l1_match(g, {"A": 0.52, "alpha": -0.05, "C": 0.21})
    assert not hp.l1_match(g, {"A": 0.5, "alpha": 0.1, "C": 0.2})


def test_l3_divergence_must_precede_the_visible_failure():
    times = np.array([0.0, 0.02, 0.1, 0.3])
    ga = {"A": np.full(4, 0.5), "alpha": np.array([0.0, 0.0, -0.1, -0.5]), "C": np.full(4, 0.2)}
    gb = {"A": np.full(4, 0.5), "alpha": np.zeros(4), "C": np.full(4, 0.2)}
    assert hp.l3_divergence(ga, gb, times, t_limit=0.35) == 0.3
    assert hp.l3_divergence(ga, gb, times, t_limit=0.25) is None
