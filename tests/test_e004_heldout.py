"""E004a Stage 1 held-out infrastructure (Amendment 4; integrity log): sealed test and shift
panels, registered seed-tree extension, pooled J_V with unequal groups, fixed-threshold warnings."""

import json
from pathlib import Path

import numpy as np
import pytest

from vdyn.e004 import audit as au
from vdyn.e004 import heldout as hd
from vdyn.e004 import panel0b as pb
from vdyn.e004 import predict as pr
from vdyn.e004 import toy

REPO = Path(__file__).resolve().parents[1]
SHA = "f" * 64


def _approve(tmp_path: Path, sha: str = SHA) -> Path:
    path = tmp_path / "HELDOUT_APPROVED"
    path.write_text(json.dumps({"predictors_frozen_sha256": sha, "scope": ["test", "shift"]}))
    return path


def test_design_generator_reproduces_the_frozen_panel():
    frozen = json.loads((REPO / "configs/e004/design_panel_0b.json").read_text())["structures"]
    again = pb.design_panel(n_per=1, yb_cap=0.2)
    for k, st in enumerate(again):
        assert st.to_dict() == frozen[k * pb.N_PER_CELL]


def test_heldout_panels_are_sealed(tmp_path):
    with pytest.raises(PermissionError):
        hd.test_panel(tmp_path / "absent", SHA, n_per=1)
    with pytest.raises(PermissionError):
        hd.shift_panel(_approve(tmp_path, "0" * 64), SHA, n_per=1)


def test_test_panel_is_fresh_and_reproducible(tmp_path):
    ok = _approve(tmp_path)
    a, b = hd.test_panel(ok, SHA, n_per=1), hd.test_panel(ok, SHA, n_per=1)
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b]
    design = pb.design_panel(n_per=1, yb_cap=0.2)
    assert len(a) == len(pb.CELLS) and all(s.sid.startswith("T-") for s in a)
    assert all(not np.allclose(s.theta0, d.theta0) for s, d in zip(a, design, strict=True))
    for st in a:
        assert toy.single(st, st.theta0)["J_G"] == pytest.approx(st.meta["targets"]["J_G"],
                                                                 abs=1e-8)  # fmt: skip


def test_shift_panel_has_the_harder_registered_difficulty(tmp_path):
    sp = hd.shift_panel(_approve(tmp_path), SHA, n_per=1)
    assert all(s.sid.startswith("S-") for s in sp)
    for st in sp:
        jg = st.meta["targets"]["J_G"]
        assert hd.SHIFT_JG[0] <= jg <= hd.SHIFT_JG[1]
        assert toy.single(st, st.theta0)["J_G"] == pytest.approx(jg, abs=1e-8)
    assert hd.SHIFT_BATCH == (4, 4)


def test_seed_tree_extension_keeps_the_design_seeds():
    runs = np.random.SeedSequence(pb.ROOT_SEED).spawn(5)[3].spawn(len(hd.ROLES))
    design = [c.spawn(4) for c in runs[0].spawn(768)]
    got = hd.run_seeds("prim_ver", "design", 768)
    assert [s.spawn_key for s in got[5]] == [s.spawn_key for s in design[5]]
    test = hd.run_seeds("prim_ver", "test", 768)
    keys_design = {s.spawn_key for ss in got for s in ss}
    assert not keys_design & {s.spawn_key for ss in test for s in ss}
    spare = runs[9].spawn(4)
    ad = [c.spawn(4) for c in spare[0].spawn(768)]
    assert [s.spawn_key for s in hd.audit_seeds("adam", "design", 768)[3]] == [
        s.spawn_key for s in ad[3]
    ]
    assert hd.audit_seeds("ng", "shift", 2)[0].spawn_key == spare[1].spawn(1538)[1536].spawn_key
    res = hd.resample_seeds()
    assert [r.spawn_key for r in res[:4]] == [r.spawn_key for r in spare[3].spawn(4)]


def test_pooled_jv_with_unequal_group_sizes():
    aud = au.Groups(np.zeros((1, 2), int), np.zeros((1, 2, 8), int), np.ones((1, 2, 8)))
    train = au.Groups(np.zeros((1, 3), int), np.zeros((1, 3, 4), int), np.zeros((1, 3, 4)))
    np.testing.assert_allclose(hd.pooled_jv(aud, train), [16 / 28])
    np.testing.assert_allclose(hd.pooled_jv(aud, None), [1.0])


def test_fixed_threshold_warnings_reproduce_the_design_procedure():
    rng = np.random.default_rng(0)
    H = [0.0, 0.01, 0.02, 0.05]
    p = rng.random((4, 200))
    success = rng.random(200) < 0.5
    fail = ~success & (rng.random(200) < 0.8)
    t_on = np.where(fail, rng.uniform(0.0, 0.3, 200), np.inf)
    ref = pr.warnings(p, H, success, fail, t_on, 0.02)
    got = hd.warnings_at(p, H, success, fail, t_on, 0.02, ref["tau"])
    for k in ("warned", "t_warn", "lead_warned"):
        np.testing.assert_array_equal(got[k], ref[k])
    for k in ("false_alarm", "median_lead", "sensitivity", "median_lead_conservative"):
        assert got[k] == pytest.approx(ref[k])
