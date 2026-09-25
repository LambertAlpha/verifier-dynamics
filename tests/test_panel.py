"""E002 panel generation, matching and split integrity (registry E002 §2)."""

import json

import numpy as np
import pytest

from vdyn.e002 import panel as pn
from vdyn.verifiers import boolean_fp as bf

F_GRID = (0.05, 0.1, 0.2)


@pytest.fixture(scope="module")
def mod_panel():
    return pn.build_panel("P-mod", q0=0.05, f_grid=F_GRID)


def test_panel_is_deterministic(mod_panel):
    again = pn.build_panel("P-mod", q0=0.05, f_grid=F_GRID)
    assert json.dumps(again, sort_keys=True) == json.dumps(mod_panel, sort_keys=True)
    other = pn.build_panel("P-rare", q0=0.002, f_grid=(0.005, 0.01, 0.02))
    assert other["slots"][0]["raw_logits"] != mod_panel["slots"][0]["raw_logits"]


def test_counts_and_split_sizes(mod_panel):
    slots = mod_panel["slots"]
    assert len(slots) == 900
    expected = {**pn.COUNTS, "SINGLE": 10, "RFP": 10}
    design = pn.design_counts()
    assert sum(design.values()) == 300
    for typ, n in expected.items():
        of_type = [s for s in slots if s["type"] == typ]
        assert len(of_type) == n, typ
        assert sum(s["split"] == "design" for s in of_type) == design[typ]
        assert abs(design[typ] - n / 3) < 1 or typ in pn.CONTROL_TYPES
    assert sum(s["split"] == "design" for s in slots) == 300
    assert sum(s["split"] == "test" for s in slots) == 600
    assert len({s["sid"] for s in slots}) == 900


@pytest.mark.parametrize("f", F_GRID)
def test_every_structure_is_matched_at_every_candidate_fpr(mod_panel, f):
    for slot in mod_panel["slots"]:
        st = pn.match(slot, f)
        s = np.asarray(st.s0)
        assert bf.fpr(st, s) == pytest.approx(f, abs=1e-12)
        assert np.all((s >= 1e-4) & (s <= 1 - 1e-4))
        if st.type == "MIX":
            assert bf.event_prob(st, s) == pytest.approx(1 - (1 - f) ** slot["rho"], abs=1e-12)
            assert st.coin == pytest.approx(1 - (1 - f) ** (1 - slot["rho"]), abs=1e-15)
        if st.type == "RFP":
            assert st.coin == f and st.n_features == 0


def test_no_near_duplicates_within_type(mod_panel):
    by_type: dict[tuple, list[np.ndarray]] = {}
    for slot in mod_panel["slots"]:
        if slot["type"] in pn.CONTROL_TYPES:
            continue
        key = (slot["type"], slot["base"])
        canon = pn.canonical(slot, F_GRID[0])
        for other in by_type.get(key, []):
            assert np.max(np.abs(canon - other)) >= 1e-3
        by_type.setdefault(key, []).append(canon)


def test_canonical_form_sorts_symmetric_groups():
    a = {"type": "AOR", "base": None, "raw_logits": [0.3, -1.0, 2.0], "rho": None}
    b = {"type": "AOR", "base": None, "raw_logits": [-1.0, 0.3, 2.0], "rho": None}
    np.testing.assert_allclose(pn.canonical(a, 0.1), pn.canonical(b, 0.1))
    c = {"type": "AOR", "base": None, "raw_logits": [2.0, 0.3, -1.0], "rho": None}
    assert np.max(np.abs(pn.canonical(a, 0.1) - pn.canonical(c, 0.1))) > 1e-3


def test_loader_refuses_the_test_split_without_approval(tmp_path, mod_panel):
    path = tmp_path / "panel.json"
    pn.write_panel(path, mod_panel)
    approval = tmp_path / "HELDOUT_APPROVED"
    design = pn.load_split(path, "design", approval_file=approval)
    assert len(design) == 300 and all(s["split"] == "design" for s in design)
    with pytest.raises(PermissionError):
        pn.load_split(path, "test", approval_file=approval)
    approval.write_text("approved\n")
    assert len(pn.load_split(path, "test", approval_file=approval)) == 600


def test_default_approval_file_is_absent_or_records_an_explicit_approval():
    # Sealed until 2026-09-25; unsealed on explicit approval (commit 83b6737). The file may only
    # exist with the recorded approval and the frozen code commit.
    if pn.DEFAULT_APPROVAL.exists():
        text = pn.DEFAULT_APPROVAL.read_text()
        assert "explicit approval" in text and "d639b09" in text


def test_design_counts_are_the_amendment_1_values():
    expected = {"AND2": 33, "AND3": 33, "AND4": 33, "OR2": 33, "OR3": 32, "THR23": 32, "AOR": 32,
                "OAND": 32, "MIX": 32, "SINGLE": 4, "RFP": 4}  # fmt: skip
    assert pn.design_counts() == expected
