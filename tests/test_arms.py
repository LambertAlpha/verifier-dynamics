"""The held-out arm runner must reproduce the design-pilot code path (same seeds, same scores)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vdyn.e002 import arms
from vdyn.e002 import panel as pn

REPO = Path(__file__).resolve().parents[1]


def _pilot():
    path = REPO / "experiments" / "e002" / "e002b_design_pilot.py"
    spec = importlib.util.spec_from_file_location("e002b_design_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("sid_type", ["AND2", "OR3", "MIX", "RFP"])
def test_run_arm_reproduces_the_design_pilot(sid_type):
    pilot = _pilot()
    slots = pn.load_split(REPO / "configs" / "e002" / "panel_P-mod.json", "design")
    slot = next(s for s in slots if s["type"] == sid_type)
    cells = [(16, 64), (64, 256)]
    ref = pilot.structure_task((0, 3, slot, 0.1, 0.05, cells, 4, 11))["scores"]
    st = pn.match(slot, 0.1)
    for c_i, (b_gold, b_roll) in enumerate(cells):
        for arm in arms.ARMS:
            for g_i, cfg in enumerate(arms.configs(arm)):
                key = f"{c_i}|{arm}|{g_i}"
                seed = (11, 0, 3, c_i)
                out = arms.run_arm(arm, cfg, st, 0.05, b_gold, b_roll, 4, seed)
                if key not in ref:
                    assert out is None, key
                else:
                    np.testing.assert_array_equal(out, np.asarray(ref[key]), err_msg=key)


def test_configs_and_plans_match_the_registered_grids():
    assert len(arms.configs("G1")) == 9 and len(arms.configs("G1-oracle")) == 3
    assert len(arms.configs("P3")) == 24 and len(arms.configs("P4")) == 3
    assert not arms.plan("P3", {"k": 1, "eta": 1.0, "observable": "fpr"}, 64, 64).feasible
