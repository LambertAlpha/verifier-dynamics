"""Helpers of the E003 post-hoc analysis, on synthetic data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/toy/e003_posthoc.py"
spec = importlib.util.spec_from_file_location("e003_posthoc", SCRIPT)
assert spec is not None and spec.loader is not None
posthoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(posthoc)


def _series(eta_slope: float) -> dict[str, np.ndarray]:
    t = np.array([0.0, 0.5, 1.0, 2.0])
    return {
        "t": t,
        "eta_F": 0.2 + eta_slope * t,
        "C_F": 0.1 + 0 * t,
        "A_F": 0.03 + 0.01 * t,
        "fpr": 0.01 * np.exp(t),
        "gold": 0.001 * np.exp(2 * t),
    }


def test_prefix_summaries_use_only_the_prefix():
    s = posthoc.prefix_summaries(_series(0.1), 1.0)
    assert s["eta(0)"] == pytest.approx(0.2)
    assert s["integral eta on [0,k]"] == pytest.approx(0.2 + 0.05)  # trapezoid is exact for lines
    assert s["mean eta on [0,k]"] == pytest.approx(0.25)
    assert s["delta log FPR on [0,k] (static metric)"] == pytest.approx(1.0)
    assert s["delta log J_G on [0,k] (early gold)"] == pytest.approx(2.0)
    assert s["delta A on [0,k]"] == pytest.approx(0.01)


def test_ranking_quality_counts_discordance_with_fixed_orientation():
    shortfall = np.array([0.0, 0.0, 0.5, 0.9])
    perfect = posthoc.ranking_quality(np.array([0.1, 0.2, 0.3, 0.4]), shortfall, +1)
    assert perfect["discordant_pairs"] == 0 and perfect["stall_success_separated"] is True
    flipped = posthoc.ranking_quality(np.array([0.1, 0.2, 0.3, 0.4]), shortfall, -1)
    assert flipped["discordant_pairs"] == flipped["comparable_pairs"] == 5
    assert flipped["stall_success_separated"] is False


def test_stable_eta_matches_closed_form_where_both_are_accurate():
    from scipy.special import expit

    from vdyn.geometry import triggered_fp as cf
    from vdyn.verifiers.triggered import Structure

    logits = np.array([[-3.0, 0.0, 2.0], [1.0, 2.5, 4.0]])
    st = Structure("x", "and", (0.1, 0.5))
    expected = [cf.eta(st, expit(logits[:, k])) for k in range(3)]
    np.testing.assert_allclose(posthoc.stable_eta("and", logits), expected, rtol=1e-12)
    # far into saturation the stable form stays exact while probability space loses digits
    deep = np.array([[25.0], [26.0]])
    assert posthoc.stable_eta("and", deep)[0] == pytest.approx(1.0, abs=1e-10)
