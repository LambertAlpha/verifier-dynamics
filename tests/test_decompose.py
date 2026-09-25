"""Generic metric decomposition: pure linear algebra, independent of any policy."""

import numpy as np
import pytest

from vdyn.geometry.decompose import decompose


def _random_spd(rng: np.random.Generator, d: int) -> np.ndarray:
    a = rng.normal(size=(d, d))
    return a @ a.T + 0.1 * np.eye(d)


def _random_cases(rng: np.random.Generator, n: int = 100):
    for _ in range(n):
        d = int(rng.integers(2, 8))
        yield rng.normal(size=d), rng.normal(size=d), _random_spd(rng, d)


def test_residual_is_metric_orthogonal_to_gold_gradient(rng):
    for g_gold, g_ver, metric in _random_cases(rng):
        dec = decompose(g_gold, g_ver, metric)
        assert dec.residual @ metric @ g_gold == pytest.approx(0.0, abs=1e-10)


def test_decomposition_reconstructs_verifier_gradient(rng):
    for g_gold, g_ver, metric in _random_cases(rng):
        dec = decompose(g_gold, g_ver, metric)
        np.testing.assert_allclose((1 + dec.alpha) * g_gold + dec.residual, g_ver, atol=1e-12)


def test_error_norm_splits_into_parallel_and_orthogonal_parts(rng):
    for g_gold, g_ver, metric in _random_cases(rng):
        dec = decompose(g_gold, g_ver, metric)
        g_err = g_ver - g_gold
        assert g_err @ metric @ g_err == pytest.approx(dec.b**2 + dec.C**2, rel=1e-10)


def test_b_is_alpha_times_signal(rng):
    for g_gold, g_ver, metric in _random_cases(rng):
        dec = decompose(g_gold, g_ver, metric)
        assert dec.b == pytest.approx(dec.alpha * dec.A, rel=1e-12)
        assert dec.A == pytest.approx(np.sqrt(g_gold @ metric @ g_gold), rel=1e-12)


def test_collinear_error_has_zero_pressure(rng):
    for g_gold, _, metric in _random_cases(rng, 20):
        dec = decompose(g_gold, 0.3 * g_gold, metric)
        assert dec.alpha == pytest.approx(-0.7, rel=1e-12)
        assert dec.C == pytest.approx(0.0, abs=1e-12)


def test_metric_matched_rate_identities(rng):
    """Proposition 1: under theta_dot = M g_V the rates follow from (A, alpha, C) in metric M."""
    for g_gold, g_ver, metric in _random_cases(rng):
        dec = decompose(g_gold, g_ver, metric)
        theta_dot = metric @ g_ver
        assert dec.gold_rate == pytest.approx(g_gold @ theta_dot, rel=1e-10, abs=1e-12)
        assert dec.verifier_rate == pytest.approx(g_ver @ theta_dot, rel=1e-10, abs=1e-12)
        assert dec.gap_rate == pytest.approx((g_ver - g_gold) @ theta_dot, rel=1e-9, abs=1e-12)


# --- degenerate case A = 0 (theory note v0.3 §1) -----------------------------------------------


def _assert_degenerate(dec, g_ver, metric):
    g_err = g_ver  # g_G contributes nothing when it is invisible to the metric
    assert dec.alpha_defined is False
    assert np.isnan(dec.alpha)
    assert np.isnan(dec.b)
    assert dec.A == 0.0
    # C is the magnitude of the whole residual, not an orthogonal component.
    assert dec.C == pytest.approx(np.sqrt(g_err @ metric @ g_err), rel=1e-12)
    # Rates follow from h_G = 0: no gold change, proxy and gap move by ||h_e||^2.
    assert dec.gold_rate == 0.0
    assert dec.verifier_rate == pytest.approx(dec.C**2, rel=1e-12)
    assert dec.gap_rate == pytest.approx(dec.C**2, rel=1e-12)


def test_zero_gold_gradient_is_degenerate(rng):
    for _, g_ver, metric in _random_cases(rng, 20):
        g_gold = np.zeros_like(g_ver)
        dec = decompose(g_gold, g_ver, metric)
        _assert_degenerate(dec, g_ver, metric)
        np.testing.assert_array_equal(dec.residual, g_ver - g_gold)


def test_metric_blind_to_gold_direction_is_degenerate():
    g_gold = np.array([0.7, 0.0])
    g_ver = np.array([0.2, -0.5])
    metric = np.diag([0.0, 3.0])  # PSD, annihilates the gold direction
    dec = decompose(g_gold, g_ver, metric)
    assert dec.alpha_defined is False
    assert np.isnan(dec.alpha)
    g_err = g_ver - g_gold
    assert dec.C == pytest.approx(np.sqrt(g_err @ metric @ g_err), rel=1e-12)
    assert dec.gold_rate == 0.0
    assert dec.gap_rate == pytest.approx(g_err @ metric @ g_ver, rel=1e-12)


def test_non_degenerate_decomposition_reports_alpha_defined(rng):
    for g_gold, g_ver, metric in _random_cases(rng, 20):
        assert decompose(g_gold, g_ver, metric).alpha_defined is True
