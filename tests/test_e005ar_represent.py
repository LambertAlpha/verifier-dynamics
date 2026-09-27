"""E005a-R representations R1-R3 (research/09_e005ar_design.md §5)."""

import numpy as np
import pytest

from vdyn.e005ar import represent as rp
from vdyn.e005ar import stats as st


def test_haar_basis_is_orthonormal_nested_and_unbiased_in_energy():
    rng = np.random.default_rng(0)
    Q = rp.haar_basis(50, 10, rng)
    assert Q.shape == (50, 10)
    assert np.allclose(Q.T @ Q, np.eye(10))
    v = np.zeros(50)
    v[3] = 1.0
    e = [np.sum((rp.haar_basis(50, 10, rng)[:, :4].T @ v) ** 2) for _ in range(3000)]
    assert np.mean(e) == pytest.approx(4 / 50, rel=0.08)


@pytest.mark.parametrize("M,d", [(20, 60), (80, 30)])
def test_top_eigenvectors_match_the_second_moment_eigendecomposition(M, d):
    rng = np.random.default_rng(M)
    Y = rng.standard_normal((M, d)) * np.linspace(3, 0.2, d)
    V = rp.top_eigvecs(Y, 6)
    w, E = np.linalg.eigh(Y.T @ Y)
    ref = E[:, ::-1][:, :6]
    assert np.allclose(V @ V.T, ref @ ref.T, atol=1e-8)
    assert rp.top_eigvecs(Y, 200).shape[1] == min(M, d)


def test_folds_are_disjoint_halves():
    a, b = rp.halves(9)
    assert list(a) == [0, 1, 2, 3] and list(b) == [4, 5, 6, 7, 8]


def test_crossfit_basis_depends_only_on_the_other_fold():
    rng = np.random.default_rng(1)
    yG = rng.standard_normal((10, 8, 20))
    yV = rng.standard_normal((10, 8, 20))
    A, B = rp.halves(10)
    bA, bB = rp.crossfit_bases(yG, yV, 5)
    yG2, yV2 = yG.copy(), yV.copy()
    yG2[B] += 5.0  # perturb fold B only
    yV2[B] -= 3.0
    bA2, bB2 = rp.crossfit_bases(yG2, yV2, 5)
    assert np.allclose(bA @ bA.T, bA2 @ bA2.T)  # the basis built from fold A is unchanged
    assert not np.allclose(bB @ bB.T, bB2 @ bB2.T)


def test_selected_subspace_estimates_are_rotation_equivariant():
    rng = np.random.default_rng(2)
    n, m, d = 12, 4, 25
    yG = rng.standard_normal((n, m, d)) + 0.3
    yV = yG + rng.standard_normal((n, m, d)) * 0.5
    Qr = np.linalg.qr(rng.standard_normal((d, d)))[0]

    def r2(yG, yV):
        sel, est = rp.halves(n)
        basis = rp.top_eigvecs(yG[sel].reshape(-1, d), 6)
        xG, xe = yG[est].mean(1) @ basis, (yV - yG)[est].mean(1) @ basis
        return st.e3(xG[None], xe[None])["C2"][0]

    assert r2(yG, yV) == pytest.approx(r2(yG @ Qr, yV @ Qr), rel=1e-8)
