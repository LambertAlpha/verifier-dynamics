"""Mutation tests for the E005a-R core logic (research/09_e005ar_design.md §11).

Each property check must PASS for the real implementation and FAIL for a plausible bug injected by
monkeypatching: a statistic that counts the diagonal, a test without residualization, a
cross-fitted basis that sees the estimation fold, a functional metric without the square root, the
uncorrected plug-in in place of E3, and an alpha scale that ignores the construction term.
"""

import numpy as np
import pytest

from vdyn.e005 import estimators as es
from vdyn.e005ar import env
from vdyn.e005ar import represent as rp
from vdyn.e005ar import stats as st


def _aligned_null(rng, R, n, k, alpha=-0.4):
    hG = np.zeros(k)
    hG[0] = 1.0
    eG = rng.standard_normal((R, n, k))
    return hG + eG, alpha * (hG + eG) + rng.standard_normal((R, n, k))


# ------------------------------------------------------------------ property checks
def fpr_ok() -> bool:
    xG, xe = _aligned_null(np.random.default_rng(0), 1500, 24, 8)
    fpr = st.signflip(xG, xe, np.random.default_rng(1), B=99)["reject"].mean()
    return bool(0.03 <= fpr <= 0.07)


def crossfit_independent() -> bool:
    rng = np.random.default_rng(2)
    yG, yV = rng.standard_normal((10, 4, 12)), rng.standard_normal((10, 4, 12))
    bA, _ = rp.crossfit_bases(yG, yV, 4)
    _, B = rp.halves(10)
    yG[B] += 7.0
    bA2, _ = rp.crossfit_bases(yG, yV, 4)
    return bool(np.allclose(bA @ bA.T, bA2 @ bA2.T))


def functional_is_fisher_root() -> bool:
    base = env.make_base(4, "ordinary", np.random.SeedSequence(3), "m")
    Phi = env.functional_map(base, "main", 10)[:4]
    return bool(np.allclose(Phi @ Phi.T, env.fisher_beh(base)))


def e3_unbiased_at_null() -> bool:
    xG, xe = _aligned_null(np.random.default_rng(4), 600, 20, 30)
    c2 = st.e3(xG, xe)["C2"]
    return bool(abs(c2.mean()) < 4 * c2.std(ddof=1) / np.sqrt(len(c2)))


def alpha_exact() -> bool:
    base = env.make_base(8, "ordinary", np.random.SeedSequence(5), "m")
    for con in env.CONSTRUCTIONS:
        sV = env.s_for_alpha(base, con, 0.6, -0.3)
        o = env.oracle(env.Context.build(base, con, 0.6, sV, 0.0, 20, "flat", 1))
        if abs(o["alpha"] + 0.3) > 1e-9:
            return False
    return True


CHECKS = [fpr_ok, crossfit_independent, functional_is_fisher_root, e3_unbiased_at_null,
          alpha_exact]  # fmt: skip


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_real_implementation_passes(check):
    assert check()


# ------------------------------------------------------------------ mutants
def _diag_in_observed(Ks, rng, B=st.B_FLIPS):
    T = np.sum([K.sum((1, 2)) for K in Ks], axis=0)  # bug: the diagonal is counted
    Tb = np.zeros((Ks[0].shape[0], B))
    for K in Ks:
        S = rng.choice(np.array([-1.0, 1.0]), size=(K.shape[0], B, K.shape[1]))
        Tb += (np.matmul(S, K) * S).sum(-1) - np.trace(K, axis1=1, axis2=2)[:, None]
    p = (1 + (Tb >= T[:, None]).sum(1)) / (B + 1)
    return {"T": T, "p": p, "reject": p <= st.LEVEL}


def _no_residual(xG, xe):
    return np.einsum("rni,rmi->rnm", xe, xe)


def _crossfit_all(yG, yV, kmax):
    d = yG.shape[-1]
    Y = np.concatenate([yG.reshape(-1, d), yV.reshape(-1, d)])
    b = rp.top_eigvecs(Y, kmax)
    return b, b


def _functional_no_sqrt(base, variant, d, lam=None, v=None):
    Phi = REAL_FUNCTIONAL(base, variant, d, lam, v)
    p = env.probs(base).reshape(-1)
    return Phi * np.sqrt(np.clip(p / env.PROMPTS, 0, None))[None, : Phi.shape[1]]


def _plugin_as_e3(xG, xe):
    out = es.estimate_all(xG, xe)["plugin"]
    return {q: np.asarray(out[q], dtype=float) for q in ("C2", "SE_C2", "A2", "alpha")}


def _alpha_without_construction(base, con, rho, alpha):
    return (1 + alpha) / ((1 - rho) * base["a1"] + rho)


REAL_FUNCTIONAL = env.functional_map

MUTANTS = [
    ("diagonal counted in the observed statistic", st, "signflip_from_grams",
     _diag_in_observed, fpr_ok),
    ("no residualization", st, "gram_residual", _no_residual, fpr_ok),
    ("cross-fit basis sees the estimation fold", rp, "crossfit_bases", _crossfit_all,
     crossfit_independent),
    ("functional metric without square root", env, "functional_map", _functional_no_sqrt,
     functional_is_fisher_root),
    ("plug-in instead of E3", st, "e3", _plugin_as_e3, e3_unbiased_at_null),
    ("alpha scale ignores the construction", env, "s_for_alpha", _alpha_without_construction,
     alpha_exact),
]  # fmt: skip


@pytest.mark.parametrize("name,module,attr,mutant,check", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(monkeypatch, name, module, attr, mutant, check):
    monkeypatch.setattr(module, attr, mutant)
    assert not check(), f"mutant survived: {name}"
