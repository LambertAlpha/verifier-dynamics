"""Path A: closed forms for feature-triggered false positives (theory note §10, Props. 8-9).

Hand-typed from the derivations; shares no code with `autodiff`. State: q = P(corr = 1) and the
feature probabilities s (an array, empty for the random-FP control). Gradients are with respect to
the logits (u, phi_1..phi_m).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from vdyn.verifiers.triggered import Structure


def exploit_rate(structure: Structure, s: np.ndarray | tuple[float, ...]) -> float:
    """S = P(event), which equals the on-policy FPR."""
    s = np.asarray(s, dtype=np.float64)
    if structure.kind == "random":
        assert structure.f is not None
        return structure.f
    if structure.kind == "single":
        return float(s[0])
    if structure.kind == "and":
        return float(np.prod(s))
    return float(1 - np.prod(1 - s))


def exploit_rate_grad(structure: Structure, s: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """dS/ds_i (with respect to the feature probabilities)."""
    s = np.asarray(s, dtype=np.float64)
    if structure.kind == "random":
        return np.zeros(0)
    if structure.kind == "single":
        return np.ones(1)
    if structure.kind == "and":
        return np.array([np.prod(np.delete(s, i)) for i in range(len(s))])
    return np.array([np.prod(np.delete(1 - s, i)) for i in range(len(s))])


def kappa2(structure: Structure, s: np.ndarray | tuple[float, ...]) -> float:
    """Squared Fisher norm of grad_phi S: sum_i (dS/ds_i)^2 s_i (1 - s_i)."""
    s = np.asarray(s, dtype=np.float64)
    return float(np.sum(exploit_rate_grad(structure, s) ** 2 * s * (1 - s)))


def static_metrics(structure: Structure, q: float, s: np.ndarray) -> dict[str, float]:
    S = exploit_rate(structure, s)
    return {
        "gold_accuracy": q,
        "fpr": S,
        "fnr": 0.0,
        "fp_mass": (1 - q) * S,
        "verifier_accuracy": 1 - (1 - q) * S,
    }


def j_verifier(structure: Structure, q: float, s: np.ndarray) -> float:
    return q + (1 - q) * exploit_rate(structure, s)


def grad_gold(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    out = np.zeros(1 + structure.n_features)
    out[0] = q * (1 - q)
    return out


def grad_verifier(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    S = exploit_rate(structure, s)
    return np.concatenate(
        [[q * (1 - q) * (1 - S)], (1 - q) * exploit_rate_grad(structure, s) * s * (1 - s)]
    )


def fisher(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return np.diag(np.concatenate([[q * (1 - q)], s * (1 - s)]))


def fisher_diagnostics(structure: Structure, q: float, s: np.ndarray) -> tuple[float, float, float]:
    """(A, alpha, C) in the Fisher metric: sqrt(q(1-q)), -FPR, (1-q) kappa."""
    return (
        float(np.sqrt(q * (1 - q))),
        -exploit_rate(structure, s),
        float((1 - q) * np.sqrt(kappa2(structure, s))),
    )


def c_max(q: float, fpr: float) -> float:
    """Cramer-Rao cap on C at a given on-policy FPR (Prop. 8)."""
    return float((1 - q) * np.sqrt(fpr * (1 - fpr)))


def eta(structure: Structure, s: np.ndarray | tuple[float, ...]) -> float:
    """Normalized accessibility (C / C_max)^2 = kappa^2 / (S(1-S)), in [0, 1]."""
    if structure.kind == "random":
        return 0.0
    S = exploit_rate(structure, s)
    return kappa2(structure, s) / (S * (1 - S))


def natural_velocity(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    """(u', phi') = F^{-1} g_V = (1 - S, (1 - q) dS/ds)."""
    return np.concatenate(
        [[1 - exploit_rate(structure, s)], (1 - q) * exploit_rate_grad(structure, s)]
    )


def natural_rates(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    """(q', s') under natural gradient."""
    s = np.asarray(s, dtype=np.float64)
    vel = natural_velocity(structure, q, s)
    return np.concatenate([[q * (1 - q) * vel[0]], s * (1 - s) * vel[1:]])


def vanilla_velocity(structure: Structure, q: float, s: np.ndarray) -> np.ndarray:
    return grad_verifier(structure, q, s)


def _symmetric(s0: tuple[float, ...]) -> bool:
    return all(x == s0[0] for x in s0)


def gold_race(
    structure: Structure, s0: tuple[float, ...], s: np.ndarray | tuple[float, ...]
) -> float:
    """Lambda = log(q/q0) as a function of the exploit state along the natural-gradient path."""
    s = np.asarray(s, dtype=np.float64)
    if structure.kind == "random":
        raise ValueError("the random-FP control has no exploit path")
    if structure.kind == "single" or (structure.kind == "or" and _symmetric(s0)):
        # single: q/s constant; symmetric 2-OR: q/a constant
        if structure.kind == "or" and len(s0) != 2:
            raise NotImplementedError("closed form only for the symmetric 2-OR")
        return float(np.log(s[0] / s0[0]))
    if structure.kind == "and" and _symmetric(s0):
        k = len(s0)

        def antiderivative(x: float) -> float:
            return float(np.log(x) - sum(1.0 / (m * x**m) for m in range(1, k)))

        return antiderivative(float(s[0])) - antiderivative(s0[0])
    if structure.kind == "and" and len(s0) == 2:
        rho = (1 - s0[0]) / (1 - s0[1])
        return float((np.log(s[1] / s0[1]) - rho * np.log(s[0] / s0[0])) / (1 - rho))
    raise NotImplementedError(f"no closed-form gold race for {structure}")


def _path_point(structure: Structure, t: float) -> np.ndarray:
    """Exploit state on the natural-gradient path, parameterized by the last feature probability."""
    s0 = structure.s0
    if structure.kind == "and" and len(s0) == 2 and not _symmetric(s0):
        rho = (1 - s0[0]) / (1 - s0[1])
        return np.array([1 - rho * (1 - t), t])
    return np.full(len(s0), t)


def predicted_outcome(structure: Structure, q0: float) -> dict[str, float | bool]:
    """Prop. 9(c): stall with q_inf = q0 exp(Lambda(1)), or success with S frozen at S_inf."""
    if structure.kind == "random":
        return {"stall": False, "q_inf": 1.0, "S_inf": exploit_rate(structure, ())}
    target = float(np.log(1 / q0))
    lam_one = gold_race(structure, structure.s0, np.ones(structure.n_features))
    if lam_one < target:
        return {"stall": True, "q_inf": float(q0 * np.exp(lam_one)), "S_inf": 1.0}
    t_star = brentq(
        lambda t: gold_race(structure, structure.s0, _path_point(structure, t)) - target,
        structure.s0[-1],
        1.0,
        xtol=1e-15,
        rtol=1e-15,
    )
    return {
        "stall": False,
        "q_inf": 1.0,
        "S_inf": exploit_rate(structure, _path_point(structure, float(t_star))),
    }


def clean_gold(u0: float, t: float | np.ndarray) -> np.ndarray:
    """Gold under natural gradient on the clean verifier V = G: u' = 1."""
    return expit(u0 + np.asarray(t))


def random_fp_gold(u0: float, f: float, t: float | np.ndarray) -> np.ndarray:
    """Gold under the random-FP control: u' = 1 - f exactly."""
    return expit(u0 + (1 - f) * np.asarray(t))
