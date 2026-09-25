"""Exact long-run gold outcome under the natural-gradient flow (Prop. 9, general form).

In the time variable tau = ∫(1 - q) dt the feature dynamics do not depend on q:

    dphi_i/dtau = dFPR/ds_i(s),   dLambda/dtau = 1 - FPR,   q = q0 exp(Lambda),   dt/dtau = 1/(1-q).

Gold stalls iff Lambda(∞) < log(1/q0) (then q_inf = q0 exp(Lambda(∞))); otherwise q -> 1 and the
exploit freezes at the FPR where Lambda = log(1/q0). Phase A carries t (for report times and t95);
phase B drops t, whose derivative diverges as q -> 1, and runs to termination.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

from vdyn.verifiers.boolean_fp import EventStructure, dfpr_ds, one_minus_fpr_from_logits

CONVERGED = 1e-12  # stop when 1 - FPR < this; the remaining tail of Lambda is added analytically
STALL_EPS = 1e-6  # registry E002 amendment 1: stall iff q_inf < 1 - 1e-6 (E003 success threshold)
NEAR_SUCCESS = 1e-9  # phase A stops this close to log(1/q0), before dt/dtau blows up


@dataclass(frozen=True)
class GoldRace:
    stall: bool
    q_inf: float
    fpr_inf: float
    lambda_inf: float
    jg_at: dict[float, float] = field(default_factory=dict)
    fpr_at: dict[float, float] = field(default_factory=dict)
    t95: float | None = None


def _rhs(st: EventStructure, q0: float, with_time: bool):
    m = st.n_features

    def f(_tau: float, y: np.ndarray) -> np.ndarray:
        phi, lam = y[:m], y[m]
        out = np.empty_like(y)
        out[:m] = dfpr_ds(st, expit(phi)) if m else []
        out[m] = one_minus_fpr_from_logits(st, phi)
        if with_time:
            out[m + 1] = 1.0 / max(-np.expm1(np.log(q0) + lam), 1e-300)
        return out

    return f


def _event(fn, terminal: bool, direction: float = 0.0):
    fn.terminal = terminal
    fn.direction = direction
    return fn


def gold_race(
    st: EventStructure,
    q0: float,
    report_times: tuple[float, ...] = (),
    rtol: float = 1e-10,
    atol: float = 1e-12,
    tau_max: float = 1e10,
) -> GoldRace:
    m, target = st.n_features, float(np.log(1 / q0))
    y0 = np.concatenate([np.log(np.asarray(st.s0) / (1 - np.asarray(st.s0))), [0.0]])
    jg_at: dict[float, float] = {}
    fpr_at: dict[float, float] = {}

    def converged(_tau, y):
        return one_minus_fpr_from_logits(st, y[:m]) - CONVERGED

    # Phase A: carry t up to the last report time.
    state = y0
    if report_times:
        times = sorted(report_times)
        near = _event(lambda _tau, y: y[m] - (target - NEAR_SUCCESS), terminal=True, direction=1)
        conv = _event(converged, terminal=True, direction=-1)
        hits = [
            _event(lambda _tau, y, T=T: y[m + 1] - T, terminal=False, direction=1) for T in times
        ]
        last = _event(lambda _tau, y: y[m + 1] - times[-1], terminal=True, direction=1)
        sol = solve_ivp(
            _rhs(st, q0, with_time=True),
            (0.0, tau_max),
            np.concatenate([y0, [0.0]]),
            method="DOP853",
            rtol=rtol,
            atol=atol,
            events=[near, conv, last, *hits],
        )
        if sol.status < 0:
            raise RuntimeError(f"gold race phase A failed for {st.sid}: {sol.message}")
        for T, ys in zip(times, sol.y_events[3:], strict=True):
            if len(ys):
                jg_at[T] = float(q0 * np.exp(ys[0][m]))
                fpr_at[T] = 1 - one_minus_fpr_from_logits(st, ys[0][:m])
        end = sol.y[:, -1]
        for T in times:  # report times beyond a stall/near-success point: state is frozen
            if T not in jg_at:
                jg_at[T] = float(q0 * np.exp(end[m]))
                fpr_at[T] = 1 - one_minus_fpr_from_logits(st, end[:m])
        state = end[: m + 1]

    # Phase B: without t, to success or convergence.
    success = _event(lambda _tau, y: y[m] - target, terminal=True, direction=1)
    conv_b = _event(converged, terminal=True, direction=-1)
    sol = solve_ivp(
        _rhs(st, q0, with_time=False),
        (0.0, tau_max),
        state,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=[success, conv_b],
        dense_output=True,
    )
    if sol.status < 0:
        raise RuntimeError(f"gold race phase B failed for {st.sid}: {sol.message}")
    end = sol.y[:, -1].copy()
    if not len(sol.t_events[0]):
        end[m] += _tail(st, sol)
    if len(sol.t_events[0]) or end[m] >= target:
        ys = sol.y_events[0][0] if len(sol.t_events[0]) else end
        return GoldRace(
            stall=False,
            q_inf=1.0,
            fpr_inf=1 - one_minus_fpr_from_logits(st, ys[:m]),
            lambda_inf=target,
            jg_at=jg_at,
            fpr_at=fpr_at,
        )
    lam_inf = float(end[m])
    q_inf = float(q0 * np.exp(lam_inf))
    if q_inf >= 1 - STALL_EPS:  # numerically at the threshold: not a stall (amendment 1)
        return GoldRace(
            stall=False,
            q_inf=q_inf,
            fpr_inf=1 - one_minus_fpr_from_logits(st, end[:m]),
            lambda_inf=lam_inf,
            jg_at=jg_at,
            fpr_at=fpr_at,
        )
    return GoldRace(
        stall=True,
        q_inf=q_inf,
        fpr_inf=1 - one_minus_fpr_from_logits(st, end[:m]),
        lambda_inf=lam_inf,
        jg_at=jg_at,
        fpr_at=fpr_at,
        t95=_time_to_level(st, q0, q0 + 0.95 * (q_inf - q0), rtol, atol, tau_max),
    )


def _tail(st: EventStructure, sol) -> float:
    """∫_{tau_e}^∞ (1 - FPR) dtau for a power-law tail c tau^-p (p from tau_e/2 -> tau_e);
    negligible for exponential decay."""
    m = st.n_features
    tau_e = float(sol.t[-1])
    if tau_e <= 0:
        return 0.0
    g_e = one_minus_fpr_from_logits(st, sol.y[:m, -1])
    g_h = one_minus_fpr_from_logits(st, sol.sol(tau_e / 2)[:m])
    if g_e <= 0 or g_h <= g_e:
        return 0.0
    p = float(np.log(g_h / g_e) / np.log(2.0))
    if p > 30:  # effectively exponential decay
        return 0.0
    if p <= 1.05:
        raise RuntimeError(f"non-integrable tail (p = {p:.3f}) for {st.sid}")
    return g_e * tau_e / (p - 1)


def _time_to_level(
    st: EventStructure, q0: float, level: float, rtol: float, atol: float, tau_max: float
) -> float:
    m = st.n_features
    lam_level = float(np.log(level / q0))
    y0 = np.concatenate([np.log(np.asarray(st.s0) / (1 - np.asarray(st.s0))), [0.0, 0.0]])
    hit = _event(lambda _tau, y: y[m] - lam_level, terminal=True, direction=1)
    sol = solve_ivp(
        _rhs(st, q0, with_time=True),
        (0.0, tau_max),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=[hit],
    )
    if not len(sol.t_events[0]):
        raise RuntimeError(f"level {level} not reached for {st.sid}")
    return float(sol.y_events[0][0][m + 1])
