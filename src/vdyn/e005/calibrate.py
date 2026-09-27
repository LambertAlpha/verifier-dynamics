"""E005a calibration engine (research/07_e005a_measurement.md §4, §6).

For one calibration point: R Monte Carlo audits at every (N, m); for every metric (I, D, F) the
estimators E0 (legacy, pooled, same-batch metric), E1, E2, E3 and the paired audit-only plug-in
('plugin', the TP1 input); per-estimator summaries against the oracle and the per-replication
C^2 values (for rank statistics). Metrics D / F: E1-E3 and 'plugin' whiten with the metric from
the independent unlabeled rollouts; E0 with the metric from all rollouts (legacy).
"""

from typing import Any

import numpy as np

from vdyn.e005 import estimators as es
from vdyn.e005 import sampling as sm

N_GRID = (32, 64, 128, 256, 512, 1024)
M_GRID = (4, 8)
R_REPS = 100
ESTIMATORS = ("E0", "E1", "E2", "E3", "plugin")
REP_CHUNK = 25
Z95, Z_ONE = 1.96, 1.645


def summarize(v: np.ndarray, se: np.ndarray, truth: float, is_c2: bool = False) -> dict[str, float]:
    ok = np.isfinite(v)
    out: dict[str, float] = {"truth": float(truth), "undefined": float(1 - ok.mean())}
    if not ok.any():
        return out
    x, s = v[ok], se[ok]
    out.update(mean=float(x.mean()), var=float(x.var(ddof=1)) if len(x) > 1 else 0.0,
               bias=float(x.mean() - truth), rmse=float(np.sqrt(np.mean((x - truth) ** 2))),
               coverage=float(np.mean(np.abs(x - truth) <= Z95 * s)),
               mean_se=float(np.nanmean(s)))  # fmt: skip
    if is_c2:
        out.update(q95=float(np.quantile(x, 0.95)), reject=float(np.mean(x - Z_ONE * s > 0)))
    return out


def _metric_W(scores: np.ndarray, name: str) -> np.ndarray | None:
    if name == "I":
        return None
    return es.metric_sqrt(es.fisher_hat(scores), {"D": "diag", "F": "full"}[name])


def _estimates(au: dict[str, np.ndarray], metric: str) -> dict[str, dict[str, np.ndarray]]:
    xG, xV, xVu = au["xG"], au["xV"], au["xVu"]
    W_u = _metric_W(au["scores_u"], metric)
    W_all = _metric_W(np.concatenate([au["scores_a"], au["scores_u"]], axis=1), metric)
    out: dict[str, dict[str, np.ndarray]] = {}
    est = es.estimate_all(es.whiten(xG, W_u), es.whiten(xV - xG, W_u))
    for name in ("E1", "E2", "E3", "plugin"):
        out[name] = est[name]
    out["E0"] = es.legacy_e0(xG, xV, xVu, W_all)
    return out


def run_point(point: dict[str, Any], seed: np.random.SeedSequence, N: tuple[int, ...] = N_GRID,
              m: tuple[int, ...] = M_GRID, R: int = R_REPS) -> dict[str, Any]:  # fmt: skip
    prep = sm.prepare(point)
    configs = [(nn, mm) for nn in N for mm in m if nn // mm >= 3]
    rngs = [np.random.default_rng(s) for s in seed.spawn(len(configs))]
    summary: dict[str, Any] = {}
    reps: dict[str, np.ndarray] = {}
    for (nn, mm), rng in zip(configs, rngs, strict=True):
        acc: dict[str, dict[str, list[np.ndarray]]] = {}
        for start in range(0, R, REP_CHUNK):
            au = sm.audit(point, rng, min(REP_CHUNK, R - start), nn, mm, prep)
            for metric in ("I", "D", "F"):
                for name, e in _estimates(au, metric).items():
                    slot = acc.setdefault(f"{metric}|{nn}|{mm}|{name}", {})
                    for q in ("A2", "alpha", "C2", "SE_A2", "SE_alpha", "SE_C2"):
                        slot.setdefault(q, []).append(np.asarray(e[q], dtype=float))
        for key, slot in acc.items():
            metric = key.split("|")[0]
            truth = point["oracle"][metric]
            arr = {q: np.concatenate(v) for q, v in slot.items()}
            summary[key] = {q: summarize(arr[q], arr[f"SE_{q}"], truth[q], is_c2=(q == "C2"))
                            for q in ("A2", "alpha", "C2")}  # fmt: skip
            reps[key] = arr["C2"].astype(np.float32)
            reps[key + "|SE"] = arr["SE_C2"].astype(np.float32)
    return {"pid": point["pid"], "summary": summary, "reps": {k: v for k, v in reps.items()
                                                              if not k.endswith("|SE")},
            "reps_se": {k[:-3]: v for k, v in reps.items() if k.endswith("|SE")}}  # fmt: skip
