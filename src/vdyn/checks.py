"""Pass/fail records comparing observations with registered predictions (E003 onwards)."""

from typing import Any

import numpy as np
from scipy.stats import rankdata


def _plain(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        run: str,
        prediction: str,
        quantity: str,
        observed: Any,
        expected: Any,
        tolerance: Any,
        passed: bool | None,
    ) -> None:
        self.rows.append(
            {
                "run": run,
                "prediction": prediction,
                "quantity": quantity,
                "observed": _plain(observed),
                "expected": _plain(expected),
                "tolerance": _plain(tolerance),
                "passed": passed,
            }
        )

    def close(
        self, run: str, prediction: str, quantity: str, observed: Any, expected: Any, tol: float
    ) -> None:
        err = float(
            np.max(np.abs(np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float)))
        )
        self.add(run, prediction, quantity, observed, expected, tol, err <= tol)

    def max_error(self, run: str, prediction: str, quantity: str, errors: Any, tol: float) -> None:
        worst = float(np.max(np.abs(errors)))
        self.add(run, prediction, quantity, worst, 0.0, tol, worst <= tol)

    def info(self, run: str, prediction: str, quantity: str, observed: Any) -> None:
        self.add(run, prediction, quantity, observed, None, None, None)

    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["passed"] is False]


def spearman_with_ties(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
