import numpy as np

from vdyn.checks import Checks, spearman_with_ties


def test_close_passes_within_tolerance_and_records_values():
    checks = Checks()
    checks.close("run", "P1", "x", np.array([1.0, 2.0]), [1.0, 2.0 + 5e-7], 1e-6)
    checks.close("run", "P1", "y", 1.0, 1.1, 1e-6)
    assert [r["passed"] for r in checks.rows] == [True, False]
    assert checks.rows[0]["observed"] == [1.0, 2.0]


def test_max_error_and_info_rows():
    checks = Checks()
    checks.max_error("run", "P2", "err", np.array([1e-9, -3e-9]), 1e-8)
    checks.info("run", "P2", "note", 0.5)
    assert checks.rows[0]["observed"] == 3e-9 and checks.rows[0]["passed"] is True
    assert checks.rows[1]["passed"] is None
    assert checks.failed() == []


def test_spearman_uses_average_ranks_for_ties():
    # identical to scipy.stats.spearmanr on tied data
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 0.0, 0.5, 0.9])
    assert spearman_with_ties(x, y) == np.corrcoef([1, 2, 3, 4, 5], [2, 2, 2, 4, 5])[0, 1]
