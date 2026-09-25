import numpy as np
import pytest

SEED = 20260924


@pytest.fixture(scope="session")
def logit_points() -> np.ndarray:
    """(u, v) logits: a 9x9 grid on [-8, 8]^2 plus 200 seeded draws on [-12, 12]^2.

    |logit| = 12 means q or s within ~6e-6 of the boundary.
    """
    grid = np.linspace(-8.0, 8.0, 9)
    uu, vv = np.meshgrid(grid, grid)
    rng = np.random.default_rng(SEED)
    rand = rng.uniform(-12.0, 12.0, size=(200, 2))
    return np.vstack([np.column_stack([uu.ravel(), vv.ravel()]), rand])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
