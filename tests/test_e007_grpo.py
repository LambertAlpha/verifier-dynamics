"""E007 MLX GRPO pieces: group advantages, masked token-mean loss, and that one step raises the
log-probability of a positively-advantaged completion (tiny random Qwen2-architecture model)."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mg = pytest.importorskip("vdyn.e007.mlx_grpo")


def test_group_advantages_match_the_toy_definition():
    V = np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1.0]])
    A = mg.group_advantages(V)
    sd = V[0].std(ddof=1)
    assert A[0] == pytest.approx((V[0] - V[0].mean()) / (sd + 1e-4))
    assert np.all(A[1] == 0)
    assert A[2, 3] > 0 > A[2, 0]


def test_one_step_increases_the_logprob_of_an_advantaged_completion():
    model = mg.tiny_model(seed=0)
    prompts = [[1, 2, 3], [4, 5]]
    comps = [[7, 8, 9, 2], [10, 11]]
    adv = np.array([1.0, -1.0])
    before = mg.sequence_logprobs(model, prompts, comps)
    opt = mg.make_optimizer(lr=1e-2)
    stats = mg.train_step(model, opt, prompts, comps, adv, micro_batch=1, clip=1.0)
    after = mg.sequence_logprobs(model, prompts, comps)
    assert after[0] > before[0] and after[1] < before[1]
    assert np.isfinite(stats["loss"]) and stats["grad_norm"] > 0
    assert mx.default_device() is not None
