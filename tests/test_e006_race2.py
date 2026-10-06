"""Race model v2: exact base output distribution and Prop. 5 dynamics."""

import numpy as np
import pytest
import torch

from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk
from vdyn.e006 import race2
from vdyn.e006 import theory as th

torch.set_num_threads(2)


def test_base_logprobs_match_the_model_and_normalize():
    net = mdl.build(mdl.GPTConfig(), seed=3)
    L0 = race2.base_logprobs(net, [(3, 4), (50, 60)], batch=2)
    assert L0.shape == (2, race2.N_VALUES + 1)
    assert np.exp(L0).sum(1) == pytest.approx(np.ones(2), abs=1e-5)
    toks = tk.encode_text("7") + [tk.EOS]
    lp = mdl.token_logprobs(net, torch.tensor([tk.encode_prompt(3, 4)]),
                            torch.tensor([toks + [tk.PAD] * (4 - len(toks))]))  # fmt: skip
    assert L0[0, 7] == pytest.approx(float(lp[0, : len(toks)].sum()), abs=1e-4)


def test_bias_update_is_prop5_and_shared_key_collapses():
    rng = np.random.default_rng(0)
    n, K = 40, 6
    L0 = np.log(rng.dirichlet(np.ones(K), size=n))
    gold = rng.integers(0, K - 1, n)
    EV = np.zeros((n, K))
    EV[np.arange(n), gold] = 1.0
    EV[:, K - 1] = 1.0  # output K-1 accepted everywhere: a master key
    out = race2.simulate(L0, gold, EV, steps=300, lr_s=0.01, lr_b=2.0)
    assert out["fpr"][-1] > 0.9 and out["gold"][-1] < out["gold"][0]
    # one step of b equals the log-linear expected update on output-specific features (Prop. 5)
    phi = np.zeros((n, K, K + 1))
    for k in range(K):
        phi[:, k, k] = 1.0
    phi[np.arange(n), gold, K] = 1.0  # skill feature
    theta = np.zeros(K + 2)
    theta[K + 1] = 1.0  # coefficient 1 on the base log-probabilities (a fixed offset)
    u = th.expected_update(theta, np.concatenate([phi, L0[..., None]], 2),
                           np.full(n, 1 / n), EV)[:K]  # fmt: skip
    pi = np.exp(L0) / np.exp(L0).sum(1, keepdims=True)
    v = (pi * EV).sum(1)
    gb = (pi * (EV - v[:, None]) / np.sqrt(v * (1 - v))[:, None]).mean(0)
    assert np.allclose(u, gb, atol=1e-9)
