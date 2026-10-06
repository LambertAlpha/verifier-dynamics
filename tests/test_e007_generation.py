"""E007c: completions used for training must include the stop token (mlx_lm.batch_generate drops
it, which left 'end the response' untrained in E007 / E007b)."""

import pytest

mx = pytest.importorskip("mlx.core")
mg = pytest.importorskip("vdyn.e007.mlx_grpo")
gen = pytest.importorskip("vdyn.e007.generation")


def test_completions_end_with_the_stop_token_when_they_stop():
    model = mg.tiny_model(seed=1)
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(0)
    prompts = [[1, 3, 4], [5, 6], [7, 8, 9, 10]] * 4
    toks, reasons = gen.generate(model, [2], prompts, max_tokens=12, sampler=make_sampler(temp=1.0))
    assert len(toks) == len(prompts) and len(reasons) == len(prompts)
    stopped = [t for t, r in zip(toks, reasons, strict=True) if r == "stop"]
    assert stopped, "some completions should stop within 12 tokens"
    for t, r in zip(toks, reasons, strict=True):
        if r == "stop":
            assert t[-1] == 2 and 2 not in t[:-1]
        else:
            assert r == "length" and len(t) == 12 and 2 not in t
