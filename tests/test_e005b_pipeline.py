"""E005b-0 model, generation, GRPO, SFT and checkpoints (research/10_e005b0_pilot.md §2-§4).

Independent references: teacher-forced recomputation of sampled log-probs, softmax sampling
frequencies, an exact policy-gradient on an enumerable softmax policy, a hand-written
cross-entropy, and bit-exact resumption.
"""

import numpy as np
import pytest
import torch

from vdyn.e005b import grpo, sft
from vdyn.e005b import model as mdl
from vdyn.e005b import task as tk

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def net():
    return mdl.build(mdl.GPTConfig(), seed=0)


def prompts(pairs):
    return torch.tensor([tk.encode_prompt(a, b) for a, b in pairs])


def test_model_is_causal_and_deterministically_initialized(net):
    x = torch.randint(3, tk.VOCAB_SIZE, (2, 10))
    y = x.clone()
    y[:, 6:] = torch.randint(3, tk.VOCAB_SIZE, (2, 4))
    with torch.no_grad():
        assert torch.allclose(net(x)[:, :6], net(y)[:, :6], atol=1e-6)
    other = mdl.build(mdl.GPTConfig(), seed=0)
    for p, q in zip(net.parameters(), other.parameters(), strict=True):
        assert torch.equal(p, q)
    n = sum(p.numel() for p in net.parameters())
    assert 300_000 < n < 600_000


def test_sampling_masks_after_end_and_logprobs_match_teacher_forcing(net):
    g = torch.Generator().manual_seed(1)
    P = prompts([(7, 45), (99, 99), (0, 3), (50, 50)] * 16)
    out = mdl.sample(net, P, tk.MAX_NEW, 1.0, g)
    tok, mask, lp = out["tokens"], out["mask"], out["logp"]
    assert tok.shape == (64, tk.MAX_NEW) and (mask[:, 0] == 1).all()
    for i in range(64):
        row = tok[i].tolist()
        if tk.EOS in row:
            e = row.index(tk.EOS)
            assert mask[i, : e + 1].eq(1).all() and mask[i, e + 1 :].eq(0).all()
            assert all(t == tk.PAD for t in row[e + 1 :])
        else:
            assert mask[i].eq(1).all()
    ref = mdl.token_logprobs(net, P, tok)
    assert torch.allclose(ref * mask, lp * mask, atol=1e-5)


def test_first_token_frequencies_follow_the_softmax(net):
    P = prompts([(12, 34)] * 20000)
    g = torch.Generator().manual_seed(2)
    out = mdl.sample(net, P, 1, 1.0, g)
    with torch.no_grad():
        probs = torch.softmax(net(P[:1])[0, -1], -1)
    freq = torch.bincount(out["tokens"][:, 0], minlength=tk.VOCAB_SIZE).float() / 20000
    se = torch.sqrt(probs * (1 - probs) / 20000)
    assert (torch.abs(freq - probs) <= 5 * se + 1e-4).all()


def test_greedy_follows_the_argmax_chain(net):
    P = prompts([(3, 4), (88, 21)])
    tok, _ = mdl.greedy(net, P, tk.MAX_NEW)
    seq = P.clone()
    with torch.no_grad():
        for t in range(tk.MAX_NEW):
            nxt = net(seq)[:, -1].argmax(-1)
            done = (
                (seq[:, tk.PROMPT_LEN :] == tk.EOS).any(1)
                if t
                else torch.zeros(2, dtype=torch.bool)
            )
            nxt = torch.where(done, torch.full_like(nxt, tk.PAD), nxt)
            assert torch.equal(nxt, tok[:, t])
            seq = torch.cat([seq, nxt[:, None]], 1)


# ------------------------------------------------------------------ GRPO
def test_group_advantages_match_a_reference_and_zero_variance_groups_are_zero():
    r = torch.tensor([[1.0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    a = grpo.group_advantages(r, scale="group")
    ref = (r.numpy() - r.numpy().mean(1, keepdims=True)) / (
        r.numpy().std(1, ddof=1, keepdims=True) + 1e-4
    )
    assert np.allclose(a.numpy(), ref, atol=1e-6)
    assert torch.equal(a[1], torch.zeros(4)) and torch.equal(a[2], torch.zeros(4))
    assert torch.allclose(grpo.group_advantages(r, scale="none"), r - r.mean(1, keepdim=True))


def test_loss_gradient_is_the_token_averaged_policy_gradient_when_on_policy():
    torch.manual_seed(0)
    logits = torch.randn(6, 3, 5, requires_grad=True)
    tok = torch.randint(0, 5, (6, 3))
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1.0]])
    adv = torch.randn(6)
    lp = torch.log_softmax(logits, -1).gather(-1, tok[..., None])[..., 0]
    loss = grpo.policy_loss(lp, lp.detach(), adv, mask, clip_eps=0.2)
    (g1,) = torch.autograd.grad(loss, logits)
    lp2 = torch.log_softmax(logits, -1).gather(-1, tok[..., None])[..., 0]
    ref = -(adv[:, None] * lp2 * mask).sum() / mask.sum()
    (g2,) = torch.autograd.grad(ref, logits)
    assert torch.allclose(g1, g2, atol=1e-7)


def test_clipping_removes_the_gradient_outside_the_trust_region():
    lp_old = torch.zeros(2, 1)
    lp_new = torch.log(torch.tensor([[1.5], [0.5]])).requires_grad_()
    adv = torch.tensor([1.0, -1.0])
    loss = grpo.policy_loss(lp_new, lp_old, adv, torch.ones(2, 1), clip_eps=0.2)
    (g,) = torch.autograd.grad(loss, lp_new)
    assert torch.allclose(g, torch.zeros(2, 1))


def test_group_estimator_is_proportional_to_the_exact_policy_gradient():
    """Softmax over K outcomes (one token): E[grad of our loss] = -(G-1)/G grad J for the
    mean baseline (scale 'none'); checked against the exact gradient by enumeration."""
    G, n_groups = 8, 20000
    theta = torch.tensor([0.3, -0.2, 0.9, 0.0, -1.1], requires_grad=True)
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5])
    pi = torch.softmax(theta, 0)
    (exact,) = torch.autograd.grad((pi * r).sum(), theta)
    g = torch.Generator().manual_seed(5)
    y = torch.multinomial(pi.detach(), n_groups * G, replacement=True, generator=g)
    y = y.view(n_groups, G)
    adv = grpo.group_advantages(r[y], scale="none").reshape(-1)
    lp = torch.log_softmax(theta, 0)[y.reshape(-1)][:, None]
    loss = grpo.policy_loss(lp, lp.detach(), adv, torch.ones_like(lp), clip_eps=0.2)
    (est,) = torch.autograd.grad(loss, theta)  # token (= sample) average over n_groups * G
    assert torch.allclose(-est, exact * (G - 1) / G, atol=0.01)


# ------------------------------------------------------------------ SFT
def test_sft_loss_is_cross_entropy_on_answer_tokens_only(net):
    pairs = [(7, 45), (99, 99), (0, 0)]
    x, y = sft.make_batch(pairs)
    loss = sft.loss(net, x, y)
    logits = net(x)
    tot, cnt = 0.0, 0
    for i, (a, b) in enumerate(pairs):
        ans = tk.encode_answer(a + b)
        for j, t in enumerate(ans):
            pos = tk.PROMPT_LEN - 1 + j
            tot += -torch.log_softmax(logits[i, pos], -1)[t].item()
            cnt += 1
    assert loss.item() == pytest.approx(tot / cnt, rel=1e-5)


# ------------------------------------------------------------------ checkpoints
def test_resumed_training_is_bit_identical(tmp_path):
    def run(steps, resume_after=None):
        net = mdl.build(mdl.GPTConfig(), seed=1)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(9)
        pairs = tk.make_splits(0)["train"][:64]
        for s in range(steps):
            if resume_after is not None and s == resume_after:
                path = tmp_path / "ck.pt"
                mdl.save_checkpoint(path, net, opt, gen, step=s)
                net = mdl.build(mdl.GPTConfig(), seed=123)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                gen = torch.Generator()
                meta = mdl.load_checkpoint(path, net, opt, gen)
                assert meta["step"] == s
            idx = torch.randint(0, len(pairs), (16,), generator=gen)
            x, y = sft.make_batch([pairs[i] for i in idx])
            opt.zero_grad()
            sft.loss(net, x, y).backward()
            opt.step()
        return net

    a, b = run(4), run(4, resume_after=2)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p, q)
    assert mdl.state_sha256(a) == mdl.state_sha256(b)
