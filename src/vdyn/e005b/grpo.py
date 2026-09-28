"""E005b-0 GRPO objective (research/10_e005b0_pilot.md §3).

Standard (DeepSeekMath, arXiv 2402.03300 §4.1): per-prompt groups of G sampled completions;
outcome advantage (r - mean(r)) / std(r); PPO-style clipped ratio with epsilon; one policy update
per rollout batch (mu = 1), so the ratio is 1 at the update and the clip is inactive.
Deliberate choices, documented in the protocol: std with ddof = 1 plus 1e-4 (as TRL); a
zero-variance group has advantage exactly 0; token-level aggregation (sum over completion tokens
divided by the batch's token count, as DAPO / TRL 'dapo') instead of the per-sequence 1/|o_i|;
no KL term (beta = 0, as TRL v1.0 default and DAPO), with the KL to the initial policy logged.
"""

import torch

STD_EPS = 1e-4


def group_advantages(r: torch.Tensor, scale: str = "group") -> torch.Tensor:
    """r: (P, G) rewards -> (P, G) advantages."""
    c = r - r.mean(1, keepdim=True)
    if scale == "none":
        return c
    if scale == "group":
        return c / (r.std(1, keepdim=True, unbiased=True) + STD_EPS)
    raise ValueError(scale)


def policy_loss(lp_new: torch.Tensor, lp_old: torch.Tensor, adv: torch.Tensor,
                mask: torch.Tensor, clip_eps: float) -> torch.Tensor:  # fmt: skip
    """-(1/sum mask) sum_{i,t} mask * min(rho A_i, clip(rho, 1 - eps, 1 + eps) A_i)."""
    ratio = torch.exp(lp_new - lp_old)
    a = adv[:, None]
    obj = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    return -(obj * mask).sum() / mask.sum()


def k3_kl(lp_new: torch.Tensor, lp_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-token k3 estimate of KL(pi || pi_ref) on sampled tokens (diagnostic only)."""
    d = lp_ref - lp_new
    return ((torch.exp(d) - d - 1) * mask).sum() / mask.sum()
