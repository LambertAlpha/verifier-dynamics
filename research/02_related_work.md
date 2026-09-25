# 02 — Related work (stub)

Seeded from proposal v2 §8/§12 (2026-09-21). Positioning is owned by the research collaborator;
this file only tracks what each work is used for in this repository.

| Work | Relevance here |
| --- | --- |
| Skalse, Howe, Krasheninnikov, Krueger. *Defining and Characterizing Reward Hacking.* NeurIPS 2022. arXiv:2209.13085 | Binary hackability over policy sets; we want a graded, policy/optimizer/horizon-conditioned counterpart. |
| Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.* ICML 2023. arXiv:2210.10760 | Proxy-vs-gold overoptimization curves; our outcome variables mirror theirs. |
| Karwowski et al. *Goodhart's Law in Reinforcement Learning.* 2023. arXiv:2310.09144 | Geometric Goodhart account and early stopping; corresponds to proposal "Route B". |
| Laidlaw, Singhal, Dragan. *Correlated Proxies.* 2024. arXiv:2403.03185 | Correlation-breakdown definition of hacking. |
| Kim et al. *RewardMATH.* arXiv:2410.01729 | Robustness-oriented static benchmark; static baseline. |
| Liu et al. *Dr. GRPO.* 2025. arXiv:2503.20783 | Std-normalization bias in GRPO; relevant to cross-prompt reweighting (R, X aggregate). |
| Huang et al. *From Accuracy to Robustness.* 2025. arXiv:2505.22203 | Static-accurate verifiers exploited in RL; adversarial pattern set baseline. |
| Ackermann et al. *Gradient Regularization Mitigates Reward Hacking.* 2026. arXiv:2602.18037 | Training-time gradient intervention. |
| Wang et al. *GRIFT.* 2026. arXiv:2604.16242 | Gradient fingerprints for hacking detection; candidate baseline. |
| Deng et al. *Directional Alignment Mitigates Reward Hacking.* 2026. arXiv:2605.25189 | Directional gradient alignment; closest in spirit to the `alpha`/`C` decomposition — needs a careful diff. |
| Qi, Wright, MacDiarmid, Hubinger. *Training a Misaligned Reward Seeker.* Anthropic, 2026. | Gradual emergence of hacking during RL. |

## Terminology note

"Empirical Fisher" in the ML literature (e.g. Kunstner, Balles, Hennig 2019) usually means the
outer product of gradients of the log-likelihood of *observed labels*, which is not the Fisher.
In this repository the Fisher estimated by sampling `y ~ pi` is called the **Monte Carlo Fisher**.
Adam's second-moment estimate is closer to an empirical-Fisher-like quantity of minibatch gradients;
keep the terms separate in Phase 2.
