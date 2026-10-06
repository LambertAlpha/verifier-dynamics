# E011 — results (2026-10-06): the ACM rule FAILS prospectively

Protocol `e011_protocol.md`, frozen at 13b4f10. Integrity:
- arms verified (FPR 0.111–0.118);
- ACM predictions committed before runs (1ebb6bd);
- 15 runs, no STOP;
- the frozen analysis ran in a clean worktree at 616fee6.

| arm | ACM_0.75 | prediction | harm (seed mean, range) | collapsed | collapse outputs |
|---|---|---|---|---|---|
| near | 0.105 | harm ≥ 0.25 | **0.663** (0.64–0.68) | 5/5 | 100, 90, 130 (near-miss round numbers) |
| far | 0.0008 | harm < 0.25 | **0.744** (0.72–0.76) | **5/5** | s11: 100, 900, 800; s12: 0, 10 |
| near50 | 0.000 | harm < 0.25 | 0.116 (0.09–0.16) | 0/5 | – |

**Verdict: FAIL.** The rule is right for near and near50 and wrong for far.

## What went wrong
- Under far, a round number accepted *wherever it is far from the sum* includes outputs that are
  far from **every** sum: 900 and 800 exceed the maximum sum of 198, and 0 / 10 are far from most
  sums. These are perfect master keys.
- The base policy almost never produced them where they are accepted, so their on-policy
  conditional acceptance, and hence ACM, was ≈ 0.
- The policy discovered them anyway. Batch FPR reached 0.5 at steps 32–176, slower than near's
  8–14, which is consistent with a reachability delay (Conjecture 6).

## Reading
- Conditioning on where the *base* policy currently produces an output is wrong. What matters is
  whether the verifier admits an output or strategy that is accepted (nearly) everywhere it would
  be used, and that the policy can reach.
- Our two static diagnostics each capture one projection of this, and each misses a case:
  - RME, which transplants constant outputs across prompts, should flag far, but cannot see
    input-dependent strategies (near's "round to the nearest ten") or region-conditional keys
    (covhard vs cov50);
  - ACM catches near / covhard / coveasy but misses far.
- A static scan of base samples cannot enumerate the strategies a policy can discover.
- This points back to the project's original question: **short policy-conditioned probes** (a few
  dozen RL steps under the verifier, with a gold audit). Earlier work in this repo found that
  short probes beat static geometry (E002, E004a). Collapse here showed up within 7–176 steps.
