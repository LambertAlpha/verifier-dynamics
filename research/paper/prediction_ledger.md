# Prediction ledger (all pre-registered predictions of the paper program, with outcomes)

Commit hashes are those of the frozen protocol. Outcomes are taken from the results records.
Failed predictions are kept as failed.

| exp | id | prediction | outcome |
|---|---|---|---|
| E006 (77d13da) | H1 | random FP ≈ clean | **pass** (−0.004) |
| E006 | H2 | per-prompt hashed table ≈ clean | **pass** (−0.018) |
| E006 | H3 | harm non-increasing in coverage; cov25 ≈ clean; cov75 and exploit collapse | **fail** (cov25 −0.085) |
| E006 | H3b | sign of the pooled initial push predicts the step 0→50 direction | **fail** (cov25, randfp) |
| E006 | H4 | a rare key's takeover is later than exploit's, or never | **pass** (never) |
| E006 | H5 | gold-free RME ranks harm over 10 verifiers (ρ ≥ 0.7) and beats FPR and J | **fail** (0.60 < 0.64) |
| E006 (A1) | H5b | RME ranks harm over the 8 matched arms (ρ ≥ 0.7) | **pass** (0.83) |
| E008 (a4fecf6) | L1 | covhard < cov50 < coveasy (difficulty-local) | **fail** (coveasy worst) |
| E008 | L2 | pre-run push: positive for covhard, non-positive for coveasy | **fail** (reasoning error) |
| E008 | R1 | harm non-decreasing in key mass; setq collapses | **fail** (by 0.0004 at the ceiling; collapse ✓) |
| E008 | R2 | rule form beyond mass and coverage does not matter | **pass** (Δ 0.002) |
| E008 | race v1 | covhard 0.31; coveasy 0.03; sets collapse | 4 / 5 (coveasy ✗) |
| E009 (39fe302) | P1, P2 | parity regions collapse | **fail** |
| E009 | P3 | a hashed half of three-digit prompts does not collapse | **pass** |
| E009 | P4 | harm thresholds | **fail** |
| E011 (13b4f10) | ACM rule | harm ≥ 0.25 iff ACM_0.75 ≥ 0.02 (near, far, near50) | **fail** (far) |
| E012 (8cf68a4) | probe rule | harm ≥ 0.25 iff the 150-step FPR rise ≥ 0.25 (6 new verifiers) | **fail as registered**, 5 / 6 (cov65 0.230) |
| E012 | race v2 | 6 predictions | 4 / 6 |
| E007b (f10832b, A1 056949b) | E1–E5, probe A1 | see e007b_protocol.md | pending |
| E010 (a9bfb56) | G1–G3 | rule graders have no response-level key; the judge does | pending |

Totals so far (scored items):
- E006–E012: 7 pass, 12 fail.
- Rules scored per arm: E011 2 / 3, E012 5 / 6, race v1 4 / 5, race v2 4 / 6.

The program's empirical claims rest on the robust findings. Its mechanistic claims are stated with
the failures that shaped them.
