Verification run 20261006T081726Z_008a864 stopped with exit 3 because the clean arm "failed"
matching (FPR 0 by design). This was a bug in experiments/e009/e009_audit.py: it was copied from
E008, which had no clean arm, and it lacks E006's `arm == "clean" or ...` exemption.

All five matched arms passed (FPR 0.109–0.119).

Fixed in the next commit. Verification is re-run on the same deterministic samples. No training
happened.
