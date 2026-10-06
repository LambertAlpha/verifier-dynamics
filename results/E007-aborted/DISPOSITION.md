E007 was aborted after its first run (clean, seed 21) for a design flaw: the 320-token generation
cap truncated 43.5% of base completions, and the clean arm degraded (greedy accuracy on 200 test
questions fell from 0.42 to 0.30).

No flawed-verifier arm was run. This run is preserved as the record of the flaw and is not
evidence for any hypothesis. The corrected rerun is E007b (research/paper/e007b_protocol.md).
