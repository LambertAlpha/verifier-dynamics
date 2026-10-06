The first E010 scan (commit at start: see meta.json) crashed after the four rule-based graders and
the judge's static rates, at the judge RME step. The cause was a KeyError: the script keyed
`sampled_on` by sample index, but `vdyn.e006.diagnostic.response_main_effect` expects positions
0..len(keys)-1. The rule-grader branch was unaffected because there keys == positions.

No result file was written; the rule-grader numbers are in stdout.log. The fix is in the next
commit, and the full scan is re-run from scratch.
