set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e005b_matched_stage/runs
VER=results/E005b0-matched-verification/20260928T215701Z_a1eada7
for s in 4 5 6; do
  for a in clean randfp exploit; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e005b/matched_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E005b0-matched-$a-s$s
    mv "$out" $ST/E005b0-matched-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
