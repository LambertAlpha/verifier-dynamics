set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e012_stage/runs
VER=results/E012-verification/20261006T100504Z_54d409e
for s in 11 12 13 14 15; do
  for a in randfp30 del50 cov65 far40 key111half hard75; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e012/e012_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E012-$a-s$s
    mv "$out" $ST/E012-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
