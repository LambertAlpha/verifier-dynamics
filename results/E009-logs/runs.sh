set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e009_stage/runs
VER=results/E009-verification/20261006T081753Z_a894515
for s in 11 12 13 14 15; do
  for a in aeven sumeven hardhalf clean cov50 covhard; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e009/e009_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E009-$a-s$s
    mv "$out" $ST/E009-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
