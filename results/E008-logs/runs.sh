set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e008_stage/runs
VER=results/E008-verification/20261006T072948Z_b33d3bd
V6=results/E006-verification/20261006T061515Z_f05da33
# integrity: E006 clean-s11 rerun with the extended verifiers module must reproduce its final hash
test -z "$(git status --porcelain)"
.venv/bin/python experiments/e006/run.py clean 11 --verification $V6 > $ST/integrity-clean-s11.log 2>&1
out=$(grep '^run directory:' $ST/integrity-clean-s11.log | sed 's/^run directory: //')
mkdir -p $ST/integrity && mv "$out" $ST/integrity/
echo "integrity done: $out" >> $ST/progress.txt
for s in 11 12 13 14 15; do
  for a in covhard coveasy set02 set05 setq; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e008/e008_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E008-$a-s$s
    mv "$out" $ST/E008-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
