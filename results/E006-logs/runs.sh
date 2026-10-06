set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e006_stage/runs
VER=results/E006-verification/20261006T061515Z_f05da33
for s in 11 12 13 14 15; do
  for a in clean randfp hashtab cov25 cov50 cov75 exploit rarekey; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e006/run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E006-$a-s$s
    mv "$out" $ST/E006-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
