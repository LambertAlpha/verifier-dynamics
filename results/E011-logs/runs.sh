set -e
cd ~/Documents/01Project/verifier-dynamics
ST=/tmp/e011_stage/runs
VER=results/E011-verification/20261006T092225Z_db8c139
for s in 11 12 13 14 15; do
  for a in near far near50; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e011/e011_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E011-$a-s$s
    mv "$out" $ST/E011-$a-s$s/
    echo "$a s$s done: $out" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
