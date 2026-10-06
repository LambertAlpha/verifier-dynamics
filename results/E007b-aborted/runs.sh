set -e
cd ~/Documents/01Project/verifier-dynamics-e007
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 VDYN_E007_EXP=e007b
ST=/tmp/e007b_stage/runs
VER=results/E007b-verification/20261006T101427Z_62ffd39
for s in 31 32 33; do
  for a in clean randfp hashtab ends0 anywhere; do
    test -z "$(git status --porcelain)"
    .venv/bin/python experiments/e007/e007_run.py $a $s --verification $VER > $ST/$a-s$s.log 2>&1
    out=$(grep '^run directory:' $ST/$a-s$s.log | sed 's/^run directory: //')
    test -n "$out" && test -d "$out"
    mkdir -p $ST/E007b-$a-s$s
    mv "$out" $ST/E007b-$a-s$s/
    echo "$a s$s done: $out $(date)" >> $ST/progress.txt
  done
done
echo ALL_DONE >> $ST/progress.txt
