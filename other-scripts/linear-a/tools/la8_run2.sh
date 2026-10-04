#!/bin/sh
# LA-8 cycles 2-3 queue (2 workers): replicate draws for LB/PE, then zoom runs (>= 8 classes), then stack (recursion) arms.
cd "$(dirname "$0")/.."
while pgrep -f la8_runall.sh >/dev/null; do sleep 20; done
printf 'LB2 1\nPE2 1\n' | xargs -P 2 -n 2 sh -c 'nice python3 tools/la8_gp.py $0 $1 2500 --tag v2 >> data/la8/runv2_$0.log 2>&1'
for c in LA LB PE LAS; do echo $c; done | xargs -P 2 -n 1 sh -c 'LA8_MINK=8 nice python3 tools/la8_gp.py $0 1 1500 --tag zoom >> data/la8/runzoom_$0.log 2>&1'
nice python3 tools/la8_report.py v2 > data/la8/report_v2.txt 2>&1
for s in 1 2; do for c in PFLAT PREC LA LB PE LAS; do for a in fsa pda; do echo "$c $a $s"; done; done; echo "PREC fsa $s truth"; echo "PREC pda $s truth"; done | \
  xargs -P 2 -L 1 sh -c 'nice python3 tools/la8_stack.py $0 $1 $2 200 $3 >> data/la8/runstack.log 2>&1'
echo ALLDONE >> data/la8/runstack.log
