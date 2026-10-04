#!/bin/sh
# LA-8 cycle 3 queue (2 workers): (a) v3 = unconstrained evolution restarted from the >=8-class zoom populations
# (same 1,500 generations for every corpus); (b) stack arms (fsa vs pda) over the best v2 classes.
cd "$(dirname "$0")/.."
for c in LA LB PE LAS; do echo $c; done | xargs -P 2 -n 1 sh -c 'nice python3 tools/la8_gp.py $0 1 1500 --tag v3 >> data/la8/runv3_$0.log 2>&1'
for s in 1 2; do for c in PREC PFLAT LA LB LB2 PE LAS; do for a in fsa pda; do echo "$c $a $s"; done; done; echo "PREC fsa $s truth"; echo "PREC pda $s truth"; done | \
  xargs -P 2 -L 1 sh -c 'nice python3 tools/la8_stack.py $0 $1 $2 300 $3 >> data/la8/runstack.log 2>&1'
echo ALLDONE >> data/la8/runstack.log
