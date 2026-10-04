#!/bin/sh
# LA-8 cycle 3 stack arms (2 workers): fsa vs pda, equal compute, over the best final classes of each corpus.
cd "$(dirname "$0")/.."
for s in 1 2; do echo "PREC fsa $s truth"; echo "PREC pda $s truth"; for c in PREC PFLAT LA LB LB2 PE LAS; do for a in fsa pda; do echo "$c $a $s"; done; done; done | \
  xargs -P 2 -L 1 sh -c 'nice python3 tools/la8_stack.py $0 $1 $2 300 $3 >> data/la8/runstack.log 2>&1'
echo ALLDONE >> data/la8/runstack.log
