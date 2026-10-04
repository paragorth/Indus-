#!/bin/sh
# LA-8 stage 1: equal compute for every corpus (same generations, population, offspring, seeds); 2 workers.
cd "$(dirname "$0")/.."
GENS=${GENS:-2500}
for s in 1 2 3; do for c in PFLAT PREC LA LB PE LAS; do echo "$c $s"; done; done | \
  xargs -P 2 -n 2 sh -c 'nice python3 tools/la8_gp.py $0 $1 '"$GENS"' --tag v2 >> data/la8/runv2_$0.log 2>&1'
