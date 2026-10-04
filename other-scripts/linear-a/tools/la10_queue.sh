#!/bin/sh
# usage: la10_queue.sh "TARGET SEED POP GENS SPELL WDIST" ... ; runs at most 2 at a time
cd "$(dirname "$0")"
for a in "$@"; do echo "$a"; done | xargs -P 2 -I{} sh -c 'python3 la10_run.py {} >> ../data/la10/queue.log 2>&1'
