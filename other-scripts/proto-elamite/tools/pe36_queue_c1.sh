#!/bin/bash
# pe36 cycle 1: real grids (PEN 1,000 restarts, others 400) and within-string shuffles (100 restarts per seed).
cd "$(dirname "$0")"
{ for c in $(seq 0 9); do echo "PEN real $c 100 100000"; done
  for s in 1 2 3 4 5 6; do echo "PEN realshW$s 0 100 100000"; done
  for k in PEA PENUM PECLS OB UR3 LINB PC; do
    for c in 0 1 2 3; do echo "$k real $c 100 100000"; done
    for s in 1 2 3; do echo "$k realshW$s 0 100 100000"; done
  done; } | xargs -P2 -L1 python3 pe36_run.py >> ../data/pe36_ckpt/c1.log 2>&1
