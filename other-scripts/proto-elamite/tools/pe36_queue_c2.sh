#!/bin/bash
# pe36 cycles 2-3 (one lean queue, 2 workers; finished chunks are skipped):
#  c2  tablet-half replication: 200-300 restarts per half, shuffled halves 2 seeds x 100 restarts
#  c3b OB names restricted to syllabic spellings
#  c3  universals ablation (ll_: tier likelihood only) and grid sizes 13x4, 18x5
cd "$(dirname "$0")"
{ for k in PEN PEA OB LINB PC UR3 PECLS; do
    for h in h0 h1; do for c in 0 1; do echo "$k $h $c 100 100000"; done; done
    for s in 1 2; do for h in h0 h1; do echo "$k ${h}shW$s 0 100 100000"; done; done
  done
  for c in 0 1 2 3; do echo "OBSYL real $c 100 100000"; done; for s in 1 2 3; do echo "OBSYL realshW$s 0 100 100000"; done
  for k in PEN OB LINB PC UR3; do for c in 0 1; do echo "$k real $c 100 100000 ll_"; done; for s in 1 2; do echo "$k realshW$s 0 100 100000 ll_"; done; done
  for g in g13x4_ g18x5_; do for k in PEN OB PC; do
    for c in 0 1; do echo "$k real $c 100 100000 $g"; done; for s in 1 2; do echo "$k realshW$s 0 100 100000 $g"; done
  done; done; } | xargs -P2 -L1 python3 pe36_run.py >> ../data/pe36_ckpt/c2.log 2>&1
