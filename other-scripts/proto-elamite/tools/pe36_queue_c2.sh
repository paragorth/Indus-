#!/bin/bash
# pe36 cycle 2: tablet-half replication, 300 restarts per half, shuffled halves 100 restarts x 3 seeds.
cd "$(dirname "$0")"
while pgrep -f pe36_queue_c1.sh >/dev/null; do sleep 30; done
{ for k in PEN PEA PECLS OB UR3 LINB PC; do
    for h in h0 h1; do for c in 0 1 2; do echo "$k $h $c 100 100000"; done; done
    for s in 1 2 3; do for h in h0 h1; do echo "$k ${h}shW$s 0 100 100000"; done; done
  done; } | xargs -P2 -L1 python3 pe36_run.py >> ../data/pe36_ckpt/c2.log 2>&1
