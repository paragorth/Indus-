#!/bin/bash
# pe36 cycle 3: universals ablation (tier likelihood only) and grid-size sensitivity.
cd "$(dirname "$0")"
while pgrep -x -f "/bin/bash ./pe36_queue_c2.sh" >/dev/null; do sleep 30; done
{ for k in PEN OB LINB UR3 PC; do
    for c in 0 1 2; do echo "$k real $c 100 100000 ll_"; done
    for s in 1 2; do echo "$k realshW$s 0 100 100000 ll_"; done
  done
  for g in g13x4_ g18x5_ g15x3_; do for k in PEN OB PC; do
    for c in 0 1; do echo "$k real $c 100 100000 $g"; done
    echo "$k realshW1 0 100 100000 $g"; echo "$k realshW2 0 100 100000 $g"
  done; done; } | xargs -P2 -L1 python3 pe36_run.py >> ../data/pe36_ckpt/c3.log 2>&1
