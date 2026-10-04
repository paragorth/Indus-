#!/bin/bash
# LA-21 cycle 3: held-out halves (LA HT vs non-HT; LB KN vs PY, disjoint, LA-sized) + shuffled halves; grid-size sensitivity.
cd "$(dirname "$0")"
while pgrep -f la21_queue_c2 >/dev/null; do sleep 20; done
{ for j in LAHT LAnonHT LBKN1 LBPY1 LBKN2 LBPY2 LAHTsh1 LAnonHTsh1 LBKNsh1 LBPYsh1; do echo "$j 200 100000 15 5 c3"; done
  for g in "13 4" "18 5" "15 3" "17 6"; do set -- $g; for j in LA LBs1 LAshW2; do echo "$j 100 100000 $1 $2 c3g$1x$2"; done; done; } | xargs -P2 -L1 python3 la21_run.py > ../data/la21_ckpt/c3.log 2>&1
