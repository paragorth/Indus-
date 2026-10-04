#!/bin/bash
# LA-21 cycle 2: null distributions at 100 restarts, LB sampling spread, LL-only ablation.
cd "$(dirname "$0")"
while pgrep -f "la21_run.py .* c1" >/dev/null; do sleep 20; done
{ for j in LA LAshW2 LAshW3 LAshW4 LAshW5 LAshW6 LAshG2 LAshG3 LAshG4 LAshG5 LAshG6 LBs3 LBs4 LBs5 LBshW2 LBshW3 LBshW4 LBshG2 LBshG3 JPN2 GRC2 HAW1; do echo "$j 100 100000 15 5 c2 1 1"; done
  for j in LA LBs1 LAshW2 LBshW2; do echo "$j 100 100000 15 5 c2ll 0 0"; done; } | xargs -P2 -L1 python3 la21_run.py > ../data/la21_ckpt/c2.log 2>&1
