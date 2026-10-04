#!/bin/sh
# LA-30 run queue (2 workers at a time). Waits for the cycle-1 chain, then runs the rest.
cd "$(dirname "$0")"
L=../data/la30_ckpt
while kill -0 ${1:-0} 2>/dev/null; do sleep 30; done
while pgrep -f "la30_c1.py round free 24$" > /dev/null; do sleep 30; done
python3 la30_c1.py exact free 24 prune > $L/c1_exact_prune.log 2>&1
python3 la30_c2.py rel10 16 > $L/c2_rel10.log 2>&1
python3 la30_c3.py 20 > $L/c3.log 2>&1
python3 la30_c1.py round free 24 prune > $L/c1_round_prune.log 2>&1
