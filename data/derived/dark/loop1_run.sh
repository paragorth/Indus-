#!/bin/bash
cd /home/user/Indus-
run(){ python3 tools/strat_dark.py --n 3000 --seed $2 --cycle $1 --families $3 $4 > data/derived/dark/loop1_cycle$1$5.log 2>&1; }
run 1 101 mid,parity "" "" &
run 2 202 set3,w500,shapefl "" "" &
run 3 303 rare,numpos,ascrun "" "" &
run 4 404 base,len,gap,rank "" "" &
run 1 101 mid,parity --control _control &
run 2 202 set3,w500,shapefl --control _control &
run 3 303 rare,numpos,ascrun --control _control &
run 4 404 base,len,gap,rank --control _control &
wait
echo ALLDONE > data/derived/dark/loop1_done.flag
