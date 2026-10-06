#!/bin/sh
# pe62 job runner: runs lines of a queue file ("CORPUS MODE NHYP"), at most 2 pe62 python jobs at a time.
Q=$1
while read c m n; do
  [ -z "$c" ] && continue
  while [ $(pgrep -f "python3 pe62_" | wc -l) -ge 2 ]; do sleep 15; done
  (python3 pe62_c1.py $c $m $n || echo FAIL $c $m >> ../data/pe62_ckpt/fail.txt) &
  sleep 5
done < $Q
wait
