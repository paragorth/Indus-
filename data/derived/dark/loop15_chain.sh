#!/bin/sh
cd /home/user/Indus-/data/derived/dark
for pfx in loop15_c1d loop15_c1; do
  out=$(echo $pfx | sed s/c1/c2/)
  for k in seq_raw seq_strong seq_all; do
    until [ -f ${pfx}_$k.json ]; do sleep 20; done
    python3 loop15_classify.py --prefix $pfx --seqkey $k --nperm 200 > ${out}_$k.log 2>&1
  done
done
echo ALLDONE > loop15_c2_done.flag
