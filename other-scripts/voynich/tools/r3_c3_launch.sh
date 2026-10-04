#!/bin/sh
# cycle 3: refinement rounds (3,000 worlds per target, proposal = perturbed accepted worlds), one target at a time, 2 workers
cd "$(dirname "$0")"
for T in VMS PE LA LB; do
  R3_V2=1 python3 r3_run.py ref_$T 3000 5000000 ../data/r3_ckpt/prop_$T.json > ../data/r3_ckpt/ref_$T.log 2>&1
done
echo ALLDONE >> ../data/r3_ckpt/ref_LB.log
