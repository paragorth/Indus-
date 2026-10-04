#!/bin/sh
# cycle 1 full runs (2 workers each, sequential)
cd "$(dirname "$0")"
python3 la12_c1.py V 100 > ../data/la12/c1_V.log 2>&1
python3 la12_c1.py C 100 > ../data/la12/c1_C.log 2>&1
echo done > ../data/la12/c1.done
