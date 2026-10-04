#!/bin/bash
# waits for cycle 1, then runs cycle 2 and the cycle-3 inspection (2 workers each, sequential)
cd $(dirname $0)
while kill -0 1919 2>/dev/null; do sleep 30; done
python3 la14_run.py c2 c2 > ../data/la14/c2_stdout.txt 2>&1
python3 la14_c3.py > ../data/la14/c3_stdout.txt 2>&1
