#!/bin/bash
# cycle 2 (random pairings) then the cycle-3 inspection, 2 workers each, sequential
cd $(dirname $0)
python3 la14_run.py c2 c2 >> ../data/la14/c2_stdout.txt 2>&1
python3 la14_c3.py > ../data/la14/c3_stdout.txt 2>&1
