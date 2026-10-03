#!/bin/sh
cd /home/user/Indus-/ai4qi
AI4QI_SWEEP=title_early python3 pipeline.py search --pass nonortho > work/title_early2.log 2>&1
AI4QI_NO_FT=1 python3 pipeline.py fetch --pass nonortho > work/fetch_early2.log 2>&1
python3 prepare_batches.py wave12 --size 100
echo CHAINDONE
