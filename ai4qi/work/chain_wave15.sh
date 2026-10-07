#!/bin/sh
cd /home/user/Indus-/ai4qi
AI4QI_SWEEP=fulltext python3 pipeline.py search --pass nonortho > work/fulltext_search.log 2>&1
AI4QI_NO_FT=1 python3 pipeline.py fetch --pass nonortho > work/fetch_w15.log 2>&1
python3 prepare_batches.py wave15 --size 100
python3 prepare_batches.py wave15o --pass ortho --size 100
echo CHAINDONE
