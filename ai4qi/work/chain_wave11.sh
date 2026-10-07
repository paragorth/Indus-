#!/bin/sh
cd /home/user/Indus-/ai4qi
until grep -q "crossref_all: .* new records in total" work/extra_harvest.log; do sleep 60; done
until grep -q "EARLYDONE" work/title_early.log; do sleep 60; done
python3 crossref_harvest.py --wave5 > work/crossref5.log 2>&1
AI4QI_NO_FT=1 python3 pipeline.py fetch --pass nonortho > work/fetch_early.log 2>&1
python3 pipeline.py screen-crossref
python3 prepare_batches.py wave11 --size 100
python3 prepare_batches.py wave11o --pass ortho --size 100
echo CHAINDONE
