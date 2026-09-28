#!/bin/sh
cd /home/user/Indus-/ai4qi
AI4QI_SWEEP=adherence python3 pipeline.py search --pass nonortho > work/adherence_search.log 2>&1
AI4QI_NO_FT=1 python3 pipeline.py fetch --pass nonortho > work/fetch_w13.log 2>&1
python3 nice_ingest.py /tmp/claude-0/-home-user-Indus-/22814f44-e7fb-51af-9fab-91adf38b3b5c/scratchpad/nice_sl/nice_shared_learning.json
python3 pipeline.py screen-crossref
python3 prepare_batches.py wave13 --size 100
python3 prepare_batches.py wave13o --pass ortho --size 100
echo CHAINDONE
