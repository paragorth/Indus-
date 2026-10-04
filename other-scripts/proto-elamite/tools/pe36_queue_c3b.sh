#!/bin/bash
# pe36 cycle 3b: OB names restricted to syllabic spellings (is the OB failure due to the logograms?)
cd "$(dirname "$0")"
while pgrep -x -f "/bin/bash ./pe36_queue_c3.sh" >/dev/null; do sleep 30; done
{ for c in 0 1 2 3; do echo "OBSYL real $c 100 100000"; done; for s in 1 2 3; do echo "OBSYL realshW$s 0 100 100000"; done; } | xargs -P2 -L1 python3 pe36_run.py >> ../data/pe36_ckpt/c3b.log 2>&1
