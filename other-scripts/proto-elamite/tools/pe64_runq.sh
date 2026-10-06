#!/bin/sh
# pe64 job queue: at most 2 workers. usage: pe64_runq.sh QUEUEFILE (one "SCRIPT ARGS..." per line)
cd "$(dirname "$0")"
grep -v '^#' "$1" | xargs -P 2 -I{} sh -c 'python3 {} > ../data/pe64_ckpt/log_$(echo "{}" | tr " /" "__").txt 2>&1 || echo "FAIL {}" >> ../data/pe64_ckpt/fail.txt'
