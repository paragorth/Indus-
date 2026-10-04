#!/usr/bin/env python3
"""LA-44 simulation bank: python3 la44_bank.py N_TARGET SEED0 COUNT OUT.jsonl [fixed-json]
Each row: params of the simulated history + the summary panel of its written type list (resumable)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la44_common as A
n, s0, cnt, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fixed = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)['P']['seed'])
        except Exception: pass
fo = open(out, 'a'); t0 = time.time(); k = 0
for seed in range(s0, s0 + cnt):
    if seed in done: continue
    P, ty = A.simulate(seed, n, fixed)
    if len(ty) < 0.8 * n: f = None
    else: f = A.panel(ty)
    fo.write(json.dumps({'P': P, 'f': f}) + '\n'); k += 1
    if k % 1000 == 0: fo.flush(); print(seed, k, round(time.time() - t0), flush=True)
fo.close()
