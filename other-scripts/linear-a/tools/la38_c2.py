#!/usr/bin/env python3
"""LA-38 cycle 2: Linear A with its LB-derived values against 10^5 relabelings per tier (R1, R2a, R2b, R3);
negative controls: LA with signs shuffled inside words (frequencies kept, order destroyed), and 100 wrong-value
maps (random relabelings of LA treated as the truth). Composites ALL7 and SEL are fixed from cycle 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la38_c1 import run, TIERS

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'la':
        run([('LA', t, 100_000, 7, None) for t in TIERS], 'c2_la.json')
    elif stage == 'neg':
        jobs = [(f'LAsh{k}', t, 10_000, 70 + k, None) for k in range(5) for t in TIERS]
        jobs += [('LA', t, 1000, 500 + f, f) for f in range(100) for t in TIERS]
        run(jobs, 'c2_neg.json')
