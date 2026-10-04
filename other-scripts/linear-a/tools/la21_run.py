#!/usr/bin/env python3
"""LA-21 job runner: la21_run.py JOB NREST MOVES R K [tag] [wocp wvin]  -> data/la21_ckpt/<tag>_<JOB>.npz
JOB: LA, LAshW<seed>, LAshG<seed>, LAnonHT, LAHT, LBs<seed> (LA-sized LB sample), LB (all), LBshW<seed>, LBshG<seed>,
     JPN<seed>, GRC<seed>, HAW<seed>, ... (la11 language code, thinned to LA bigram count)."""
import sys, os, random, time, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la21_common import *

NSIGN = 65

def build(job):
    m = re.match(r'^([A-Za-z]+?)(\d*)$', job); base, sd = m.group(1), int(m.group(2) or 0)
    la = la_types()
    if base == 'LA': return counts_from_types(la, NSIGN), 'lb'
    if base == 'LAshW': return counts_from_types(shuffle_within(la, sd), NSIGN), 'lb'
    if base == 'LAshG': return counts_from_types(shuffle_global(la, sd), NSIGN), 'lb'
    if base in ('LAnonHT', 'LAHT'):
        ws = set(w for _, site, w in C.words_of(C.la_docs(admin_only=False)) if (site == 'Haghia Triada') == (base == 'LAHT'))
        return counts_from_types(sorted(ws), 45 if base == 'LAnonHT' else NSIGN), 'lb'
    lb = lb_types()
    if base == 'LB': return counts_from_types(lb, NSIGN), 'lb'
    if base == 'LBs': return counts_from_types(random.Random(sd).sample(lb, len(la)), NSIGN), 'lb'
    if base == 'LBshW': return counts_from_types(shuffle_within(random.Random(sd).sample(lb, len(la)), sd), NSIGN), 'lb'
    if base == 'LBshG': return counts_from_types(shuffle_global(random.Random(sd).sample(lb, len(la)), sd), NSIGN), 'lb'
    # la11 language, thinned to the LA in-word bigram + boundary count
    _, B, I, F = counts_from_types(la, NSIGN)
    tgt = int(B.sum() + I.sum() + F.sum())
    return la11_counts(base.lower(), 'DROP', NSIGN, tgt, sd), 'syll'

if __name__ == '__main__':
    job, nrest, moves, R1, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    tag = sys.argv[6] if len(sys.argv) > 6 else 'c1'
    wocp = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    wvin = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    d, tk = build(job)
    t = time.time()
    import zlib; seed = zlib.crc32((job + tag).encode()) % 100000 + 1
    rows, cols, sc = anneal(d, R1, K, nrest, moves, seed, wocp=wocp, wvin=wvin)
    np.savez_compressed(os.path.join(CK, f'{tag}_{job}.npz'), rows=rows, cols=cols, sc=sc, signs=np.array(d[0]),
                        B=d[1], I=d[2], F=d[3], truth=tk, R1=R1, K=K)
    print(job, tag, f'{time.time() - t:.0f}s', f'score {sc[:, 0].mean():.1f}', flush=True)
