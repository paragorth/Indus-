#!/usr/bin/env python3
"""pe38 cycle 3: are forced roles stable on held-out tablets, and beyond a null?
 4 random tablet splits (halves H1/H2) of PE; the cycle-1 system (rules + type-checker, 12 chains x 500k) is
 run on each half separately. Held-out agreement = share of signs forced in one half (>= 3 tokens in the
 other) that are forced to the same role in the other half. Nulls: PE with signs shuffled across entry slots,
 split the same way (x4). Control: proto-cuneiform at PE size split the same way, so stability can be
 compared with correctness (scored on known roles).
Output: data/pe38_ckpt/c3.json
"""
import collections, json, os, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe38_common as C

STEPS = int(os.environ.get('STEPS', 500_000))
NCH = int(os.environ.get('NCH', 12))


def job(args):
    kind, sp, half = args
    fn = os.path.join(C.CK, 'c3_%s_%d_%d.json' % (kind, sp, half))
    if os.path.exists(fn):
        return json.load(open(fn))
    out = _job(kind, sp, half)
    json.dump(out, open(fn, 'w'))
    return out


def _job(kind, sp, half):
    PE = C.pe_docs()
    if kind == 'PE':
        docs = PE
    elif kind == 'NULL':
        docs = C.shuffle_entries(PE, random.Random(C.seed('pe38c3-null-%d' % sp)), 'entry')
    else:
        docs = C.sample_size(C.pc_docs(), 2 * C.ntok(PE), random.Random(C.seed('pe38c3-pc')))
    idx = list(range(len(docs)))
    random.Random(C.seed('pe38c3-split-%s-%d' % ('PC' if kind == 'PC' else 'PE', sp))).shuffle(idx)
    h = idx[:len(idx) // 2] if half == 0 else idx[len(idx) // 2:]
    sub = [docs[i] for i in sorted(h)]
    B = C.build(sub)
    S, viol = C.run_chains(B, C.rule_cost(B), nchains=NCH, steps=STEPS, seed0=C.seed('%s%d%d' % (kind, sp, half)) % 1000)
    frc, mode, freq, agree, P = C.forced(B, S)
    import numpy as np
    cm = [[int(np.bincount(S[c][:, t], minlength=C.R).argmax()) for t in range(len(B['types']))] for c in range(len(S))]
    out = dict(types=B['types'], chain_modes=cm, viol=viol, kind=kind, sp=sp, half=half, forced=frc, mode={w: C.ROLES[mode[t]] for t, w in enumerate(B['types'])},
               nocc={w: int(B['nocc'][t]) for t, w in enumerate(B['types'])}, ids=[docs[i]['id'] for i in sorted(h)])
    if kind == 'PC':
        out['score'] = C.score_labels(B, frc, mode, C.PC_LAB)
    print(kind, sp, half, len(frc), flush=True)
    return out


def main():
    jobs = [(k, sp, h) for k in ('PE', 'NULL') for sp in range(4) for h in (0, 1)] + [('PC', sp, h) for sp in range(2) for h in (0, 1)]
    t = time.time()
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(C.CK, 'c3.json'), 'w'))
    print('done %.0fs' % (time.time() - t))


if __name__ == '__main__':
    main()
