#!/usr/bin/env python3
"""LA-67 kill sweep, test K-318: the la61 grade-C guess '*318 is a commodity marker (+15.6 bits)'.
Stated test (la61): give *318 the COM role alone in the la60 grammar and measure the held-out gain;
support needs >= 10 bits against 20 single decoy signs given the COM role one at a time; kill = a gain
inside the decoy range.  Fresh splits (seed names la67-*), 4 evaluation splits.
usage: la67_s318.py real | dec K0 K1 | report"""
import sys, os, json, random, glob, time
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from la60_common import load_la, admin_docs, prior_reading, split, seed
from la60_model import Grammar, doc_bits
CK = os.path.join(HERE, '..', 'data', 'la67_ckpt'); os.makedirs(CK, exist_ok=True)
NSPL = int(os.environ.get('NSPL', 4)); TARGET = '*318'


def setup():
    A = admin_docs(load_la()); R0 = prior_reading()
    for w in ['RA', 'PA', 'PA₃', 'TU', 'ME', TARGET]:
        R0['roles'].pop(w, None)
    ev = [split(A, 'la67-ev-%d' % k) for k in range(NSPL)]
    return A, R0, ev


def bits(R, ev):
    return sum(sum(doc_bits(Grammar(tr, R, order=False), d) for d in te) for tr, te in ev)


def gain(R0, w, ev, base):
    import copy
    R = copy.deepcopy(R0); R['roles'][w] = 'COM'
    return base - bits(R, ev)


def decoys(A, R0, n=20):
    cnt = Counter(t[1] for d in A for t in d['toks'] if t[0] == 'W' and '-' not in t[1])
    f = cnt[TARGET]
    pool = sorted(w for w, c in cnt.items() if w not in R0['roles'] and w != TARGET and c >= 2)
    pool.sort(key=lambda w: (abs(np.log(cnt[w] / f)), w))
    return f, [(w, cnt[w]) for w in pool[:n]]


if __name__ == '__main__':
    m = sys.argv[1]
    A, R0, ev = setup()
    if m == 'real' or m == 'dec':
        t0 = time.time()
        base = bits(R0, ev)
        f, dec = decoys(A, R0)
        ws = [TARGET] if m == 'real' else [w for w, _ in dec[int(sys.argv[2]):int(sys.argv[3])]]
        for w in ws:
            g = gain(R0, w, ev, base)
            json.dump(dict(w=w, gain=g, base=base, sec=time.time() - t0), open(os.path.join(CK, 's318_%s.json' % w.replace('*', 'x')), 'w'), ensure_ascii=False)
            print(w, round(g, 1), round(time.time() - t0), flush=True)
    else:
        f, dec = decoys(A, R0)
        r = json.load(open(os.path.join(CK, 's318_x318.json')))
        dg = []
        for w, c in dec:
            fn = os.path.join(CK, 's318_%s.json' % w.replace('*', 'x'))
            if os.path.exists(fn):
                dg.append((w, c, json.load(open(fn))['gain']))
        print(json.dumps(dict(target=TARGET, freq=f, gain=r['gain'], decoys=dg), ensure_ascii=False))
