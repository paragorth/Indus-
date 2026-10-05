"""pe46 cycle 2: deep schema search on PE with every external variable (tablet id, P-number batch,
header sign, publication volume; class sign, number system, quantity; line order; previous entry),
base signs and ~variant signs, against three nulls with the same budget (signs shuffled within
strings, strings shuffled across tablets, Markov resynthesis). Survivors: PE schemas whose held-out
CODE beats the 95th percentile of the pooled null top-20 held-out scores, on both tablet splits.
Positional test: for each variable, the best slot's held-out E in PE vs in the sign-shuffled null.
usage: python3 pe46_cycle2.py [N]
"""
import sys, os, json, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe46_lib as L

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
EXT = ['TABID', 'BATCH', 'HDR', 'PUB', 'CLS', 'SYS', 'QTY', 'ORD', 'PREV']


def build():
    out = {}
    for tag, bs in (('PE', True), ('PEv', False)):
        t = L.pe_tokens(base_signs=bs)
        out[tag] = t
        out[tag + '~sig'] = L.null_signshuf(t, 21)
        out[tag + '~tab'] = L.null_tabshuf(t, 22)
        out[tag + '~mkv'] = L.null_markov(t, 23)
    return out


def run(arg):
    k, toks = arg
    f = os.path.join(L.CK, f'c2_{k}.json')
    if os.path.exists(f):
        return k, json.load(open(f))
    C = L.make(toks, k); t0 = time.time(); res = []
    for split in (0, 1):
        out = L.search(C, N // 2, seed=200 + split, split_seed=split, top=20, varlist=EXT)
        res.append([dict(sch=o['sch'], s=L.sch_str(o['sch']), fit=o['fit'], held=o['held']) for o in out])
    d = dict(corpus=k, n=C['n'], res=res, secs=time.time() - t0)
    json.dump(d, open(f, 'w'), default=str)
    return k, d


if __name__ == '__main__':
    B = build()
    with Pool(2) as p:
        for k, d in p.imap_unordered(run, list(B.items())):
            hb = [[o['held']['CODE'] for o in r] for r in d['res']]
            print(k, d['n'], 'held_med', [round(float(np.median(h)), 3) for h in hb],
                  'held_max', [round(max(h), 3) for h in hb], round(d['secs']), flush=True)
