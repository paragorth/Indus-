#!/usr/bin/env python3
"""LA-57 cycle 1 readout: universal random linear scorers (AUC >= THR on every known system carrying the role)
run on PLANT, Linear A and shuffled Linear A.  Usage: la57_c1_read.py [tag]"""
import os, sys, json, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C
from la57_read import readout

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c1'
OUT = os.path.join(C.CK, TAG)


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    C.wlog(os.path.join(OUT, 'readout.txt'), s)


def lin(f, w):
    f = np.array(f); w = np.array(w)
    return lambda X: X[:, f] @ w


def main():
    res = json.load(open(os.path.join(OUT, 'res.json')))
    F = pickle.load(open(os.path.join(C.CK, 'feats.pkl'), 'rb'))
    models = {}
    for role, r in res.items():
        u = r['univ'][:500]
        models[role] = [lin(f, w) for f, w in u]
        log(role, 'universal', r['nuniv'], 'null', r['null_univ'][:10], '... used', len(u))
    ro = readout(models, F, [r for r in C.ROLES if r in models], log)
    json.dump(ro, open(os.path.join(OUT, 'readout.json'), 'w'), default=str)


if __name__ == '__main__':
    main()
