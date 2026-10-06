#!/usr/bin/env python3
"""LA-57 cycle 2b: refit the cycle-2 survivor specs (per role, real labels, all known systems) and read them out on
Linear A ADMINISTRATIVE documents only (LA_ADM) and its S1/S2 shuffles; also list anchor-type ranks.
Usage: la57_c2b_read.py [la_key]"""
import os, sys, json, pickle, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C
from la57_read import readout, type_table
LAK = sys.argv[1] if len(sys.argv) > 1 else 'LA_ADM'
sys.argv = sys.argv[:1]
from la57_c2 import fit, scorefn, G
OUT = os.path.join(C.CK, 'c2b')
os.makedirs(OUT, exist_ok=True)
ANCH = ['KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'SA-RA₂', 'A-DU', 'PA-DE', 'TE', 'L:GRA', 'L:VIN', 'L:OLE', 'L:VIR', 'L:CYP', 'L:NI', 'L:OLIV']


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    C.wlog(os.path.join(OUT, 'log_%s.txt' % LAK), s)


def main():
    F, sysd = C.load()
    sysd['KH']['kh'] = True
    rng = np.random.default_rng(1)
    G['X'] = {}
    for k in C.KNOWN:
        rows = np.array([i for i, l in enumerate(sysd[k]['labs']) if l is not None])
        G['X'][k] = sysd[k]['X'][rows]
    models = {}
    for role in C.ROLES:
        p = os.path.join(C.CK, 'c2_' + role, 'summary.json')
        if not os.path.exists(p):
            continue
        S = json.load(open(p))[role]
        ys = [C.label_sets(sysd[k], role, 0, rng)[1][0] for k in S['elig']]
        ms = []
        for f, fam, par in S['surv_specs'][:300]:
            sp = (np.array(f), fam, par)
            m = fit(sp, S['elig'], ys)
            if m is not None:
                ms.append(scorefn(m, sp[0]))
        models[role] = ms
        log(role, 'survivors', len(ms), 'null', S['null_surv'])
    ro = readout(models, F, [r for r in C.ROLES if models.get(r)], log, ntop=10, la_key=LAK)
    la = F[(LAK, 0)]
    for role, ms in models.items():
        if not ms:
            continue
        tt = type_table(la, ms)
        order = list(np.argsort(-tt['tmean']))
        rk = {tt['types'][i]: r + 1 for r, i in enumerate(order)}
        log('%s anchors (rank of %d types): %s' % (role, len(order), ', '.join('%s %d' % (w, rk[w]) for w in ANCH if w in rk)))
    json.dump(ro, open(os.path.join(OUT, 'readout_%s.json' % LAK), 'w'), default=str)


if __name__ == '__main__':
    main()
