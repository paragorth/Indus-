#!/usr/bin/env python3
"""la51 cycle runner. usage: la51_c1.py CORPUS PART [reps]
CORPUS: la | lb | lbsmall (LB subsampled to LA size: same number of deposits-with-terms docs)
PART: real | n1 | n2 | plant
Writes data/la51_ckpt/c1_<corpus>_<part>.json
"""
import sys, json, os, numpy as np
import la51_common as L, la51_engine as E

corpus, part = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
OUT = os.path.join(L.CK, f'c1_{corpus}_{part}.json')


def get_M(seed=0):
    if corpus == 'la':
        return L.build_matrices(L.load_la())
    docs = L.load_lb()
    if corpus == 'lbsmall':
        rng = np.random.default_rng(1000 + seed)
        docs = [docs[i] for i in rng.choice(len(docs), 1541, replace=False)]   # LA pool size
    return L.build_matrices(docs)


res = {'corpus': corpus, 'part': part, 'runs': []}
if part == 'real':
    for s in range(3 if corpus != 'lbsmall' else reps):
        M = get_M(s)
        r = E.run(M, seed=100 + s)
        r['n_docs'] = len(M['docs']); r['n_deps'] = len(M['dep_ids']); r['n_terms'] = len(M['terms'])
        res['runs'].append(r)
        print(s, r['survivors'], r['surv_terms'][:20], flush=True)
elif part in ('n1', 'n2'):
    M = get_M(0)
    for s in range(reps):
        rng = np.random.default_rng(500 + s)
        if part == 'n1': r = E.run(M, seed=100 + s, dep_of=E.null_within_site(M, rng))
        else: r = E.run(M, seed=100 + s, C=E.null_cross_site(M, rng))
        r.pop('best')
        res['runs'].append(r)
        print(s, r['survivors'], len(r['surv_terms']), flush=True)
elif part == 'plant':
    M = get_M(0)
    for cls in (['METAL', 'RIT', 'STOR'] if corpus == 'la' else ['METAL', 'STOR', 'RIT']):
        for p_in in (0.2, 0.4, 0.6):
            for s in range(reps):
                rng = np.random.default_rng(900 + s)
                M2, n, nin = E.plant(M, rng, cls=cls, p_in=p_in, p_out=0.01)
                r = E.run(M2, seed=300 + s, track=('w:PLANTED', cls))
                pb = [b for b in r['best'] if b[0] == 'w:PLANTED']
                hit_exact = any(t[2] >= 0.5 and t[1] >= E.MINTEST for t in r.get('track', []))
                # 'consistent' = surviving predicate is true for the planted class deposits
                res['runs'].append(dict(cls=cls, p_in=p_in, seed=s, n=n, n_in=nin, any=bool(pb),
                                        exact=bool(hit_exact), best=pb, survivors=r['survivors']))
                print(cls, p_in, s, n, nin, bool(pb), hit_exact, pb, flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
