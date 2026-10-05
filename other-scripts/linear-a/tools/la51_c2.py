#!/usr/bin/env python3
"""la51 cycle 2: Linear B positive control (DAMOS find areas). Pre-registered commodity-context probes
(fixed before running): OLE~STOR, VIN~STOR, ROTA~METAL, OVIS~ANIM, CAP~ANIM, SUS~ANIM, BOS~ANIM,
and for Linear A the same logograms where present (OLE, VIN, GRA, OLIV, CYP ~ STOR; *CAP/HIDE ~ ANIM).
usage: la51_c2.py lb|lbsmall|la PART reps   PART: real | n1 | n2 | plant
"""
import sys, json, os, numpy as np
import la51_common as L, la51_engine as E
corpus, part = sys.argv[1], sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
PROBES = [('L:OLE', 'STOR'), ('L:VIN', 'STOR'), ('L:ROTA', 'METAL'), ('L:OVIS', 'ANIM'), ('L:CAP', 'ANIM'),
          ('L:SUS', 'ANIM'), ('L:BOS', 'ANIM'), ('L:GRA', 'STOR'), ('L:OLIV', 'STOR'), ('L:CYP', 'STOR'),
          ('L:HIDE', 'ANIM'), ('L:AES', 'METAL'), ('L:TELA', 'TEXT'), ('L:LANA', 'TEXT')]
OUT = os.path.join(L.CK, f'c2_{corpus}_{part}.json')

def get_M(seed):
    if corpus == 'la': return L.build_matrices(L.load_la())
    docs = L.load_lb()
    if corpus == 'lbsmall':
        rng = np.random.default_rng(1000 + seed)
        docs = [docs[i] for i in rng.choice(len(docs), 1541, replace=False)]
    return L.build_matrices(docs)

res = dict(corpus=corpus, part=part, runs=[])
for s in range(reps):
    M = get_M(s if corpus == 'lbsmall' else 0)
    rng = np.random.default_rng(500 + s)
    kw = {}
    if part == 'n1': kw['dep_of'] = E.null_within_site(M, rng)
    if part == 'n2': kw['C'] = E.null_cross_site(M, rng)
    if part == 'plant':
        cls = ['METAL', 'STOR', 'ANIM'][s % 3]
        M, n, nin = E.plant(M, rng, cls=cls, p_in=0.2, p_out=0.01)
        r = E.run(M, seed=300 + s, probes=[('w:PLANTED', cls)])
        res['runs'].append(dict(cls=cls, n=n, n_in=nin, probe=r['probes'], survivors=r['survivors']))
        print(s, cls, n, nin, r['probes'], flush=True); continue
    r = E.run(M, seed=100 + s, probes=PROBES, **kw)
    r.pop('best', None)
    r['n_docs'] = len(M['docs']); r['n_deps'] = len(M['dep_ids']); r['n_terms'] = len(M['terms'])
    res['runs'].append(r)
    print(s, r['survivors'], len(r['surv_terms']), {k: (round(v['z'], 1), v['rate'], v['surv']) for k, v in r['probes'].items()}, flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
