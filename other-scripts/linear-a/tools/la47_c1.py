#!/usr/bin/env python3
"""la47 cycle 1: THE LAYOUT IS THE GRAMMAR, coarse layout (physical lines, sides; no SigLA geometry).
Corpora: LA tablets; LB (DAMOS) subsamples matched to LA size; planted spatial rule on LA pages;
nulls: identities permuted within page (P), reading order kept but re-flowed onto a donor page's line
breaks (R = shape fixed, contents swapped).  Targets: word type, la45 class, number magnitude.
Output: data/la47_ckpt/c1.json"""
import json, os, random, sys, collections
from multiprocessing import Pool
import la47_common as C, la47_engine as E

G = int(os.environ.get('G', 3000))


def job(args):
    name, seed = args
    rng = random.Random(seed)
    LA = C.la_pages()
    la = True
    if name == 'LA': P = LA
    elif name.startswith('LB'):
        B = C.lb_pages(sites={'KN', 'PY'}); rng.shuffle(B)
        n = sum(len(p['items']) for p in LA); P = []; c = 0
        for p in B:
            if c >= n: break
            P.append(p); c += len(p['items'])
        la = False
    elif name.startswith('PLANT'): P = C.plant_rule(LA, rng)
    elif name.startswith('NULLP'): P = C.null_permute(LA, rng)
    elif name.startswith('NULLR'): P = C.null_reflow(LA, rng)
    elif name.startswith('LBNULLR'):
        B = C.lb_pages(sites={'KN', 'PY'}); rng.shuffle(B); P = C.null_reflow(B[:300], rng); la = False
    rows = C.table(P, la)
    if name.startswith('PLANT'):
        X = P[0]['planted']
        for r, (pi, it) in zip([r for r in rows], [(pi, i) for pi, p in enumerate(P) for i, s, o in C.featurize(p)]):
            if r['kind'] == 'w': r['planted'] = 'X' if it['id'] in X else 'Y'
    out = {}
    for tgt in ('type', 'cls', 'mag') + (('planted',) if name.startswith('PLANT') else ()):
        w = [r for r in rows if r['kind'] == ('n' if tgt == 'mag' else 'w')]
        out[tgt] = E.run(w, tgt, G=G, seed=seed)
    return name, out


if __name__ == '__main__':
    jobs = [('LA', 1), ('LB1', 11), ('LB2', 12), ('LB3', 13), ('PLANT1', 21), ('PLANT2', 22),
            ('NULLP1', 31), ('NULLP2', 32), ('NULLP3', 33), ('NULLR1', 41), ('NULLR2', 42), ('NULLR3', 43)]
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    res = {}
    with Pool(2) as pool:
        for name, out in pool.imap_unordered(job, jobs):
            res[name] = out; print(name, json.dumps({t: {k: v for k, v in o.items() if not k.startswith('grammar')} for t, o in out.items()}), flush=True)
    path = os.path.join(C.CK, 'c1.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(res); json.dump(old, open(path, 'w'), indent=1)
