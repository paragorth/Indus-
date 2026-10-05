#!/usr/bin/env python3
"""la47 cycle 3b: (c') totals with a fixed definition: a number is a total if the nearest preceding
syllabic (non-logogram) word is KU-RO / PO-TO-KU-RO (LA) or to-so / to-sa (LB).  S vs O grammars,
held-out bits, also reported as a share of the target entropy; nulls: identities permuted within page.
(a') which layout cells carry la45 classes: for each spatial feature value, share of commodity / header
words vs the within-page identity permutation (2,000 reps), LA and LB (LB: logogram vs word).
Output data/la47_ckpt/c3b.json"""
import json, os, random, collections, math
import numpy as np
from multiprocessing import Pool
import la47_common as C, la47_engine as E


def tot_rows(P, la):
    tw = C.TOTAL if la else {'to-so', 'to-sa'}
    rows = C.table(P, la); allit = [i for p in P for i, s, o in C.featurize(p)]
    prev = None; pp = None
    for r, i in zip(rows, allit):
        if r['p'] != pp: prev = None; pp = r['p']
        if r['kind'] == 'n': r['tot'] = 'T' if prev in tw else 'N'
        elif not i['logo']: prev = i['id']
    return [r for r in rows if r['kind'] == 'n']


def job(a):
    name, seed = a; rng = random.Random(seed)
    if 'LB' in name:
        B = C.lb_pages(sites={'KN', 'PY'}); rng.shuffle(B); P = B[:400]; la = False
    else:
        P = C.la_pages(); la = True
    if 'NULL' in name: P = C.null_permute(P, rng)
    nr = tot_rows(P, la); p = np.mean([r['tot'] == 'T' for r in nr])
    H = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    o = E.run(nr, 'tot', G=G, seed=seed)
    return name, {'n_num': len(nr), 'n_tot': int(sum(r['tot'] == 'T' for r in nr)), 'H_bits': round(H, 4), 'res': o}


def cells_test(rows, label_fn, feats, rng, reps=2000):
    w = [r for r in rows if r['kind'] == 'w']
    lab = np.array([label_fn(r) for r in w]); pid = np.array([r['p'] for r in w])
    groups = [np.where(pid == p)[0] for p in np.unique(pid)]
    perms = []
    for _ in range(reps):
        idx = np.arange(len(w))
        for k in groups: idx[k] = k[rng.permutation(len(k))]
        perms.append(idx)
    out = {}
    for f in feats:
        vals = np.array([str(r['s'][f]) for r in w])
        for v in sorted(set(vals)):
            m = vals == v
            if m.sum() < 15: continue
            for c in sorted(set(lab)):
                if c == 'other': continue
                obs = float((lab[m] == c).mean()); base = float((lab == c).mean())
                null = np.array([(lab[ix][m] == c).mean() for ix in perms])
                z = (obs - null.mean()) / (null.std() + 1e-9)
                if abs(z) >= 2.5:
                    out['%s=%s|%s' % (f, v, c)] = {'n': int(m.sum()), 'share': round(obs, 3), 'base': round(base, 3), 'z': round(float(z), 2),
                                                    'P': round(float(min(1, 2 * min((null >= obs).mean(), (null <= obs).mean()))), 4)}
    return out


G = int(os.environ.get('G', 1500))
if __name__ == '__main__':
    out = {}
    rng = np.random.default_rng(9)
    LA = C.la_pages(); rows = C.table(LA)
    out['cells_LA'] = cells_test(rows, lambda r: r['cls'], C.SFEAT, rng)
    out['cells_LA_logo'] = cells_test(rows, lambda r: 'logo' if r['type'] in C.COMMOD or r['type'].startswith('VIR') or not r['type'].replace('-', '').isupper() or len(r['type'].split('-')) == 1 and len(r['type']) > 2 else 'other', C.SFEAT, rng)
    B = C.lb_pages(sites={'KN', 'PY'}); random.Random(4).shuffle(B); B = B[:600]
    out['cells_LB'] = cells_test(C.table(B, la=False), lambda r: r['cls'], C.SFEAT, rng)
    for k in ('cells_LA', 'cells_LB'): print(k, json.dumps(out[k]))
    jobs = [('C_LA', 6), ('C_LB', 7), ('C_NULL1', 8), ('C_NULL2', 9), ('C_LBNULL1', 10)]
    with Pool(2) as pool:
        for name, o in pool.imap_unordered(job, jobs):
            out[name] = o; r = o['res']
            print(name, o['n_num'], o['n_tot'], o['H_bits'], {k: r[k] for k in ('sat_S', 'sat_O', 'top10_S', 'top10_O', 'best_S', 'best_O', 'incr_S_over_O')}, r['grammar_S'][:160], flush=True)
    json.dump(out, open(os.path.join(C.CK, 'c3b.json'), 'w'), indent=1)
