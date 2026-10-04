#!/usr/bin/env python3
"""LA-37 cycle 3b: extra controls.
(c) Outside-check significance: AUC of T for same-consonant pairs under LB-derived values of the LA signs, against
    10,000 permutations of the consonant labels among the LA signs (vowel labels kept), and the same for vowels.
(d) Is LA's 0 flagged pairs (vs 2-6 in LB) an artefact of word length? LB draws matched to LA's word-length mix
    (words sampled by length class to LA proportions, ~3,900 tokens), replication flag as in c2b (10 draws).
(e) Planted token doublets at lower rates (0.15, 0.25) in LA: replication recovery (6 hosts each).
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import la37_common as K, la37_c2b as R2
import la32_common as C

LOG = os.path.join(K.CK, 'c3b.log')


def outside_perm(nperm=10000, seed=0):
    c2 = json.load(open(os.path.join(K.CK, 'c2.json')))
    al = c2['alph']; T = np.array(c2['T']); iu = np.triu_indices(len(al), 1)
    cv = [C.lb_cv(s.lower()) for s in al]
    ok = [i for i, x in enumerate(cv) if x is not None]
    cons = np.array([cv[i][0] if cv[i] else None for i in range(len(al))], dtype=object)
    vow = np.array([cv[i][1] if cv[i] else None for i in range(len(al))], dtype=object)
    m = np.array([cv[a] is not None and cv[b] is not None for a, b in zip(*iu)])
    def aucs(cons, vow):
        sc = (cons[iu[0]] == cons[iu[1]]) & m; sv = (vow[iu[0]] == vow[iu[1]]) & m & ~sc
        oth = m & ~sc & ~sv
        return K.auc(T[sc], T[oth]), K.auc(T[sv], T[oth])
    obs = aucs(cons, vow)
    rng = np.random.default_rng(seed)
    nc, nv = [], []
    for _ in range(nperm):
        c = cons.copy(); c[ok] = rng.permutation(cons[ok])
        v = vow.copy(); v[ok] = rng.permutation(vow[ok])
        nc.append(aucs(c, vow)[0]); nv.append(aucs(cons, v)[1])
    nc, nv = np.array(nc), np.array(nv)
    return dict(sameC=(obs[0], float(nc.mean()), float((1 + (nc >= obs[0]).sum()) / (nperm + 1))),
                sameV=(obs[1], float(nv.mean()), float((1 + (nv >= obs[1]).sum()) / (nperm + 1))))


def lb_lenmatched(seed):
    rng = np.random.default_rng(seed)
    LAu = K.la_units()
    lc = collections.Counter(min(len(r['w']), 6) for r in LAu)
    B = K.lb_units()
    byL = collections.defaultdict(list)
    for r in B:
        byL[min(len(r['w']), 6)].append(r)
    # sample whole documents first (keeps document structure), then thin to LA's length mix
    u = K.lb_draw(B, 3918 * 3, rng)
    have = collections.defaultdict(list)
    for r in u:
        have[min(len(r['w']), 6)].append(r)
    tot = sum(lc.values()); out = []
    scale = min(len(have[L]) / (lc[L] / tot) for L in lc if lc[L])
    ntarget = min(scale, len(LAu))
    for L in lc:
        k = int(round(ntarget * lc[L] / tot))
        idx = rng.choice(len(have[L]), min(k, len(have[L])), replace=False)
        out += [have[L][i] for i in idx]
    ntok = sum(len(r['w']) for r in out)
    al, c = K.alphabet(out, 8)
    iu, r, ch = R2.rep(out, al, seed=seed)
    flag = r >= 0.5
    labs = np.array([K.lb_label(al[a], al[b]) for a, b in zip(*iu)], dtype=object)
    rel = np.isin(labs, ['doublet', 'sameC', 'sameV'])
    return dict(kind='LBlen', seed=seed, ntok=int(ntok), nwords=len(out), nflag=int(flag.sum()),
                prec=float(rel[flag].mean()) if flag.sum() else None)


def planted(arg):
    host, rate, seed = arg
    rng = np.random.default_rng(seed)
    u = K.plant(K.la_units(), host, rate, 'token', rng)
    al, c = K.alphabet(u, 8)
    if 'X*' not in al:
        return dict(kind='PL', host=host, rate=rate, rep=None)
    iu, r, ch = R2.rep(u, al, seed=seed)
    j = [k for k, (a, b) in enumerate(zip(*iu)) if {al[a], al[b]} == {host, 'X*'}][0]
    return dict(kind='PL', host=host, rate=rate, rep=float(r[j]), nX=c['X*'])


def job(a):
    if a[0] == 'LBlen':
        return lb_lenmatched(a[1])
    if a[0] == 'PL':
        return planted(a[1:])
    return dict(kind='outside', **outside_perm())


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('outside',)] + [('LBlen', s) for s in range(10)]
    jobs += [('PL', h, rt, 300 + i) for i, h in enumerate(['KA', 'SI', 'TA', 'NA', 'DA', 'RE']) for rt in (0.15, 0.25)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(K.CK, 'c3b.json'), 'w'), default=str)
    for r in res:
        K.log(LOG, json.dumps(r, default=str))
    K.log(LOG, f'done {time.time() - t0:.0f}s')
