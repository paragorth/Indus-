#!/usr/bin/env python3
"""LA-55 cycle 1: scaling classes; controls (LB, Ur III) must sort known classes; planted classes recovered.
usage: la55_c1.py CORPUS NREP   (CORPUS in LA, LB, LBS, UR, PLANT)"""
import sys, json, collections
import numpy as np
from la55_common import *

corpus = sys.argv[1]; NREP = int(sys.argv[2])
rng = np.random.default_rng(seed('la55-c1-' + corpus))
truth = {}
if corpus == 'LA' or corpus == 'PLANT':
    docs = la_units(); min_k = 3
elif corpus == 'LB':
    docs = lb_units(); min_k = 3; truth = lb_truth_classes()
elif corpus == 'LBS':      # LB cut to Linear A size: whole units drawn until ~1,574 documents
    docs = lb_units(); truth = lb_truth_classes(); min_k = 3
    cnt = collections.Counter(d['unit'] for d in docs)
    us = sorted(u for u in cnt if cnt[u] >= 3); r = random.Random(seed('la55-lbs')); r.shuffle(us)
    keep, n = set(), 0
    for u in us:
        if n >= 1574: break
        keep.add(u); n += cnt[u]
    docs = [d for d in docs if d['unit'] in keep]
elif corpus == 'LAG':      # signs (syllabograms inside words) instead of words
    docs = la_units(signs=True); min_k = 3
elif corpus == 'LBG':
    docs = lb_units(signs=True); min_k = 3
elif corpus == 'LASITE':   # whole sites as archives
    docs = la_units(level='site'); min_k = 3
elif corpus == 'UR':
    docs = ur_units(max_docs=12000); min_k = 5; truth = ur_truth_classes(docs)

planted = {}
if corpus == 'PLANT':
    r = random.Random(seed('la55-plant'))
    cnt = collections.Counter(d['unit'] for d in docs)
    units = [u for u in cnt if cnt[u] >= 3]
    byu = collections.defaultdict(list)
    for i, d in enumerate(docs):
        if d['unit'] in units: byu[d['unit']].append(i)
    alli = [i for u in units for i in byu[u]]
    for cls in ('SAT', 'LIN', 'SUP', 'THR'):
        for j in range(15):
            m = r.randint(6, 25); w = 'P:%s%d' % (cls, j)
            chosen = set()
            while len(chosen) < m:
                if cls == 'SAT':
                    u = r.choice(units); i = r.choice(byu[u])
                elif cls == 'LIN':
                    i = r.choice(alli)
                elif cls == 'SUP':
                    u = r.choices(units, weights=[cnt[x] ** 1.6 for x in units])[0]; i = r.choice(byu[u])
                else:
                    u = r.choices([x for x in units if cnt[x] >= 30], weights=[cnt[x] for x in units if cnt[x] >= 30])[0]
                    i = r.choice(byu[u])
                chosen.add(i)
            for i in chosen:
                docs[i]['terms'].add(w)
            planted[w] = cls

data = Data(docs, min_unit=3, min_k=min_k)
K, N = data.counts()
real = features(K, N)
res = {'corpus': corpus, 'ndocs': len(data.docs), 'units': data.A, 'types': data.types, 'N': N,
       'Ktot': data.Ktot, 'real': real}
for strat in (False, True):
    nul = null_features(data, NREP, rng, strat)
    z = zscores(real, nul)
    cls = classify(z)
    key = 'N3' if strat else 'N1'
    res[key] = {'z': z, 'cls': cls.tolist(), 'null_mean_beta': nul['beta'].mean(0)}
    # bootstrap stability (N2) of beta
# N2: bootstrap of documents within units -> beta CI
bs = []
for _ in range(min(NREP, 300)):
    idx = data.boot(rng)
    X = data.X[idx]; u = data.u[idx]
    U = sp.csr_matrix((np.ones(len(u)), (u, np.arange(len(u)))), shape=(data.A, len(u)))
    Kb = np.asarray((U @ X).todense()).T
    bs.append(fit_beta(Kb, N))
bs = np.array(bs)
res['N2_beta_lo'] = np.percentile(bs, 2.5, 0); res['N2_beta_hi'] = np.percentile(bs, 97.5, 0)

# held-out leave-one-unit-out: scaling vs proportional, real vs N1 shuffles
g_real, nwu = louo_gain(K, N)
g_null = []
for _ in range(min(NREP, 100)):
    Kn, _ = data.counts(data.shuffle(rng))
    g_null.append(louo_gain(Kn, N)[0])
g_null3 = []
for _ in range(min(NREP, 100)):
    Kn, _ = data.counts(data.shuffle(rng, True))
    g_null3.append(louo_gain(Kn, N)[0])
res['louo'] = {'real': g_real, 'n': nwu, 'N1': g_null, 'N3': g_null3}

# truth-class tests (controls)
def truth_tests(tr, cls, z):
    out = {}
    lab = [tr.get(t) for t in data.types]
    tab = collections.defaultdict(collections.Counter)
    for l, c in zip(lab, cls):
        if l: tab[l][c] += 1
    out['table'] = {k: dict(v) for k, v in tab.items()}
    def pick(c, key):
        return [z[key][i] for i, l in enumerate(lab) if l == c]
    for a, b in (('FUNC', 'PERSON'), ('COMM', 'PERSON'), ('FUNC', 'COMM'), ('PLACE', 'PERSON')):
        out['%s>%s occ' % (a, b)] = auc(pick(a, 'occ'), pick(b, 'occ'))
        out['%s<%s beta' % (a, b)] = auc([-v for v in pick(a, 'beta')], [-v for v in pick(b, 'beta')])
        out['%s>%s thr' % (a, b)] = auc(pick(a, 'thr'), pick(b, 'thr'))
    # label-permutation null for the FUNC<PERSON beta AUC and FUNC>PERSON occ AUC and COMM>PERSON thr
    labs = np.array([l if l else '' for l in lab], dtype=object)
    m = labs != ''
    idx = np.where(m)[0]
    perm = {'FUNC<PERSON beta': [], 'FUNC>PERSON occ': [], 'COMM>PERSON thr': []}
    rr = np.random.default_rng(seed('la55-perm'))
    for _ in range(2000):
        L2 = labs.copy(); L2[idx] = rr.permutation(labs[idx])
        f = [i for i in idx if L2[i] == 'FUNC']; p = [i for i in idx if L2[i] == 'PERSON']
        c_ = [i for i in idx if L2[i] == 'COMM']
        perm['FUNC<PERSON beta'].append(auc(-z['beta'][f], -z['beta'][p]))
        perm['FUNC>PERSON occ'].append(auc(z['occ'][f], z['occ'][p]))
        perm['COMM>PERSON thr'].append(auc(z['thr'][c_], z['thr'][p]))
    for k, v in perm.items():
        obs = out[k.replace('<', '<').replace('>', '>')] if k in out else None
        a_, b_ = k.split(' ')[0], k.split(' ')[1]
        key = k
        o = out.get(key)
        v = np.array(v)
        out['perm_p ' + key] = float(((v >= o).sum() + 1) / (len(v) + 1)) if o == o else None
    out['n'] = dict(collections.Counter(l for l in lab if l))
    return out

if truth:
    for key in ('N1', 'N3'):
        z = res[key]['z']; cls = np.array(res[key]['cls'])
        res[key]['truth'] = truth_tests(truth, cls, z)
if planted:
    for key in ('N1', 'N3'):
        cls = res[key]['cls']
        tab = collections.defaultdict(collections.Counter)
        for t, c in zip(data.types, cls):
            if t in planted: tab[planted[t]][c] += 1
        res[key]['planted'] = {k: dict(v) for k, v in tab.items()}
        exp = {'SAT': 'SATURATING', 'LIN': 'LINEAR', 'SUP': 'SUPERLINEAR', 'THR': 'THRESHOLD'}
        res[key]['planted_recall'] = {k: tab[k][exp[k]] / max(1, sum(tab[k].values())) for k in exp}
        # false classes on real LA words
        res[key]['real_classes'] = dict(collections.Counter(c for t, c in zip(data.types, cls) if not t.startswith('P:')))
jdump(res, 'c1_%s.json' % corpus)

# short printout
print('==', corpus, 'docs', len(data.docs), 'units', data.A, 'types', len(data.types))
print('   louo bits/word-unit real %.4f  N1 %.4f+-%.4f  N3 %.4f+-%.4f' % (g_real, np.mean(g_null), np.std(g_null),
                                                                   np.mean(g_null3), np.std(g_null3)))
for key in ('N1', 'N3'):
    print('  ', key, dict(collections.Counter(res[key]['cls'])))
    if 'truth' in res[key]:
        t = res[key]['truth']
        print('    table', t['table'])
        print('    ', {k: round(v, 3) for k, v in t.items() if isinstance(v, float)})
    if 'planted' in res[key]:
        print('    planted', res[key]['planted'], res[key]['planted_recall'], 'real', res[key]['real_classes'])
