"""LA-78 cycle 2: planted worlds. Can time travel find a known true grammar, and does it
find it better than inside-corpus tests (random-document CV, leave-one-site-out)?

Each world keeps the real tokens, documents, sites and publication years and replaces
'what follows the word' by labels drawn from a true hypothesis h* (random final-sign or
initial-sign partition) plus real per-site label frequencies (the site confound):
   logit P(y | token) = log p_site(y) + beta * E[class_h*(token), y],  E ~ N(0,1).
Variant 'drift': the truth applies only to documents published <= 1960; later documents
follow a different random partition (a grammar that does not carry into the future).
h* is placed in the pool; we record its rank (percentile) under each selector.
"""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la78_engine import *  # noqa

N_PER = int(os.environ.get('N_PER', 300))
NW = int(os.environ.get('NW', 12))
docs = load()
T = tokens(docs)
pool, signs, words, F = make_pool(T, N_PER, seed=7801)
y1r = np.array([t['y1'] for t in T]); y2 = np.array([t['init'] for t in T])
docid = np.array([t['doc'] for t in T])
year = np.array([docs[t['doc']]['year'] for t in T])
sitegrp = np.array([docs[t['doc']]['site'] for t in T])
WIN = [(1950, 1976), (1976, 2100), (1988, 2100)]
sites = sorted(set(sitegrp))
psite = {}
for s in sites:
    c = np.bincount(y1r[sitegrp == s], minlength=4) + 1.0; psite[s] = c / c.sum()
LP = np.log(np.array([psite[s] for s in sitegrp]))


def ari(a, b):
    from collections import Counter
    n = len(a); comb = lambda x: x * (x - 1) / 2
    ct = Counter(zip(a, b)); sa = Counter(a); sb = Counter(b)
    s_ij = sum(comb(v) for v in ct.values()); s_a = sum(comb(v) for v in sa.values()); s_b = sum(comb(v) for v in sb.values())
    e = s_a * s_b / comb(n); m = (s_a + s_b) / 2
    return (s_ij - e) / (m - e) if m != e else 0.0


rng = np.random.RandomState(78)
res = []
for variant in ('stable', 'drift'):
    for beta in (0.7, 1.4):
        for w in range(NW):
            fam = 'FIN' if w % 2 == 0 else 'INI'
            K = 3
            part = rng.randint(0, K, len(signs))
            key = F['last'] if fam == 'FIN' else F['first']
            ftrue = part[key]
            E = rng.normal(0, 1, (K, 4))
            logit = LP + beta * E[ftrue]
            if variant == 'drift':
                part2 = rng.randint(0, K, len(signs)); f2 = part2[key]; E2 = rng.normal(0, 1, (K, 4))
                late = year > 1960
                logit[late] = LP[late] + beta * E2[f2[late]]
            p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            y1 = np.array([rng.choice(4, p=pi) for pi in p])
            Y = [(y1, 4)]
            P = list(pool) + [dict(name='TRUE', fam=fam, K=K, f=ftrue.astype(np.int64), part=part.tolist())]
            for mode in ('raw', 'prior', 'site'):
                MODE['mode'] = mode; MODE['grp'] = sitegrp
                cv, fut, ins, lo = evaluate(P, Y, year, docid, WIN, grp=sitegrp)
                pct = lambda v: float((v[:-1] < v[-1]).mean())
                credit = fut[:, 0]
                r = dict(mode=mode, variant=variant, beta=beta, fam=fam,
                         true_gain_W=[round(float(x), 4) for x in fut[-1]],
                         pct_time=pct(credit), pct_cv=pct(cv[:, 0]), pct_loso=pct(lo[:, 0]), pct_ins=pct(ins[:, 0]),
                         pct_W2=pct(fut[:, 1]), pct_W3=pct(fut[:, 2]))
                # do the selectors' top-20 resemble the truth? (ARI over sign partitions, same family only)
                same = [i for i, h in enumerate(P[:-1]) if h['fam'] == fam]
                def top_ari(score):
                    idx = sorted(same, key=lambda i: -score[i])[:20]
                    return float(np.mean([ari(P[i]['part'], part.tolist()) for i in idx]))
                r['ari_top20_time'] = round(top_ari(credit), 4); r['ari_top20_cv'] = round(top_ari(cv[:, 0]), 4)
                r['ari_top20_loso'] = round(top_ari(lo[:, 0]), 4)
                r['rho_cv_fut'] = [round(spearman(cv[:, k], fut[:, k]), 3) for k in range(3)]
                r['rho_loso_fut'] = [round(spearman(lo[:, k], fut[:, k]), 3) for k in range(3)]
                res.append(r); print(json.dumps(r), flush=True)

summ = {}
for mode, variant, beta in itertools.product(('raw', 'prior', 'site'), ('stable', 'drift'), (0.7, 1.4)):
    if True:
        R = [r for r in res if r['variant'] == variant and r['beta'] == beta and r['mode'] == mode]
        variant_ = variant; variant = f'{mode}_{variant_}'
        summ[f'{variant}_b{beta}'] = {k: round(float(np.mean([r[k] for r in R])), 3) for k in
                                      ('pct_time', 'pct_cv', 'pct_loso', 'pct_ins', 'pct_W2', 'pct_W3', 'ari_top20_time', 'ari_top20_cv', 'ari_top20_loso')}
        summ[f'{variant}_b{beta}']['true_gain_W1W2W3'] = [round(float(np.mean([r['true_gain_W'][k] for r in R])), 3) for k in range(3)]
        summ[f'{variant}_b{beta}']['rho_cv_fut'] = [round(float(np.mean([r['rho_cv_fut'][k] for r in R])), 3) for k in range(3)]
        summ[f'{variant}_b{beta}']['rho_loso_fut'] = [round(float(np.mean([r['rho_loso_fut'][k] for r in R])), 3) for k in range(3)]
        summ[f'{variant}_b{beta}']['top1_time'] = sum(r['pct_time'] == 1.0 for r in R)
        summ[f'{variant}_b{beta}']['top1_cv'] = sum(r['pct_cv'] == 1.0 for r in R)
        summ[f'{variant}_b{beta}']['n'] = len(R)
        variant = variant_
json.dump(dict(summary=summ, worlds=res), open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
print('SUMMARY', json.dumps(summ, indent=1))
