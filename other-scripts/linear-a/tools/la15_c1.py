"""LA-15 cycle 1: how many words and signs did the Minoan administration use?
Richness estimators (Chao1, ACE, Chao2, jackknife-2), a bootstrap community ensemble,
and a massive random-guessing ABC over Zipf-Mandelbrot communities.
Calibration: (a) planted communities of known size sampled at LA's size;
(b) Linear B syllabary (known: 89 signs read in DAMOS) estimated from LA-sized subsamples;
(c) Linear B Knossos: from an LA-sized word sample, predict the number of word types
observed in the full Knossos corpus (an out-of-sample count).
"""
import sys, json, random, collections, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la15_common import *

rng = np.random.default_rng(15)
prng = random.Random(15)
res = {}


def community(A):
    """Chao-style estimated community: detected species scaled by coverage + f0 unseen."""
    A = np.array([x for x in A if x > 0], float)
    n = A.sum()
    C = coverage(A)
    f0 = max(int(round(chao1(A) - len(A))), 0)
    p = A / n * C
    if f0 > 0:
        p = np.concatenate([p, np.full(f0, (1 - C) / f0)])
    return p / p.sum(), len(A)


def boot_predict(A, m, B=2000, stat=None):
    """Parametric bootstrap ensemble: resample n from the community, refit, then draw m more
    tokens; return arrays of (refit chao1, new types in m tokens)."""
    p, S = community(A)
    n = int(sum(A))
    c1s, news = [], []
    for _ in range(B):
        x = rng.multinomial(n, p)
        Ab = x[x > 0]
        c1s.append(chao1(Ab))
        pb, Sb = community(Ab)
        seen = np.zeros(len(pb), bool)
        seen[:Sb] = True
        y = rng.multinomial(m, pb)
        news.append(int(((y > 0) & ~seen).sum()))
    return np.array(c1s), np.array(news)


def zm_probs(S, a, q):
    i = np.arange(1, S + 1, dtype=float)
    w = (i + q) ** (-a)
    return w / w.sum()


def summ(A):
    A = np.asarray(A)
    A = A[A > 0]
    return np.array([len(A), (A == 1).sum(), (A == 2).sum(), (A == 3).sum(), A.max()], float)


def abc(A_obs, n_sims=20000, keep=200, m_future=None):
    """Massive random guessing: n_sims random communities, keep the closest `keep`."""
    n = int(sum(A_obs))
    so = summ(A_obs)
    sc = np.array([so[0], so[1], max(so[2], 5), max(so[3], 5), max(so[4], 5)])
    sims = []
    for _ in range(n_sims):
        S = int(np.exp(rng.uniform(np.log(max(so[0], 50) * 1.0), np.log(so[0] * 100))))
        a = rng.uniform(0.3, 1.6)
        q = rng.uniform(0, 30)
        p = zm_probs(S, a, q)
        x = rng.multinomial(n, p)
        d = np.sqrt((((summ(x) - so) / sc) ** 2).sum())
        sims.append((d, S, a, q))
    sims.sort()
    acc = sims[:keep]
    out = dict(S=[s[1] for s in acc], a=[s[2] for s in acc], q=[s[3] for s in acc], dmax=acc[-1][0])
    if m_future:
        fut = []
        for d, S, a, q in acc:
            p = zm_probs(S, a, q)
            x = rng.multinomial(n, p)
            y = rng.multinomial(m_future, p)
            fut.append(int(((x + y) > 0).sum()))  # types at n + m_future
        out['types_future'] = fut
    return out


def q(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    return f"{np.median(v):.0f} [{np.percentile(v, lo):.0f}-{np.percentile(v, hi):.0f}]"


t0 = time.time()
LA = load_la()
LB = load_lb()
laW = list(collections.Counter(w for d in LA for w in d['words']).values())
laS = list(collections.Counter(s for d in LA for s in d['signs']).values())
lbW = list(collections.Counter(w for d in LB for w in d['words']).values())
lbS = list(collections.Counter(s for d in LB for s in d['signs']).values())
rows = []

# ---- 1a. point estimates and bootstrap intervals
E = {}
for name, docs, key in [('LA words', LA, 'words'), ('LA signs', LA, 'signs'),
                        ('LB words', LB, 'words'), ('LB signs', LB, 'signs')]:
    e = all_estimates(docs, key)
    A = list(collections.Counter(w for d in docs for w in d[key]).values())
    c1b, _ = boot_predict(A, 1, B=1000)
    e['chao1_boot'] = q(c1b)
    E[name] = e
res['estimates'] = E
print(json.dumps(E, indent=1, default=str), time.time() - t0, flush=True)

# ---- 1b. planted communities at LA's size: does each estimator cover the known truth?
planted = []
for S_true, a, qq in [(1500, 0.6, 2), (4000, 0.7, 5), (8000, 0.9, 10), (20000, 1.0, 10), (5000, 0.4, 0)]:
    p = zm_probs(S_true, a, qq)
    hits = collections.Counter()
    vals = collections.defaultdict(list)
    for r in range(200):
        x = rng.multinomial(1332, p)
        A = x[x > 0]
        vals['Sobs'].append(len(A))
        vals['f1share'].append((A == 1).sum() / len(A))
        vals['chao1'].append(chao1(A))
        vals['ace'].append(ace(A))
    # ABC on one planted sample
    x = rng.multinomial(1332, p)
    ab = abc(x[x > 0], n_sims=4000, keep=80)
    lo, hi = np.percentile(ab['S'], [2.5, 97.5])
    planted.append(dict(S_true=S_true, a=a, q=qq, Sobs=np.mean(vals['Sobs']), f1share=np.mean(vals['f1share']),
                        chao1=np.mean(vals['chao1']), chao1_ratio=np.mean(vals['chao1']) / S_true,
                        ace_ratio=np.mean(vals['ace']) / S_true, abc_med=float(np.median(ab['S'])),
                        abc_lo=float(lo), abc_hi=float(hi), abc_cover=bool(lo <= S_true <= hi)))
    print(planted[-1], flush=True)
res['planted'] = planted

# ---- 1c. Linear B syllabary from LA-sized sign samples (truth: full DAMOS reads 89 sign types)
lb_sign_tokens = [s for d in LB for s in d['signs']]
sg = []
for r in range(300):
    smp = prng.sample(lb_sign_tokens, sum(laS))
    A = list(collections.Counter(smp).values())
    sg.append((len(A), chao1(A), ace(A)))
sg = np.array(sg)
res['lb_signs_from_la_size'] = dict(Sobs=q(sg[:, 0]), chao1=q(sg[:, 1]), ace=q(sg[:, 2]),
                                    cover89=float(np.mean(sg[:, 1] >= 89 * 0.95)))
print(res['lb_signs_from_la_size'], flush=True)

# ---- 1d. Linear B Knossos: LA-sized word sample (by documents) -> predict KN types at full KN size
KN = [d for d in LB if d['site'] == 'KN' and d['words']]
knW = collections.Counter(w for d in KN for w in d['words'])
N_kn, S_kn = sum(knW.values()), len(knW)
pred = []
for r in range(40):
    prng.shuffle(KN)
    sub, tok = [], 0
    for d in KN:
        if tok >= 1332:
            break
        sub.append(d)
        tok += len(d['words'])
    A = list(collections.Counter(w for d in sub for w in d['words']).values())
    m = N_kn - sum(A)
    _, nw = boot_predict(A, m, B=100)
    ab = abc(A, n_sims=1500, keep=60, m_future=m) if r < 10 else None
    pred.append(dict(Sobs=len(A), boot_pred=len(A) + np.median(nw), boot_lo=len(A) + np.percentile(nw, 2.5),
                     boot_hi=len(A) + np.percentile(nw, 97.5), rare_tail=float(np.mean(np.array(A) == 1)),
                     abc_pred=float(np.median(ab['types_future'])) if ab else None,
                     abc_lo=float(np.percentile(ab['types_future'], 2.5)) if ab else None,
                     abc_hi=float(np.percentile(ab['types_future'], 97.5)) if ab else None))
    print(r, pred[-1], flush=True)
bp = np.array([p['boot_pred'] for p in pred])
cov = np.mean([p['boot_lo'] <= S_kn <= p['boot_hi'] for p in pred])
abp = [p for p in pred if p['abc_pred']]
res['kn_extrap'] = dict(truth=S_kn, N=N_kn, boot_pred=q(bp), boot_cover=float(cov),
                        abc_pred=q([p['abc_pred'] for p in abp]),
                        abc_cover=float(np.mean([p['abc_lo'] <= S_kn <= p['abc_hi'] for p in abp])))
print(res['kn_extrap'], flush=True)

# ---- 1e. ABC on LA itself (the main estimate)
ab = abc(laW, n_sims=30000, keep=300)
res['la_abc'] = dict(S=q(ab['S']), a=q(np.array(ab['a']) * 100), dmax=ab['dmax'])
abS = abc(laS, n_sims=20000, keep=200)
res['la_sign_abc'] = dict(S=q(abS['S']))
print(res['la_abc'], res['la_sign_abc'], flush=True)
json.dump(res, open(os.path.join(OUT, 'c1.json'), 'w'), indent=1, default=float)
print('done', time.time() - t0)
