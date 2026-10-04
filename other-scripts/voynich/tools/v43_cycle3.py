"""v43 cycle 3: massive random guessing of the island detector itself.  3,000 random language-vs-generator
classifiers (random 3-12 of the 32 v31 features, logistic, LANG+INVENT vs GEN+GIBB+MAGIC, trained on a random 70%
of corpora); survivors must reach >= 0.85 balanced accuracy on the held-out 30% of corpora.  Each survivor scans
the Voynich (ZL and IT2a, W = 12 chunks) with its own chunk-permutation null; island candidates must also hold on
held-out lines (survivor scans the odd-line stream; the same pages' even-line chunks are scored).  Agreement of the
survivors on one place is tested against one shared chunk permutation applied to all of them.
Controls: planted Latin island (1,200 tokens at chunk 60), uniform trigram text (cycle 1 features), 10 languages.
Usage: python3 v43_cycle3.py [nworkers]"""
import sys, time, random
import numpy as np
from collections import Counter, defaultdict
import v43_lib as L
import v31_lib as V

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 2
t0 = time.time()
def log(*a): print(round(time.time() - t0), *a, flush=True)
NCLF, W = 3000, 12

R = V.load('feats_N100.json')
KEYS = sorted(R[0]['F'].keys())
R = [r for r in R if r['cls'] in V.CLASSES]
X = np.array([[r['F'][k] for k in KEYS] for r in R], float); X[~np.isfinite(X)] = 0
Y = np.array([r['cls'] in ('LANG', 'INVENT') for r in R], int)
CORP = np.array([r['corpus'] for r in R])
corpora = sorted(set(CORP))
mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1; Xs = (X - mu) / sd


def fmat(F):
    M = np.array([[f.get(k, 0.0) for k in KEYS] for f in F], float); M[~np.isfinite(M)] = 0
    return (M - mu) / sd


# ---- texts (features cached by cycles 1/2)
P = L.voynich('ZL3b')
S = L.stream_of_pages(P); cpage = [S[i * 100 + 50][1] for i in range(len(S) // 100)]
T = {'ZL': fmat(L.load('F_ZL.json')), 'U_tri': fmat(L.load('F_U_tri.json'))}
PI = L.voynich('IT2a')
SI = L.stream_of_pages(PI)
FI = L.load('F_IT.json')
if FI is None:
    FI = L.feats_pool(L.chunks(SI), NW); L.save('F_IT.json', FI)
T['IT'] = fmat(FI)
cpageI = [SI[i * 100 + 50][1] for i in range(len(SI) // 100)]
# planted Latin (cycle 1 plant, 1200 tokens at chunk 60)
pl = L.load('F_P_L_msI_Lat_1200_60.json'); Tp = T['ZL'].copy(); Tp[60:72] = fmat(pl); T['plant'] = Tp
# odd / even line streams (cycle 2)
A = fmat(L.load('F2_ZL_0.json')); B = fmat(L.load('F2_ZL_1.json'))
def half_pages(k):
    out = []
    for pi, p in enumerate(P):
        li = 0
        for pa in p['paras']:
            for l in pa:
                if li % 2 == k: out += [pi] * len(l)
                li += 1
    return out
hpA, hpB = half_pages(0), half_pages(1)
cpA = [hpA[i * 100 + 50] for i in range(len(A))]; cpB = [hpB[i * 100 + 50] for i in range(len(B))]
LT = {n: fmat(L.load(f'F_{n}.json')) for n in L.LANGS}

from sklearn.linear_model import LogisticRegression
rng = random.Random(0); nrng = np.random.default_rng(0)
surv = []
SC = defaultdict(list)
for c in range(NCLF):
    k = rng.randint(3, 12); fs = sorted(rng.sample(range(len(KEYS)), k))
    test = set(rng.sample(corpora, int(0.3 * len(corpora))))
    tr = np.array([x not in test for x in CORP])
    m = LogisticRegression(C=0.5, max_iter=500, class_weight='balanced').fit(Xs[tr][:, fs], Y[tr])
    pr = m.predict(Xs[~tr][:, fs]); yt = Y[~tr]
    bacc = 0.5 * ((pr[yt == 1] == 1).mean() + (pr[yt == 0] == 0).mean())
    if bacc < 0.85: continue
    sc = {n: m.predict_proba(M[:, fs])[:, 1] for n, M in T.items()}
    langmin = min(L.window_scores(m.predict_proba(M[:, fs])[:, 1], W)[::W].min() for n, M in LT.items()
                  if n not in test or True)
    langp5 = np.percentile(np.concatenate([L.window_scores(m.predict_proba(M[:, fs])[:, 1], W)[::W] for M in LT.values()]), 5)
    tau = min(0.5, langp5)
    d = dict(fs=fs, bacc=float(bacc), tau=float(tau), langmin=float(langmin))
    for n, cs in sc.items():
        ws = L.window_scores(cs, W); i = int(ws.argmax())
        null = L.scan_null(cs, W, 200, seed=c)
        d[n] = dict(max=float(ws.max()), i=i, p=float((null >= ws.max()).mean()), nabove=int((ws >= tau).sum()),
                    nwin=len(ws))
    # held-out: island chosen on odd lines, tested on even lines of the same pages
    sa = m.predict_proba(A[:, fs])[:, 1]; sb = m.predict_proba(B[:, fs])[:, 1]
    wa = L.window_scores(sa, W); i = int(wa.argmax()); pg = {cpA[j] for j in range(i, i + W)}
    inB = np.array([cpB[j] in pg for j in range(len(sb))])
    d['held'] = dict(pages=[P[p]['id'] for p in sorted(pg)], A=float(wa.max()), B=float(sb[inB].mean()) if inB.any() else None,
                     Brest=float(sb[~inB].mean()), B_above_tau=bool(inB.any() and sb[inB].mean() >= tau),
                     Bpct=float((sb[inB].mean() > np.array([sb[nrng.permutation(len(sb))[:inB.sum()]].mean() for _ in range(200)])).mean()) if inB.any() else None)
    surv.append(d)
    for n in ('ZL', 'IT', 'U_tri', 'plant'): SC[n].append(sc[n].astype(np.float32))
    if len(surv) % 50 == 0: log('clf', c, 'survivors', len(surv))
log('survivors', len(surv), 'of', NCLF)
L.save('cycle3_surv.json', surv)

# ---- agreement on a place: argmax concentration vs one shared permutation applied to all survivors
def conc(idx, n):
    h = np.zeros(n + W)
    for i in idx: h[i:i + W] += 1
    return h.max() / len(idx), int(h.argmax())
agree = {}
for n in ('ZL', 'IT', 'U_tri', 'plant'):
    M = np.array(SC[n]); nch = M.shape[1]
    obs, where = conc([d[n]['i'] for d in surv], nch)
    nulls = []
    for r in range(200):
        perm = nrng.permutation(nch)
        idx = [int(L.window_scores(M[j, perm], W).argmax()) for j in range(len(M))]
        nulls.append(conc(idx, nch)[0])
    nulls = np.array(nulls)
    cp = cpage if n != 'IT' else cpageI; PP = P if n != 'IT' else PI
    agree[n] = dict(obs=obs, where=where, p=float((nulls >= obs).mean()), null95=float(np.percentile(nulls, 95)),
                    pages=[PP[cp[j]]['id'] for j in range(where, min(where + 1, len(cp)))],
                    frac_p05=float(np.mean([d[n]['p'] < 0.05 for d in surv])),
                    frac_above=float(np.mean([d[n]['nabove'] > 0 for d in surv])))
    log('agree', n, agree[n])
held = [d['held'] for d in surv if d['held']['B'] is not None]
agree['held'] = dict(n=len(held), frac_B_above_tau=float(np.mean([h['B_above_tau'] for h in held])),
                     mean_Bpct=float(np.mean([h['Bpct'] for h in held])),
                     frac_Bpct95=float(np.mean([h['Bpct'] >= 0.95 for h in held])),
                     top_pages=Counter(p for h in held for p in h['pages']).most_common(15))
log('held', agree['held'])
L.save('cycle3_agree.json', agree)
log('done')
