"""v43 cycle 2: the most language-like SUBSET, chosen on half of each page's lines (odd lines, A) and tested on the
other half (even lines, B).  Searches: top-k A-chunks (non-contiguous), best contiguous run of k A-chunks, and
5,000 random page subsets (random size, random make-up) with the top 1% kept.  A subset's held-out score is the
classifier score of its pages' B lines, against random page sets of the same B token count.
Controls: planted Latin / German pages (scattered, 12 pages; and contiguous, 12 pages) must be picked on A and hold
on B; uniform trigram text must give no held-out gain.  Also the reverse direction (choose on B, test on A).
Usage: python3 v43_cycle2.py [nworkers]"""
import sys, time, random
import numpy as np
from collections import Counter, defaultdict
import v43_lib as L

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 2
t0 = time.time()
def log(*a): print(round(time.time() - t0), *a, flush=True)

C1 = L.load('cycle1.json')
TAU6 = min(0.5, C1['tau']['6']['p5'])
P0 = L.voynich('ZL3b')


def half_stream(P, k):
    S = []
    for pi, p in enumerate(P):
        li = 0
        for pa in p['paras']:
            for l in pa:
                if li % 2 == k:
                    for w in l: S.append((w, pi, (pi, li)))
                li += 1
    return S


def page_lines(P, pi, k):
    out, li = [], 0
    for pa in P[pi]['paras']:
        for l in pa:
            if li % 2 == k: out.append(l)
            li += 1
    return out


class Scorer:
    """scores the k-half of a set of pages: concatenate the pages' k-lines in reading order, 100-token chunks,
    classifier mean; chunk features cached by (page tuple)."""
    def __init__(self, P, k, excl=(), tag=''):
        self.P, self.k, self.excl, self.tag = P, k, excl, tag
        self.cache = {}

    def score(self, pages):
        pages = tuple(sorted(pages))
        if pages in self.cache: return self.cache[pages]
        S = []
        for pi in pages:
            for li, l in enumerate(page_lines(self.P, pi, self.k)):
                for w in l: S.append((w, pi, (pi, li)))
        ch = L.chunks(S)
        if not ch: return None
        F = L.chunk_feats(ch, seed=len(pages))
        s = float(L.lscore(L.probs(F, self.excl)).mean())
        self.cache[pages] = (s, len(S))
        return self.cache[pages]


def experiment(P, excl=(), name='ZL', truth=None, nrand=5000, seed=0):
    rng = random.Random(seed)
    out = {}
    for k_choose in (0, 1):
        k_test = 1 - k_choose
        SA = half_stream(P, k_choose)
        chA = L.chunks(SA)
        F = L.load(f'F2_{name}_{k_choose}.json')
        if F is None or len(F) != len(chA):
            F = L.feats_pool(chA, NW); L.save(f'F2_{name}_{k_choose}.json', F)
        cs = L.lscore(L.probs(F, excl))
        cpages = [sorted({SA[j][1] for j in range(i * 100, i * 100 + 100)}) for i in range(len(chA))]
        TS = Scorer(P, k_test, excl)
        # null: random page sets matched on B token count (by page count of the chosen set)
        NUL = {}
        def rand_null(npg, nn=15):
            b = max(4, int(round(npg / 4.0)) * 4)
            if b in NUL: return NUL[b]
            v = []
            for _ in range(nn):
                r = TS.score(rng.sample(range(len(P)), min(b, len(P))))
                if r: v.append(r[0])
            NUL[b] = np.array(v)
            return NUL[b]
        res = {}
        picks = {}
        for kk in (6, 12, 24):
            top = np.argsort(-cs)[:kk]
            picks[f'top{kk}'] = sorted({p for i in top for p in cpages[i]})
            ws = L.window_scores(cs, kk); i = int(ws.argmax())
            picks[f'run{kk}'] = sorted({p for j in range(i, i + kk) for p in cpages[j]})
        # random page subsets scored on the choosing half (pages' own chunks: mean of chunks fully inside pages)
        pg_cs = defaultdict(list)
        for i, pg in enumerate(cpages):
            for p in pg: pg_cs[p].append(cs[i])
        pmean = {p: np.mean(v) for p, v in pg_cs.items()}
        keys = ('sec', 'quire', 'hand', 'lang')
        best = []
        for _ in range(nrand):
            mode = rng.random()
            if mode < 0.5:
                sub = rng.sample(sorted(pmean), rng.randint(6, 40))
            else:   # union of random metadata cells
                kx = rng.choice(keys); vals = sorted({P[p][kx] for p in pmean})
                pick = set(rng.sample(vals, rng.randint(1, max(1, len(vals) // 3))))
                sub = [p for p in pmean if P[p][kx] in pick]
                if len(sub) < 6: continue
                if len(sub) > 40: sub = rng.sample(sub, 40)
            best.append((np.mean([pmean[p] for p in sub]), tuple(sorted(sub))))
        best.sort(reverse=True)
        surv = best[:max(1, len(best) // 100)]
        picks['rand_top1pct'] = None
        for nm, pg in picks.items():
            if pg is None: continue
            r = TS.score(pg)
            if r is None: continue
            nul = rand_null(len(pg))
            d = dict(npages=len(pg), choose=float(np.mean([pmean.get(p, np.nan) for p in pg])), test=r[0], ntok=r[1],
                     null_mean=float(nul.mean()), z=float((r[0] - nul.mean()) / (nul.std() + 1e-9)),
                     above_tau=r[0] >= TAU6, secs=dict(Counter(P[p]['sec'] for p in pg)),
                     hands=dict(Counter(P[p]['hand'] for p in pg)), quires=dict(Counter(P[p]['quire'] for p in pg)))
            if truth is not None:
                tr = set(truth); d['recall'] = len(tr & set(pg)) / len(tr); d['precision'] = len(tr & set(pg)) / len(pg)
            res[nm] = d
        # random survivors: held-out score of each of the top 1% (cap 25)
        sv = []
        for sc, pg in surv[:15]:
            r = TS.score(pg)
            if r: sv.append((sc, r[0], len(pg)))
        rnd = [TS.score(rng.sample(range(len(P)), rng.randint(6, 40))) for _ in range(15)]
        rnd = np.array([x[0] for x in rnd if x])
        sv = np.array(sv)
        res['rand_survivors'] = dict(n=len(sv), choose_mean=float(sv[:, 0].mean()), test_mean=float(sv[:, 1].mean()),
                                     rand_test_mean=float(rnd.mean()), z=float((sv[:, 1].mean() - rnd.mean()) / (rnd.std() / np.sqrt(len(sv)) + 1e-9)),
                                     frac_above_tau=float((sv[:, 1] >= TAU6).mean()))
        out[f'choose{k_choose}'] = res
        log(name, 'choose', k_choose, {k: (round(v.get('test', v.get('test_mean', 0)), 3), round(v['z'], 2), v.get('recall')) for k, v in res.items()})
    return out


def planted(P, lname, pages, seed=0):
    ll = L.corpus_lines(lname, maxtok=20000)
    vw = [w for p in P for pa in p['paras'] for l in pa for w in l]
    vmap = L.translit_map([w for l in ll for w in l], vw)
    lw = iter([''.join(vmap.get(c, c) for c in w) for l in ll for w in l])
    Q = []
    for pi, p in enumerate(P):
        if pi in pages:
            q = dict(p); q['paras'] = [[[next(lw) for _ in l] for l in pa] for pa in p['paras']]; Q.append(q)
        else: Q.append(p)
    return Q


R = {}
R['ZL'] = experiment(P0, (), 'ZL')
L.save('cycle2.json', R)
rng = random.Random(5)
scat = sorted(rng.sample(range(len(P0)), 12))
cont = list(range(150, 162))
for lname in ('L_msI_Lat', 'L_msG_Alem'):
    for nm, pg in (('scat', scat), ('cont', cont)):
        Q = planted(P0, lname, set(pg))
        R[f'plant_{lname}_{nm}'] = experiment(Q, (lname,), f'pl_{lname}_{nm}', truth=pg, nrand=2000)
        L.save('cycle2.json', R)
# uniform trigram
import v33_lib as G
lines = [l for p in P0 for pa in p['paras'] for l in pa]
tri = G.Trigram(lines); it = iter(tri.gen(40000, random.Random(3)))
U = []
for p in P0:
    q = dict(p); q['paras'] = [[[next(it) for _ in l] for l in pa] for pa in p['paras']]; U.append(q)
R['uniform_tri'] = experiment(U, (), 'utri', nrand=2000)
L.save('cycle2.json', R)
log('done')
