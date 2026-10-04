"""v46 cycle 3: stranger forms.
(A) GLYPH-WEIGHT NUMERALS (massive random guessing): a numeral word = a word made only of 'digit' glyphs;
    value = sum of glyph weights (additive, Roman-like). 20,000 random weight vectors (each EVA letter a digit
    with p 0.25, weight in {1,2,3,5,10,20,50}); page features: F1 sum of numeral values (log, size-residualised),
    F2 largest numeral on the page, F3 value of the page's first word, F4 of its last word. Score = pooled
    within-section Spearman with the object count. Fit half -> top 20 -> 150-step hill climb on fit half ->
    held-out half. Null: same with counts permuted in section x size strata (20 replicates).
    Controls: Hyginus (faithful counts: Roman numerals i v x l must be found); planted numerals in the Voynich
    (n = 5a + b written as g^a m^b on 60% of pages).
(B) CROSS-SECTION TRANSFER: a number word means the same in every section. Patterns fitted on one
    eye-counted section (top 20 by |S1|) are scored, sign-matched, in each other section; null = 300 random
    patterns of matched frequency."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import rankdata
import v46_lib as L

log = []
def out(s):
    print(s, flush=True); log.append(s)

WVALS = np.array([1, 2, 3, 5, 10, 20, 50], float)

class Numerals:
    def __init__(self, units):
        self.units = units
        M, types, tot = L.type_matrix(units)
        self.M, self.types = M, types
        self.chars = sorted(set(''.join(types)))
        ci = {c: i for i, c in enumerate(self.chars)}
        G = np.zeros((len(types), len(self.chars)), np.float32)
        for i, w in enumerate(types):
            for ch in w: G[i, ci[ch]] += 1
        self.G, self.Z = G, (G > 0).astype(np.float32)
        ti = {w: i for i, w in enumerate(types)}
        self.page_types = [np.array(sorted(set(ti[w] for w in u['toks']))) for u in units]
        self.first = np.array([ti[u['toks'][0]] for u in units]); self.last = np.array([ti[u['toks'][-1]] for u in units])
        self.T = np.array([len(u['toks']) for u in units], float)
        self.secs = [u['sec'] for u in units]

    def features(self, W):            # W: chars x B
        valid = (self.Z @ (W == 0).astype(np.float32)) == 0      # types x B
        vals = (self.G @ W) * valid
        F1 = self.M @ vals
        F1 = np.log1p(F1) - 0 * np.log(self.T)[:, None]
        F2 = np.stack([vals[pt].max(0) for pt in self.page_types])
        return {'F1': F1, 'F2': F2, 'F3': vals[self.first], 'F4': vals[self.last]}

def score(F, n, secs, idx):
    s = [secs[i] for i in idx]
    Zf = L.zrank_within(F[idx], s); zn = L.zrank_within(n[idx], s)[:, 0]
    return (Zf * zn[:, None]).mean(0)

def random_W(nc, B, rng):
    W = np.where(rng.random((nc, B)) < 0.25, WVALS[rng.integers(0, len(WVALS), (nc, B))], 0).astype(np.float32)
    return W

def search(N, n, fit, test, rng, nrand=20000, topk=20, steps=150):
    best = {}
    nc = len(N.chars)
    for b in range(nrand // 1000):
        W = random_W(nc, 1000, rng)
        Fs = N.features(W)
        for f, F in Fs.items():
            s = score(F, n, N.secs, fit)
            for j in np.argsort(-np.abs(s))[:topk]:
                best.setdefault(f, []).append((abs(s[j]), np.sign(s[j]), W[:, j].copy()))
    res = {}
    for f, lst in best.items():
        lst = sorted(lst, key=lambda x: -x[0])[:topk]
        held = []
        for a, sg, w in lst:
            cur = a
            for _ in range(steps):
                w2 = w.copy(); k = rng.integers(nc)
                w2[k] = 0 if (w2[k] > 0 and rng.random() < .5) else WVALS[rng.integers(len(WVALS))]
                s2 = score(N.features(w2[:, None])[f], n, N.secs, fit)[0]
                if abs(s2) > cur: cur, w, sg = abs(s2), w2, np.sign(s2)
            t = score(N.features(w[:, None])[f], n, N.secs, test)[0] * sg
            held.append((cur, t, w))
        held.sort(key=lambda x: -x[0])
        res[f] = held
    return res

def halves(secs, rng):
    secs = np.array(secs); fit = np.zeros(len(secs), bool)
    for s in set(secs):
        ii = np.nonzero(secs == s)[0]; ii = ii[rng.permutation(len(ii))]; fit[ii[:len(ii) // 2]] = True
    return np.nonzero(fit)[0], np.nonzero(~fit)[0]

def run_A(tag, units, nrep=20, nrand=20000):
    N = Numerals(units)
    n = np.array([u['n'] for u in units], float)
    rng = np.random.default_rng(0)
    fit, test = halves(N.secs, rng)
    R = search(N, n, fit, test, rng, nrand=nrand)
    strat = L.strata(units)
    P = L.perms_within(strat, nrep, 11)
    nulls = defaultdict(list)
    for k, p in enumerate(P):
        Rn = search(N, n[p], fit, test, np.random.default_rng(100 + k), nrand=nrand // 4, steps=60)
        for f, h in Rn.items(): nulls[f].append(np.mean([x[1] for x in h[:5]]))
    for f in ('F1', 'F2', 'F3', 'F4'):
        h = R[f]; real = np.mean([x[1] for x in h[:5]]); nl = np.array(nulls[f])
        w = h[0][2]; digits = ''.join('%s=%d ' % (c, v) for c, v in zip(N.chars, w) if v > 0)
        out('%s %s: fit |rho| top %.3f, held-out mean of top 5 %.3f vs null %.3f +- %.3f (p %.3f); best digits: %s' % (
            tag, f, h[0][0], real, nl.mean(), nl.std(), (1 + (nl >= real).sum()) / (1 + len(nl)), digits.strip()))

def plant_numerals(units, share=0.6, seed=0):
    rng = np.random.default_rng(seed); outu = []
    for u in units:
        t = list(u['toks'])
        if rng.random() < share and u['n'] > 0:
            a, b = divmod(int(u['n']), 5); w = 'g' * a + 'm' * b
            t[rng.integers(len(t))] = w
        outu.append(dict(u, toks=t))
    return outu

def run_B(V):
    secs = ['bio', 'zodiac', 'pharma', 'stars']
    data = {}
    M, types, tot = L.type_matrix(V)
    pats = L.make_patterns(types, tot, n_random=4000, seed=0, min_tok=15)
    C = L.counts_for(M, pats); names = [p[0] for p in pats]
    T = np.array([len(u['toks']) for u in V], float); n = np.array([u['n'] for u in V], float)
    S = np.array([u['sec'] for u in V])
    per = {}
    for s in secs:
        m = S == s
        a, _, _ = L.s1_matrix(C[m], T[m], n[m], list(S[m])); per[s] = a
    rng = np.random.default_rng(0)
    for s in secs:
        top = np.argsort(-np.abs(per[s]))[:20]
        for t in secs:
            if t == s: continue
            real = np.mean(np.sign(per[s][top]) * per[t][top])
            null = []
            for _ in range(300):
                r = rng.choice(len(names), 20, replace=False); sg = rng.choice([-1, 1], 20)
                null.append(np.mean(sg * per[t][r]))
            null = np.array(null)
            data[(s, t)] = (real, (1 + (null >= real).sum()) / 301)
    rows = ['%s->%s %.3f (p %.2f)' % (s, t, *data[(s, t)]) for (s, t) in data]
    out('CROSS-SECTION TRANSFER (top 20 S1 patterns of the source section, sign-matched S1 in target): ' + ' ; '.join(rows))
    # patterns consistent in all four sections
    Z = np.stack([per[s] for s in secs])
    cons = np.min(np.abs(Z), 0) * (np.abs(np.sign(Z).sum(0)) == 4)
    o = np.argsort(-cons)[:5]
    out('   most consistent patterns across all 4 sections (min |S1|, same sign): ' + ' ; '.join('%s %s' % (names[j], np.round(Z[:, j], 2).tolist()) for j in o))
    # null for consistency: permuted counts
    P = L.perms_within(L.strata(V), 200, 5); mx = []
    for p in P:
        Zp = np.stack([L.s1_matrix(C[S == s], T[S == s], n[p][S == s], [s] * int((S == s).sum()))[0] for s in secs])
        mx.append((np.min(np.abs(Zp), 0) * (np.abs(np.sign(Zp).sum(0)) == 4)).max())
    mx = np.array(mx)
    out('   max consistency real %.3f vs permuted-count null %.3f +- %.3f, p %.3f' % (cons.max(), mx.mean(), mx.std(), (1 + (mx >= cons.max()).sum()) / 201))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'hyg'):
        H = L.hyginus_units()
        run_A('HYGINUS faithful', [dict(u, n=u['stated']) for u in H])
        run_A('HYGINUS Ptolemy', H)
    if which in ('all', 'voy'):
        V = [u for u in L.voy_units() if u['sec'] != 'herbal']
        run_A('PLANT g/m numerals on 60% of eye-counted pages', plant_numerals(V))
        run_A('VOYNICH eye-counted sections', V)
        run_A('VOYNICH herbal (auto leaves)', [u for u in L.voy_units() if u['sec'] == 'herbal'])
    if which in ('all', 'B'):
        run_B([u for u in L.voy_units() if u['sec'] != 'herbal'])
    open(os.path.join(L.CK, 'c3_%s.log' % which), 'w').write('\n'.join(log))
