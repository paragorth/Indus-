"""v8 cycle 2: is Currier A -> B a continuum (stages of one drifting process) or two discrete states?

Page B-score = mean per-token log-likelihood ratio B-model vs A-model (unigram, add-0.5, models
trained on all OTHER pages; leave-one-page-out). Computed separately on even and odd lines.
 V-2.1 graded test: within a language group, do even-line and odd-line B-scores of the same page agree
       (a page has its own stable position between A and B)? Stratum-demeaned (section x hand).
       Controls on Isidore pages with the same templates: (G) drifting key over the whole book, labels
       A = first block, B = second; (D) two fixed keys, abrupt switch at the same point (discrete).
 V-2.2 gap / bimodality: share of pages in the middle third between the A and B medians.
 V-2.3 oddities as stages: within A and within B, does a page's B-score (computed from words NOT carrying
       the feature) predict the feature? Features: line-final m/g quota, q-initial share, 'ed' share,
       -ol/-or share, mean line length. Calibrated with the same feature logic on (G) and (D).
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import spearmanr
from v8_lib import *

rng = random.Random(7)


def models(pages, labels, excl=lambda w, pos: False):
    """Per-page counts by group (A/B) of words passing the exclusion filter; pos = (line index, word index, line length)."""
    pc = []
    for p in pages:
        c = [Counter(), Counter()]  # even, odd lines
        for i, l in enumerate(p['lines']):
            for j, w in enumerate(l):
                if not excl(w, (i, j, len(l))):
                    c[i % 2][w] += 1
        pc.append(c)
    tot = {'A': Counter(), 'B': Counter()}
    for c, g in zip(pc, labels):
        if g in tot:
            tot[g].update(c[0]); tot[g].update(c[1])
    return pc, tot


def bscores(pages, labels, excl=lambda w, pos: False):
    pc, tot = models(pages, labels, excl)
    vocab = set(tot['A']) | set(tot['B'])
    V = len(vocab) + 1
    NA, NB = sum(tot['A'].values()), sum(tot['B'].values())
    out = []
    for c, g in zip(pc, labels):
        res = []
        for half in (0, 1, None):
            cc = c[0] + c[1] if half is None else c[half]
            s, n = 0.0, 0
            for w, k in cc.items():
                a = tot['A'][w] - (c[0][w] + c[1][w] if g == 'A' else 0)
                b = tot['B'][w] - (c[0][w] + c[1][w] if g == 'B' else 0)
                na = NA - (sum(c[0].values()) + sum(c[1].values()) if g == 'A' else 0)
                nb = NB - (sum(c[0].values()) + sum(c[1].values()) if g == 'B' else 0)
                s += k * (math.log((b + .5) / (nb + .5 * V)) - math.log((a + .5) / (na + .5 * V)))
                n += k
            res.append(s / n if n else float('nan'))
        out.append(res)
    return np.array(out)


def demean(x, strata):
    x = np.array(x, float); out = x.copy()
    for s in set(strata):
        idx = [i for i, t in enumerate(strata) if t == s]
        out[idx] = x[idx] - np.nanmean(x[idx])
    return out


def graded(pages, labels, strata):
    S = bscores(pages, labels)
    r = {}
    for g in ('A', 'B'):
        idx = [i for i, l in enumerate(labels) if l == g]
        e = demean(S[idx, 0], [strata[i] for i in idx]); o = demean(S[idx, 1], [strata[i] for i in idx])
        ok = ~(np.isnan(e) | np.isnan(o))
        r[g] = spearmanr(e[ok], o[ok]).correlation
    return r, S


def middle_share(S, labels):
    s = S[:, 2]
    ma = np.median([v for v, l in zip(s, labels) if l == 'A']); mb = np.median([v for v, l in zip(s, labels) if l == 'B'])
    lo, hi = ma + (mb - ma) / 3, ma + 2 * (mb - ma) / 3
    return np.mean([(lo < v < hi) for v in s]), ma, mb


# ---------------- data ----------------
Vall = voynich_pages()
V = [p for p in Vall if p['lang'] in ('A', 'B')]
lab = [p['lang'] for p in V]
strata = [(p['illus'], p['hand']) for p in V]
nA = lab.count('A')
print('pages', len(V), 'A', nA, 'B', len(V) - nA)

LAT = latin_words()
# template: A pages first, then B pages, so the controls have the same group sizes
tmpl = [p for p in V if p['lang'] == 'A'] + [p for p in V if p['lang'] == 'B']
Lp = text_to_pages(LAT, tmpl, offset=2000)
G = drift_cipher(Lp, seed=3)
clab = ['A'] * nA + ['B'] * (len(V) - nA)
# discrete: key1 for A block (all letters variant 0), key2 for B block (random half of letters variant 1)
half = set(random.Random(5).sample(list('abcdefghiklmnopqrstuxyz'), 11))
D = []
for i, p in enumerate(Lp):
    q = dict(p)
    if i >= nA:
        q['lines'] = [[''.join(ch.upper() if ch in half else ch for ch in w) for w in l] for l in p['lines']]
    D.append(q)
cstrata = [(i * 8) // len(Lp) for i in range(len(Lp))]  # 8 contiguous 'sections' for demeaning

rows = []
rv, Sv = graded(V, lab, strata)
rG, SG = graded(G, clab, cstrata)
rD, SD = graded(D, clab, cstrata)
# null: within-stratum redeal for Voynich
nulls = defaultdict(list)
for s in range(10):
    Vr = redeal(V, strata_key=lambda p: (p['lang'], p['illus'], p['hand']), seed=200 + s)
    rr, _ = graded(Vr, lab, strata)
    for g in 'AB':
        nulls[g].append(rr[g])
print('graded Voynich', rv, 'null', {g: (np.mean(v), np.std(v)) for g, v in nulls.items()})
print('graded drift control', rG, 'discrete control', rD)
mv, mG, mD = middle_share(Sv, lab), middle_share(SG, clab), middle_share(SD, clab)
print('middle share V', mv, 'G', mG, 'D', mD)
# where are the middle pages?
s = Sv[:, 2]
lo, hi = mv[1] + (mv[2] - mv[1]) / 3, mv[1] + 2 * (mv[2] - mv[1]) / 3
mids = [(V[i]['id'], V[i]['lang'], V[i]['illus'], V[i]['hand'], round(s[i], 2)) for i in range(len(V)) if lo < s[i] < hi]
print('middle pages', mids)
# A-group B-score by section: is A pharma/other nearer to B?
for g in 'AB':
    by = defaultdict(list)
    for i in range(len(V)):
        if lab[i] == g:
            by[(V[i]['illus'], V[i]['hand'])].append(s[i])
    print(g, {k: (len(v), round(float(np.median(v)), 2)) for k, v in by.items()})

zA = (rv['A'] - np.mean(nulls['A'])) / (np.std(nulls['A']) + 1e-9)
zB = (rv['B'] - np.mean(nulls['B'])) / (np.std(nulls['B']) + 1e-9)
rows.append(('V-2.1', 'Graded-position test. Page B-score = mean token LLR (B vs A unigram, leave-one-page-out), from even lines and from odd lines separately; Spearman of the two inside each language after demeaning by section x hand. Null: tokens redealt within language x section x hand (10x). Controls: Isidore pages, same templates and group sizes, (G) one key drifting through the whole book, (D) two fixed keys with an abrupt switch',
             'Voynich within A rho %.2f (null %.2f, z %.1f); within B %.2f (null %.2f, z %.1f). Control G (drift): A %.2f, B %.2f. Control D (discrete): A %.2f, B %.2f' % (rv['A'], np.mean(nulls['A']), zA, rv['B'], np.mean(nulls['B']), zB, rG['A'], rG['B'], rD['A'], rD['B']),
             None))
rows.append(('V-2.2', 'Gap test: share of pages whose whole-page B-score lies in the middle third between the A and B medians. Controls G and D as above',
             'Voynich %.3f (%d pages); G %.3f; D %.3f' % (mv[0], len(mids), mG[0], mD[0]), None))

# ---------------- oddities as stages ----------------
def feats(pages):
    F = defaultdict(list)
    for p in pages:
        ends = [l[-1] for l in p['lines'] if l]
        toks = [w for l in p['lines'] for w in l]
        F['m/g line-end quota'].append(np.mean([w[-1] in 'mg' for w in ends]))
        F['q-initial share'].append(np.mean([w.startswith('q') for w in toks]))
        F['ed share'].append(np.mean(['ed' in w for w in toks]))
        F['-ol/-or share'].append(np.mean([w.endswith('ol') or w.endswith('or') for w in toks]))
        F['mean line length'].append(np.mean([len(l) for l in p['lines']]))
    return F

EXCL = {
    'm/g line-end quota': lambda w, pos: pos[1] == pos[2] - 1,
    'q-initial share': lambda w, pos: w.startswith('q'),
    'ed share': lambda w, pos: 'ed' in w,
    '-ol/-or share': lambda w, pos: w.endswith('ol') or w.endswith('or'),
    'mean line length': lambda w, pos: False,
}
F = feats(V)
res = []
for name, ex in EXCL.items():
    Sx = bscores(V, lab, ex)[:, 2]
    line = []
    for g in 'AB':
        idx = [i for i in range(len(V)) if lab[i] == g]
        st = [strata[i] for i in idx]
        x = demean(Sx[idx], st); y = demean(np.array(F[name])[idx], st)
        ok = ~(np.isnan(x) | np.isnan(y))
        r = spearmanr(x[ok], y[ok])
        line.append((g, r.correlation, r.pvalue))
    # direction of A->B difference
    dAB = np.mean([F[name][i] for i in range(len(V)) if lab[i] == 'B']) - np.mean([F[name][i] for i in range(len(V)) if lab[i] == 'A'])
    res.append((name, dAB, line))
    print(name, 'B-A diff %.3f' % dAB, line)

# calibration of the stage test on controls: feature = share of tokens with a variant letter X, X excluded from score
def ctrl_stage(P, labels, letter):
    up = letter.upper()
    Sx = bscores(P, labels, lambda w, pos: up in w)[:, 2]
    y = np.array([np.mean([up in w for l in p['lines'] for w in l]) for p in P])
    out = []
    for g in 'AB':
        idx = [i for i in range(len(P)) if labels[i] == g]
        st = [cstrata[i] for i in idx]
        x = demean(Sx[idx], st); yy = demean(y[idx], st)
        out.append(spearmanr(x, yy).correlation)
    return out
cg = [ctrl_stage(G, clab, L) for L in 'aeiost']
cd = [ctrl_stage(D, clab, L) for L in sorted(half)[:6]]
print('control G stage rho (A,B) per letter', np.round(cg, 2))
print('control D stage rho (A,B) per letter', np.round(cd, 2))

txt = '; '.join(f"{n} (B-A {d:+.3f}): A rho {l[0][1]:+.2f} p={l[0][2]:.2g}, B rho {l[1][1]:+.2f} p={l[1][2]:.2g}" for n, d, l in res)
rows.append(('V-2.3', 'Oddities as stages: inside A and inside B (demeaned by section x hand), Spearman of page B-score computed WITHOUT the feature-carrying words vs the feature. Calibration: same logic on controls with feature = share of tokens carrying a variant letter (letter excluded from score)',
             txt + ' || control G (drift) median rho A %.2f / B %.2f; control D (discrete) median rho A %.2f / B %.2f' % (np.median([c[0] for c in cg]), np.median([c[1] for c in cg]), np.median([c[0] for c in cd]), np.median([c[1] for c in cd])),
             None))
import pickle
pickle.dump(dict(rows=rows, rv=rv, nulls=dict(nulls), rG=rG, rD=rD, mv=mv, mG=mG, mD=mD, mids=mids, res=res, cg=cg, cd=cd, S=Sv, ids=[p['id'] for p in V]),
            open(os.path.join(DATA, 'results', 'v8_cycle2.pkl'), 'wb'))
for r in rows:
    print(r)
