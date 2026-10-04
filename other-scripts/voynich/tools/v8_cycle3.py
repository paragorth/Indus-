"""v8 cycle 3: which physical unit was written in one sitting, and where are the jumps?

Residual similarity: Hellinger cosine between pages, double-centred inside a homogeneous group
(same Currier language, hand and illustration type) so page size and group effects drop out.

 V-3.1 Physical-unit test. Pair classes inside a quire (homogeneous group only):
   LEAF      recto + verso of one leaf (physically fixed)
   SIDE      conjugate leaves of one bifolio, same face of the open sheet (r of first-half leaf with v of
             the second-half leaf, or v with r)  -> sessions on flat, unfolded sheets
   CROSS     conjugate leaves, opposite faces (r-r or v-v)
   OPEN      facing pages in the current binding (v of leaf k, r of leaf k+1), not conjugate
   OTHER     every other same-quire pair
   Null: permute page identities within (quire x group) 2000x and recompute class means.
   Control: Isidore text poured into the same physical layout in leaf order (recto, verso, next leaf).
 V-3.2 Jumps. Along the binding order, boundary score between consecutive leaves = residual similarity of
   the 2 pages before vs 2 after. Which boundaries are the sharpest? Do quire edges, bifolio changes,
   hand changes stand out? Calibration: plant a partial key change (3 letters) in Isidore pages at a
   random point, 30 trials: is the planted boundary in the lowest 5%?
"""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict, Counter
from v8_lib import *

rng = np.random.default_rng(11)
Vall = voynich_pages(min_tokens=20)


def resid_sim(pages):
    X, _ = matrix(pages, 2)
    S = hellinger_sim(X)
    n = len(S)
    np.fill_diagonal(S, np.nan)
    m = np.nanmean(S, 1)
    R = S - m[:, None] - m[None, :] + np.nanmean(S)
    return R


def groupkey(p):
    return (p['lang'], p['hand'], p['illus'])


def leaf_half(p, quire_leaves):
    """Is this leaf in the first half of its quire (by $F letter a,b,c,d,e = first; v,w,x,y,z = second)?"""
    return p['leaf'] in 'abcde'


def pair_class(a, b):
    if a['quire'] != b['quire']:
        return None
    if a['leafnum'] == b['leafnum'] and a['leaf'] == b['leaf'] and re.sub(r'\d+$', '', a['id'])[:-1] == re.sub(r'\d+$', '', b['id'])[:-1]:
        return 'LEAF' if a['side'] != b['side'] else None
    if a['bifolio'] == b['bifolio'] and a['bifolio'] is not None and a['leaf'] != b['leaf']:
        return 'SIDE' if a['side'] != b['side'] else 'CROSS'
    # facing pages in the current binding
    lo, hi = (a, b) if a['leafnum'] < b['leafnum'] else (b, a)
    if hi['leafnum'] - lo['leafnum'] == 1 and lo['side'] == 'v' and hi['side'] == 'r':
        return 'OPEN'
    return 'OTHER'


def class_means(pages, R, ids_by_group):
    acc = defaultdict(list)
    for g, ids in ids_by_group.items():
        for i, j in itertools.combinations(ids, 2):
            c = pair_class(pages[i], pages[j])
            if c:
                acc[c].append(R[i, j])
    return {c: (np.mean(v), len(v)) for c, v in acc.items()}


def unit_test(pages, label, nperm=2000):
    groups = defaultdict(list)
    for i, p in enumerate(pages):
        groups[groupkey(p)].append(i)
    Rg = {}
    out_real = defaultdict(list)
    # residuals inside each homogeneous group
    full = np.full((len(pages), len(pages)), np.nan)
    for g, ids in groups.items():
        if len(ids) < 4:
            continue
        R = resid_sim([pages[i] for i in ids])
        for a, i in enumerate(ids):
            for b, j in enumerate(ids):
                full[i, j] = R[a, b]
    # pairs by quire x group
    qg = defaultdict(list)
    for i, p in enumerate(pages):
        if not np.isnan(full[i]).all():
            qg[(p['quire'], groupkey(p))].append(i)
    real = class_means(pages, full, qg)
    # null: permute which page sits in which physical slot inside quire x group
    null = defaultdict(list)
    for t in range(nperm):
        perm = np.arange(len(pages))
        for k, ids in qg.items():
            perm[ids] = rng.permutation(ids)
        Rp = full[np.ix_(perm, perm)]
        cm = class_means(pages, Rp, qg)
        for c, (m, n) in cm.items():
            null[c].append(m)
        null['SIDE-CROSS'].append(cm.get('SIDE', (0,))[0] - cm.get('CROSS', (0,))[0])
        null['LEAF-SIDE'].append(cm.get('LEAF', (0,))[0] - cm.get('SIDE', (0,))[0])
    real['SIDE-CROSS'] = (real['SIDE'][0] - real['CROSS'][0], 0)
    real['LEAF-SIDE'] = (real['LEAF'][0] - real['SIDE'][0], 0)
    res = {}
    for c, (m, n) in real.items():
        nv = np.array(null[c])
        res[c] = (m, n, nv.mean(), (m - nv.mean()) / (nv.std() + 1e-12), np.mean(nv >= m))
    print(label, {c: tuple(round(float(x), 3) for x in v) for c, v in res.items()})
    return res


def fmt(res):
    return ', '.join(f'{c} {res[c][0]:+.3f} (n={res[c][1]}, z {res[c][3]:+.1f}, p={res[c][4]:.3g})' for c in ['LEAF', 'SIDE', 'CROSS', 'OPEN', 'OTHER', 'SIDE-CROSS', 'LEAF-SIDE'] if c in res)


rows = []
rv = unit_test(Vall, 'Voynich all groups')
rAh = unit_test([p for p in Vall if p['lang'] == 'A' and p['illus'] == 'H'], 'A herbal')
rB = unit_test([p for p in Vall if p['lang'] == 'B'], 'B')
# IT2a transcription
rIT = unit_test(voynich_pages(min_tokens=20, name='IT2a'), 'IT2a all', nperm=1000)
# control: Isidore poured into the same physical layout in leaf order
LAT = latin_words()
Lp = text_to_pages(LAT, Vall, offset=2000)
for q, p in zip(Lp, Vall):
    q['lang'], q['hand'], q['illus'] = p['lang'], p['hand'], p['illus']
rL = unit_test(Lp, 'Latin control (book order)')
# control 2: Isidore written sheet-side by sheet-side: reassign text so that SIDE partners get consecutive text
# (simulate: within each bifolio, order slots as outer face [first-leaf r, second-leaf v], inner face [first-leaf v, second-leaf r])
def sheetwise(pages):
    slots = sorted(range(len(pages)), key=lambda i: (pages[i]['order'] if False else 0))
    byb = defaultdict(list)
    for i, p in enumerate(pages):
        byb[(p['quire'], p['bifolio'])].append(i)
    order = []
    for k in sorted(byb, key=lambda k: min(byb[k])):
        ids = byb[k]
        first = [i for i in ids if (pages[i]['leaf'] or 'a') in 'abcde']; second = [i for i in ids if i not in first]
        outer = [i for i in first if pages[i]['side'] == 'r'] + [i for i in second if pages[i]['side'] == 'v']
        inner = [i for i in first if pages[i]['side'] == 'v'] + [i for i in second if pages[i]['side'] == 'r']
        order += outer + inner
    tmpl = [pages[i] for i in order]
    poured = text_to_pages(LAT, tmpl, offset=2000)
    out = [None] * len(pages)
    for i, q in zip(order, poured):
        out[i] = q
    return out
Ls = sheetwise(Vall)
for q, p in zip(Ls, Vall):
    q['lang'], q['hand'], q['illus'] = p['lang'], p['hand'], p['illus']
rLs = unit_test(Ls, 'Latin control (sheet-side order)')

rows.append(('V-3.1', 'Physical writing unit. Residual page similarity (Hellinger, double-centred within language x hand x illustration) for pair classes inside a quire: LEAF (r+v of one leaf), SIDE (conjugate leaves, same face of the unfolded sheet), CROSS (conjugate, opposite faces), OPEN (facing pages in current binding), OTHER. Null: page identities permuted within quire x group (2000x). Controls: Isidore poured into the same layout in book order, and in sheet-face order',
             'Voynich ZL: ' + fmt(rv) + ' | A herbal: ' + fmt(rAh) + ' | B: ' + fmt(rB) + ' | IT2a: ' + fmt(rIT) + ' || control book order: ' + fmt(rL) + ' | control sheet-face order: ' + fmt(rLs),
             None))

# ---------------- V-3.2 jumps ----------------
def boundary_scores(pages, R):
    """Boundaries between consecutive pages in binding order (index order)."""
    n = len(pages); out = []
    for k in range(1, n):
        a = [i for i in (k - 2, k - 1) if i >= 0]; b = [i for i in (k, k + 1) if i < n]
        v = [R[i, j] for i in a for j in b if not np.isnan(R[i, j])]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)

X, _ = matrix(Vall, 2)
S = hellinger_sim(X); np.fill_diagonal(S, np.nan)
m = np.nanmean(S, 1); R = S - m[:, None] - m[None, :] + np.nanmean(S)
bs = boundary_scores(Vall, R)
def btype(k):
    a, b = Vall[k - 1], Vall[k]
    t = []
    if a['quire'] != b['quire']: t.append('quire')
    if a['leafnum'] == b['leafnum']: t.append('same-leaf')
    if a['hand'] != b['hand']: t.append('hand')
    if a['lang'] != b['lang']: t.append('lang')
    if a['illus'] != b['illus']: t.append('section')
    if a['quire'] == b['quire'] and a['bifolio'] != b['bifolio'] and a['leafnum'] != b['leafnum']: t.append('bifolio')
    return t
types = [btype(k) for k in range(1, len(Vall))]
qs = np.nanpercentile(bs, 10)
tab = {}
for t in ['quire', 'hand', 'lang', 'section', 'bifolio', 'same-leaf']:
    idx = [i for i, ty in enumerate(types) if t in ty]
    tab[t] = (len(idx), np.nanmean(bs[idx]) if idx else np.nan, np.mean([bs[i] <= qs for i in idx]) if idx else np.nan)
# 'clean' boundaries: same quire, same hand/lang/section, different bifolio
clean = [i for i, ty in enumerate(types) if ty == ['bifolio']]
sameb = [i for i, ty in enumerate(types) if ty == [] ]  # same bifolio, different leaf (and same hand etc.)
print('boundary table', tab)
print('clean bifolio change', len(clean), np.nanmean(bs[clean]), 'same-bifolio leaf change', len(sameb), np.nanmean(bs[sameb]))
# sharpest within-homogeneous jumps (no quire/hand/lang/section change)
hom = [i for i, ty in enumerate(types) if not set(ty) & {'quire', 'hand', 'lang', 'section'}]
worst = sorted(hom, key=lambda i: bs[i])[:8]
wl = [f"{Vall[i]['id']}|{Vall[i + 1]['id']} {bs[i]:+.3f}" for i in worst]
print('sharpest homogeneous jumps', wl)
# permutation p for clean bifolio change < same-bifolio change
from scipy.stats import mannwhitneyu
mw = mannwhitneyu(bs[clean], bs[sameb], alternative='less') if clean and sameb else None

# calibration: planted partial key change in Isidore at random points
det = []
for t in range(30):
    Lpp = [dict(p) for p in Lp]
    k0 = int(rng.integers(20, len(Lpp) - 20))
    lets = set(random.Random(t).sample(list('aeioustnrlm'), 3))
    for i in range(k0, len(Lpp)):
        Lpp[i]['lines'] = [[''.join(ch.upper() if ch in lets else ch for ch in w) for w in l] for l in Lpp[i]['lines']]
    Xl, _ = matrix(Lpp, 2); Sl = hellinger_sim(Xl); np.fill_diagonal(Sl, np.nan)
    ml = np.nanmean(Sl, 1); Rl = Sl - ml[:, None] - ml[None, :] + np.nanmean(Sl)
    bl = boundary_scores(Lpp, Rl)
    rank = np.sum(bl < bl[k0 - 1]) / len(bl)
    det.append(rank)
det = np.array(det)
print('planted key change rank quantile: median', np.median(det), 'share in lowest 5%', np.mean(det <= 0.05))

rows.append(('V-3.2', 'Jumps along the current binding order: boundary score = residual similarity of the 2 pages before vs the 2 after (whole book, double-centred). Boundary types: quire edge, hand change, language change, section change, bifolio change in a quire with nothing else changing, and leaf change inside one bifolio. Calibration: partial key change (3 letters get a variant) planted at a random point in Isidore pages (30 trials)',
             'mean score / share in sharpest 10%%: ' + '; '.join(f'{t} n={v[0]} {v[1]:+.3f} / {v[2]:.2f}' for t, v in tab.items()) +
             f'; clean bifolio change n={len(clean)} {np.nanmean(bs[clean]):+.3f} vs same-bifolio leaf change n={len(sameb)} {np.nanmean(bs[sameb]):+.3f} (MW p={mw.pvalue:.2g}); sharpest homogeneous jumps: ' + ', '.join(wl) +
             f' || planted key change: median rank quantile {np.median(det):.3f}, in lowest 5% in {np.mean(det <= 0.05) * 100:.0f}% of trials',
             None))
import pickle
pickle.dump(dict(rows=rows, rv=rv, rAh=rAh, rB=rB, rIT=rIT, rL=rL, rLs=rLs, tab=tab, wl=wl, det=det), open(os.path.join(DATA, 'results', 'v8_cycle3.pkl'), 'wb'))
for r in rows:
    print(r)
