"""v41 cycle 1: calibrate the drift clock.
(a) planted drift in Voynich hand-1 A herbal pages (hidden times), several strengths
(b) shuffled traits (independent permutation) -> no trajectory
(c) real dated medieval German manuscripts (ReF MLU/RUB, 1350-1650): traits chosen on
    training texts from earliest vs latest third, clock tested on held-out texts against
    their dates; within-text coherence = topic-only baseline (one scribe, one time).
"""
import sys, json, random, math
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, __import__('os').path.dirname(__file__))
from v41_lib import *

rows = []
res = {}
rng = random.Random(41)

# ---------- (a) planted drift ----------
P = vpages()
H1 = [p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H']
A = [p for p in P if p['lang'] == 'A']; B = [p for p in P if p['lang'] == 'B']
o = orient_from(A, B, VTRAITS)
res['orient'] = dict(zip(VTRAITS, o.tolist()))


def plant(word, s, r):
    if r.random() >= s:
        return word
    w = word
    for b, g in (('cth', 't'), ('ckh', 'k'), ('cph', 'p'), ('cfh', 'f')):
        if w.startswith(b):
            return g + w[3:]
    for a, b in (('chol', 'chedy'), ('chor', 'chedy'), ('chy', 'chedy'), ('shy', 'shedy'), ('ol', 'edy'), ('or', 'edy')):
        if w.endswith(a):
            return w[:-len(a)] + b
    return w


def planted_pages(pages, smax, seed):
    r = random.Random(seed)
    t = np.array([r.random() for _ in pages])
    out = []
    for p, ti in zip(pages, t):
        lines = [[plant(w, smax * ti, r) for w in l] for l in p['lines']]
        out.append(dict(lines=lines, all=[w for l in lines for w in l],
                        h=[[w for i, l in enumerate(lines) if i % 2 == k for w in l] for k in (0, 1)]))
    return out, t


def evaluate(pages, o, strata=None, truth=None, seed=0, nperm=300, restarts=12):
    (X1, X2), names = rate_matrix(pages, VTRAITS)
    coh = coherence_null(X1, X2, o, strata, nperm=nperm, seed=seed)
    D1 = demean(X1, strata) if strata is not None else X1 - X1.mean(0)
    D2 = demean(X2, strata) if strata is not None else X2 - X2.mean(0)
    order, f, orders = seriate(D1, restarts=restarts, iters=15000, seed=seed)
    pos = position_of(order)
    # orient the order so that later = more B on half 1
    comp1 = (((D1 - D1.mean(0)) / (D1.std(0) + 1e-12)) * o).mean(1)
    if spearmanr(pos, comp1)[0] < 0:
        pos = pos.max() - pos
    comp2 = (((D2 - D2.mean(0)) / (D2.std(0) + 1e-12)) * o).mean(1)
    held = spearmanr(pos, comp2)[0]
    signs = [spearmanr(pos, D2[:, j])[0] * o[j] for j in range(len(names))]
    stab = np.mean([abs(spearmanr(position_of(orders[i]), position_of(orders[j]))[0])
                    for i in range(len(orders)) for j in range(i + 1, len(orders))])
    out = dict(R=coh['R'], S=coh['S'], ratio=coh['ratio'], z=coh['z'], p=coh['p'],
               held_rho=float(held), held_signs_ok=int(sum(1 for s in signs if s > 0)),
               k=len(names), restart_stab=float(stab), pos=pos.tolist())
    if truth is not None:
        out['truth_rho'] = float(spearmanr(pos, truth)[0])
    return out


for s in (0.0, 0.1, 0.2, 0.4):
    for seed in (0, 1):
        pp, t = planted_pages(H1, s, 100 + seed)
        e = evaluate(pp, o, truth=t, seed=seed)
        e.pop('pos')
        res[f'plant_{s}_{seed}'] = e
        print('plant', s, seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()}, flush=True)

# a jump (two fixed states, no drift) inside the same pages
for seed in (0, 1):
    r = random.Random(200 + seed)
    t = np.array([1.0 if r.random() < 0.5 else 0.0 for _ in H1])
    pp = []
    for p, ti in zip(H1, t):
        lines = [[plant(w, 0.3 * ti, r) for w in l] for l in p['lines']]
        pp.append(dict(lines=lines, all=[w for l in lines for w in l],
                       h=[[w for i, l in enumerate(lines) if i % 2 == k for w in l] for k in (0, 1)]))
    e = evaluate(pp, o, truth=t, seed=seed); e.pop('pos')
    res[f'jump_{seed}'] = e
    print('jump', seed, {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()}, flush=True)

pl = [res[f'plant_{s}_{q}'] for s in (0.1, 0.2, 0.4) for q in (0, 1)]
p0 = [res[f'plant_0.0_{q}'] for q in (0, 1)]
rows.append('| V-41.1.1 | Planted drift clock: hand-1 A herbal pages (n=%d) given hidden times t~U(0,1); each word rewritten A->B (bench drop at word start, -chy/-chol/-chor/-ol/-or -> -edy forms) with prob s*t. 10 fixed v30/v36/v8 traits, oriented A->B on the real A/B split; co-movement S across odd/even line halves vs independent-trait permutation (300x); SA seriation (12 restarts x 15k swaps) on odd lines; order checked against truth and against even-line traits | s=0 (no plant): S %s, z %s, truth rho %s; s=0.1: z %s/%s truth rho %s/%s; s=0.2: z %s/%s rho %s/%s; s=0.4: z %s/%s rho %s/%s; S/R at s=0.4 %s/%s | ' % (
    len(H1),
    '/'.join(f"{e['S']:+.3f}" for e in p0), '/'.join(f"{e['z']:+.1f}" for e in p0), '/'.join(f"{e['truth_rho']:+.2f}" for e in p0),
    *[f"{e['z']:+.1f}" for e in pl[0:2]], *[f"{e['truth_rho']:+.2f}" for e in pl[0:2]],
    *[f"{e['z']:+.1f}" for e in pl[2:4]], *[f"{e['truth_rho']:+.2f}" for e in pl[2:4]],
    *[f"{e['z']:+.1f}" for e in pl[4:6]], *[f"{e['truth_rho']:+.2f}" for e in pl[4:6]],
    *[f"{e['ratio']:.2f}" for e in pl[4:6]]) + 'filled below |')

# ---------- (c) dated German ----------
M = [m for m in ref_meta() if m['year'] and m['medium'].startswith('Hand')]
print('German texts', len(M), flush=True)
for m in M:
    m['pages'] = chunk_pages(ref_words(m['path']), 300, maxpages=12)
M = [m for m in M if len(m['pages']) >= 4]
years = np.array([m['year'] for m in M])
print('usable', len(M), 'years', years.min(), years.max(), flush=True)
cand = Counter()
for m in M:
    m['gc'] = Counter(); m['nw'] = 0
    for p in m['pages']:
        for w in p['all']:
            s = '^' + w + '$'
            m['gc'].update({s[i:i + n] for n in (1, 2, 3) for i in range(len(s) - n + 1)} - {'^', '$'})
            m['nw'] += 1
    cand.update(m['gc'])
ng = [g for g, c in cand.most_common(400)]
gres = []
for split in range(4):
    r = random.Random(split)
    idx = list(range(len(M))); r.shuffle(idx)
    tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]
    ys = sorted(tr, key=lambda i: years[i])
    early, late = ys[:len(ys) // 3], ys[-(len(ys) // 3):]
    # pooled log-odds of 'word contains g'
    def lo(ids, g):
        k = sum(M[i]['gc'][g] for i in ids); n = sum(M[i]['nw'] for i in ids)
        return math.log((k + .5) / (n - k + .5))
    eff = sorted(((lo(late, g) - lo(early, g), g) for g in ng), key=lambda x: -abs(x[0]))
    pick = []
    for e, g in eff:
        if any(g in h or h in g for _, h in pick):
            continue
        pick.append((e, g))
        if len(pick) == 10:
            break
    T = ngram_traits([g for _, g in pick]); og = np.array([np.sign(e) for e, _ in pick])
    pages, strata, tyear = [], [], []
    for i in te:
        for p in M[i]['pages']:
            pages.append(p); strata.append(i); tyear.append(years[i])
    (X1, X2), _ = rate_matrix(pages, T)
    Z = ((X1 + X2) / 2 - ((X1 + X2) / 2).mean(0)) / (((X1 + X2) / 2).std(0) + 1e-12) * og
    clock = Z.mean(1)
    st = np.array(strata)
    tclock = np.array([clock[st == i].mean() for i in te])
    rho_date = spearmanr(tclock, years[te])[0]
    across = coherence_null(X1, X2, og, None, nperm=200, seed=split)
    within = coherence_null(X1, X2, og, strata, nperm=200, seed=split)
    # seriation of held-out TEXTS by their mean traits, many restarts
    Xt = np.array([((X1 + X2) / 2)[st == i].mean(0) for i in te])
    order, f, ords = seriate(Xt - Xt.mean(0), restarts=12, iters=15000, seed=split)
    pos = position_of(order)
    ser_rho = abs(spearmanr(pos, years[te])[0])
    g = dict(split=split, traits=[g for _, g in pick], rho_date=float(rho_date), ser_rho=float(ser_rho),
             across_S=across['S'], across_z=across['z'], across_ratio=across['ratio'],
             within_S=within['S'], within_z=within['z'], within_R=within['R'], within_ratio=within['ratio'], n_te=len(te))
    gres.append(g)
    print('german', g, flush=True)
res['german'] = gres
mean = lambda k: np.mean([g[k] for g in gres])
rows.append('| V-41.1.2 | Real dated control: %d ReF manuscripts (Bavarian/Upper/Middle German, 1350-1650, dated by header) cut into 300-word pages (<=12/text); 10 non-nested 1-3-gram spelling traits chosen on a random half of the texts (latest third vs earliest third), tested on the other half (4 splits): (i) text clock (oriented mean trait z) vs date, (ii) SA seriation of held-out texts (12 restarts) vs date, (iii) co-movement S across texts, and within texts (one scribe, one time = topic-only baseline) | clock~date rho %.2f (splits %s); seriation |rho| with date %.2f; across-text S %.3f z %.1f S/R %.2f; within-text S %.3f z %.1f (R %.3f) | Positive control passes if clock~date and seriation >> 0 and across-text S > 0; within-text S is the topic-only floor |' % (
    len(M), mean('rho_date'), '/'.join(f"{g['rho_date']:.2f}" for g in gres), mean('ser_rho'),
    mean('across_S'), mean('across_z'), mean('across_ratio'), mean('within_S'), mean('within_z'), mean('within_R')))
json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=float)
write_rows(os.path.join(CK, 'c1_rows.txt'), rows)
print('\n'.join(rows))
