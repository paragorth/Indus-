"""R1 families (b) random sign->number assignments, (d) random sign->meaning-class assignments,
(e) random cross-script word pairings. Entry point run(job, log).

Stage 1 always uses documents of splits A+B, stage 2 the held-out documents of split C.
"""
import json, math, os, random, hashlib
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import norm, binom
import r1_lib as L

ALPHA = 0.05


def bonf_z(n):
    return float(norm.isf(ALPHA / max(n, 1)))


def stage(split):
    return 1 if split in ('A', 'B') else 2


# =====================================================================================
# (d) sign -> meaning class, scored by co-occurrence with commodity / section labels
# =====================================================================================
def units_d(script, corpus):
    """Return list of units: (cluster, split, label, stratum, features:set)."""
    U = []
    if script == 'voynich':
        ng = Counter()
        for d in corpus['docs']:
            for lw in d['words']:
                for w in lw:
                    ng.update(feat_ngrams(w))
        keep = {g for g, n in ng.items() if n >= 50}
        for ci, d in enumerate(corpus['docs']):
            if d['stratum'] not in ('A', 'B'):
                continue
            for lw in d['words']:
                for w in lw:
                    U.append((ci, d['split'], d['label'], d['stratum'], frozenset(feat_ngrams(w) & keep)))
    elif script == 'linear_a':
        for ci, d in enumerate(corpus['docs']):
            for line in d['lines']:
                logos = [t for t in line if t.startswith('L:')]
                if not logos:
                    continue
                lab = logos[0][2:]
                w, words = [], []
                for t in line + ['_']:
                    if t == '_':
                        if w and not any(x.startswith('L:') or x in L.FIX for x in w):
                            words.append(w)
                        w = []
                    else:
                        w.append(t)
                for w in words:
                    U.append((ci, d['split'], lab, None, frozenset(w)))
    else:
        for ci, d in enumerate(corpus['docs']):
            for e in d['entries']:
                mid = [s for s in e['signs'][:-1] if s != '?']
                if not mid or not e['system']:
                    continue
                U.append((ci, d['split'], e['system'], None, frozenset(mid)))
    return U


def feat_ngrams(w):
    w = list(w)
    out = set()
    for n in (1, 2, 3):
        for i in range(len(w) - n + 1):
            out.add(''.join(w[i:i + n]))
    return out


class DScorer:
    def __init__(self, U, st):
        U = [u for u in U if stage(u[1]) == st]
        self.U = U
        cl = sorted({u[0] for u in U})
        cmap = {c: i for i, c in enumerate(cl)}
        self.cl = np.array([cmap[u[0]] for u in U])
        self.ncl = len(cl)
        self.lab = np.array([u[2] for u in U], dtype=object)
        strata = sorted({str(u[3]) for u in U})
        self.st_of_cl = np.zeros(self.ncl, int)
        smap = {s: i for i, s in enumerate(strata)}
        for u in U:
            self.st_of_cl[cmap[u[0]]] = smap[str(u[3])]
        self.nst = len(strata)
        self.feat_units = defaultdict(list)
        for i, u in enumerate(U):
            for f in u[4]:
                self.feat_units[f].append(i)
        self.feat_units = {f: np.array(v) for f, v in self.feat_units.items()}
        self.n = len(U)

    def z(self, S, c, minsup=5):
        x = np.zeros(self.n, bool)
        for f in S:
            ix = self.feat_units.get(f)
            if ix is not None:
                x[ix] = True
        if x.sum() < minsup:
            return 0.0, 0, 0.0
        y = (self.lab == c).astype(float)
        n1 = np.bincount(self.cl, weights=x, minlength=self.ncl)
        y1 = np.bincount(self.cl, weights=x * y, minlength=self.ncl)
        n0 = np.bincount(self.cl, weights=~x, minlength=self.ncl)
        y0 = np.bincount(self.cl, weights=(~x) * y, minlength=self.ncl)
        num = den = 0.0
        effs = []
        for s in range(self.nst):
            m = self.st_of_cl == s
            N1, N0 = n1[m].sum(), n0[m].sum()
            if N1 < 2 or N0 < 2:
                continue
            p1, p0 = y1[m].sum() / N1, y0[m].sum() / N0
            v1 = np.sum((y1[m] - p1 * n1[m]) ** 2) / N1 ** 2
            v0 = np.sum((y0[m] - p0 * n0[m]) ** 2) / N0 ** 2
            pb = (y1[m].sum() + y0[m].sum()) / (N1 + N0)
            if pb <= 0 or pb >= 1:
                continue          # label absent (or universal) in this stratum: no information
            vb = pb * (1 - pb) * (1 / N1 + 1 / N0)
            v = max(v1 + v0, vb, 1e-12)
            num += (p1 - p0) / v
            den += 1 / v
            effs.append(p1 - p0)
        if den == 0:
            return 0.0, int(x.sum()), 0.0
        return float(num / math.sqrt(den)), int(x.sum()), float(num / den)


def shuffle_labels_d(U, rng):
    """Permute whole-document label vectors among documents of the same stratum and unit count."""
    by_cl = defaultdict(list)
    for i, u in enumerate(U):
        by_cl[u[0]].append(i)
    groups = defaultdict(list)
    for c, ix in by_cl.items():
        groups[(str(U[ix[0]][3]), len(ix))].append(c)
    newlab = {}
    for g, cls in groups.items():
        perm = list(cls)
        rng.shuffle(perm)
        for a, b in zip(cls, perm):
            for i, j in zip(by_cl[a], by_cl[b]):
                newlab[i] = U[j][2]
    return [(u[0], u[1], newlab[i], u[3], u[4]) for i, u in enumerate(U)]


def plant_d(U, rng, feats, labels):
    """One true rule: units of label c* carry feature f* with extra probability 0.5."""
    c = labels[int(rng.integers(0, len(labels)))]
    f = feats[int(rng.integers(min(10, len(feats) - 1), min(40, len(feats))))]
    out = []
    for u in U:
        fs = u[4]
        if u[2] == c and rng.random() < 0.5:
            fs = fs | {f}
        out.append((u[0], u[1], u[2], u[3], fs))
    return out, (f, c)


def run_d(j, log):
    script = j['script']
    corpus = L.load(script)
    rng = np.random.default_rng(3000 + j['seed'])
    U = units_d(script, corpus)
    lc = Counter(u[2] for u in U)
    labels = [l for l, n in lc.items() if n >= 20]
    fc = Counter(f for u in U for f in u[4])
    feats = [f for f, n in fc.most_common() if n >= 10]
    plant = None
    if j['mode'] in ('shuffle', 'planted'):
        U = shuffle_labels_d(U, random.Random(j['seed']))
    if j['mode'] == 'planted':
        U, plant = plant_d(U, rng, feats, labels)
    s1, s2 = DScorer(U, 1), DScorer(U, 2)
    hyps = []
    for h in range(j['N']):
        m = int(rng.integers(1, 4))
        S = tuple(sorted(set(feats[int(i)] for i in rng.integers(0, len(feats), m))))
        c = labels[int(rng.integers(0, len(labels)))]
        z, sup, eff = s1.z(S, c)
        hyps.append((z, S, c, sup, eff))
    N = len(hyps)
    thr1 = bonf_z(N)
    # dedupe
    seen, surv = set(), []
    for h in sorted(hyps, key=lambda x: -x[0]):
        if h[0] <= thr1:
            break
        if (h[1], h[2]) in seen:
            continue
        seen.add((h[1], h[2])); surv.append(h)
    tested = surv[:200]
    thr2 = bonf_z(len(tested))
    rep = []
    for z1, S, c, sup, eff in tested:
        z2, sup2, eff2 = s2.z(S, c)
        rep.append({'S': list(S), 'label': c, 'z1': z1, 'eff1': eff, 'sup1': sup, 'z2': z2, 'eff2': eff2,
                    'sup2': sup2, 'replicated': bool(z2 > thr2)})
    out = {'N': N, 'n_units': len(U), 'labels': {l: lc[l] for l in labels}, 'n_feats': len(feats), 'thr1': thr1,
           'n_stage1': len(surv), 'n_tested2': len(tested), 'thr2': thr2,
           'n_replicated': sum(r['replicated'] for r in rep),
           'z_quantiles': np.quantile([h[0] for h in hyps], [.5, .9, .99, 1]).tolist(), 'stage2': rep[:100]}
    if plant:
        out['plant'] = list(plant)
        out['recovered'] = any(r['replicated'] and r['label'] == plant[1] and plant[0] in r['S'] for r in rep)
        out['n_replicated_not_planted'] = sum(r['replicated'] and not (r['label'] == plant[1] and plant[0] in r['S'])
                                              for r in rep)
    return out


# =====================================================================================
# (b) sign -> number assignments scored by arithmetic of totals / numbering
# =====================================================================================
GRID_PE = [1 / 120, 1 / 60, 1 / 30, 1 / 24, 1 / 12, 1 / 10, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2,
           2, 3, 4, 5, 6, 8, 10, 12, 20, 24, 30, 60, 100, 120, 300, 600, 720, 1000, 3600]
GRID_LA = [1 / 2, 1 / 3, 2 / 3, 1 / 4, 3 / 4, 1 / 5, 1 / 6, 1 / 8, 3 / 8, 1 / 10, 1 / 12, 1 / 16, 1 / 20,
           1 / 24, 1 / 32, 1 / 48, 1 / 60, 1 / 64]


def pe_cases():
    """Total-bearing tablets from attack_arith.json. Tablets whose every (sum, total) pair is
    identical code by code balance under any values and are dropped (uninformative)."""
    d = json.load(open(os.path.join(L.PE, 'attack_arith.json')))
    cases = []
    for t in d['tablets']:
        pairs = [(p['sum'], p['tot'], p['cls']) for p in t['pairs']]
        if all(sm == tt for sm, tt, _ in pairs):
            continue
        cases.append({'id': t['id'], 'pairs': pairs, 'cls': t['classes'][0] if t['classes'] else '?',
                      'split': L.split_of(t['id'])})
    return cases


def la_cases():
    d = json.load(open(os.path.join(L.LA, 'attack_fractions.json')))
    return [{'id': s['id'], 'const': s['const'], 'coef': s['coef'], 'split': L.split_of(s['id']), 'cls': 'x'}
            for s in d['sections']]


def balanced_pe(case, val, tots=None):
    for k, (sm, tt, _) in enumerate(case['pairs']):
        if tots is not None:
            tt = tots[k]
        a = sum(val[c] * n for c, n in sm.items())
        b = sum(val[c] * n for c, n in tt.items())
        if abs(a - b) > 1e-9 * max(1, abs(a), abs(b)):
            return False
    return True


def balanced_la(case, val, const=None):
    c = case['const'] if const is None else const
    return abs(c + sum(v * val[k] for k, v in case['coef'].items())) < 1e-9


def score_b(cases, val, kind, rng, R=20):
    """Number of balanced cases; null = totals (PE) / constants (LA) shuffled among cases of
    the same class (R shuffles). Returns (x, n, p0, p)."""
    if kind == 'pe':
        x = sum(balanced_pe(c, val) for c in cases)
        tot_pool = defaultdict(list)
        for i, c in enumerate(cases):
            tot_pool[c['cls']].append(i)
        hits = 0
        for _ in range(R):
            for cls, ix in tot_pool.items():
                perm = rng.permutation(ix)
                for i, j in zip(ix, perm):
                    ci, cj = cases[i], cases[j]
                    if len(ci['pairs']) != len(cj['pairs']):
                        continue
                    hits += balanced_pe(ci, val, [p[1] for p in cj['pairs']])
    else:
        x = sum(balanced_la(c, val) for c in cases)
        hits = 0
        for _ in range(R):
            perm = rng.permutation(len(cases))
            for i, j in enumerate(perm):
                hits += balanced_la(cases[i], val, cases[j]['const'])
    n = len(cases)
    p0 = max(hits / (R * n), 1 / (R * n * 2))
    p = float(binom.sf(x - 1, n, p0)) if x > 0 else 1.0
    return x, n, p0, p


def zodiac_cases():
    d = json.load(open(os.path.join(L.VOY, 'derived', 'v4_zodiac_labels.json')))
    g = defaultdict(list)
    for x in d:
        if x['label'] and '?' not in x['label']:
            g[(x['page'], x['ring'])].append(x)
    cases = []
    for (pg, ring), xs in g.items():
        if len(xs) < 4:
            continue
        xs.sort(key=lambda r: r['angle'])
        cases.append({'id': f'{pg}:{ring}', 'labels': [L.vglyphs(r['label']) for r in xs], 'split': L.split_of(pg),
                      'cls': 'z'})
    return cases


def zod_value(lab, gv, rule):
    if rule == 'sum':
        return sum(gv.get(g, 0) for g in lab)
    if rule == 'pos_lr':
        v = 0
        for g in lab:
            v = v * 10 + gv.get(g, 0)
        return v
    v = 0
    for g in lab[::-1]:
        v = v * 10 + gv.get(g, 0)
    return v


def zod_stat(cases, gv, rule, rng, R=30):
    """Successor statistic: cyclically adjacent labels whose values differ by exactly 1.
    Null: labels permuted within each ring (R permutations). Returns z."""
    obs = 0
    nulls = np.zeros(R)
    for c in cases:
        v = np.array([zod_value(l, gv, rule) for l in c['labels']])
        obs += int(np.sum(np.abs(v - np.roll(v, 1)) == 1))
        for r in range(R):
            w = rng.permutation(v)
            nulls[r] += np.sum(np.abs(w - np.roll(w, 1)) == 1)
    sd = nulls.std(ddof=1)
    if sd == 0:
        return 0.0, obs
    return float((obs - nulls.mean()) / sd), obs


def run_b(j, log):
    script = j['script']
    rng = np.random.default_rng(4000 + j['seed'])
    plant = None
    if script == 'proto_elamite':
        cases = pe_cases()
        codes = sorted({c for cs in cases for p in cs['pairs'] for v in (p[0], p[1]) for c in v})
        if j['mode'] == 'shuffle':
            cases = shuffle_b_pe(cases, random.Random(j['seed']))
        elif j['mode'] == 'planted':
            cases, plant = plant_b_pe(cases, codes, rng)
        return run_b_generic(j, cases, codes, 'pe', rng, plant, log)
    if script == 'linear_a':
        cases = la_cases()
        codes = sorted({k for c in cases for k in c['coef']})
        if j['mode'] == 'shuffle':
            consts = [c['const'] for c in cases]
            random.Random(j['seed']).shuffle(consts)
            cases = [dict(c, const=k) for c, k in zip(cases, consts)]
        elif j['mode'] == 'planted':
            vstar = {k: GRID_LA[int(rng.integers(0, len(GRID_LA)))] for k in codes}
            cases = [dict(c, const=-sum(v * vstar[k] for k, v in c['coef'].items())) for c in cases]
            plant = vstar
        return run_b_generic(j, cases, codes, 'la', rng, plant, log)
    # voynich zodiac numbering
    cases = zodiac_cases()
    glyphs = sorted({g for c in cases for l in c['labels'] for g in l})
    if j['mode'] == 'shuffle':
        # labels shuffled across all rings: destroys any position-number link
        pool = [l for c in cases for l in c['labels']]
        random.Random(j['seed']).shuffle(pool)
        it = iter(pool)
        cases = [dict(c, labels=[next(it) for _ in c['labels']]) for c in cases]
    elif j['mode'] == 'planted':
        digits = list(glyphs)
        rng.shuffle(digits)
        gstar = {i: digits[i] for i in range(10)}
        start = 0
        newc = []
        for c in cases:
            n = len(c['labels'])
            o = int(rng.integers(0, n))
            labs = []
            for i in range(n):
                k = 1 + (i + o) % n
                labs.append([gstar[int(ch)] for ch in str(k)])
            newc.append(dict(c, labels=labs))
        cases = newc
        plant = {v: k for k, v in gstar.items()}
        glyphs = sorted({g for c in cases for l in c['labels'] for g in l} | set(glyphs))
    c1 = [c for c in cases if stage(c['split']) == 1]
    c2 = [c for c in cases if stage(c['split']) == 2]
    hyps = []
    for h in range(j['N']):
        gv = {g: int(rng.integers(0, 10)) for g in glyphs}
        rule = ['sum', 'pos_lr', 'pos_rl'][int(rng.integers(0, 3))]
        z, obs = zod_stat(c1, gv, rule, rng, R=20)
        hyps.append((z, gv, rule, obs))
        if h % 2000 == 0:
            log(f'  zod {j["mode"]} {h}')
    N = len(hyps)
    thr1 = bonf_z(N)
    surv = sorted([h for h in hyps if h[0] > thr1], key=lambda x: -x[0])
    tested = surv[:200]
    thr2 = bonf_z(len(tested))
    rep = []
    for z1, gv, rule, obs in tested:
        z2, obs2 = zod_stat(c2, gv, rule, rng, R=200)
        r = {'z1': z1, 'z2': z2, 'rule': rule, 'gv': gv, 'replicated': bool(z2 > thr2)}
        if plant:
            r['digit_match'] = sum(gv.get(g) == d for g, d in plant.items()) / len(plant)
        rep.append(r)
    out = {'N': N, 'n_cases1': len(c1), 'n_cases2': len(c2), 'thr1': thr1, 'n_stage1': len(surv),
           'n_tested2': len(tested), 'thr2': thr2, 'n_replicated': sum(r['replicated'] for r in rep),
           'z_quantiles': np.quantile([h[0] for h in hyps], [.5, .9, .99, 1]).tolist(), 'stage2': rep[:50]}
    if plant:
        out['plant'] = plant
        out['recovered'] = any(r['replicated'] for r in rep)
    return out


def shuffle_b_pe(cases, rnd):
    by = defaultdict(list)
    for i, c in enumerate(cases):
        by[(c['cls'], len(c['pairs']))].append(i)
    new = [dict(c) for c in cases]
    for k, ix in by.items():
        perm = list(ix)
        rnd.shuffle(perm)
        for i, j in zip(ix, perm):
            new[i]['pairs'] = [(a[0], b[1], a[2]) for a, b in zip(cases[i]['pairs'], cases[j]['pairs'])]
    return new


def plant_b_pe(cases, codes, rng):
    vstar = {c: GRID_PE[int(rng.integers(0, len(GRID_PE)))] for c in codes}
    vstar['N01'] = 1.0
    out = []
    for c in cases:
        pairs, ok = [], True
        for sm, tt, cls in c['pairs']:
            val = sum(vstar[k] * n for k, n in sm.items())
            use = sorted(set(tt) | set(sm), key=lambda k: -vstar[k])
            rem, tot = val, {}
            for k in use:
                q = int(rem // vstar[k] + 1e-9)
                if q:
                    tot[k] = q; rem -= q * vstar[k]
            if abs(rem) > 1e-9:
                ok = False
                break
            pairs.append((sm, tot, cls))
        if ok:
            out.append(dict(c, pairs=pairs))
    return out, vstar


def run_b_generic(j, cases, codes, kind, rng, plant, log):
    c1 = [c for c in cases if stage(c['split']) == 1]
    c2 = [c for c in cases if stage(c['split']) == 2]
    grid = GRID_PE if kind == 'pe' else GRID_LA
    hyps = []
    for h in range(j['N']):
        val = {k: grid[int(rng.integers(0, len(grid)))] for k in codes}
        if kind == 'pe':
            val['N01'] = 1.0
        x, n, p0, p = score_b(c1, val, kind, rng, R=10)
        hyps.append((p, x, n, p0, val))
        if h % 2000 == 0:
            log(f'  b {kind} {j["mode"]} {h}')
    N = len(hyps)
    surv = sorted([h for h in hyps if h[0] < ALPHA / N], key=lambda x: x[0])
    tested = surv[:200]
    rep = []
    for p, x, n, p0, val in tested:
        x2, n2, p02, p2 = score_b(c2, val, kind, rng, R=40)
        r = {'p1': p, 'x1': x, 'n1': n, 'p0_1': p0, 'x2': x2, 'n2': n2, 'p0_2': p02, 'p2': p2,
             'replicated': bool(p2 < ALPHA / max(len(tested), 1)), 'val': val}
        if kind == 'pe':
            r['N14'] = val.get('N14')
        rep.append(r)
    out = {'N': N, 'n_cases1': len(c1), 'n_cases2': len(c2), 'n_stage1': len(surv), 'n_tested2': len(tested),
           'n_replicated': sum(r['replicated'] for r in rep),
           'p_quantiles': np.quantile([h[0] for h in hyps], [0, .01, .1, .5]).tolist(), 'stage2': rep[:60]}
    if kind == 'pe':
        reps = [r for r in rep if r['replicated']]
        out['replicated_with_N14_eq_10'] = sum(abs(r['val'].get('N14', 0) - 10) < 1e-9 for r in reps)
        allx = [h for h in hyps if abs(h[4].get('N14', 0) - 10) < 1e-9]
        out['stage1_mean_x_N14_10'] = float(np.mean([h[1] for h in allx])) if allx else None
        out['stage1_mean_x_other'] = float(np.mean([h[1] for h in hyps if h not in allx]))
    if plant:
        out['plant'] = plant
        # recovered: a replicated survivor equals the planted values on every code used in >= 3 test cases
        use = Counter()
        for c in c2:
            if kind == 'pe':
                for sm, tt, _ in c['pairs']:
                    use.update(set(sm) | set(tt))
            else:
                use.update(c['coef'])
        key = [k for k, n in use.items() if n >= 3 and k != 'N01']
        out['plant_key_codes'] = key
        out['recovered'] = any(r['replicated'] and all(abs(r['val'][k] - plant[k]) < 1e-9 for k in key) for r in rep)
    return out


# =====================================================================================
# (e) cross-script pairings with matched shape, scored by context similarity
# =====================================================================================
def shape(w):
    m, p = {}, []
    for g in w:
        m.setdefault(g, len(m)); p.append(m[g])
    return (len(w), tuple(p))


def occurrences(script, corpus):
    """Occurrence records: (type, split, relpos, firstline, lastline, lineinit, fnum, doclen)."""
    occ = []
    if script == 'voynich':
        for d in corpus['docs']:
            nl = len(d['words'])
            for li, lw in enumerate(d['words']):
                for wi, w in enumerate(lw):
                    occ.append((w, d['split'], li / max(nl - 1, 1), li == 0, li == nl - 1, wi == 0,
                                d['para_start'][li], nl))
    elif script == 'linear_a':
        for d in corpus['docs']:
            nl = len(d['words_ctx'])
            for li, lw in enumerate(d['words_ctx']):
                for wi, (w, nt, nv) in enumerate(lw):
                    occ.append((w, d['split'], li / max(nl - 1, 1), li == 0, li == nl - 1, wi == 0,
                                nt == 'num', nl))
    else:
        for d in corpus['docs']:
            nl = len(d['lines'])
            for e in d['entries']:
                mid = tuple(s for s in e['signs'][:-1])
                if not mid or '?' in mid:
                    continue
                li = e['line']
                occ.append((mid, d['split'], li / max(nl - 1, 1), li == 1, li == nl - 1, True,
                            len(e['signs']) > 2, nl))
    return occ


def profiles(occ, types, st):
    agg = defaultdict(list)
    for o in occ:
        if stage(o[1]) == st and o[0] in types:
            agg[o[0]].append(o[2:])
    prof = {}
    for t, rows in agg.items():
        a = np.array(rows, float)
        prof[t] = a.mean(0)
    # percentile-rank each feature within the script so scripts are comparable
    ts = sorted(prof)
    M = np.array([prof[t] for t in ts])
    R = np.zeros_like(M)
    for k in range(M.shape[1]):
        col = M[:, k]
        R[:, k] = (np.argsort(np.argsort(col, kind='stable'), kind='stable') + 0.5) / len(col)
        # ties: average rank
        for v in np.unique(col):
            m = col == v
            R[m, k] = R[m, k].mean()
    return {t: R[i] for i, t in enumerate(ts)}


def plant_e(occs, rng, pair, noise=0.3):
    """Insert 3 synthetic length-2 word types into two scripts with the same context profile.
    Profiles sit at the top of most features (so they are comparable after per-script percentile
    ranking); with probability `noise` an occurrence copies a random real occurrence's context."""
    planted = []
    for k in range(3):
        for s in pair:
            occ = occs[s]
            dl = [o[7] for o in occ]
            prof = [(1.0, True, True, True, True, max(dl)), (0.0, True, True, True, True, max(dl)),
                    (1.0, True, True, True, True, min(dl))][k]
            t = (f'P{k}a', f'P{k}b')
            for i in rng.choice(len(occ), 40, replace=False):
                o = occ[int(i)]
                if rng.random() < noise:
                    r = occ[int(rng.integers(0, len(occ)))]
                    occ.append((t, o[1]) + tuple(r[2:]))
                else:
                    occ.append((t, o[1]) + prof)
        planted.append(((pair[0], (f'P{k}a', f'P{k}b')), (pair[1], (f'P{k}a', f'P{k}b'))))
    return planted


def run_e(j, log):
    """Script = 'voynich'/'linear_a'/'proto_elamite' names the job slot only; every job pairs all three scripts."""
    rng = np.random.default_rng(5000 + j['seed'] + 7 * ['voynich', 'linear_a', 'proto_elamite'].index(j['script']))
    names = ['linear_a', 'proto_elamite', 'voynich']
    occs = {s: occurrences(s, L.load(s)) for s in names}
    planted = None
    if j['mode'] == 'shuffle':
        for s in names:
            types = [o[0] for o in occs[s]]
            random.Random(j['seed']).shuffle(types)
            occs[s] = [(t,) + o[1:] for t, o in zip(types, occs[s])]
    elif j['mode'] == 'planted':
        pair = [names[i] for i in rng.choice(3, 2, replace=False)]
        planted = plant_e(occs, rng, pair)
    P1, P2 = {}, {}
    for s in names:
        c1 = Counter(o[0] for o in occs[s] if stage(o[1]) == 1)
        c2 = Counter(o[0] for o in occs[s] if stage(o[1]) == 2)
        types = {t for t in c1 if c1[t] >= 4 and c2[t] >= 2}
        P1[s] = profiles(occs[s], types, 1)
        P2[s] = profiles(occs[s], types, 2)
    # candidate pairs: same shape, different scripts
    pairs = []
    for a in range(3):
        for b in range(a + 1, 3):
            sa, sb = names[a], names[b]
            bysh = defaultdict(list)
            for t in P1[sb]:
                bysh[shape(t)].append(t)
            for t in P1[sa]:
                for u in bysh.get(shape(t), []):
                    pairs.append((sa, t, sb, u))
    rng.shuffle(pairs)
    pairs = pairs[:j['N']]
    if planted:
        for (sa, ta), (sb, tb) in planted:
            if (sa, ta, sb, tb) not in pairs and (sb, tb, sa, ta) not in pairs:
                pairs.append((sa, ta, sb, tb))

    def dist(P, p):
        return float(np.linalg.norm(P[p[0]][p[1]] - P[p[2]][p[3]]))

    def pool_null(P, p, K=4000):
        # matched null: same script pair, same shape, random partners
        return null_cache.setdefault((id(P), p[0], p[2], shape(p[1])), _null(P, p, K))

    def _null(P, p, K):
        A = [t for t in P[p[0]] if shape(t) == shape(p[1])]
        B = [t for t in P[p[2]] if shape(t) == shape(p[3])]
        ia = rng.integers(0, len(A), K); ib = rng.integers(0, len(B), K)
        return np.sort([np.linalg.norm(P[p[0]][A[i]] - P[p[2]][B[k]]) for i, k in zip(ia, ib)])

    null_cache = {}
    sc = []
    for p in pairs:
        d1 = dist(P1, p)
        nl = pool_null(P1, p)
        r1 = (np.searchsorted(nl, d1, side='right') + 1) / (len(nl) + 1)
        sc.append((r1, d1, p))
    sc.sort(key=lambda x: x[0])
    K = 100
    tested = sc[:K]
    rep = []
    for r1, d1, p in tested:
        d2 = dist(P2, p)
        nl = pool_null(P2, p)
        r2 = (np.searchsorted(nl, d2, side='right') + 1) / (len(nl) + 1)
        rep.append({'pair': [p[0], '-'.join(p[1]), p[2], '-'.join(p[3])], 'rank1': r1, 'd1': d1, 'rank2': float(r2),
                    'd2': d2, 'replicated': bool(r2 < ALPHA / K)})
    out = {'N': len(pairs), 'n_types': {s: len(P1[s]) for s in names}, 'stage1': 'top-100 by matched-null rank',
           'n_stage1': len(tested), 'n_tested2': len(tested), 'thr2_rank': ALPHA / K,
           'n_replicated': sum(r['replicated'] for r in rep), 'min_rank1': float(sc[0][0]) if sc else None,
           'stage2': rep}
    if planted:
        hit = [r for r in rep if r['replicated'] and r['pair'][1].startswith('P') and r['pair'][1][1].isdigit()
               and r['pair'][3].startswith('P') and r['pair'][3][1].isdigit()]
        out['planted_pairs'] = [[a[0], b[0], '-'.join(a[1])] for a, b in planted]
        out['n_planted_recovered'] = len(hit)
        out['recovered'] = len(hit) > 0
        out['n_replicated_not_planted'] = out['n_replicated'] - len(hit)
    return out


def run(j, log):
    return {'b': run_b, 'd': run_d, 'e': run_e}[j['fam']](j, log)
