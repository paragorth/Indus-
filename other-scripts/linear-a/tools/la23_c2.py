"""LA-23 cycle 2: count the people of the Hagia Triada tablets by capture-recapture.

Individuals = person-like word types (c1 probabilities) on HT tablets. Occasions:
tablets (frequency estimators: Chao1, coverage, Bayesian beta-binomial Mh), Villa vs Casa
(Chapman / Lincoln-Petersen), 4 findspots and 4 commodity classes (log-linear M0/Mt/Mh/Mth/Mt2).
Uncertainty: thousands of draws, each one (a) samples class membership from p (with its
ensemble sd), (b) applies name ambiguity (homonym split rate eta ~ U(0, .3); one-sign spelling
variants merged with nu ~ U(0, .5)), (c) (no tablet bootstrap: duplicated tablets fake recaptures); sampling
uncertainty comes from the Bayesian Mh posterior draw.
Controls: planted populations of known size on the HT tablet frame (heterogeneous, local,
with homonyms and variants); Linear B KN and PY LA-sized draws vs the full-site distinct count;
findspot labels permuted; non-person words through the same pipeline.
"""
import sys, json, itertools
from la23_common import *

MODE = sys.argv[1] if len(sys.argv) > 1 else 'all'
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rng = np.random.default_rng(seed('la23c2' + MODE))
C1 = json.load(open(os.path.join(CK, 'c1.json')))
P = {r['w']: (r['p'], r['sd']) for r in C1['la']}

HT = la_docs()
OCC = occurrences(HT)
SET = os.environ.get('LA23_SET', 'clf')


def strict_rule(occ):
    """Blind operational person-like rule: mostly in entry position, at least once followed directly
    (no logogram) by a number <= 5, and on <= 5 tablets (frequent repeated terms excluded)."""
    by = collections.defaultdict(list)
    for o in occ: by[o['w']].append(o)
    out = set()
    for w, L in by.items():
        ent = [x for x in L if x['entry']]
        if len(ent) >= 0.5 * len(L) and any(x['direct'] and x['num'] is not None and x['num'] <= 5 for x in ent) \
                and len(set(x['doc'] for x in L)) <= 5:
            out.add(w)
    return out


if SET == 'strict':
    _S = strict_rule(OCC)
    P = {w: (1.0 if w in _S else 0.0, 0.0) for w in set(o['w'] for o in OCC)}


def fs_group(f):
    if f.startswith('Casa'): return f
    if f in ('Villa Magazine', 'Portico 11 and Room 13', 'Corridor 9 and Vestibule 26'): return 'Villa Magazine'
    return None


def ct_group(c):
    if c in ('GRA',): return 'GRA'
    if c in ('OLE', 'OLIV'): return 'OLE'
    if c in ('VIR',): return 'VIR'
    return 'other'


FS4 = ['Villa Magazine', 'Casa Room 7', 'Casa del Lebete', 'Casa Room 9']
CT4 = ['GRA', 'OLE', 'VIR', 'other']
DOCF = [fs_group(d['findspot']) for d in HT]
DOCC = [ct_group(d['ctype']) for d in HT]
# per tablet: set of word types (2+ signs) present, any slot
TAB = [set() for _ in HT]
for o in OCC: TAB[o['doc']].add(o['w'])
WORDS = sorted(set(o['w'] for o in OCC))


def one_sign_pairs(words):
    out = []
    ws = [w.split('-') for w in words]
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i], ws[j]
            if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1 and len(a) >= 3:
                out.append((words[i], words[j]))
            elif abs(len(a) - len(b)) == 1 and min(len(a), len(b)) >= 2:
                s, l = (a, b) if len(a) < len(b) else (b, a)
                if any(l[:k] + l[k + 1:] == s for k in range(len(l))):
                    out.append((words[i], words[j]))
    return out


PAIRS = one_sign_pairs(WORDS)


def estimators(tabs, docf, docc, rng, mcmc=True):
    """tabs: list of sets of individual ids per tablet. Returns dict of estimates."""
    T = len(tabs)
    cnt = collections.Counter(x for t in tabs for x in t)
    S = len(cnt)
    r = dict(S=S, n=sum(cnt.values()))
    freqs = list(cnt.values())
    r['chao1'] = chao1(freqs) if S else 0
    r['cov'] = coverage(freqs) if S else 0
    r['q1'] = sum(1 for v in freqs if v == 1)
    # two lists Villa vs Casa
    A = set(x for t, f in zip(tabs, docf) if f == 'Villa Magazine' for x in t)
    B = set(x for t, f in zip(tabs, docf) if f and f.startswith('Casa') for x in t)
    r['villa'], r['casa'], r['both'] = len(A), len(B), len(A & B)
    r['chapman'] = chapman(len(A), len(B), len(A & B))
    for lab, groups, G in (('fs', docf, FS4), ('ct', docc, CT4)):
        sets = [set(x for t, g in zip(tabs, groups) if g == gg for x in t) for gg in G]
        ids = set().union(*sets)
        H = [tuple(int(x in s) for s in sets) for x in ids]
        ll = loglinear(H, len(G))
        for m, (N, aic, dev) in ll.items():
            r['%s_%s' % (lab, m)] = N; r['%s_%s_aic' % (lab, m)] = aic
        if ll:
            best = min(ll, key=lambda m: ll[m][1]); r['%s_best' % lab] = ll[best][0]; r['%s_bestm' % lab] = best
    if mcmc and S > 2:
        fk = fk_from_counts(freqs, T)
        dr = mh_bayes(fk, T, rng, n_iter=2500, burn=1000, thin=50)
        r['mh_bayes'] = float(rng.choice(dr))
        r['mh_bayes_med'] = float(np.median(dr))
    return r


def la_draw(rng, p_scale=1.0, eta=None, nu=None, boot=False, person=True, permute_fs=False):
    """One posterior/bootstrap draw of the HT person population."""
    memb = {}
    for w in WORDS:
        p, sd = P.get(w, (0.0, 0.0))
        pp = np.clip(p + rng.normal(0, sd), 0, 1) if sd > 0 else p
        isp = rng.random() < pp
        memb[w] = isp if person else (not isp)
    eta = rng.uniform(0, 0.3) if eta is None else eta
    nu = rng.uniform(0, 0.5) if nu is None else nu
    # spelling variants: union-find merge
    par = {w: w for w in WORDS}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in PAIRS:
        if memb[a] and memb[b] and rng.random() < nu:
            par[find(a)] = find(b)
    # homonyms: names seen on >1 tablet split into two people with prob eta (tablets assigned at random)
    cnt = collections.Counter(find(w) for t in TAB for w in t if memb[w])
    split = {k for k, v in cnt.items() if v > 1 and rng.random() < eta}
    idx = rng.integers(0, len(HT), len(HT)) if boot else np.arange(len(HT))
    tabs, docf, docc = [], [], []
    for j, i in enumerate(idx):
        s = set()
        for w in TAB[i]:
            if not memb[w]: continue
            k = find(w)
            if k in split: k = k + ('#a' if (zlib.crc32((k + str(i)).encode()) & 1) else '#b')
            s.add(k)
        # bootstrap duplicates of the same tablet are the same capture occasion content: tag copy
        tabs.append(s); docf.append(DOCF[i]); docc.append(DOCC[i])
    if permute_fs:
        perm = rng.permutation(len(docf)); docf = [docf[k] for k in perm]
    return tabs, docf, docc, dict(eta=eta, nu=nu)


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])], float)
        if len(v) == 0: continue
        out[k] = dict(med=float(np.median(v)), lo=float(np.percentile(v, 5)), hi=float(np.percentile(v, 95)), n=len(v))
    return out


KEYS = ['S', 'n', 'chao1', 'cov', 'q1', 'villa', 'casa', 'both', 'chapman', 'fs_M0', 'fs_Mt', 'fs_Mh', 'fs_Mth', 'fs_Mt2',
        'fs_best', 'ct_M0', 'ct_Mt', 'ct_Mh', 'ct_Mth', 'ct_best', 'mh_bayes', 'mh_bayes_med']


# ---------------------------------------------------------------- planted populations
def planted(N, rng, sigma=1.0, local=0.7, hom=0.15, var=0.1):
    """N people with lognormal capture weights and a home findspot; each HT tablet keeps its
    observed number of person slots (from the real data at p>=.5) and its findspot.
    Names: Zipf pool so some people share a name (rate ~hom); spelling variant: a capture is
    written under a variant name with prob var."""
    slots = [sum(1 for w in t if P.get(w, (0, 0))[0] >= 0.5) for t in TAB]
    w = np.exp(rng.normal(0, sigma, N))
    home = rng.choice(FS4, N, p=[0.5, 0.2, 0.2, 0.1])
    # names: each person gets a name; with prob hom reuse an existing person's name
    names = []
    for i in range(N):
        names.append(names[rng.integers(0, i)] if (i > 0 and rng.random() < hom) else 'n%d' % i)
    tabs = []
    for t, k in enumerate(slots):
        f = DOCF[t] or 'Villa Magazine'
        ww = w * np.where(home == f, local / max(1e-9, (home == f).mean()), (1 - local) / max(1e-9, (home != f).mean()))
        k = min(k, N)
        ch = rng.choice(N, k, replace=False, p=ww / ww.sum()) if k else []
        s = set()
        for c in ch:
            nm = names[c]
            if rng.random() < var: nm = nm + "'"
            s.add(nm)
        tabs.append(s)
    return tabs


def run_planted(nrep=40):
    out = {}
    for N in (150, 400, 1000, 3000):
        for cfg in ('easy', 'hard'):
            rows = []
            for rep in range(nrep):
                if cfg == 'easy':
                    tabs = planted(N, rng, sigma=0.3, local=0.25, hom=0.0, var=0.0)
                else:
                    tabs = planted(N, rng, sigma=1.2, local=0.7, hom=0.15, var=0.1)
                rows.append(estimators(tabs, DOCF, DOCC, rng))
            s = summarize(rows, KEYS)
            out['%d_%s' % (N, cfg)] = s
            print('planted', N, cfg, {k: round(s[k]['med']) for k in ('S', 'chao1', 'chapman', 'fs_best', 'ct_best', 'mh_bayes') if k in s}, flush=True)
    return out


# ---------------------------------------------------------------- Linear B control
def run_lb(nrep=60):
    LB = lb_docs(); L = lb_name_label()
    c1 = C1
    out = {}
    target = len(OCC)
    for site in ('KN', 'PY'):
        docs = [d for d in LB if d['site'] == site]
        ser = sorted(set(d['series'] for d in docs))
        full = set(v for d in docs for ln in d['lines'] for k, v in ln if k == 'W' and v in L)
        rows = []; prec = []
        for rep in range(nrep):
            idx = rng.permutation(len(docs)); sel, n = [], 0
            for i in idx:
                d = docs[i]; k = sum(1 for ln in d['lines'] for t, _ in ln if t == 'W')
                if k == 0: continue
                sel.append(d); n += k
                if n >= target: break
            tabs = [set(v for ln in d['lines'] for k, v in ln if k == 'W' and v in L) for d in sel]
            rule = strict_rule(occurrences(sel)); allw = set(v for d in sel for ln in d['lines'] for k, v in ln if k == 'W')
            prec.append((len([w for w in rule if w in L]) / max(1, len(rule)), len([w for w in allw if w in L]) / max(1, len(allw)),
                         len([w for w in rule if w in L]) / max(1, len([w for w in allw if w in L]))))
            # occasions: 'findspot' := hand groups (top 3 hands + rest), 'ctype' := series letter groups
            hands = collections.Counter(d['scribe'] for d in sel).most_common(3)
            hg = {h: i for i, (h, _) in enumerate(hands)}
            docf = [FS4[hg.get(d['scribe'], 3)] for d in sel]
            sl = collections.Counter(d['ctype'] for d in sel).most_common(3)
            cg = {h: i for i, (h, _) in enumerate(sl)}
            docc = [CT4[cg.get(d['ctype'], 3)] for d in sel]
            # Villa/Casa analogue: hand group 0 vs rest
            docf2 = ['Villa Magazine' if f == FS4[0] else 'Casa x' for f in docf]
            r = estimators(tabs, docf, docc, rng)
            r2 = estimators(tabs, docf2, docc, rng, mcmc=False)
            r['chapman'] = r2['chapman']
            rows.append(r)
        s = summarize(rows, KEYS)
        s['rule_precision_base_recall'] = np.mean(prec, 0).tolist()
        print('LB', site, 'strict rule precision, base rate, recall', s['rule_precision_base_recall'], flush=True)
        s['full_distinct'] = len(full); s['full_tablets'] = len(docs); s['draw_tablets'] = len(sel)
        out[site] = s
        print('LB', site, 'full distinct', len(full), {k: (round(s[k]['med']), round(s[k]['lo']), round(s[k]['hi'])) for k in ('S', 'chao1', 'chapman', 'fs_best', 'ct_best', 'mh_bayes') if k in s}, flush=True)
    return out


def run_la(ndraw):
    rows, rows_np, rows_perm, rows_fix = [], [], [], []
    for i in range(ndraw):
        tabs, docf, docc, par = la_draw(rng)
        r = estimators(tabs, docf, docc, rng); r.update(par); rows.append(r)
        if i % 4 == 0 and i < ndraw // 2:
            tabs, docf, docc, _ = la_draw(rng, person=False)
            rows_np.append(estimators(tabs, docf, docc, rng))
            tabs, docf, docc, _ = la_draw(rng, permute_fs=True)
            rows_perm.append(estimators(tabs, docf, docc, rng, mcmc=False))
            tabs, docf, docc, _ = la_draw(rng, eta=0.0, nu=0.0, boot=False)
            rows_fix.append(estimators(tabs, docf, docc, rng))
        if i % 250 == 249: print('la draw', i + 1, flush=True)
    return dict(person=summarize(rows, KEYS + ['eta', 'nu']), nonperson=summarize(rows_np, KEYS),
                fs_permuted=summarize(rows_perm, KEYS), no_ambiguity_noboot=summarize(rows_fix, KEYS),
                raw_eta_nu_N=[(r['eta'], r['nu'], r.get('mh_bayes'), r['chao1'], r['chapman']) for r in rows[:3000]])


if __name__ == '__main__':
    res = {}
    if MODE in ('all', 'planted'):
        res['planted'] = run_planted()
    if MODE in ('all', 'lb'):
        res['lb'] = run_lb()
    if MODE in ('all', 'la'):
        res['la'] = run_la(NDRAW)
        s = res['la']['person']
        print('LA person', {k: (round(s[k]['med'], 1), round(s[k]['lo'], 1), round(s[k]['hi'], 1)) for k in s})
        print('LA nonperson', {k: (round(v['med'], 1), round(v['lo'], 1), round(v['hi'], 1)) for k, v in res['la']['nonperson'].items()})
        print('LA fs-permuted', {k: (round(v['med'], 1), round(v['lo'], 1), round(v['hi'], 1)) for k, v in res['la']['fs_permuted'].items()})
    json.dump(res, open(os.path.join(CK, 'c2_%s_%s.json' % (MODE, SET)), 'w'), indent=1)
