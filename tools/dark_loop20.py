"""S-DARK-20: ALLOGRAPH CLOCK. Do sign-form variants (Wells forms merged in S268, seq_raw vs merged form) drift with
depth / period within Mohenjo-daro and within Harappa, in the same direction? If so, build a clock from the variants,
test it out of sample, and apply it to undated texts.
Note: data/raw/inscriptions.csv carries no ~a/~b variant letters (0 of 5,680 text fields); the only variant structure
in the data is the S268 merge classes (data/derived/sign_allographs_levels.json) plus the S-DARK-11.3 candidate pairs.
Usage: python3 tools/dark_loop20.py CYCLE   (1 drift test; 2 clock; 3 clock on undated texts; 4 by object class)
"""
import json, csv, re, sys, random, collections, math
import numpy as np
CYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TYPEF = next((a[5:] for a in sys.argv if a.startswith('type=')), None)
CONTROL = 'control' in sys.argv
CSEED = next((int(a[5:]) for a in sys.argv if a.startswith('seed=')), 7)
NPERM = 2000
rng = np.random.default_rng(20 + CYCLE)
OUT = open(f'data/derived/dark/loop20_cycle{CYCLE}{"_"+TYPEF if TYPEF else ""}{"_control"+str(CSEED) if CONTROL else ""}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

C = json.load(open('data/derived/merged-corpus-canonical.json'))
raw = {r['cisi']: r for r in csv.DictReader(open('data/raw/inscriptions.csv')) if r['cisi']}
LEV = json.load(open('data/derived/sign_allographs_levels.json'))['merges']
# variant sets: canonical -> [forms], with level
SETS = collections.OrderedDict()
for m in LEV:
    SETS.setdefault(m['into'], {'forms': [m['into']], 'level': {}})
    SETS[m['into']]['forms'].append(m['form']); SETS[m['into']]['level'][m['form']] = m['level']
# S-DARK-11.3 allograph-like candidate pairs, not merged (extra sets, labelled 'cand')
for a, b in [(384, 388), (27, 28), (525, 526), (850, 851), (390, 407)]:
    key = f'{a}x{b}'
    SETS[key] = {'forms': [a, b], 'level': {b: 'cand'}, 'cand': True}

FINE = {'Period 2': 0, 'Period 3B-1': 1, 'Period 3B-2': 2, 'Period 3C-1': 4, 'Period 3C-2': 5, 'Period 3C-3': 6,
        'Period 3C-4': 7, 'Period 4': 8, 'Period 5A': 9}
COARSE = {'Early': 0, 'Intermediate': 1, 'Late': 2, 'Late ': 2}   # Mohenjo-daro Marshall/Mackay levels
def depth_ft(s):
    m = re.match(r'^-([\d.]+)\s*ft', s, re.I)
    if not m: return None
    try: return float(m.group(1).replace('..', '.'))
    except ValueError: return None

OBJ = []
for r in C:
    if not r.get('seq_raw'): continue
    if r['site'] not in ('Mohenjo-daro', 'Harappa'): continue
    x = raw.get(r['cisi'], {})
    t = r['type'].split(':')[0]
    if TYPEF and t != TYPEF: continue
    tm = x.get('time', '')
    OBJ.append(dict(cisi=r['cisi'], site=r['site'], area=r.get('area-section') or '--', type=r['type'], t0=t,
                    depth=depth_ft(x.get('depth', '')), fine=FINE.get(tm),
                    coarse=COARSE.get(x.get('period', '')) if r['site'] == 'Mohenjo-daro' else FINE.get(tm),
                    seq=r['seq_raw'], seq_all=r['seq_all']))
# time axis per city: 'depth' (ft below datum, larger = older) and 'period' (ordinal; MD = Early/Int/Late when fine is
# missing, HP = HARP fine period). Both oriented so larger = LATER for the report: age = -depth, period as given.
for o in OBJ:
    o['age'] = -o['depth'] if o['depth'] is not None else None
    o['per'] = o['fine'] if o['fine'] is not None else o['coarse']
if CONTROL:  # whole-machine control: shuffle time labels within site x area x type
    g = collections.defaultdict(list)
    for o in OBJ: g[(o['site'], o['area'], o['type'])].append(o)
    for os_ in g.values():
        for key in ('age', 'per'):
            vals = [o[key] for o in os_]; random.Random(CSEED * 1000 + len(os_)).shuffle(vals)
            for o, v in zip(os_, vals): o[key] = v

def strata(objs):
    g = collections.defaultdict(list)
    for i, o in enumerate(objs): g[(o['area'], o['type'])].append(i)
    return [np.array(v) for v in g.values()]

def rankavg(x):
    x = np.asarray(x, float); n = len(x)
    order = np.argsort(x, kind='mergesort'); r = np.empty(n); sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(a, b):
    ra, rb = rankavg(a), rankavg(b)
    if ra.std() == 0 or rb.std() == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])

def perm_time(objs, key, strat):
    """return a function that yields permuted time vectors (shuffled within stratum, texts without time stay None)."""
    t = np.array([np.nan if o[key] is None else o[key] for o in objs])
    def gen():
        tp = t.copy()
        for idx in strat:
            sub = idx[~np.isnan(t[idx])]
            if len(sub) > 1: tp[sub] = t[sub][rng.permutation(len(sub))]
        return tp
    return t, gen

# ---------------- cycle 1: drift of each variant form vs time, per city ----------------
def tokens_for_set(objs, forms):
    """token list: (text index, form) for every token of any form in the set (seq_raw)."""
    fs = set(forms); tok = []
    for i, o in enumerate(objs):
        for s in o['seq']:
            if s in fs: tok.append((i, s))
    return tok

def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps); q = np.empty(n); prev = 1.0
    for k in range(n - 1, -1, -1):
        prev = min(prev, ps[o[k]] * n / (k + 1)); q[o[k]] = prev
    return q

def cycle1():
    say(f'# S-DARK-20.1 drift test: variant form share vs time, per city; token level, permutation {NPERM}x of time labels '
        f'across texts within site x area x object type; Spearman(form indicator, time residualised on stratum mean); larger = later '
        f'(age = -depth ft; period = HARP fine period at Harappa, Marshall Early/Int/Late at Mohenjo-daro when fine missing)')
    say(f'# texts: MD {sum(o["site"]=="Mohenjo-daro" for o in OBJ)}, HP {sum(o["site"]=="Harappa" for o in OBJ)}; '
        f'with depth MD {sum(o["site"]=="Mohenjo-daro" and o["age"] is not None for o in OBJ)} HP {sum(o["site"]=="Harappa" and o["age"] is not None for o in OBJ)}; '
        f'with period MD {sum(o["site"]=="Mohenjo-daro" and o["per"] is not None for o in OBJ)} HP {sum(o["site"]=="Harappa" and o["per"] is not None for o in OBJ)}')
    results = {}  # (set, form, city, key) -> (rho, p, n, nminor)
    for city in ('Mohenjo-daro', 'Harappa'):
        objs = [o for o in OBJ if o['site'] == city]; strat = strata(objs)
        for key in ('age', 'per'):
            t, gen = perm_time(objs, key, strat)
            t = residualize(objs, t, strat)   # within-stratum deviation, so the null is centred on 0
            arrows = []
            for sk, S in SETS.items():
                tok = tokens_for_set(objs, S['forms'])
                tok = [(i, s) for i, s in tok if not np.isnan(t[i])]
                if len(tok) < 30: continue
                ti = np.array([i for i, _ in tok]); forms = np.array([s for _, s in tok])
                for f in S['forms'][1:]:
                    y = (forms == f).astype(float)
                    if y.sum() < 5 or (len(y) - y.sum()) < 5: continue
                    arrows.append((sk, f, ti, y))
            if not arrows: continue
            # permutations: shared across arrows for speed
            obs = [spearman(y, t[ti]) for _, _, ti, y in arrows]
            cnt = np.zeros(len(arrows))
            for _ in range(NPERM):
                tp = gen()
                for k, (_, _, ti, y) in enumerate(arrows):
                    if abs(spearman(y, tp[ti])) >= abs(obs[k]) - 1e-12: cnt[k] += 1
            ps = (cnt + 1) / (NPERM + 1); qs = bh(ps)
            say(f'\n## {city} vs {"depth (age)" if key=="age" else "period"}: {len(arrows)} arrows')
            say('set\tform\tlevel\tn_tok\tn_form\trho(form,later)\tp\tq_BH')
            for k, (sk, f, ti, y) in enumerate(arrows):
                lev = SETS[sk]['level'].get(f, 'strong')
                say(f'{sk}\t{f}\t{lev}\t{len(y)}\t{int(y.sum())}\t{obs[k]:+.3f}\t{ps[k]:.4f}\t{qs[k]:.3f}')
                results[(sk, f, city, key)] = (obs[k], ps[k], qs[k], len(y), int(y.sum()))
            say(f'survivors q<0.1: {int((qs<0.1).sum())} of {len(arrows)}; p<0.05: {int((ps<0.05).sum())} (expected {0.05*len(arrows):.1f})')
    # replication across the two cities
    say('\n## Cross-city agreement (sets testable in both cities)')
    for key in ('age', 'per'):
        pairs = [(sk, f) for (sk, f, c, k) in results if c == 'Mohenjo-daro' and k == key and (sk, f, 'Harappa', key) in results]
        same = 0; both_q = 0; both_p05 = 0
        say(f'\n### {"depth" if key=="age" else "period"}: {len(pairs)} shared arrows')
        say('set\tform\tMD rho\tMD p\tHP rho\tHP p\tagree')
        for sk, f in pairs:
            a = results[(sk, f, 'Mohenjo-daro', key)]; b = results[(sk, f, 'Harappa', key)]
            ag = (a[0] > 0) == (b[0] > 0) and a[0] != 0 and b[0] != 0
            same += ag
            if a[2] < 0.1 and b[2] < 0.1 and ag: both_q += 1
            if a[1] < 0.05 and b[1] < 0.05 and ag: both_p05 += 1
            say(f'{sk}\t{f}\t{a[0]:+.3f}\t{a[1]:.3f}\t{b[0]:+.3f}\t{b[1]:.3f}\t{"same" if ag else "opp"}')
        if pairs:
            from math import comb
            n = len(pairs); pbin = sum(comb(n, k) for k in range(same, n + 1)) / 2 ** n
            say(f'same-direction {same} of {n} (binomial P(>= {same}) = {pbin:.3f}); both q<0.1 same sign: {both_q}; both p<0.05 same sign: {both_p05} '
                f'(expected under null ~{n*0.05*0.05/2:.2f})')
    return results

# ---------------- cycle 2: the clock ----------------
def features(objs, sets):
    """per text: for each set, +1 if a non-canonical form occurs, -1 if only the canonical form, 0 if absent."""
    X = np.zeros((len(objs), len(sets)))
    for j, sk in enumerate(sets):
        S = SETS[sk]; can = S['forms'][0]; var = set(S['forms'][1:])
        for i, o in enumerate(objs):
            has_c = can in o['seq']; has_v = any(s in var for s in o['seq'])
            X[i, j] = 1 if has_v else (-1 if has_c else 0)
    return X

def residualize(objs, t, strat):
    """time minus stratum mean (within site x area x type), NaN kept."""
    r = t.copy()
    for idx in strat:
        sub = idx[~np.isnan(t[idx])]
        if len(sub): r[sub] = t[sub] - t[sub].mean()
    return r

def ridge_fit(X, y, lam=1.0):
    Xc = np.c_[X, np.ones(len(X))]
    A = Xc.T @ Xc + lam * np.eye(Xc.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, Xc.T @ y)
def ridge_pred(X, w): return np.c_[X, np.ones(len(X))] @ w

def clock_eval(objs, X, t, strat, label, nperm=500, folds=5):
    """k-fold within-city: Spearman(predicted, residual time) on held-out; null = features permuted across texts
    within stratum (shuffled-variant clock), same folds."""
    ok = ~np.isnan(t); idx = np.where(ok)[0]
    r = residualize(objs, t, strat)
    def cv_rho(Xm):
        pred = np.full(len(objs), np.nan); perm = rng.permutation(idx); fold = np.array_split(perm, folds)
        for k in range(folds):
            te = fold[k]; tr = np.concatenate([fold[j] for j in range(folds) if j != k])
            w = ridge_fit(Xm[tr], r[tr]); pred[te] = ridge_pred(Xm[te], w)
        return spearman(pred[idx], r[idx])
    obs = cv_rho(X)
    null = []
    for _ in range(nperm):
        Xp = X.copy()
        for s in strat:
            sub = s[ok[s]]
            if len(sub) > 1: Xp[sub] = X[sub][rng.permutation(len(sub))]
        null.append(cv_rho(Xp))
    null = np.array(null); p = (np.sum(null >= obs) + 1) / (nperm + 1)
    say(f'{label}: n={len(idx)} held-out Spearman(pred, residual time) = {obs:+.3f}; shuffled-variant clock mean {null.mean():+.3f} sd {null.std():.3f}; P = {p:.3f}')
    return obs, p

def cross_city(objsA, XA, tA, stratA, objsB, XB, tB, stratB, label):
    rA = residualize(objsA, tA, stratA); rB = residualize(objsB, tB, stratB)
    okA = ~np.isnan(tA); okB = ~np.isnan(tB)
    w = ridge_fit(XA[okA], rA[okA]); pred = ridge_pred(XB[okB], w)
    obs = spearman(pred, rB[okB])
    # null: permute B's features within stratum
    null = []
    for _ in range(500):
        Xp = XB.copy()
        for s in stratB:
            sub = s[okB[s]]
            if len(sub) > 1: Xp[sub] = XB[sub][rng.permutation(len(sub))]
        null.append(spearman(ridge_pred(Xp[okB], w), rB[okB]))
    null = np.array(null); p = (np.sum(null >= obs) + 1) / 501
    say(f'{label}: train n={okA.sum()} test n={okB.sum()}; Spearman(pred, residual time) = {obs:+.3f}; null mean {null.mean():+.3f} sd {null.std():.3f}; P = {p:.3f}')
    return w, obs, p

def cycle2():
    say(f'# S-DARK-20.2 allograph clock: per text, one feature per variant set (+1 variant form present, -1 canonical only, 0 absent), '
        f'{len(SETS)} sets; ridge regression (lambda 1) to stratum-residualised time (site x area x object type); '
        f'5-fold CV within city and train-one-city/test-the-other; control = shuffled-variant clock (features permuted within stratum, 500x)')
    sets = list(SETS)
    say('sets:', ' '.join(f'{k}:{"/".join(map(str,SETS[k]["forms"]))}' for k in sets))
    data = {}
    for city in ('Mohenjo-daro', 'Harappa'):
        objs = [o for o in OBJ if o['site'] == city]; strat = strata(objs); X = features(objs, sets)
        data[city] = (objs, X, strat)
        for key in ('age', 'per'):
            t = np.array([np.nan if o[key] is None else o[key] for o in objs])
            clock_eval(objs, X, t, strat, f'{city} vs {"depth" if key=="age" else "period"} (5-fold CV)')
    for key in ('age', 'per'):
        for A, B in (('Mohenjo-daro', 'Harappa'), ('Harappa', 'Mohenjo-daro')):
            oA, XA, sA = data[A]; oB, XB, sB = data[B]
            tA = np.array([np.nan if o[key] is None else o[key] for o in oA]); tB = np.array([np.nan if o[key] is None else o[key] for o in oB])
            w, obs, p = cross_city(oA, XA, tA, sA, oB, XB, tB, sB, f'train {A} -> test {B} vs {"depth" if key=="age" else "period"}')
            if key == 'age':
                say('   weights (later = +): ' + ' '.join(f'{sets[j]}:{w[j]:+.2f}' for j in np.argsort(-np.abs(w[:-1]))[:8]))
    # pooled two-city clock, out-of-fold predictions saved for cycle 3
    objs = [o for o in OBJ]; strat = strata(objs)
    # strata must be within site too
    g = collections.defaultdict(list)
    for i, o in enumerate(objs): g[(o['site'], o['area'], o['type'])].append(i)
    strat = [np.array(v) for v in g.values()]
    X = features(objs, sets)
    t = np.array([np.nan if o['age'] is None else o['age'] for o in objs])
    clock_eval(objs, X, t, strat, 'both cities pooled vs depth (5-fold CV)')
    r = residualize(objs, t, strat); ok = ~np.isnan(t)
    w = ridge_fit(X[ok], r[ok]); score = ridge_pred(X, w)
    json.dump({'sets': sets, 'weights': w.tolist(), 'scores': {o['cisi']: float(s) for o, s in zip(objs, score)}},
              open('data/derived/dark/loop20_clock.json', 'w'))
    say('pooled clock saved: data/derived/dark/loop20_clock.json; weights: ' + ' '.join(f'{sets[j]}:{w[j]:+.2f}' for j in range(len(sets))))

# ---------------- cycle 3: clock applied to undated texts ----------------
def cycle3():
    say('# S-DARK-20.3 clock applied to UNDATED texts (no depth): does W390 quota, text length, closer mix or tablet reuse move along the '
        'clock score? Stratified permutation (site x area x object type) of the clock score across texts, 2000x; the W390 arrow uses a '
        'clock refitted WITHOUT the 390 set and the 390x407 pair (the clock would otherwise contain its own outcome); length arrows '
        'also reported against a per-token clock (score / number of set tokens) since more signs = more variant tokens')
    ck = json.load(open('data/derived/dark/loop20_clock.json')); sets = ck['sets']
    objs = [o for o in OBJ]
    g = collections.defaultdict(list)
    for i, o in enumerate(objs): g[(o['site'], o['area'], o['type'])].append(i)
    strat = [np.array(v) for v in g.values()]
    X = features(objs, sets); t = np.array([np.nan if o['age'] is None else o['age'] for o in objs])
    r = residualize(objs, t, strat); ok = ~np.isnan(t)
    w = ridge_fit(X[ok], r[ok]); score = ridge_pred(X, w)
    no390 = [j for j, s in enumerate(sets) if s not in (390, '390x407')]
    w2 = ridge_fit(X[ok][:, no390], r[ok]); score390 = ridge_pred(X[:, no390], w2)
    ntok = np.abs(X).sum(1)
    undated = np.array([o['age'] is None for o in objs]) & (ntok > 0)
    say(f'undated texts with at least one variant-set token: {undated.sum()} (MD {sum(undated[i] for i,o in enumerate(objs) if o["site"]=="Mohenjo-daro")}, HP {sum(undated[i] for i,o in enumerate(objs) if o["site"]=="Harappa")})')
    MIDCOUNT = collections.Counter(tuple(o['seq']) for o in C_all())
    arrows = {
        'w390_count': lambda o: sum(x in (390, 405, 406, 407) for x in o['seq']),
        'len': lambda o: len(o['seq']),
        'jar_final': lambda o: int(o['seq'][-1] == 740),
        'arrow_final': lambda o: int(o['seq'][-1] == 520),
        'closer_other': lambda o: int(o['seq'][-1] not in (740, 520, 390, 405, 400)),
        'tablet_text_reuse_log': lambda o: math.log(MIDCOUNT[tuple(o['seq'])]) if o['t0'] == 'TAB' else None,
    }
    for sub_name, mask in (('undated', undated), ('dated (sanity: clock was fitted on these)', ok & (ntok > 0))):
        say(f'\n## {sub_name} texts, n = {mask.sum()}')
        say('arrow\tclock\tn\trho\tp_perm')
        for name, fn in arrows.items():
            for cname, sc in (('full', score), ('no390', score390), ('per-token', score / np.maximum(ntok, 1))):
                if name == 'w390_count' and cname == 'full': continue
                if name != 'w390_count' and cname == 'no390': continue
                if name not in ('len',) and cname == 'per-token': continue
                vals = np.array([np.nan if fn(o) is None else fn(o) for o in objs], float)
                m = mask & ~np.isnan(vals)
                if m.sum() < 30: continue
                obs = spearman(sc[m], vals[m]); cnt = 0
                for _ in range(NPERM):
                    sp = sc.copy()
                    for s in strat:
                        s2 = s[m[s]]
                        if len(s2) > 1: sp[s2] = sc[s2][rng.permutation(len(s2))]
                    if abs(spearman(sp[m], vals[m])) >= abs(obs) - 1e-12: cnt += 1
                say(f'{name}\t{cname}\t{m.sum()}\t{obs:+.3f}\t{(cnt+1)/(NPERM+1):.4f}')
def C_all():
    return [dict(seq=r['seq_raw']) for r in C if r.get('seq_raw')]

if CYCLE == 1: cycle1()
elif CYCLE == 2: cycle2()
elif CYCLE == 3: cycle3()
