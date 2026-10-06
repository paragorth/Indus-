#!/usr/bin/env python3
"""LA-67 cycle 3 (part 1, cheap): family-wise and universe corrections of two cycle-2 tests, and the
remaining existing-data kill tests: la51 room links, la23 Casa Room 9, la47 layout profiles (fresh seed,
1,000 permutations), la49 groups (proxy: count-based embeddings, duplicates and stone vessels held out)."""
from la67_lib import *
import la67_c2 as C2
OUT = os.path.join(LOOPS, 'la67_cycle3.txt')
NH = 20
RES = {}


def t_kaja(D):
    """family-wise: was KA/JA (the best of all sign pairs in la13) beyond the max over all pairs under relabeling?"""
    docs = D
    rng = random.Random(seed('kaja-fw'))
    tab_site = {d['tab']: d['site'] for d in docs}
    tabs = sorted(tab_site)
    nht = sum(1 for t in tabs if tab_site[t] == 'HT')
    def C_of(ht):
        g = defaultdict(set)
        for d in docs:
            l = 'HT' if d['tab'] in ht else 'X'
            for t in d['toks']:
                if t[0] == 'W' and len(t[1].split('-')) == 2:
                    g[l].add(t[1])
        return C2.subs_pairs(g['HT'], g['X'], 0)
    real = C_of(set(t for t in tabs if tab_site[t] == 'HT'))
    obs = real[('KA', 'JA')]
    mx = []; pairs_pass = Counter()
    nulls = defaultdict(list)
    for _ in range(300):
        u = tabs[:]; rng.shuffle(u)
        c = C_of(set(u[:nht]))
        mx.append(max(c.values()) if c else 0)
        for k in real:
            nulls[k].append(c.get(k, 0))
    p_fw = pct_rank(obs, mx)
    pass_rate = np.mean([pct_rank(v, nulls[k]) <= 0.05 for k, v in real.items()])
    RES['kaja'] = dict(obs=obs, max_null=float(np.mean(mx)), p_fw=p_fw, pass_rate=float(pass_rate), npairs=len(real))
    wlog(OUT, '| LA-67.3a | Family-wise check of LA-67.2c (la13 HT KA vs other JA was chosen as the best of all substitution rules): KA/JA count vs the MAXIMUM over all sign pairs '
              'under 300 fresh tablet relabelings; decoys = all %d observed first-slot pairs through the single-pair test | KA/JA %d; null maximum %.2f, family-wise P %.3f; decoy '
              'pairs passing the single-pair P <= 0.05: %.2f | %s |' % (len(real), obs, np.mean(mx), p_fw, pass_rate,
                                                                     'SURVIVES family-wise' if p_fw <= 0.05 else 'KILLED (the single-pair P 0.037 does not survive the selection; random relabelings give a pair this strong)'))


def t_figsolives(D):
    T = defaultdict(lambda: [set(), None])
    for d in admin(D):
        for t in d['toks']:
            if t[0] == 'L' or (t[0] == 'W' and t[1] == 'NI'):
                T[d['tab']][0].add(re.sub(r'\+.*', '', t[1]))
        T[d['tab']][1] = d['site']
    T = {t: v for t, v in T.items() if any(x != 'NI' for x in v[0]) or 'NI' in v[0]}
    T = {t: v for t, v in T.items() if v[0]}
    F = [t for t, v in T.items() if 'NI' in v[0]]; O = [t for t, v in T.items() if 'OLIV' in v[0]]
    obs = len(set(F) & set(O))
    rng = random.Random(seed('fo2'))
    bys = defaultdict(list)
    for t, v in T.items():
        bys[v[1]].append(t)
    null = []
    for _ in range(5000):
        f2 = set()
        for s, m in Counter(T[t][1] for t in F).items():
            f2 |= set(rng.sample(bys[s], m))
        null.append(len(f2 & set(O)))
    p = float((np.sum(np.array(null) <= obs) + 1) / 5001)
    # decoy pairs: every pair of commodity bases with >= 5 tablets each, same test (lower tail)
    bases = Counter(x for v in T.values() for x in v[0])
    cand = [b for b, c in bases.items() if c >= 5]
    dp = []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            Fa = [t for t, v in T.items() if a in v[0]]; Ob = set(t for t, v in T.items() if b in v[0])
            o = len(set(Fa) & Ob); nl = []
            for _ in range(300):
                f2 = set()
                for s, m in Counter(T[t][1] for t in Fa).items():
                    f2 |= set(rng.sample(bys[s], m))
                nl.append(len(f2 & Ob))
            dp.append((np.sum(np.array(nl) <= o) + 1) / 301)
    RES['fo'] = dict(obs=obs, null=float(np.mean(null)), p=p, dfalse=float(np.mean(np.array(dp) <= 0.05)), nd=len(dp))
    wlog(OUT, '| LA-67.3b | Corrected universe for LA-67.2k figs/olives (la50 C: NI and OLIV share fewer tablets than chance): universe = administrative tablets with a logogram or NI '
              '(%d); NI tablets re-dealt within site 5,000 times (fresh seed); decoys = all %d pairs of commodity bases with >= 5 tablets, same lower-tail test | NI & OLIV tablets %d vs %.2f, '
              'P(low) %.3f; decoy pairs with P(low) <= 0.05: %.2f | %s |' % (len(T), len(dp), obs, np.mean(null), p, np.mean(np.array(dp) <= 0.05),
                                                                         'SURVIVES' if p <= 0.05 else 'KILLED (no avoidance beyond within-site re-dealing)'))


def t_la51(D):
    import la51_common as M
    docs = [d for d in M.load_la() if d['site'] == 'Haghia Triada' and d['deposit'] in ('HT_VILLA_MAG', 'HT_CASA7', 'HT_CASA9', 'HT_CDL')]
    mag = np.array([d['deposit'] == 'HT_VILLA_MAG' for d in docs])
    terms = Counter(x for d in docs for x in d['terms'])
    tg = {'L:QA2': 1, 's:TE': 1, 's:NI': 0, 'L:CYP': 0, 's:O': 0, 'L:*86': 0, 'w:A-DU': 0}
    rng = np.random.default_rng(seed('la51'))
    def pv(t, idx, mg, want):
        has = np.array([t in docs[i]['terms'] for i in idx])
        if has.sum() == 0:
            return None, None
        o = (mg[has] == want).mean()
        nl = np.array([(rng.permutation(mg)[has] == want).mean() for _ in range(1000)])
        return o, float((np.sum(nl >= o) + 1) / 1001)
    allidx = np.arange(len(docs))
    out = {}
    pool = [t for t, c in terms.items() if c >= 3 and t not in tg]
    for t in list(tg) + pool:
        want = tg.get(t)
        if want is None:
            o1, p1 = pv(t, allidx, mag, 1); o0, p0 = pv(t, allidx, mag, 0)
            want, o, p = (1, o1, p1) if p1 <= p0 else (0, o0, p0)
            p = min(1, p * 2)
        else:
            o, p = pv(t, allidx, mag, want)
        rep = tested = 0
        for k in range(NH):
            r = random.Random(seed('la51-h%d' % k)); ids = list(range(len(docs))); r.shuffle(ids)
            for h in (ids[:len(ids) // 2], ids[len(ids) // 2:]):
                h = np.array(sorted(h)); has = np.array([t in docs[i]['terms'] for i in h])
                if has.sum() == 0:
                    continue
                tested += 1; rep += (mag[h][has] == want).mean() > (mag[h] == want).mean()
        out[t] = dict(n=int(terms[t]), want='MAG' if want else 'HOUSE', share=o, p=p, rep=rep, tested=tested,
                      surv=bool(p is not None and p <= 0.05 and tested and rep / tested >= 0.8))
    dfalse = float(np.mean([out[t]['surv'] for t in pool]))
    RES['la51'] = out
    wlog(OUT, '| LA-67.3c | la51 C: QA2 and TE go with the HT Villa magazines; NI, CYP, O, *86 and A-DU with the houses (Casa 7, Casa 9, Casa del Lebete). Kill line: new finds in the '
              'other kind of room. Existing-data test: HT documents in these deposits (%d; %d in magazines); share of the term\'s documents in the predicted kind vs 1,000 permutations '
              'of deposit labels (fresh seed); held-out: direction right in >= 0.8 of %d fresh document halves (x2). Decoys: all %d terms on >= 3 of these documents, two-sided, same rule | '
              '%s; decoy false-survival %.2f | %s |' % (len(docs), int(mag.sum()), NH, len(pool),
                                                       '; '.join('%s n %d %s %.2f P %.3f halves %d/%d' % (t, v['n'], v['want'], v['share'] or 0, v['p'] if v['p'] is not None else 1, v['rep'], v['tested']) for t, v in out.items() if t in tg),
                                                       dfalse, '; '.join('%s %s' % (t.split(':')[1], 'SURVIVES' if out[t]['surv'] else 'KILLED') for t in tg)))


def t_casa9(D):
    import la51_common as M
    docs = [d for d in M.load_la() if d['site'] == 'Haghia Triada' and d['deposit']]
    allw = Counter()
    for d in M.load_la():
        for x in d['terms']:
            if x.startswith('w:'):
                allw[x] += 1
    deps = Counter(d['deposit'] for d in docs)
    lab = [d['deposit'] for d in docs]
    def stat(lab):
        by = defaultdict(set)
        for d, l in zip(docs, lab):
            by[l] |= set(x for x in d['terms'] if x.startswith('w:'))
        # a word is 'met nowhere else' if all its documents are in this deposit
        dep_of = defaultdict(set)
        for d, l in zip(docs, lab):
            for x in d['terms']:
                if x.startswith('w:'):
                    dep_of[x].add(l)
        out = {}
        for l, ws in by.items():
            if ws:
                out[l] = np.mean([dep_of[x] == {l} and allw[x] == sum(1 for d2, l2 in zip(docs, lab) if l2 == l and x in d2['terms']) for x in ws])
        return out
    obs = stat(lab)
    rng = random.Random(seed('casa9'))
    null = defaultdict(list)
    for _ in range(1000):
        l2 = lab[:]; rng.shuffle(l2)
        s = stat(l2)
        for k in obs:
            null[k].append(s.get(k, 0))
    P = {k: pct_rank(obs[k], null[k]) for k in obs if deps[k] >= 5}
    dfalse = float(np.mean([P[k] <= 0.05 for k in P if k != 'HT_CASA9']))
    RES['casa9'] = dict(obs=obs, P=P)
    wlog(OUT, '| LA-67.3d | la23 C: Casa Room 9 lists words met nowhere else (P 0.045, Holm 0.18). Kill line (persons): needs a new person list. Existing-data test: share of a deposit\'s '
              'word types found in no other document of the corpus, vs 1,000 re-dealings of HT documents among HT deposits (fresh seed); decoys = the other HT deposits with >= 5 '
              'documents through the same test | %s | %s |' % ('; '.join('%s %.2f P %.3f' % (k, obs[k], P[k]) for k in P), 'SURVIVES' if P.get('HT_CASA9', 1) <= 0.05 else 'KILLED'))


def t_la47(D):
    import la47_common as L, la47_c3 as L3
    rows = L.table(L.la_pages())
    r = L3.part_a(rows, random.Random(seed('la47')), reps=1000)
    RES['la47'] = r
    s = r['S']
    wlog(OUT, '| LA-67.3e | la47 C: layout profiles separate la45 classes where order profiles do not (bal. acc. 0.436 vs 0.328, P 0.035, 200 permutations). Kill line: P > 0.2. '
              'Re-run with a fresh seed and 1,000 within-page identity permutations (same SigLA-aligned pages) | layout %.3f vs null %.3f +- %.3f, P %.3f (%s); order %.3f vs %.3f, P %.3f | %s |' % (
                  s['balacc'], s['null_mean'], s['null_sd'], s['P'], s['per_class'], r['O']['balacc'], r['O']['null_mean'], r['O']['P'],
                  'SURVIVES (P <= 0.05)' if s['P'] <= 0.05 else 'KILLED (P > 0.2)' if s['P'] > 0.2 else 'NOT KILLED, NOT SUPPORTED (0.05 < P <= 0.2): demote'))


def t_la49(D):
    """proxy for la49's groups: PPMI-SVD embeddings over within-line +-2 token windows on bootstrapped documents,
    duplicates (HT 86/95, HT 114/121 second copy) and stone vessels held out; cohesion of the group vs 500 frequency-
    matched random groups of the same size."""
    groups = {'commodity': ['NI', '*304', '*306', '*308', 'E', 'SU', 'CYP', 'OLE', 'VIN'],
              'libation': ['A-TA-I-*301-WA-JA', 'JA-SA-SA-RA-ME', 'SI-RU-TE', 'I-PI-NA-MA', 'PA', 'NA', 'TU', 'NE', 'QE', 'DI']}
    drop_tabs = {'HT95', 'HT121'}
    out = {}
    for hold in (False, True):
        docs = [d for d in D if not (hold and (d['tab'] in drop_tabs or d['support'] == 'Stone vessel'))]
        rng = np.random.default_rng(seed('la49-%s' % hold))
        res = {g: [] for g in groups}
        for b in range(40):
            bs = [docs[i] for i in rng.integers(0, len(docs), len(docs))]
            cnt = Counter(re.sub(r'\+.*', '', t[1]) for d in bs for t in d['toks'] if t[0] in 'WL')
            V = [w for w, c in cnt.items() if c >= 3]; vi = {w: i for i, w in enumerate(V)}
            M = np.zeros((len(V), len(V)))
            for d in bs:
                for ln in lines(d):
                    s = [re.sub(r'\+.*', '', t[1]) if t[0] in 'WL' else '#N' for t in ln]
                    for i, x in enumerate(s):
                        if x not in vi:
                            continue
                        for j in range(max(0, i - 2), min(len(s), i + 3)):
                            if j != i and s[j] in vi:
                                M[vi[x], vi[s[j]]] += 1
            tot = M.sum(); r_ = M.sum(1, keepdims=True); c_ = M.sum(0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                P = np.log(M * tot / (r_ @ c_)); P[~np.isfinite(P)] = 0; P = np.maximum(P, 0)
            U, S, _ = np.linalg.svd(P, full_matrices=False)
            E = U[:, :20] * np.sqrt(S[:20]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
            def coh(ws):
                ix = [vi[w] for w in ws if w in vi]
                if len(ix) < 3:
                    return None
                G = E[ix] @ E[ix].T
                return (G.sum() - len(ix)) / (len(ix) * (len(ix) - 1))
            lf = {w: math.log(cnt[w]) for w in V}
            for g, ws in groups.items():
                o = coh(ws)
                if o is None:
                    res[g].append(None); continue
                ws_in = [w for w in ws if w in vi]
                nl = []
                for _ in range(200):
                    pick = []
                    for w in ws_in:
                        c = [x for x in V if abs(lf[x] - lf[w]) < 0.5 and x not in pick and x not in ws]
                        pick.append(c[rng.integers(len(c))] if c else V[rng.integers(len(V))])
                    nl.append(coh(pick))
                res[g].append((np.sum(np.array(nl) >= o) + 1) / 201)
        out['held-out' if hold else 'all'] = {g: (float(np.mean([p <= 0.05 for p in v if p is not None])) if any(p is not None for p in v) else None,
                                                  float(np.median([p for p in v if p is not None])) if any(p is not None for p in v) else None,
                                                  sum(p is not None for p in v)) for g, v in res.items()}
    RES['la49'] = out
    h = out['held-out']
    wlog(OUT, '| LA-67.3f | la49 C: a commodity group (NI *304 *306 *308 E SU with CYP OLE VIN) and a libation group (A-TA-I-*301-WA-JA, JA-SA-SA-RA-ME, SI-RU-TE, I-PI-NA-MA, PA NA TU NE QE DI) '
              'are stable. Kill line: the groups dissolving when duplicate tablets and stone vessels are held out. Proxy (no transformer re-run): 20-dim PPMI-SVD embeddings from '
              '+-2 token windows within lines, 40 document bootstraps per setting; cohesion vs 200 frequency-matched random groups per bootstrap | %s | %s |' % (
                  '; '.join('%s: %s' % (k, ', '.join('%s cohesive (P <= 0.05) in %.2f of %d bootstraps, median P %.3f' % (g, v[0], v[2], v[1]) for g, v in vv.items() if v[0] is not None)) for k, vv in out.items()),
                  '; '.join('%s %s' % (g, 'SURVIVES' if v[0] is not None and v[0] >= 0.8 else 'KILLED (dissolves when held out)') for g, v in h.items())))


if __name__ == '__main__':
    D = docs_all()
    if not os.path.exists(OUT):
        wlog(OUT, '# LA-67 cycle 3: corrections, the remaining existing-data kill tests, and fresh-seed re-runs of earlier machinery (6 Oct 2026). Scripts tools/la67_c3.py, la67_c3b.py.')
    only = sys.argv[1:] or None
    for f in (t_kaja, t_figsolives, t_la51, t_casa9, t_la47, t_la49):
        if only and f.__name__ not in only:
            continue
        f(D); print(f.__name__, 'done', flush=True)
        json.dump(RES, open(os.path.join(CK, 'c3_%s.json' % f.__name__), 'w'), default=str, ensure_ascii=False)
