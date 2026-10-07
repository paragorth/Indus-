"""LA-80 cycle 2: PROPHESY THE LUMP. Which signs a sealing lump bears -> the lump's outline.
select: HT nodules/roundels only; folds = inscription types (all copies of one inscription in one fold);
null = inscription-block permutation (text of whole inscription groups swapped among groups of similar size);
freeze; test: non-HT lumps measured after the freeze."""
import json, os, sys, collections, hashlib, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L
from la80_c1 import resid, survivors, SHAPES
NH = int(os.environ.get('LA80_NH', 20000))

def lump_table(ids, shape_fn):
    C = {d['id']: d for d in json.load(open(os.path.join(L.DATA, 'corpus_ra.json')))}
    shp = json.load(open(shape_fn)) if shape_fn else {}
    rows = []
    for i in ids:
        d = C[i]; r = shp.get(i)
        if not r or not r.get('ok'): continue
        toks = [t for t in d['tokens'] if t['t'] in ('word', 'logo', 'num', 'unk')]
        S = set()
        for t in toks:
            if t['t'] == 'word': S |= set(t['s'])
            if t['t'] == 'logo': S |= set(t.get('v', '').split('+'))
        key = '|'.join(('-'.join(t['s']) if t['t'] == 'word' else str(t.get('v'))) for t in toks)
        F = dict(n_signs=len(S), n_tok=len(toks), has_logo=int(any(t['t'] == 'logo' for t in toks)),
                 has_num=int(any(t['t'] == 'num' for t in toks)), multi=int(any(t['t'] == 'word' and len(t['s']) > 1 for t in toks)),
                 damaged=int(any(t.get('st') in ('damaged', 'illegible') for t in toks)))
        rows.append(dict(id=i, site=d['site'], sup=d['support'], S=S, F=F, key=key, shape=[r.get(s, np.nan) for s in SHAPES]))
    return rows

def tvec(h, rows):
    k, a = h
    if k == 'scalar': return np.array([r['F'][a] for r in rows], float)
    return np.array([len(r['S'] & set(a)) > 0 for r in rows], float)

def covs(rows, site_dummies=False):
    X = [[np.log1p(r['F']['n_signs']), r['sup'] == 'Roundel', r['sup'] == 'Sealing', r['F']['damaged']] for r in rows]
    X = np.array(X, float)
    if site_dummies:
        us = sorted(set(r['site'] for r in rows))
        X = np.c_[X, np.array([[r['site'] == u for u in us[1:]] for r in rows], float).reshape(len(rows), -1)]
    X = X[:, X.std(0) > 0]
    return X

def score(pool, TV, M, X, folds, min_pos=5):
    Rs = []
    for j in range(M.shape[1]):
        col = M[:, j]; ok = ~np.isnan(col); Rs.append((ok, resid(col[ok], X[ok])))
    uf = sorted(set(folds)); fidx = {f: np.array([x == f for x in folds]) for f in uf}
    out = []
    for ii, (h, j) in enumerate(pool):
        t = TV[ii]; ok, rs = Rs[j]; tt = t[ok]
        if len(set(tt)) < 2 or (set(tt) <= {0, 1} and min(tt.sum(), len(tt) - tt.sum()) < min_pos):
            out.append(None); continue
        rt = resid(tt, X[ok]); r = float(rt @ rs); gs = []
        for f in uf:
            m = fidx[f][ok]
            if m.sum() >= 6 and len(set(tt[m])) > 1:
                a = rt[m] - rt[m].mean(); b = rs[m] - rs[m].mean(); den = np.linalg.norm(a) * np.linalg.norm(b)
                gs.append(float(a @ b / den) if den > 0 else 0.0)
        out.append((r, gs, int(ok.sum())))
    return out

def select():
    rng = np.random.default_rng(802)
    ids = json.load(open(os.path.join(L.CK, 'ht_lump_ids.json')))
    rows = lump_table(ids, os.path.join(L.CK, 'shape_ht_lump.json'))
    M = np.array([r['shape'] for r in rows]); X = covs(rows)
    keys = sorted(set(r['key'] for r in rows))
    folds = [zlib.crc32(r['key'].encode()) % 4 for r in rows]
    cnt = collections.Counter()
    for r in rows: cnt.update(r['S'])
    pool_s = sorted([s for s, c in cnt.items() if c >= 8 and s])
    Fn = sorted(rows[0]['F'])
    import itertools
    hs = [('scalar', f) for f in Fn] + [('signs', c) for k in (1, 2, 3) for c in itertools.combinations(pool_s, k)]
    pool = [(h, j) for h in hs for j in range(len(SHAPES))]   # exhaustive: the sign pool is small
    TV = [tvec(h, rows) for h, j in pool]
    sc = score(pool, TV, M, X, folds); S_real = survivors(sc)
    log = dict(n=len(rows), n_keys=len(keys), sign_pool=len(pool_s), valid=sum(s is not None for s in sc), real=len(S_real))
    # block null: whole inscription groups receive the text of another group of similar size
    grp = collections.defaultdict(list)
    for i, r in enumerate(rows): grp[r['key']].append(i)
    gl = sorted(grp, key=lambda k: len(grp[k]))
    tert = np.array_split(np.arange(len(gl)), 3)
    shuf, shuf_sets = [], []
    for s in range(20):
        mapping = {}
        for t in tert:
            perm = rng.permutation(t)
            for a, b in zip(t, perm): mapping[gl[a]] = gl[b]
        # index of a representative member of the donor group, per row
        src = np.array([grp[mapping[r['key']]][0] for r in rows])
        TV2 = [v[src] for v in TV]
        X2 = X[src]
        sc2 = score(pool, TV2, M, X2, folds); s2 = survivors(sc2); shuf.append(len(s2)); shuf_sets.append(s2)
    log['shuffle'] = shuf
    planted = []
    for w in range(10):
        Mp = M.copy(); links = []
        while len(links) < 3:
            i = int(rng.integers(len(pool)))
            if sc[i] is None: continue
            h, j = pool[i]; t = TV[i]
            col = Mp[:, j]; ok = ~np.isnan(col); zt = (t - t.mean()) / t.std()
            Mp[ok, j] = col[ok] + 0.35 * np.nanstd(col) * zt[ok]; links.append(i)
        s3 = set(survivors(score(pool, TV, Mp, X, folds))); planted.append(sum(l in s3 for l in links))
    log['planted_of_3'] = planted
    def hj(i):
        (h, j) = pool[i]; return {'h': [h[0], list(h[1]) if isinstance(h[1], tuple) else h[1]], 'shape': SHAPES[j], 'r_ht': sc[i][0], 'groups': sc[i][1]}
    valid = [i for i, s in enumerate(sc) if s is not None]
    frozen = {'plan': 'LA-80 c2. Score survivors on non-HT nodules/roundels/sealings measured after this freeze; covariates log signs, support, damage, site. '
                      'Primary: share of survivors whose sign repeats vs the 20 block-null survivor sets and 2,000 random pairs. Kill: share <= 0.60 or not above the null 95th pct.',
              'survivors': [hj(i) for i in S_real], 'shuffle_survivors': [[hj(i) for i in s] for s in shuf_sets],
              'random_ref': [hj(i) for i in rng.choice(valid, min(2000, len(valid)), replace=False)], 'log': log}
    fn = os.path.join(L.DATA, 'la80_frozen_c2.json'); json.dump(frozen, open(fn, 'w'), indent=0, default=float)
    h = hashlib.sha256(open(fn, 'rb').read()).hexdigest(); open(fn.replace('.json', '.sha256'), 'w').write(h + '  la80_frozen_c2.json\n')
    print(json.dumps(log, default=float)); print('sha256', h)

def test():
    fz = os.path.join(L.DATA, 'la80_frozen_c2.json'); fr = json.load(open(fz))
    assert open(fz.replace('.json', '.sha256')).read().split()[0] == hashlib.sha256(open(fz, 'rb').read()).hexdigest()
    ids = json.load(open(os.path.join(L.CK, 'test_lump_ids.json')))
    rows_all = lump_table(ids, os.path.join(L.CK, 'shape_test_lump.json'))
    out = {}
    for name, filt in [('all', lambda r: True), ('khania', lambda r: r['site'] == 'Khania'), ('not_khania', lambda r: r['site'] != 'Khania'),
                       ('roundels', lambda r: r['sup'] == 'Roundel'), ('nodules', lambda r: r['sup'] != 'Roundel')]:
        rows = [r for r in rows_all if filt(r)]
        M = np.array([r['shape'] for r in rows]); X = covs(rows, site_dummies=True)
        def agree(entries, Mm=M):
            v = []
            for x in entries:
                h = (x['h'][0], tuple(x['h'][1]) if isinstance(x['h'][1], list) else x['h'][1]); j = SHAPES.index(x['shape'])
                t = tvec(h, rows); col = Mm[:, j]; ok = ~np.isnan(col); tt = t[ok]
                if len(set(tt)) < 2 or (set(tt) <= {0, 1} and min(tt.sum(), len(tt) - tt.sum()) < 2): continue
                v.append(np.sign(x['r_ht']) * float(resid(tt, X[ok]) @ resid(col[ok], X[ok])))
            v = np.array(v); return (float(np.mean(v > 0)) if len(v) else None, len(v), float(np.mean(v)) if len(v) else None)
        res = dict(n=len(rows), real=agree(fr['survivors']), shuffle=[agree(s) for s in fr['shuffle_survivors']], random=agree(fr['random_ref']))
        # power: plant every survivor at 0.15 sd
        v = []
        for x in fr['survivors'][::3]:
            h = (x['h'][0], tuple(x['h'][1]) if isinstance(x['h'][1], list) else x['h'][1]); j = SHAPES.index(x['shape'])
            t = tvec(h, rows); col = M[:, j].copy(); ok = ~np.isnan(col); tt = t[ok]
            if len(set(tt)) < 2 or min(tt.sum(), len(tt) - tt.sum()) < 2: continue
            col[ok] += np.sign(x['r_ht']) * 0.15 * np.nanstd(col) * (tt - tt.mean()) / tt.std()
            v.append(np.sign(x['r_ht']) * float(resid(tt, X[ok]) @ resid(col[ok], X[ok])) > 0)
        res['power_0.15'] = (float(np.mean(v)) if v else None, len(v))
        out[name] = res
        sh = [s[0] for s in res['shuffle'] if s[0] is not None]
        print(name, 'n', res['n'], 'real', res['real'], 'null mean %.3f max %.3f' % (np.mean(sh), np.max(sh)) if sh else '', 'rank', sum(np.array(sh) >= (res['real'][0] or 0)), '/', len(sh), 'random', res['random'], 'power', res['power_0.15'])
    json.dump(out, open(os.path.join(L.CK, 'c2_test.json'), 'w'), indent=1, default=float)

if __name__ == '__main__':
    {'select': select, 'test': test}[sys.argv[1]]()
