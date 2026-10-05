#!/usr/bin/env python3
"""la47 cycle 2: fine 2-D geometry from SigLA boxes (x indentation of line-initial items, gap before an
item, item size residualised on sign code/logogram identity, vertical drift).  Does adding geometry to
the coarse spatial features let random grammars predict held-out word identities / la45 classes better,
against (P) identities permuted within page with slot geometry kept, and the planted control (a size
rule planted on real geometry: tokens of a random 20 % of types inflated by 25 %)?
Also direct tests: size and gap by la45 class, within-page permutation null.
Output data/la47_ckpt/c2.json"""
import json, os, random, collections, sys
import numpy as np
from multiprocessing import Pool
import la47_common as C, la47_engine as E

GEO = ['indent', 'gapb', 'size', 'drift']


def geo_pages():
    P = [p for p in C.la_pages(supports=('Tablet',)) if p.get('vb')]
    # residual size: log(h / page median syllabogram h) minus identity mean
    rel = collections.defaultdict(list)
    for p in P:
        hs = [b[3] for i in p['items'] if i.get('box') for b in i['box'] if b[5] == 'syllabogram']
        ws = [b[2] for i in p['items'] if i.get('box') for b in i['box'] if b[5] == 'syllabogram']
        if len(hs) < 3: p['skip'] = 1; continue
        p['mh'] = float(np.median(hs)); p['mw'] = float(np.median(ws))
        for i in p['items']:
            for b in i.get('box') or []:
                key = b[4] if b[5] == 'syllabogram' and b[4] else ('L:' + i['id'])
                rel[key].append(np.log(b[3] / p['mh']))
    mean = {k: float(np.mean(v)) for k, v in rel.items() if len(v) >= 3}
    P = [p for p in P if not p.get('skip')]
    for p in P:
        xs = [b[0] for i in p['items'] if i.get('box') for b in i['box']]
        x0 = min(xs)
        for i in p['items']:
            bx = i.get('box')
            if not bx: continue
            r = []
            for b in bx:
                key = b[4] if b[5] == 'syllabogram' and b[4] else ('L:' + i['id'])
                if key in mean: r.append(np.log(b[3] / p['mh']) - mean[key])
            i['sres'] = float(np.mean(r)) if r else None
            i['x0'] = (bx[0][0] - x0) / p['mw']; i['xr'] = (bx[-1][0] + bx[-1][2]); i['xl'] = bx[0][0]
            i['yc'] = float(np.mean([b[1] + b[3] / 2 for b in bx]))
    return P


def bin3(v, a, b, names):
    return names[0] if v < a else (names[1] if v < b else names[2])


def geo_rows(P, la=True):
    rows = C.table(P, la)
    k = 0
    # align rows with featurize order
    allit = [(pi, i) for pi, p in enumerate(P) for i, s, o in C.featurize(p)]
    # size tertiles over corpus
    sz = [i['sres'] for _, i in allit if i.get('sres') is not None]
    q1, q2 = np.quantile(sz, [1 / 3, 2 / 3])
    for r, (pi, i) in zip(rows, allit):
        p = P[pi]
        s = r['s']
        if i.get('box') is None:
            s.update(indent='nb', gapb='nb', size='nb', drift='nb'); continue
        s['size'] = 'na' if i.get('sres') is None else bin3(i['sres'], q1, q2, ['small', 'mid', 'large'])
        if s['linit']:
            s['indent'] = bin3(i['x0'], 0.6, 2.0, ['flush', 'ind1', 'ind2']); s['gapb'] = 'start'
        else:
            s['indent'] = 'inner'
            its = p['items']; k = its.index(i)
            prev = next((x for x in reversed(its[:k]) if x['k'] != 'd'), None)
            if prev is None or prev['k'] != 'w' or prev.get('xr') is None: s['gapb'] = 'num'
            else: s['gapb'] = bin3((i['xl'] - prev['xr']) / p['mw'], 0.3, 1.0, ['tight', 'norm', 'wide'])
        # vertical drift: item centre vs its line's median centre (raised/lowered/level)
        same = [x['yc'] for x in p['items'] if x.get('yc') is not None and x['pl'] == i['pl']]
        d = (i['yc'] - float(np.median(same))) / p['mh']
        s['drift'] = bin3(d, -0.25, 0.25, ['up', 'level', 'down'])
    return rows


def permute_rows(rows, rng):
    """identity targets shuffled among word slots within page (all slot features kept)"""
    out = [dict(r) for r in rows]
    by = collections.defaultdict(list)
    for k, r in enumerate(out): by[r['p']].append(k)
    for ks in by.values():
        tg = [(rows[k]['type'], rows[k]['cls']) for k in ks]; rng.shuffle(tg)
        for k, (a, b) in zip(ks, tg): out[k]['type'] = a; out[k]['cls'] = b
    return out


def plant_size(rows, P, rng):
    out = permute_rows(rows, rng)
    types = sorted({r['type'] for r in out}); X = set(rng.sample(types, len(types) // 5))
    for r in out:
        r['planted'] = 'X' if r['type'] in X else 'Y'
        if r['type'] in X and r['s']['size'] in ('small', 'mid') and rng.random() < 0.5:
            r['s'] = dict(r['s'], size='large' if r['s']['size'] == 'mid' else 'mid')
    return out


def job(a):
    name, seed = a
    rng = random.Random(seed)
    P = geo_pages(); rows = [r for r in geo_rows(P) if r['kind'] == 'w' and r['s']['size'] != 'nb']
    if name.startswith('NULLP'): rows = permute_rows(rows, rng)
    if name.startswith('PLANT'): rows = plant_size(rows, P, rng)
    out = {'pages': len(P), 'rows': len(rows)}
    for tgt in ('type', 'cls') + (('planted',) if name.startswith('PLANT') else ()):
        out[tgt + '_base'] = E.run(rows, tgt, G=G, seed=seed, sfeat=C.SFEAT)
        out[tgt + '_geo'] = E.run(rows, tgt, G=G, seed=seed, sfeat=C.SFEAT + GEO)
        out[tgt + '_geoonly'] = E.run(rows, tgt, G=G, seed=seed, sfeat=GEO)
    return name, out


def direct(P):
    rng = np.random.default_rng(5)
    rows = [r for r in geo_rows(P) if r['kind'] == 'w']
    allit = [(pi, i) for pi, p in enumerate(P) for i, s, o in C.featurize(p) if i['k'] == 'w']
    data = [(pi, i['sres'], C.la45_class(i['id']), i['logo']) for pi, i in allit if i.get('sres') is not None]
    pid = np.array([d[0] for d in data]); s = np.array([d[1] for d in data]); cl = np.array([d[2] for d in data])
    res = {}
    for c in ('commodity', 'header', 'total', 'other'):
        m = cl == c
        if m.sum() < 5: continue
        obs = s[m].mean() - s[~m].mean(); null = []
        for _ in range(2000):
            sp = s.copy()
            for p in np.unique(pid):
                k = np.where(pid == p)[0]; sp[k] = sp[rng.permutation(k)]
            null.append(sp[m].mean() - sp[~m].mean())
        null = np.array(null)
        res['size_' + c] = [int(m.sum()), round(float(obs), 4), round(float((np.abs(null) >= abs(obs)).mean()), 4)]
    # line-initial items larger?
    li = np.array([r['s']['linit'] for r, (pi, i) in zip(rows, allit) if i.get('sres') is not None])
    obs = s[li == 1].mean() - s[li == 0].mean(); null = []
    for _ in range(2000):
        sp = s.copy()
        for p in np.unique(pid):
            k = np.where(pid == p)[0]; sp[k] = sp[rng.permutation(k)]
        null.append(sp[li == 1].mean() - sp[li == 0].mean())
    res['size_lineinitial'] = [int((li == 1).sum()), round(float(obs), 4), round(float((np.abs(np.array(null)) >= abs(obs)).mean()), 4)]
    # indentation table: class x indent for line-initial words
    tab = collections.Counter((r['cls'], r['s']['indent']) for r in rows if r['s'].get('indent') in ('flush', 'ind1', 'ind2'))
    res['indent_by_class'] = {'%s|%s' % k: v for k, v in sorted(tab.items())}
    tab = collections.Counter((r['cls'], r['s']['gapb']) for r in rows if r['s'].get('gapb') in ('tight', 'norm', 'wide', 'num', 'start'))
    res['gap_by_class'] = {'%s|%s' % k: v for k, v in sorted(tab.items())}
    return res


G = int(os.environ.get('G', 1500))
if __name__ == '__main__':
    P = geo_pages()
    print('pages', len(P), 'word items with geometry', sum(1 for p in P for i in p['items'] if i['k'] == 'w' and i.get('box')))
    out = {'direct': direct(P)}
    print(json.dumps(out['direct']))
    jobs = [('LA', 1), ('NULLP1', 31), ('NULLP2', 32), ('NULLP3', 33), ('PLANT1', 21), ('PLANT2', 22)]
    with Pool(2) as pool:
        for name, o in pool.imap_unordered(job, jobs):
            out[name] = o
            print(name, json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a in ('top10_S', 'best_S', 'top10_O', 'incr_S_over_O', 'n')}) for k, v in o.items()}), flush=True)
    json.dump(out, open(os.path.join(C.CK, 'c2.json'), 'w'), indent=1)
