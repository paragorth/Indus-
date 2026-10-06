"""v76 cycle 3: (a) the slot analysis inside the starred recipe section (S) and in herbal first paragraphs
(H, paragraph 0) separately; (b) linked attributes: in template groups of >= 3 cross-page members aligned to a
medoid, pairs of variable columns whose variation co-occurs (phi) beyond a member-permutation null;
(c) the identifier stream: page-own slot fillers per page; concentration (one item per record) and sharing with
the next page (runs of related records) vs random same-section pages. Every statistic for real corpora and the
fitted generator panel (same code)."""
import os, sys, json, gzip, random, math, glob
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
import v76_c1 as C1
import v76_c2 as C2


def subset_pairs(P, pairs, kind):
    if kind == 'S': f = lambda p: p['sec'] == 'S'
    elif kind == 'H0': f = lambda p: p['sec'] == 'H' and p['pidx'] == 0
    else: f = lambda p: True
    return [(i, j) for i, j in pairs if f(P[i]) and f(P[j])]


def linked(P, name, rng, PA=None):
    PA = PA or P
    d = np.load(os.path.join(V.CK, 'c1_%s.npz' % name)); sc, npair = d['sc'][:8000], d['npair'][:8000]
    r0 = random.Random(7600); tmpl = [C1.rand_template(r0) for _ in range(len(sc))]
    order = np.argsort(-np.where(npair[:, 1] >= 5, sc[:, 1], -1e9))[:C2.K]
    views = [C1.para_view(p) for p in P]
    S = V.sim_matrix(P)
    seen = set(); tests = sig = 0; groups = 0
    for k in order:
        lab = C1.signatures(views, tmpl[k], {})
        grp = defaultdict(list)
        for i, g in enumerate(lab): grp[g].append(i)
        for g, ii in grp.items():
            pages = {P[i]['page'] for i in ii}
            if len(pages) < 3 or len(ii) > 30: continue
            key = tuple(sorted(ii))
            if key in seen: continue
            seen.add(key); groups += 1
            sub = np.nan_to_num(S[np.ix_(ii, ii)])
            med = ii[int(np.argmax(sub.sum(1)))]
            others = [i for i in ii if i != med and P[i]['page'] != P[med]['page']]
            if len(others) < 3: continue
            n = len(PA[med]['toks'])
            M = np.full((len(others), n), -1, np.int8)        # 1 varies, 0 same, -1 gap
            for r, i in enumerate(others):
                for x, y in C2.align(PA[med]['toks'], PA[i]['toks']):
                    if x is None: continue
                    M[r, x] = -1 if y is None else int(PA[med]['toks'][x] != PA[i]['toks'][y])
            cols = [c for c in range(n) if (M[:, c] >= 0).sum() == len(others) and 0 < M[:, c].sum() < len(others)]
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    x, y = M[:, cols[a]].astype(float), M[:, cols[b]].astype(float)
                    obs = abs(np.corrcoef(x, y)[0, 1])
                    nul = [abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(50)]
                    nul = [v for v in nul if not np.isnan(v)]
                    if np.isnan(obs) or not nul: continue
                    tests += 1
                    if obs >= 0.999 and np.mean(np.array(nul) >= obs) < 0.05: sig += 1
    return dict(groups=groups, tests=tests, linked=sig, rate=sig / max(1, tests))


def idstream(name, P):
    fill = json.load(gzip.open(os.path.join(V.CK, 'c2b_fill_%s.json.gz' % name), 'rt'))
    own = defaultdict(Counter)
    for X, ta, Y, tb, pos, v in fill:
        if v >= 0.5:      # at least one side page-own; credit each side that is
            pass
    pages = defaultdict(Counter)
    for p in P: pages[p['page']].update(p['toks'])
    for X, ta, Y, tb, pos, v in fill:
        if pages[X][ta] > 1 and pages[Y][ta] == 0: own[X][ta] += 1
        if pages[Y][tb] > 1 and pages[X][tb] == 0: own[Y][tb] += 1
    conc = [c.most_common(1)[0][1] / sum(c.values()) for c in own.values() if sum(c.values()) >= 3]
    order = []
    for p in P:
        if p['page'] not in order: order.append(p['page'])
    sec = {p['page']: p['sec'] for p in P}
    ids = {pg: set(c) for pg, c in own.items() if c}
    adj, rnd = [], []
    rng = random.Random(3)
    for a, b in zip(order, order[1:]):
        if a in ids and b in ids and sec[a] == sec[b]:
            adj.append(len(ids[a] & ids[b]) / len(ids[a] | ids[b]))
            cand = [x for x in ids if sec[x] == sec[a] and x not in (a, b)]
            for _ in range(5):
                c = rng.choice(cand) if cand else b
                rnd.append(len(ids[a] & ids[c]) / len(ids[a] | ids[c]))
    top = {pg: c.most_common(2) for pg, c in own.items()}
    return dict(npages=len(own), conc=float(np.mean(conc)) if conc else None, nconc=len(conc),
                adj=float(np.mean(adj)) if adj else None, rnd=float(np.mean(rnd)) if rnd else None, nadj=len(adj),
                top={k: v for k, v in list(top.items())[:400]})


def job(name):
    fn = os.path.join(V.CK, 'c3_%s.json' % name)
    if os.path.exists(fn): return name
    P = json.load(gzip.open(os.path.join(V.CK, 'par_%s.json.gz' % name), 'rt'))
    pr, _ = C2.pairs_for(name, P, 'ALL')
    PA = C2.strip_d(P)
    out = {}
    for kind in ('S', 'H0'):
        sp = subset_pairs(P, pr, kind)
        if len(sp) > 6000: sp = random.Random(5).sample(sp, 6000)
        out[kind] = C2.analyse(name, PA, sp)[0] if len(sp) >= 20 else None
    out['LINK'] = linked(P, name, np.random.default_rng(1), PA)
    out['ID'] = idstream(name, PA)
    json.dump(out, open(fn, 'w'))
    return name


if __name__ == '__main__':
    names = sorted(os.path.basename(f)[3:-4] for f in glob.glob(os.path.join(V.CK, 'c1_*.npz')))
    names = [n for n in names if os.path.exists(os.path.join(V.CK, 'c2b_fill_%s.json.gz' % n))]
    with Pool(2) as pool:
        for n in pool.imap_unordered(job, names): print('done', n, flush=True)
