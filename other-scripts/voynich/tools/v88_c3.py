"""v88 cycle 3.
(a) Kill test for the through-the-leaf line-mood lead (V-88.1.5): glyph profiles residualised
    by section x language x line role (paragraph-first / last / body) x length bin; LEAF row
    alignment vs same-stratum random leaves (hand/section-only model), both vocabulary halves,
    both transcriptions; planted 'leaf-mood' generator as positive control (rows of a verso copy
    the glyph mood of the recto row at the same height), Markov as negative.
(b) Outside test: text-only prediction of which recto faces each verso (row-aligned J + S),
    frozen as sha256 in data/v88_frozen_predictions.json BEFORE reading the physical claims;
    then compared with codicological evidence fetched from voynich.nu (sp_origin, 7 Oct 2026):
    f78v/f81r integrated design (Q13 centre originally), f33v drawing disappears into the
    gutter (f33v/f40r), paint transfers 'mainly consistent with the present binding'.
"""
import sys, os, json, math, random, re
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v88_lib as G, v65_lib as V, vlib

NREP = int(os.environ.get('NREP', 1000))


def pages_roles(name):
    L = vlib.load_voynich(name, ltypes=('P',))
    pages = defaultdict(list); roles = defaultdict(list); meta = {}
    for r in L:
        if not re.match(r'^f\d+[rv]$', r['folio']):
            continue
        ws = [''.join(vlib.glyphs(w)) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w and '*' not in w]
        if not ws:
            continue
        k = V.side_of(r['folio'])
        pages[k].append(ws)
        roles[k].append('F' if r['para_start'] else ('L' if r['para_end'] else 'B'))
        meta.setdefault(k, {'sec': r['illus'], 'lang': r['lang'], 'hand': r['hand']})
    keep = {k for k in pages if len(pages[k]) >= 3}
    return {k: pages[k] for k in keep}, {k: roles[k] for k in keep}, meta


def resid_vectors(pages, roles, meta, half, resid=True):
    alpha = {}
    for L in pages.values():
        for line in L:
            for w in line:
                for g in w:
                    alpha.setdefault(g, len(alpha))
    vec = {}; grp = defaultdict(list)
    for k, L in pages.items():
        for i, line in enumerate(L):
            v = np.zeros(len(alpha))
            for w in line:
                if half is None or G.wclass(w) == half:
                    for g in w:
                        v[alpha[g]] += 1
            s = v.sum()
            v = v / s if s else v
            lb = 0 if len(line) <= 5 else (1 if len(line) <= 8 else 2)
            g = (meta[k]['sec'], meta[k]['lang'], roles[k][i], lb)
            vec[(k, i)] = v; grp[g].append((k, i))
    if resid:
        for g, items in grp.items():
            m = np.mean([vec[x] for x in items], axis=0)
            for x in items:
                vec[x] = vec[x] - m
    for x in vec:
        n = np.linalg.norm(vec[x]); vec[x] = vec[x] / n if n else vec[x]
    return vec


def leaf_stat(pages, vec, pairs, D=(-2, -1, 0, 1, 2)):
    prof = {}
    for d in D:
        a = []
        for P, Q in pairs:
            for k in range(len(pages[P])):
                j = k + d
                if 0 <= j < len(pages[Q]):
                    a.append(float(vec[(P, k)] @ vec[(Q, j)]))
        prof[d] = np.mean(a)
    return prof[0], prof[0] - np.mean([prof[d] for d in D if d]), prof


def leaf_test(pages, roles, meta, quires, half, resid, seed=0):
    rng = random.Random(seed)
    vec = resid_vectors(pages, roles, meta, half, resid)
    pairs = G.pair_classes(quires, pages)['LEAF']
    strat = defaultdict(list)
    for k in pages:
        strat[(meta[k]['sec'], meta[k]['lang'], meta[k]['hand'])].append(k)
    T0, D, prof = leaf_stat(pages, vec, pairs)
    nT, nD = [], []
    for _ in range(NREP):
        fake = []
        for P, Q in pairs:
            pool = [k for k in strat[(meta[Q]['sec'], meta[Q]['lang'], meta[Q]['hand'])] if k[1] == 'v' and k != Q]
            fake.append((P, rng.choice(pool) if pool else Q))
        a, b, _ = leaf_stat(pages, vec, fake)
        nT.append(a); nD.append(b)
    return {'T0': T0, 'D': D, 'zT': (T0 - np.mean(nT)) / np.std(nT), 'zD': (D - np.mean(nD)) / np.std(nD),
            'prof': {str(k): float(v) for k, v in prof.items()}}


def leafmood_plant(pages, roles, meta, quires, rate, seed=3):
    """positive control: each verso row k is replaced, with prob `rate`, by a random same-stratum
    line whose glyph mood is nearest the recto row k (a row-height mood shared through the leaf)."""
    rng = random.Random(seed)
    vec = resid_vectors(pages, roles, meta, None, False)
    pool = defaultdict(list)
    for k, L in pages.items():
        for i, line in enumerate(L):
            pool[(meta[k]['sec'], meta[k]['lang'], roles[k][i])].append((k, i))
    out = {k: [list(l) for l in L] for k, L in pages.items()}
    for P, Q in G.pair_classes(quires, pages)['LEAF']:
        for k in range(min(len(pages[P]), len(pages[Q]))):
            if rng.random() < rate:
                cand = rng.sample(pool[(meta[Q]['sec'], meta[Q]['lang'], roles[Q][k])], min(20, len(pool[(meta[Q]['sec'], meta[Q]['lang'], roles[Q][k])])))
                best = max(cand, key=lambda x: float(vec[x] @ vec[(P, k)]))
                out[Q][k] = list(pages[best[0]][best[1]])
    return out


def facing_scores(pages, quires):
    """text-only score for every (verso, recto) pair: row-aligned contrast of J + standardised S."""
    import v88_c1 as C
    fJ, fS, _, _ = C.make_f(pages)
    vs = sorted(k for k in pages if k[1] == 'v'); rs = sorted(k for k in pages if k[1] == 'r')
    MJ = np.zeros((len(vs), len(rs))); MS = np.zeros_like(MJ)
    for i, P in enumerate(vs):
        for j, Q in enumerate(rs):
            pj = G.row_profile(pages, [(P, Q)], fJ, D=(-1, 0, 1)); ps = G.row_profile(pages, [(P, Q)], fS, D=(-1, 0, 1))
            MJ[i, j] = 0 if math.isnan(pj[0]) else pj[0] - np.nanmean([pj[-1], pj[1]])
            MS[i, j] = 0 if math.isnan(ps[0]) else ps[0] - np.nanmean([ps[-1], ps[1]])
    M = (MJ - MJ.mean()) / MJ.std() + (MS - MS.mean()) / MS.std()
    return vs, rs, M


def main():
    quires, _ = V.structure('ZL3b')
    out = {'leaf': {}}
    for name in ('ZL3b', 'IT2a'):
        p, ro, m = pages_roles(name)
        for resid in (False, True):
            for half in (None, 0, 1):
                r = leaf_test(p, ro, m, quires, half, resid)
                out['leaf']['%s|resid=%s|half=%s' % (name, resid, half)] = r
                print(name, 'resid', resid, 'half', half, 'zT %.2f zD %.2f' % (r['zT'], r['zD']), flush=True)
    p, ro, m = pages_roles('ZL3b')
    tmpl_meta = m
    for rate in (0.15, 0.3):
        pl = leafmood_plant(p, ro, m, quires, rate)
        r = leaf_test(pl, ro, m, quires, None, True)
        out['leaf']['PLANT-leafmood-%.2f' % rate] = r
        print('PLANT leafmood', rate, 'zT %.2f zD %.2f' % (r['zT'], r['zD']), flush=True)
    mk = V.markov_pages(p, m, 5)
    r = leaf_test(mk, ro, m, quires, None, True); out['leaf']['Markov'] = r
    print('Markov', 'zT %.2f zD %.2f' % (r['zT'], r['zD']), flush=True)

    # (b) frozen text-only facing predictions, then the outside comparison
    preds = {}
    for name in ('ZL3b', 'IT2a'):
        pg, _ = G.vpages(name)
        vs, rs, M = facing_scores(pg, quires)
        preds[name] = {'%d%s' % P: ['%d%s' % rs[j] for j in np.argsort(-M[i])] for i, P in enumerate(vs)}
    blob = json.dumps(preds, sort_keys=True)
    import hashlib
    h = hashlib.sha256(blob.encode()).hexdigest()
    fz = os.path.join(G.DATA, 'v88_frozen_predictions.json')
    json.dump({'sha256': h, 'note': 'v88 text-only ranked facing rectos per verso (row-aligned junction + line mood), frozen before the physical comparison', 'pred': preds}, open(fz, 'w'))
    print('FROZEN sha256', h, flush=True)
    # physical claims (voynich.nu sp_origin fetched 7 Oct 2026)
    claims = {'78v': '81r', '33v': '40r', '2v': '3r', '3v': '4r', '5v': '6r', '19v': '20r', '14v': '15r', '10v': '15r'}
    comp = {}
    for name, pr in preds.items():
        rk = {}
        for v_, r_ in claims.items():
            if v_ in pr and r_ in pr[v_]:
                rk[v_ + '/' + r_] = (pr[v_].index(r_) + 1, len(pr[v_]))
        # current facing pages in the binding (paint transfers 'mainly consistent with present binding')
        cur = [('%d%s' % P, '%d%s' % Q) for P, Q in G.pair_classes(quires, G.vpages(name)[0])['GUT']]
        cr = [pr[a].index(b) + 1 for a, b in cur if a in pr and b in pr[a]]
        n = len(next(iter(pr.values())))
        comp[name] = {'claims': rk, 'current_facing_mean_rank': float(np.mean(cr)), 'n_current': len(cr), 'chance_mean_rank': (n + 1) / 2,
                      'current_top5': int(sum(r <= 5 for r in cr)), 'chance_top5': len(cr) * 5 / n}
        print(name, json.dumps(comp[name]), flush=True)
    out['outside'] = comp; out['frozen_sha256'] = h
    json.dump(out, open(os.path.join(G.CK, 'c3.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
