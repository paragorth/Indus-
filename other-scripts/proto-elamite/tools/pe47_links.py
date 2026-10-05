"""pe47 cycle 2: printouts of one sheet should leave arithmetic fingerprints BETWEEN tablets.

(a) SUMMARY -> DETAIL: an entry value on tablet S equals the (computed, unwritten) sum of tablet D's
    obverse entries of the same system, or of D's entries sharing a last token (>= 2 entries).
(b) SAME ROW PRINTED TWICE: two tablets share >= 2 identical (token string, value) entries.
Only 'rare' values count (value seen <= RARE times among all entries of its system, value >= VMIN),
so each match carries surprisal.
Nulls: N1 values shuffled across tablets within (last token, system); N2 whole entries re-dealt
between tablets. The number of links is compared with the nulls; then the links are VALIDATED with
information the numbers never saw: token overlap of S and D (PE, PLANT, Ur III) and same
(king, year, month) dating (Ur III), against the links that the null worlds produce between the same
real tablets.
"""
import collections, json, math, os, random, sys
import numpy as np
from pe47_common import CK, build_pe, build_ur3, sample_like, ent_lengths
from pe47_search import null_shuffle_values, null_permute_entries

RARE = 3
VMIN = 5.0


def plant2(n_target, seed, U=30, C=20, P=6, survive=0.06):
    """Master table printed as UP slices (header unit+period, entries = commodities) and PU totals
    (header period, entries = units, value = row sum over commodities = total of the UP slice)."""
    r = random.Random(seed)
    act = {u: set(r.sample(range(C), r.randint(3, 8))) for u in range(U)}
    su = [math.exp(r.gauss(0, 0.8)) for _ in range(U)]
    bc = [math.exp(r.gauss(1.5, 1.0)) for _ in range(C)]
    X = {(u, c, p): max(1, int(round(su[u] * bc[c] * math.exp(r.gauss(0, 0.5)))))
         for u in range(U) for c in act[u] for p in range(P)}
    tabs = []
    n_print = int(n_target / survive)
    for i in range(n_print):
        if r.random() < 0.85:
            u, p = r.randrange(U), r.randrange(P)
            ents = [[['Z%d' % r.randrange(60), 'C%d' % c], float(X[(u, c, p)]), 'cnt', False] for c in sorted(act[u])]
            tabs.append(dict(id='', ctx=['U%d' % u, 'P%d' % p], ents=ents, meta=dict(kind='UP', u=u, p=p)))
        else:
            p = r.randrange(P)
            us = r.sample(range(U), r.randint(3, 10))
            ents = [[['Z%d' % r.randrange(60), 'U%d' % u], float(sum(X[(u, c, p)] for c in act[u])), 'cnt', False] for u in us]
            tabs.append(dict(id='', ctx=['P%d' % p], ents=ents, meta=dict(kind='PU', p=p, us=us)))
    r.shuffle(tabs)
    tabs = tabs[:n_target]
    for i, t in enumerate(tabs):
        t['id'] = 'PL%05d' % i
    return tabs


def vk(v):
    return round(v, 4)


def find_links(corpus):
    freq = collections.Counter()
    for t in corpus:
        for e in t['ents']:
            freq[(e[2], vk(e[1]))] += 1
    rare = lambda s, v: v >= VMIN and freq[(s, v)] <= RARE
    # detail sums
    sums = collections.defaultdict(list)   # (sys, value) -> [(D, kind)]
    for di, t in enumerate(corpus):
        ob = [e for e in t['ents'] if not e[3]]
        bys = collections.defaultdict(list)
        for e in ob:
            bys[e[2]].append(e)
        for s, L in bys.items():
            if len(L) >= 2:
                sums[(s, vk(sum(e[1] for e in L)))].append((di, 'all'))
                byl = collections.defaultdict(list)
                for e in L:
                    byl[e[0][-1]].append(e)
                for lt, LL in byl.items():
                    if 2 <= len(LL) < len(L):
                        sums[(s, vk(sum(e[1] for e in LL)))].append((di, 'tok'))
    summ = []
    for si, t in enumerate(corpus):
        for ei, e in enumerate(t['ents']):
            k = (e[2], vk(e[1]))
            if not rare(*k):
                continue
            for di, kind in sums.get(k, []):
                if di != si:
                    summ.append((si, ei, di, kind))
    # same row printed twice
    cell = collections.defaultdict(set)
    for ti, t in enumerate(corpus):
        for e in t['ents']:
            if rare(e[2], vk(e[1])):
                cell[(e[0][-1], e[2], vk(e[1]))].add(ti)
    pairc = collections.Counter()
    for k, S in cell.items():
        S = sorted(S)
        if len(S) > 8:
            continue
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                pairc[(S[a], S[b])] += 1
    twins = [p for p, n in pairc.items() if n >= 2]
    # value-only twins (identical rare value, any tokens)
    vcell = collections.defaultdict(set)
    for ti, t in enumerate(corpus):
        for e in t['ents']:
            if rare(e[2], vk(e[1])):
                vcell[(e[2], vk(e[1]))].add(ti)
    vpair = collections.Counter()
    for k, S in vcell.items():
        S = sorted(S)
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                vpair[(S[a], S[b])] += 1
    vtwins = [p for p, n in vpair.items() if n >= 2]
    return summ, twins, vtwins


def toks(t):
    s = set(t['ctx'])
    for e in t['ents']:
        s |= set(e[0])
    return s


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def validate(corpus, summ, twins, vtwins):
    TK = [toks(t) for t in corpus]
    out = {}
    pairs = sorted({(s, d) for s, _, d, _ in summ})
    out['n_summ_links'] = len(summ)
    out['n_summ_pairs'] = len(pairs)
    out['summ_jac'] = float(np.mean([jac(TK[s], TK[d]) for s, d in pairs])) if pairs else None
    # does the summary entry's token string name something in D (header or entries)?
    hit = [len(set(corpus[s]['ents'][e][0]) & TK[d]) > 0 for s, e, d, _ in summ]
    out['summ_entry_names_D'] = float(np.mean(hit)) if hit else None
    out['n_twins'] = len(twins)
    out['twin_jac'] = float(np.mean([jac(TK[a], TK[b]) for a, b in twins])) if twins else None
    out['n_vtwins'] = len(vtwins)
    out['vtwin_jac'] = float(np.mean([jac(TK[a], TK[b]) for a, b in vtwins])) if vtwins else None
    if 'month' in corpus[0]['meta']:
        def same(a, b):
            ma, mb = corpus[a]['meta'], corpus[b]['meta']
            return ma['king'] == mb['king'] and ma['year'] == mb['year'] and ma['year'] not in ('', '00')
        def samem(a, b):
            return same(a, b) and corpus[a]['meta']['month'] == corpus[b]['meta']['month'] and corpus[a]['meta']['month'] > 0
        for nm, P in (('summ', pairs), ('twin', twins), ('vtwin', vtwins)):
            out[nm + '_sameyear'] = float(np.mean([same(a, b) for a, b in P])) if P else None
            out[nm + '_samemonth'] = float(np.mean([samem(a, b) for a, b in P])) if P else None
        r = random.Random(0)
        rp = [tuple(r.sample(range(len(corpus)), 2)) for _ in range(20000)]
        out['rand_sameyear'] = float(np.mean([same(a, b) for a, b in rp]))
        out['rand_samemonth'] = float(np.mean([samem(a, b) for a, b in rp]))
    if 'kind' in corpus[0]['meta']:
        def true_link(s, e, d):
            ms, md = corpus[s]['meta'], corpus[d]['meta']
            if ms['kind'] == 'PU' and md['kind'] == 'UP':
                u = int(corpus[s]['ents'][e][0][-1][1:])
                return md['u'] == u and md['p'] == ms['p']
            return False
        tl = [true_link(s, e, d) for s, e, d, _ in summ]
        out['summ_true_share'] = float(np.mean(tl)) if tl else None
        out['summ_true_n'] = int(sum(tl))
    r = random.Random(1)
    rp = [tuple(r.sample(range(len(corpus)), 2)) for _ in range(20000)]
    out['rand_jac'] = float(np.mean([jac(TK[a], TK[b]) for a, b in rp]))
    return out


def get(name, seed):
    pe = build_pe()
    if name == 'PE':
        return build_pe()
    if name == 'PLANT2':
        return plant2(len(pe), 3000 + seed)
    if name == 'UR3D':
        return sample_like(build_ur3('drehem'), len(pe), 77 + seed)
    if name == 'UR3U':
        return sample_like(build_ur3('umma'), len(pe), 77 + seed)
    if name == 'UR3D_FULL':
        return build_ur3('drehem')
    if name == 'UR3U_FULL':
        return build_ur3('umma')


def job(a):
    name, null, seed = a
    c = get(name, seed if null == 'real' else 0)
    if null == 'N1':
        c = null_shuffle_values(c, 900 + seed)
    elif null == 'N2':
        c = null_permute_entries(c, 900 + seed)
    summ, tw, vtw = find_links(c)
    v = validate(c, summ, tw, vtw)
    v.update(name=name, null=null, seed=seed)
    if name == 'PE' and null == 'real':
        v['links'] = [(c[s]['id'], ' '.join(c[s]['ents'][e][0]), c[s]['ents'][e][1], c[d]['id'], k) for s, e, d, k in summ]
        v['twins_ids'] = [(c[a]['id'], c[b]['id']) for a, b in tw]
    return v


if __name__ == '__main__':
    from multiprocessing import Pool
    out = sys.argv[1]
    jobs = []
    for name in ('PLANT2', 'PE', 'UR3D', 'UR3U', 'UR3D_FULL', 'UR3U_FULL'):
        nseeds = 1 if name in ('PE',) or 'FULL' in name else 3
        for s in range(nseeds):
            jobs.append((name, 'real', s))
        nn = 3 if 'FULL' in name else 20
        for s in range(nn):
            jobs.append((name, 'N1', s))
            jobs.append((name, 'N2', s))
    res = []
    with Pool(2) as p:
        for v in p.imap_unordered(job, jobs):
            res.append(v)
            json.dump(res, open(out, 'w'))
    print('done', len(res))
