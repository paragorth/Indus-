"""pe7 cycle 4: is the PE co-tablet resemblance kinship (households) or a tablet formula?

(a) herd office re-test with the flag = pe5 herd-office tablets (no sign in the flag).
(b) coverage: for every element shared by >= 2 distinct names on a tablet (>= 4 names),
    the share of the tablet's names that carry it, and the number of different shared
    elements per tablet.  Households (several kin groups per tablet) -> several shared
    elements of partial coverage; a tablet formula -> one element covering most names.
    Reference: the same PE tablets with planted kin groups (2-3 names sharing one
    PE-frequency element, real shared structure first removed by shuffling), Linear B.
(c) residual co-tablet excess after deleting the shared signs significant at p <= 0.001
    in cycle 3 (and after deleting the 25 most frequent signs).
(d) distance decay: share of pairs sharing an element at line distance 1, 2, 3-4, 5+ on
    the same tablet, PE vs Linear B; families written as blocks decay, a formula is flat."""
import sys, os, json, random
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import *  # noqa
from pe7_cycle3 import cotablet
import pe7_cycle2 as c2
from common import load, entries
from pe4_common import FINAL


def pe_lines():
    """per tablet, multi-sign middles in line order (consecutive duplicates kept)."""
    T = load()
    E = entries(T, require_clean=True)
    by = defaultdict(list)
    for e in E:
        s = e['signs']
        mid = tuple(s[:-1]) if (s[-1] in FINAL and len(s) >= 2) else tuple(s)
        if len(mid) >= 2:
            by[e['tablet']].append(mid)
    return by


def linb_lines():
    C = corpora()
    by = defaultdict(list)
    for x in C['LINB']:
        if len(x['seq']) >= 2:
            for t in x['tablets']:
                by[t].append(tuple(x['seq']))
    return by   # order within a tablet = first-appearance order in DAMOS (line order)


def coverage(tabs):
    cov, nshared, top = [], [], []
    for names in tabs:
        names = list(dict.fromkeys(names))
        if len(names) < 4:
            continue
        c = Counter(e for w in names for e in set(w))
        sh = {e: k for e, k in c.items() if k >= 2}
        nshared.append(len(sh))
        for e, k in sh.items():
            cov.append(k / len(names))
        top.append(max(sh.values()) / len(names) if sh else 0)
    cov = np.array(cov)
    return {'tablets': len(nshared), 'shared_elements_per_tablet': float(np.mean(nshared)),
            'events': len(cov), 'share_cov_ge50': float((cov >= .5).mean()) if len(cov) else None,
            'median_cov': float(np.median(cov)) if len(cov) else None, 'mean_top_cov': float(np.mean(top))}


def planted_tabs(tabs, rng):
    allw = [w for v in tabs for w in v]
    f = Counter(s for w in allw for s in w)
    el, wt = zip(*f.items())
    toks = [s for w in allw for s in w]
    rng.shuffle(toks)
    out, k = [], 0
    for v in tabs:
        v = list(dict.fromkeys(v))
        nv = []
        for w in v:
            nv.append(list(toks[k:k + len(w)]))
            k += len(w)
        idx = list(range(len(nv)))
        rng.shuffle(idx)
        i = 0
        while i < len(idx) - 1:
            g = idx[i:i + rng.choice((2, 3))]
            e = rng.choices(el, wt)[0]
            for j in g:
                nv[j][rng.randrange(len(nv[j]))] = e
            i += len(g)
        out.append([tuple(w) for w in nv])
    return out


def decay(tabs, nperm=500, seed=3):
    rng = np.random.default_rng(seed)
    bins = {1: '1', 2: '2', 3: '3-4', 4: '3-4'}

    def stat(tl):
        a, n = Counter(), Counter()
        for v in tl:
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if v[i] == v[j]:
                        continue
                    b = bins.get(j - i, '5+')
                    n[b] += 1
                    a[b] += bool(set(v[i]) & set(v[j]))
        return {b: a[b] / n[b] for b in n}, n
    obs, n = stat(tabs)
    nulls = defaultdict(list)
    for _ in range(nperm):
        tl = [[v[i] for i in rng.permutation(len(v))] for v in tabs]
        s, _ = stat(tl)
        for b, x in s.items():
            nulls[b].append(x)
    return {b: {'obs': obs[b], 'within_tablet_shuffle': float(np.mean(nulls[b])), 'pairs': n[b],
                'p_high': float((1 + sum(x >= obs[b] for x in nulls[b])) / (nperm + 1))} for b in sorted(obs)}


def strip(tabs, drop):
    out = []
    for v in tabs:
        w = [tuple(s for s in x if s not in drop) for x in v]
        w = [x for x in w if x]
        if len(w) >= 2:
            out.append(w)
    return out


def partB():
    r = ckpt('c4b')
    if r:
        return r
    res = {}
    pe = pe_lines()
    lb = linb_lines()
    pe_t = [v for v in pe.values() if len(set(v)) >= 2]
    lb_t = [v for v in lb.values() if len(v) >= 2]
    res['coverage_PE'] = coverage(pe_t)
    res['coverage_LINB'] = coverage(lb_t)
    rng = random.Random(4)
    res['coverage_PE_planted_kin'] = coverage(planted_tabs(pe_t, rng))
    res['coverage_PE_shuffled'] = coverage(planted_tabs(pe_t, random.Random(5)) and
                                           [list(t) for t in [shuffle_elements(v, random.Random(6)) for v in pe_t]])
    B = ckpt('c3b')
    sig = {r_[0] for r_ in B['PE_cotablet']['elements'] if r_[4] <= 0.001 and not r_[0].startswith('<')}
    res['drop_sig'] = sorted(sig)
    dist = [list(dict.fromkeys(v)) for v in pe_t]
    res['resid_drop_sig'] = {k: v for k, v in cotablet(strip(dist, sig), nperm=1000).items() if k != 'elements'}
    f = Counter(s for v in dist for w in v for s in w)
    top25 = {s for s, _ in f.most_common(25)}
    res['resid_drop_top25'] = {k: v for k, v in cotablet(strip(dist, top25), nperm=1000).items() if k != 'elements'}
    res['decay_PE'] = decay(pe_t)
    res['decay_LINB'] = decay(lb_t)
    res['decay_PE_planted_blocks'] = decay([[w for w in v] for v in planted_block(pe_t, random.Random(8))])
    save_ckpt('c4b', res)
    return res


def planted_block(tabs, rng):
    """shuffled PE tablets in which adjacent lines form kin blocks of 2-3 sharing an element."""
    allw = [w for v in tabs for w in v]
    f = Counter(s for w in allw for s in w)
    el, wt = zip(*f.items())
    out = []
    for v in tabs:
        nv = [list(w) for w in shuffle_elements(v, rng)]
        i = 0
        while i < len(nv) - 1:
            L = rng.choice((2, 3))
            e = rng.choices(el, wt)[0]
            for j in range(i, min(i + L, len(nv))):
                nv[j][rng.randrange(len(nv[j]))] = e
            i += L
        out.append([tuple(w) for w in nv])
    return out


if __name__ == '__main__':
    with Pool(2) as pool:
        b = pool.apply_async(partB)
        herd = {}
        for k, o in pool.imap_unordered(c2.run, [('PE_herdtab', r) for r in range(c2.NREP)]):
            herd[k] = o
            print(k, {l: (v['PS']['p'], v['dist'] and round(v['dist']['z'], 2)) for l, v in o.items() if isinstance(v, dict) and 'PS' in v}, flush=True)
        res = {'B': b.get(), 'herd': herd}
    json.dump(res, open(os.path.join(DATA, 'pe7_cycle4.json'), 'w'))
    print(json.dumps(res['B'], indent=1)[:6000])
