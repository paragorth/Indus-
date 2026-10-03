#!/usr/bin/env python3
"""PE-1.2: role consistency of recurring designations across tablets.
For every pair of occurrences of the same designation on DIFFERENT tablets:
same final class sign? same header opener? same number system? quantity
distance |log10 a - log10 b| and exact-equal value (same system only).
Controls: (A) designation labels permuted over all designation entries;
(B) labels permuted within (final, system) strata - quantity beyond class;
(C) tablet-kept: partner replaced by a random designation entry from the
SAME partner tablet (keeps tablet pairing, breaks designation identity)."""
import json, random, math
from pe1_lib import *
T = load(); meta = tablet_meta(T)
R = 1000
def run(minlen, strip_prefix=False):
    DE = designation_entries(T, minlen=minlen, strip_prefix=strip_prefix)
    for e in DE: e['hdr'] = meta[e['tablet']]['header']
    bt = collections.defaultdict(list)
    for i, e in enumerate(DE): bt[e['tablet']].append(i)
    def pairs(labels):
        g = collections.defaultdict(list)
        for i, l in enumerate(labels): g[l].append(i)
        P = []
        for l, ix in g.items():
            if len({DE[i]['tablet'] for i in ix}) < 2: continue
            for a in range(len(ix)):
                for b in range(a + 1, len(ix)):
                    i, j = ix[a], ix[b]
                    if DE[i]['tablet'] != DE[j]['tablet']: P.append((i, j))
        return P
    def stats(P):
        n = len(P); s = {'pairs': n}
        if not n: return s
        s['same_final'] = sum(DE[i]['final'] == DE[j]['final'] for i, j in P) / n
        hp = [(i, j) for i, j in P if DE[i]['hdr'] and DE[j]['hdr']]
        s['same_header'] = sum(DE[i]['hdr'] == DE[j]['hdr'] for i, j in hp) / max(1, len(hp))
        sp = [(i, j) for i, j in P if DE[i]['sys'] and DE[j]['sys']]
        s['same_system'] = sum(DE[i]['sys'] == DE[j]['sys'] for i, j in sp) / max(1, len(sp))
        qp = [(i, j) for i, j in sp if DE[i]['sys'] == DE[j]['sys'] and DE[i]['val'] and DE[j]['val']]
        s['n_qty_pairs'] = len(qp)
        s['qty_dlog'] = sum(abs(math.log10(DE[i]['val']) - math.log10(DE[j]['val'])) for i, j in qp) / max(1, len(qp))
        s['qty_equal'] = sum(DE[i]['val'] == DE[j]['val'] for i, j in qp) / max(1, len(qp))
        return s
    lab = [e['des'] for e in DE]
    P = pairs(lab); obs = stats(P)
    rng = random.Random(3)
    nA, nB, nC = [], [], []
    strata = collections.defaultdict(list)
    for i, e in enumerate(DE): strata[(e['final'], e['sys'])].append(i)
    for r in range(R):
        l2 = lab[:]; rng.shuffle(l2); nA.append(stats(pairs(l2)))
        l3 = lab[:]
        for k, ix in strata.items():
            v = [lab[i] for i in ix]; rng.shuffle(v)
            for i, x in zip(ix, v): l3[i] = x
        nB.append(stats(pairs(l3)))
        PC = [(i, rng.choice(bt[DE[j]['tablet']])) for i, j in P]
        nC.append(stats(PC))
    out = {'obs': obs}
    for k in ['same_final', 'same_header', 'same_system', 'qty_dlog', 'qty_equal']:
        for nm, nl in [('A_perm', nA), ('B_strat', nB), ('C_tabletkept', nC)]:
            if nm == 'C_tabletkept' and k == 'same_header': continue
            v = [x[k] for x in nl if k in x]
            out[f'{k}|{nm}'] = zp(obs[k], v, greater=(k != 'qty_dlog'))
    return out
res = {}
for lab, kw in [('len>=2', dict(minlen=2)), ('len>=3', dict(minlen=3)), ('len>=2 prefix-stripped', dict(minlen=2, strip_prefix=True))]:
    res[lab] = o = run(**kw)
    print('==', lab, o['obs'])
    for k, v in o.items():
        if '|' in k: print('  %-28s obs %.3f null %.3f sd %.3f z %6.2f p %.4f' % (k, v['obs'], v['null_mean'], v['null_sd'], v['z'], v['p']))
json.dump(res, open(os.path.join(DATA, 'pe1_c2_roles.json'), 'w'), indent=1)
