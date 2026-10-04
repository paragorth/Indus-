"""pe12 cycle 3c: ADJACENT ANTI-REPETITION (side result of the SEQ family).
In cycles 1-2 the previous/next entry's slot sign predicted the slot sign WORSE than a random
tablet-mate does (STRICT null above real, z -3 to -8).  Direct test: share of adjacent entry
pairs with the same slot sign vs 1,000 within-(tablet, system) permutations of entry order.
Controls: proto-cuneiform (same engine), Ur III weight lists, and a planted PE corpus where
one tablet in three is re-ordered so that equal signs are grouped (runs) -> must go UP.
"""
import os, sys, json
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe12_common import pe_entries, PEDATA  # noqa
from common import norm_code  # noqa

rng = np.random.default_rng(33)


def adj_rate(E, slot, nperm=1000):
    by = defaultdict(list)
    for e in E:
        by[(e['tab'], e['sys'])].append(e)
    seqs = []
    for k, L in by.items():
        L = sorted(L, key=lambda e: e['idx'])
        if len(L) >= 3:
            seqs.append([(e['signs'][-1] if slot == 'LAST' else e['signs'][0]) for e in L])
    def rate(S):
        same = tot = 0
        for s in S:
            same += sum(a == b for a, b in zip(s, s[1:])); tot += len(s) - 1
        return same / tot
    real = rate(seqs)
    null = np.array([rate([list(rng.permutation(s)) for s in seqs]) for _ in range(nperm)])
    return dict(n_seq=len(seqs), real=round(real, 4), null=round(float(null.mean()), 4),
                z=round(float((real - null.mean()) / null.std()), 2),
                p_low=float((1 + (null <= real).sum()) / (1 + nperm)),
                p_high=float((1 + (null >= real).sum()) / (1 + nperm)))


def grouped(E, frac=1 / 3):
    by = defaultdict(list)
    for e in E:
        by[e['tab']].append(e)
    out = []
    for t, L in by.items():
        if rng.random() < frac:
            L = sorted(L, key=lambda e: (e['signs'][-1], e['idx']))
            L = [dict(e, idx=j) for j, e in enumerate(L)]
        out.extend(L)
    return out


if __name__ == '__main__':
    res = {}
    E = pe_entries()
    for slot in ['LAST', 'FIRST']:
        res['PE_' + slot] = adj_rate(E, slot)
    res['PE_LAST_planted_runs'] = adj_rate(grouped(E), 'LAST')
    PC = json.load(open(os.path.join(PEDATA, 'pe2_pc_corpus.json')))
    for t in PC:
        for l in t['lines']:
            l['numerals'] = [[n, norm_code(c)] for n, c in l['numerals']]
    EP = pe_entries(PC, corpus='PC')
    for slot in ['LAST', 'FIRST']:
        res['PC_' + slot] = adj_rate(EP, slot)
    from pe12_ur3 import ur3_entries
    EU = ur3_entries()
    res['UR3_unit'] = adj_rate(EU, 'FIRST')
    res['UR3_commodity'] = adj_rate(EU, 'LAST')
    for k, v in res.items():
        print(k, v, flush=True)
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle3c.json'), 'w'), indent=1)


def lag_profile(E, slot='LAST', lags=range(1, 9), nperm=300):
    """Same-sign rate at lag L vs within-(tablet, system) permutation: a fixed-order
    multi-line record template gives a deficit at lag 1 and an excess at its period."""
    by = defaultdict(list)
    for e in E:
        by[(e['tab'], e['sys'])].append(e)
    seqs = [[(e['signs'][-1] if slot == 'LAST' else e['signs'][0]) for e in sorted(L, key=lambda e: e['idx'])]
            for L in by.values() if len(L) >= 4]
    def rate(S, L):
        same = tot = 0
        for s in S:
            same += sum(a == b for a, b in zip(s, s[L:])); tot += max(len(s) - L, 0)
        return same / max(tot, 1)
    out = {}
    for L in lags:
        r = rate(seqs, L)
        nl = np.array([rate([list(rng.permutation(s)) for s in seqs], L) for _ in range(nperm)])
        out[L] = dict(real=round(r, 4), null=round(float(nl.mean()), 4), z=round(float((r - nl.mean()) / nl.std()), 1))
    return out


if __name__ == '__main__':
    res2 = {'PE_LAST': lag_profile(E, 'LAST'), 'PE_FIRST': lag_profile(E, 'FIRST'),
            'PE_planted_runs': lag_profile(grouped(E), 'LAST'),
            'PC_LAST': lag_profile(EP, 'LAST'), 'UR3_unit': lag_profile(EU, 'FIRST')}
    for k, v in res2.items():
        print('lag', k, {L: (d['real'], d['null'], d['z']) for L, d in v.items()}, flush=True)
    res['lag'] = res2
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle3c.json'), 'w'), indent=1)
