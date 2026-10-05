"""pe44 cycle 4: QUOTA CEILINGS.  A norm leaves series that never exceed it: entries of the same item on
one tablet sit AT a round value or a little BELOW it, never above (P009258: M297 = 10, 10, 9, 8, 8, 8, 7).
Series = entries sharing the final sign (PE) / item word (Ur III) on one tablet, >= 3 entries, same system.
CAPPED: max is one-denomination (round, >= 2nd denomination), max occurs >= 2 times, every other value
in [0.6 max, max).
Nulls (200 reps): S1 values shuffled among the tablet's own entries (same system): keeps the tablet's
value set, breaks the item grouping;  N3 values shuffled corpus-wide within (system, log2 size).
Ur III control: tablets with per-head rations ('-ta' capacity lines) should carry more capped series
than livestock / labour tablets (known norm vs counted).  Planted: 25 PE series forced to a cap.
"""
import json, math, os, random, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle2 import shuffle_within_size


def series(recs):
    g = defaultdict(list)
    for r in recs:
        if r['item'] not in ('-', None, 'x'):
            g[(r['item'], r['sys'])].append(r['val'])
    return {k: v for k, v in g.items() if len(v) >= 3}


def capped(vals, den):
    m = max(vals)
    if m < den[1] or value_feats(m, den)[0] != 1 or vals.count(m) < 2:
        return 0
    return int(all(v == m or 0.6 * m <= v < m for v in vals))


def n_capped(tabs, detail=False):
    n, hits = 0, []
    for t, recs in tabs.items():
        for k, v in series(recs).items():
            c = capped(v, DEN[k[1]])
            n += c
            if c and detail:
                hits.append((t, k[0], v))
    return (n, hits) if detail else n


def shuffle_tablet(tabs, rng):
    out = {}
    for t, recs in tabs.items():
        by = defaultdict(list)
        for i, r in enumerate(recs):
            by[r['sys']].append(i)
        new = [dict(r) for r in recs]
        for s, idx in by.items():
            vals = [recs[i]['val'] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                new[i]['val'] = v
        out[t] = new
    return out


def test(tabs, rng, reps=200):
    real, hits = n_capped(tabs, True)
    out = {'real': real}
    for tag, nf in (('S1', shuffle_tablet), ('N3', shuffle_within_size)):
        a = np.array([n_capped(nf(tabs, random.Random(int(rng.integers(1e9))))) for _ in range(reps)])
        out[tag] = {'null': float(a.mean()), 'p_hi': float((1 + (a >= real).sum()) / (1 + reps)),
                    'z': float((real - a.mean()) / (a.std() + 1e-9))}
    return out, hits


def plant(tabs, rng, n=25):
    tb = {t: [dict(r) for r in recs] for t, recs in tabs.items()}
    cand = [(t, k) for t, recs in tb.items() for k in series(recs)]
    for t, (item, sy) in rng.sample(cand, n):
        den = DEN[sy]
        idx = [i for i, r in enumerate(tb[t]) if r['item'] == item and r['sys'] == sy]
        m = max(den[1], max(tb[t][i]['val'] for i in idx))
        hi = max(k for k, d in enumerate(den) if d <= m)
        cap = den[hi] * int(math.ceil(m / den[hi]))
        cap = cap if value_feats(cap, den)[0] == 1 else den[min(hi + 1, len(den) - 1)]
        for j, i in enumerate(idx):
            tb[t][i]['val'] = cap if j < 2 else max(1, int(cap * rng.uniform(0.6, 1.0 - 1e-9)))
    return tb


if __name__ == '__main__':
    rng = np.random.default_rng(41)
    res = {}
    pe = pe_tabs()
    res['PE'], hits = test(pe, rng)
    res['PE_hits'] = hits
    print('PE', res['PE'], len(hits), flush=True)
    for h in hits:
        print('  ', h, flush=True)
    res['PE_planted'], _ = test(plant(pe, random.Random(3)), rng, reps=100)
    print('PLANT', res['PE_planted'], flush=True)
    u = ur3_tabs()
    rt = [t for t, d in u.items() if any(r['lab'] == 'RATION' for r in d['recs'])]
    ct = [t for t, d in u.items() if any(r['lab'] in ('LIVESTOCK', 'PEOPLE') for r in d['recs'])
          and not any(r['lab'] == 'RATION' for r in d['recs'])]
    rr = random.Random(5)
    for name, ids in (('UR3_ration', rt), ('UR3_counted', rr.sample(ct, min(len(ct), 3000)))):
        tabs = {t: u[t]['recs'] for t in ids}
        nser = sum(len(series(r)) for r in tabs.values())
        res[name], _ = test(tabs, rng, reps=100)
        res[name]['n_tabs'] = len(ids); res[name]['n_series'] = nser
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
