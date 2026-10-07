"""v90: freeze the surviving wheel decompositions and their predicted content-wheel values for the
held-out (test) folios, with a sha256; then score the frozen predictions on GC2a (a third,
independent transcription never used in the search).
Prediction per test folio: every rare wheel value (ZL count 2-20) seen on the folio's odd lines
recurs on the folio's even lines more often than on the even lines of other same-stratum folios."""
import os, sys, json, hashlib
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L


def wheel_str(t, cols, scheme=0):
    f = L.fields_of(t['u'], scheme)
    return '|'.join(f[c] for c in cols)


def survivors():
    out = []
    for name in ('VOY', 'VOY_IT'):
        r = json.load(open(os.path.join(L.CK, 'c2_%s.json' % name)))
        for x in sorted(r['top'], key=lambda x: -x['test']['z'])[:3]:
            if any(x['cmap'][c] is not None for c in x['cols']):
                continue      # freeze only uncoarsened wheels (values readable as glyph strings)
            out.append({'from': name, 'scheme': x['scheme'], 'cols': x['cols'], 'fields': x['fields'],
                        'test_z': x['test']['z'], 'decomp': x['decomp']})
    return out


def freeze():
    toks, order = L.voynich_tokens('ZL3b')
    tr, te = L.split_pages(order)
    S = survivors()
    preds = []
    for s in S:
        vals = [wheel_str(t, s['cols'], s['scheme']) for t in toks]
        tot = Counter(vals)
        per = defaultdict(set)
        for t, v in zip(toks, vals):
            if t['page'] in te and t['line'] % 2 == 0 and 2 <= tot[v] <= 20:
                per[t['page']].add(v)
        preds.append({'wheel': s, 'pred': {p: sorted(v) for p, v in sorted(per.items())}})
    obj = {'loop': 'v90', 'date': '2026-10-07', 'rule': 'rare wheel values on odd lines (0-based even line index) of each test folio recur on the other lines of the folio above the same-stratum rate',
           'test_folios': sorted(te), 'predictions': preds}
    h = L.sha(obj)
    obj['sha256'] = h
    json.dump(obj, open(os.path.join(L.DATA, 'v90_frozen_predictions.json'), 'w'), indent=0)
    print('frozen', h, [len(p['pred']) for p in preds])
    return obj


def score_gc(obj, wrong=False, seed=0):
    import random
    rng = random.Random(seed)
    toks, order = L.voynich_tokens('GC2a')
    meta = {}
    for t in toks:
        meta.setdefault(t['page'], (t['sec'], t['hand'], t['lang']))
    res = []
    for P in obj['predictions']:
        s = P['wheel']
        other = defaultdict(set)
        for t in toks:
            if t['line'] % 2 == 1:
                other[t['page']].add(wheel_str(t, s['cols'], s['scheme']))
        H = E = V = n = 0
        for p, vals in P['pred'].items():
            if p not in other:
                continue
            peers = [q for q in other if q != p and meta.get(q) == meta.get(p)]
            if len(peers) < 2:
                continue
            if wrong:   # control: the folio's predictions checked on a random same-stratum folio
                p2 = rng.choice(peers)
                peers = [q for q in other if q != p2 and meta.get(q) == meta.get(p2)]
                p = p2
            for v in vals:
                h = 1.0 if v in other[p] else 0.0
                e = sum(v in other[q] for q in peers) / len(peers)
                H += h; E += e; V += e * (1 - e); n += 1
        res.append({'fields': s['fields'], 'from': s['from'], 'n': n, 'hits': H, 'exp': E,
                    'lift': H / max(E, 1e-9), 'z': (H - E) / max(V, 1e-9) ** 0.5})
    return res


if __name__ == '__main__':
    obj = freeze()
    out = {'sha256': obj['sha256'], 'gc2a': score_gc(obj), 'wrong_folio': [score_gc(obj, True, k) for k in range(20)]}
    json.dump(out, open(os.path.join(L.CK, 'freeze_gc.json'), 'w'))
    for r in out['gc2a']:
        print(json.dumps(r))
    for i in range(len(out['gc2a'])):
        zs = [w[i]['z'] for w in out['wrong_folio']]
        print('wrong-folio control', i, 'z mean %.2f max %.2f' % (np.mean(zs), max(zs)))
