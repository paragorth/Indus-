"""pe23 group scan: does a group's numbers carry more (or less) of a fingerprint than other
numbers of the same system and size?  Per-entry indicators, compared with the mean of their
stratum; z uses a cluster-robust (tablet) variance, never smaller than the iid variance.
Strata: 'mag' = (system, log2 size bin); 'tab' = (tablet, system, log4 size bin) [within-tablet].
"""
import math
from collections import defaultdict, Counter
import numpy as np
from pe23_common import pe_den, canon_arrays, OFFICES, bh
from scipy.stats import norm


def indicators(ents):
    """Adds per-entry fingerprint indicators (None where undefined)."""
    rep = Counter((e['tab'], e['sys'], e['val']) for e in ents)
    for s in {e['sys'] for e in ents}:
        idx = [i for i, e in enumerate(ents) if e['sys'] == s]
        den = pe_den(s, ents[idx[0]].get('vset', 'B'))
        vals = np.array([ents[i]['val'] for i in idx], dtype=np.int64)
        nlev, low, lead = canon_arrays(vals, den)
        for j, i in enumerate(idx):
            e = ents[i]
            big = e['val'] >= den[1]
            e['ROUND'] = float(nlev[j] == 1) if big else None
            e['LOW'] = float(low[j]) if big else None
            e['LEAD1'] = float(lead[j] == 1) if big else None
            e['FIVE'] = float(e['val'] % 5 == 0) if e['val'] >= 10 else None
            e['REP'] = float(rep[(e['tab'], e['sys'], e['val'])] > 1)
        # composite 'roundness' (entries >= 2nd denomination): mean of standardised ROUND, LOW, FIVE
        sub = [ents[i] for i in idx if ents[i]['ROUND'] is not None]
        if len(sub) > 5:
            st = {}
            for f in ('ROUND', 'LOW', 'FIVE'):
                a = np.array([e[f] for e in sub if e[f] is not None])
                st[f] = (a.mean(), a.std() + 1e-9)
            for e in sub:
                zs = [(e[f] - st[f][0]) / st[f][1] for f in ('ROUND', 'LOW', 'FIVE') if e[f] is not None]
                e['COMP'] = float(np.mean(zs))
    for e in ents:
        e.setdefault('COMP', None)
    return ents


def memberships(e, key):
    if key == 'final':
        return [e['final']] if e['final'] else []
    if key == 'first':
        return [e['first']] if e['first'] and e['nsign'] > 1 else []
    if key == 'anysign':
        return sorted(set(e['signs']))
    if key == 'hdr':
        return [e['hdr'] or 'NONE']
    if key == 'site':
        return [e['site']]
    if key == 'surface':
        return [e['surface']]
    if key == 'office':
        return [o for o, S in OFFICES.items() if set(e['signs']) & S]
    if key == 'size':
        n = e['nent']
        return ['ent1-3' if n <= 3 else 'ent4-9' if n <= 9 else 'ent10-19' if n <= 19 else 'ent20+']
    if key == 'sys':
        return [e['sys'] + ('@' if e['hatched'] else '')]
    if key == 'nsign':
        return [f"ns{min(e['nsign'], 4)}"]
    return []


def stratum(e, how):
    b = int(math.log2(max(1, e['val'])))
    if how == 'mag':
        return (e['sys'], b)
    return (e['tab'], e['sys'], b // 2)


def scan(ents, keys=('final',), feats=('ROUND',), strata='mag', min_n=15, min_tabs=5,
         by_system=True):
    ents = indicators([dict(e) for e in ents])
    out = []
    for f in feats:
        sub = [e for e in ents if e.get(f) is not None]
        sm = defaultdict(list)
        for e in sub:
            sm[stratum(e, strata)].append(e[f])
        mean = {k: np.mean(v) for k, v in sm.items()}
        var = {k: np.var(v) for k, v in sm.items()}
        sizes = {k: len(v) for k, v in sm.items()}
        for key in keys:
            groups = defaultdict(list)
            for e in sub:
                for g in memberships(e, key):
                    sy = e['sys'] if by_system else 'ALL'
                    groups[(g, sy)].append(e)
            for (g, sy), ge in groups.items():
                # entries only in strata that also have outsiders
                ge = [e for e in ge if sizes[stratum(e, strata)] > 1]
                tabs = Counter(e['tab'] for e in ge)
                if len(ge) < min_n or len(tabs) < min_tabs:
                    continue
                res = np.array([e[f] - mean[stratum(e, strata)] for e in ge])
                obs = float(np.mean([e[f] for e in ge]))
                exp = obs - float(res.mean())
                vi = sum(var[stratum(e, strata)] for e in ge)
                ct = defaultdict(float)
                for e, r in zip(ge, res):
                    ct[e['tab']] += r
                G = len(ct)
                vc = sum(x * x for x in ct.values()) * G / max(1, G - 1)
                v = max(vi, vc)
                z = float(res.sum() / math.sqrt(v)) if v > 0 else 0.0
                out.append({'key': key, 'group': g, 'sys': sy, 'feat': f, 'n': len(ge), 'tabs': G,
                            'obs': obs, 'exp': exp, 'z': z, 'p': float(2 * norm.sf(abs(z)))})
    if out:
        q = bh([r['p'] for r in out])
        for r, qq in zip(out, q):
            r['q'] = float(qq)
    return out
