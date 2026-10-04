"""pe14 cycle 3c: IS THERE A WEAVE BEYOND THE FIXED TWO-LINE RECORDS? (non-circular)
Removing the tablets that show a fixed record (an identical entry at every second
position >= 3 times) selects on period 2 itself, so the null must be treated the
same way: 300 DIP surrogates of the whole corpus (entries drawn one by one, weight
rho^shared signs: the pe13 dip), the SAME removal rule applied to every surrogate,
comb statistics (periods 2-6) computed on the surviving tablets of real and surrogate.
Also: run-length distribution of fixed records (>= 3, 4, 5, 6, 8 repeats) real vs DIP.
usage: python3 pe14_cycle3c.py <config>
"""
import json, os, random, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK, LAGS  # noqa
from pe14_cycle1 import corpus, run_spaced  # noqa
from pe14_cycle3 import prep, tune, dip_perm  # noqa
from pe14_sum1b import comb, PS  # noqa

NS = 300
RUNS = [3, 4, 5, 6, 8]


def featfns(T):
    lc = [x for x, _ in Counter(u['toks'][-1] for t in T for u in t['u']).most_common(20)]
    fc = [x for x, _ in Counter(u['toks'][0] for t in T for u in t['u']).most_common(20)]
    return {'last': lambda u: u['toks'][-1] if u['toks'][-1] in lc else None,
            'first': lambda u: u['toks'][0] if u['toks'][0] in fc else None,
            'lead': lambda u: u['lead'], 'size': lambda u: u['size'], 'sys': lambda u: u['sys'],
            'len': lambda u: min(len(u['toks']), 5),
            'M288': lambda u: 1 if 'M288' in u['toks'] else None,
            'M346': lambda u: 1 if 'M346' in u['toks'] else None,
            'M376': lambda u: 1 if 'M376' in u['toks'] else None,
            'entry': lambda u: tuple(u['toks'])}


def lagvec(tabs_units, fn):
    out = np.zeros(len(LAGS))
    for U in tabs_units:
        v = [fn(u) for u in U]
        n = len(v)
        for j, p in enumerate(LAGS):
            out[j] += sum(1 for i in range(n - p) if v[i] is not None and v[i] == v[i + p])
    return out


def stats(Us, F):
    keep = [U for U in Us if run_spaced(U, 2) < 3]
    runs = [run_spaced(U, 2) for U in Us]
    return ({k: lagvec(keep, fn) for k, fn in F.items()},
            {k: lagvec(Us, fn) for k, fn in F.items()},
            np.array([sum(1 for r in runs if r >= m) for m in RUNS]), len(keep))


def main(cfg):
    T = [t for t in corpus(cfg) if len(t['u']) >= 3]
    rng = random.Random(77 + sum(map(ord, cfg)))
    _, tabs = prep(T)
    rho, _ = tune(tabs, rng)
    F = featfns(T)
    real_c, real_all, real_runs, nkeep = stats([t['u'] for t in T], F)
    sc, sa, sr, sk = [], [], [], []
    for s in range(NS):
        Us = [[t['u'][i] for i in dip_perm(tb, rho, rng)] for t, tb in zip(T, tabs)]
        c, a, r, k = stats(Us, F)
        sc.append(c); sa.append(a); sr.append(r); sk.append(k)
        if s % 50 == 0:
            print(cfg, 'sur', s, flush=True)
    sr = np.array(sr)
    out = {'cfg': cfg, 'rho': rho, 'ntab': len(T), 'nkeep': nkeep, 'nkeep_sur': float(np.mean(sk)),
           'runs': {'obs': real_runs.tolist(), 'dip_m': sr.mean(0).tolist(),
                    'p': [float((1 + np.sum(sr[:, j] >= real_runs[j])) / (NS + 1)) for j in range(len(RUNS))]},
           'cond': {}, 'all': {}}
    for tag, real, sur in (('cond', real_c, sc), ('all', real_all, sa)):
        for k in F:
            S = np.array([x[k] for x in sur])
            m = S.mean(0)
            o = real[k]
            r = {'O/E': (o / np.maximum(m, 1e-9)).round(3).tolist()}
            for p in PS:
                cs, co = comb(S, m, p), comb(o, m, p)
                r['z%d' % p] = float((co - cs.mean()) / (cs.std() + 1e-12))
                r['p%d' % p] = float((1 + np.sum(cs >= co)) / (NS + 1))
            out[tag][k] = r
            print(cfg, tag, '%-6s' % k, 'O/E 1-8', ' '.join('%.2f' % x for x in r['O/E'][:8]),
                  '| ' + ' '.join('z%d %.1f' % (p, r['z%d' % p]) for p in PS), flush=True)
    print(cfg, 'runs >=', RUNS, 'obs', real_runs.tolist(), 'dip', sr.mean(0).round(1).tolist(), 'p', out['runs']['p'])
    json.dump(out, open(os.path.join(CK, 'c3c_%s.json' % cfg), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
