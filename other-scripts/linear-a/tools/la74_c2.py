#!/usr/bin/env python3
"""la74 cycle 2: (a) held-out sites: train on Haghia Triada, test on all other sites; shuffled control.
(b) magnitude test of the pouring picture: survivors' fitted vessel sizes (strings only, no integers)
predict that fractions on big integers use the larger vessels. Null: integers permuted among
fraction quantities within site."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
import la74_c1 as C1
from multiprocessing import Pool

def run_site(name, st, sites, signs, V):
    train = [s for s, g in zip(st, sites) if g == 'Haghia Triada']
    test = [s for s, g in zip(st, sites) if g != 'Haghia Triada']
    with Pool(2) as P:
        out = P.map(C1.job, [(900 + w, train, test, signs, V) for w in range(2)])
    res = sorted([r for o in out for r in o], key=lambda r: -r['gtr'])
    top = res[:len(res) // 100]
    summ = dict(name=name, n_train=len(train), n_test=len(test), best_gtr=top[0]['gtr'],
                top_gte_mean=float(np.mean([r['gte'] for r in top])), top_gte_pos=int(sum(r['gte'] > 0 for r in top)),
                ntop=len(top), frac_e1=float(np.mean([r['e'] == 1.0 for r in res])),
                best=dict(map=top[0]['map'], sizes=top[0]['H']['sizes'], e=top[0]['e'], gte=top[0]['gte']))
    json.dump(dict(summ=summ, top=top), open(os.path.join(CK, f'c2_{name}.json'), 'w'), default=float)
    print(json.dumps(summ, default=float), flush=True)

def magnitude(top, S, V, rng, nperm=20000):
    """per survivor: sign -> log size; statistic = mean log size (N>=5) - mean log size (N==0)"""
    st = [norm(r['s'], V) for r in S]; N = np.array([r['v'] for r in S]); site = [r['site'] for r in S]
    out = []
    for r in top:
        ls = {x: math.log(v) for x, v in zip(r['map'], r['H']['sizes'])}
        val = np.array([np.mean([ls[x] for x in s if x in ls]) if any(x in ls for x in s) else np.nan for s in st])
        ok = ~np.isnan(val)
        def stat(Nv):
            a = val[ok & (Nv >= 5)]; b = val[ok & (Nv == 0)]
            return a.mean() - b.mean() if len(a) and len(b) else np.nan
        obs = stat(N)
        sites = np.array(site); null = []
        for _ in range(nperm // len(top)):
            Np = N.copy()
            for g in set(site):
                idx = np.where(sites == g)[0]; Np[idx] = rng.permutation(N[idx])
            null.append(stat(Np))
        null = np.array(null)
        out.append(dict(obs=float(obs), p=float((np.sum(null >= obs) + 1) / (len(null) + 1)), n_big=int((ok & (N >= 5)).sum()),
                        n_zero=int((ok & (N == 0)).sum())))
    return out

if __name__ == '__main__':
    S = load_strings(); V = vocab([r['s'] for r in S]); C1.VFULL = V
    st = [norm(r['s'], V) for r in S]; sites = [r['site'] for r in S]; signs = V[:12]
    rng = np.random.default_rng(742)
    which = sys.argv[1:] or ['site', 'siteshuf', 'mag']
    if 'site' in which: run_site('site', st, sites, signs, V)
    if 'siteshuf' in which: run_site('siteshuf', shuffle_signs(st, rng), sites, signs, V)
    if 'mag' in which:
        for nm in ['real', 'real2', 'shuf', 'nvworld']:
            top = json.load(open(os.path.join(CK, f'c1_{nm}.json')))['top']
            # magnitude test always uses the REAL integers and REAL strings; for control corpora
            # the survivors' size maps are scored on real data (they carry no real information)
            m = magnitude(top, S, V, rng, nperm=6000)
            obs = [x['obs'] for x in m]; ps = [x['p'] for x in m]
            print(json.dumps(dict(src=nm, n=len(m), obs_mean=float(np.nanmean(obs)), obs_pos=int(np.sum(np.array(obs) > 0)),
                                  p_median=float(np.median(ps)), p_lt05=int(np.sum(np.array(ps) < 0.05)), detail=m[:5])), flush=True)
