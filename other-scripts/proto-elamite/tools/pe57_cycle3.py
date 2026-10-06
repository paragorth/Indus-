"""pe57 cycle 3 (disruptive part): do the unit ratios of the PE grain-capacity ladder line up with the ratios
between vessel-capacity modes beyond chance?
For a ladder V (unit values) and vessel modes M, the score at scale s is the mean over modes of
min_j |ln(m_i / (s v_j))|; S* = min over s (grid, log-uniform).  Tests:
 (a) RANDOM ASSIGNMENTS: 5,000 random scales (N39C or N39a log-uniform 0.01-50 l): share of scales at which
     every mode sits within x1.15 of some unit (chance hit rate), and where the pe55 scale falls.
 (b) LADDER vs RANDOM LADDERS: S* of the real ladder vs 5,000 random ladders with the same number of units and
     random step factors from {2,3,4,5,6,10}; p = share of random ladders with S* <= real.
 (c) MODES vs RANDOM MODES: S* for the real modes vs 2,000 random mode sets (same number of modes, each
     log-uniform 0.2-5 l, sorted, separated by > x1.4).
Control: the proto-cuneiform grain ladder (N30c 1/10, N28 1/4, N39a 1, N1 5, N14 30, N45 90; Englund 2004
values) against the Uruk vessels (Jebel Aruda) and the Jemdet Nasr bowls."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pe57_common import *

PE = {'apriori': [1, 2, 4, 12, 24, 120, 720], 'alt_a': [1, 2, 4, 8, 24, 120, 720], 'alt_b': [1, 2, 4, 12, 24, 144, 1440],
      'alt_c': [1, 3, 6, 12, 24, 120, 1200], 'alt_d': [1, 2, 6, 12, 60, 300, 1800]}
PC = {'proto_cuneiform_SZE': [0.1, 0.25, 1, 5, 30, 90]}
S = np.exp(np.linspace(np.log(0.002), np.log(50), 6000))
TOL = np.log(1.15)

def score_grid(V, M):
    V = np.asarray(V, float); M = np.asarray(M, float)
    d = np.abs(np.log(M[None, :, None] / (S[:, None, None] * V[None, None, :])))  # scale x mode x unit
    return d.min(2)  # scale x mode

def sstar(V, M):
    return float(score_grid(V, M).mean(1).min())

def rand_ladder(r, k):
    f = r.choice([2, 3, 4, 5, 6, 10], k - 1)
    return np.r_[1, np.cumprod(f)]

def run(V, M, r, pe55=None):
    g = score_grid(V, M)
    allhit = (g <= TOL).all(1)
    # random assignments: scale log-uniform 0.01-50 l for the base unit
    sc = np.exp(r.uniform(np.log(0.01), np.log(50), 5000))
    idx = np.searchsorted(S, sc).clip(0, len(S) - 1)
    hit_rate = float(allhit[idx].mean())
    real = float(g.mean(1).min()); best_s = float(S[np.argmin(g.mean(1))])
    rl = np.array([sstar(rand_ladder(r, len(V)), M) for _ in range(5000)])
    out = {'modes': list(map(float, M)), 'S_star': real, 'best_scale_base_unit_l': best_s,
           'chance_all_modes_hit_rate': hit_rate, 'p_vs_random_ladders': float((rl <= real).mean()),
           'random_ladder_S_median': float(np.median(rl))}
    if len(M) >= 2:
        rm = []
        for _ in range(2000):
            while True:
                m = np.sort(10 ** r.uniform(np.log10(0.2), np.log10(5), len(M)))
                if np.all(np.diff(np.log(m)) > np.log(1.4)): break
            rm.append(sstar(V, m))
        out['p_vs_random_modes'] = float((np.array(rm) <= real).mean())
    if pe55:
        lo, hi = pe55
        sel = (S >= lo) & (S <= hi)
        out['pe55_scale_hits_all_modes'] = bool(allhit[sel].any())
        out['pe55_scale_best_score'] = float(g[sel].mean(1).min())
    return out

if __name__ == '__main__':
    c2 = json.load(open(os.path.join(CK, 'cycle2.json')))
    r = np.random.default_rng(11)
    res = {}
    # mode sets (KDE modes of cycle 2; bootstrap-stable modes only are reported in the loop file)
    msets = {k: c2[k]['kde_modes'] for k in c2 if not k.startswith('null') and 'kde_modes' in c2[k]}
    for ms, M in msets.items():
        for lname, V in {**PE, **PC}.items():
            pe55 = [(0.6, 0.8), (5, 7)] if lname == 'apriori' else None
            o = run(V, M, r, pe55=pe55[0] if pe55 else None)
            if pe55:
                o['pe55_alias_5_7'] = run(V, M, r, pe55=pe55[1])['pe55_scale_hits_all_modes']
            res[f'{ms}|{lname}'] = o
            print(ms, lname, json.dumps(o), flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1)
