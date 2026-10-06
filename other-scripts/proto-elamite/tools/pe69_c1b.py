"""pe69 cycle 1b: the decisive test. On runs where the team rule and the per-line rules make DIFFERENT predictions
(>= 2 clean count lines, sum != last and sum != 2 x last), how often does 60 x team hit, against
 (i) re-dealt M288 values (Monte Carlo, 10,000x) and
 (ii) the rate the team rule would have if it held as reliably as it does on single-line runs (binomial power check),
 (iii) a planted control: the real corpus with 60 x team written into a random 25% of ALL slots; how often is the
     discriminating-run test passed (hits >= 3 and p < 0.05)?
Repeated for bare 'M288 n' lines only, and for the joined tablets (where the rule was first seen) separately."""
import os, sys, json, math
from collections import Counter
import numpy as np
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe69_lib import *  # noqa


def run(T, tag, bare_only=False, rng=None):
    S = slots(T)
    m, A = feats(S)
    nc = np.array([nclean(s) for s in S])
    bare = np.array([s['m_signs'] == ['M288'] for s in S])
    s_, l_ = A['SUM_BEFORE'], A['LAST']
    base = np.isfinite(m) & (~bare_only | bare)
    disc = base & (nc >= 2) & np.isfinite(s_) & (s_ != l_) & (s_ != 2 * l_)
    single = base & (nc == 1)
    hd = int((disc & (np.abs(m - 60 * s_) < 1e-9)).sum())
    hs = int((single & (np.abs(m - 60 * s_) < 1e-9)).sum())
    rate1 = hs / max(1, single.sum())
    pool = m[np.isfinite(m)]
    nul = np.array([int((disc & (np.abs(redeal_m(m, rng) - 60 * s_) < 1e-9)).sum())
                    for _ in range(10000)])
    out = {'tag': tag, 'bare_only': bare_only, 'disc_runs': int(disc.sum()), 'disc_team_hits': hd,
           'null_mean': round(float(nul.mean()), 2), 'p_vs_redeal': float(((nul >= hd).sum() + 1) / 10001),
           'single_line_runs': int(single.sum()), 'single_hit_rate': round(rate1, 3),
           'expected_if_team_rule_holds': round(rate1 * disc.sum(), 1),
           'p_kill_binom(<=obs | rule holds)': float(binom.cdf(hd, int(disc.sum()), rate1)),
           'disc_last120_hits': int((disc & (np.abs(m - 120 * l_) < 1e-9)).sum()),
           'disc_last60_hits': int((disc & (np.abs(m - 60 * l_) < 1e-9)).sum())}
    return out, (S, m, A, nc, disc)


def planted(T, rng, frac=0.25, reps=500):
    """control: write 60 x team into a random share of all clean slots with a defined team; does the test see it?"""
    S = slots(T)
    m, A = feats(S)
    nc = np.array([nclean(s) for s in S])
    s_, l_ = A['SUM_BEFORE'], A['LAST']
    ok = np.isfinite(m) & np.isfinite(s_)
    disc = ok & (nc >= 2) & (s_ != l_) & (s_ != 2 * l_)
    passed = 0
    hits = []
    for _ in range(reps):
        mv = m.copy()
        sel = ok & (rng.random(len(m)) < frac)
        mv[sel] = 60 * s_[sel]
        hd = int((disc & (np.abs(mv - 60 * s_) < 1e-9)).sum())
        nul = np.array([int((disc & (np.abs(redeal_m(mv, rng) - 60 * s_) < 1e-9)).sum()) for _ in range(200)])
        p = ((nul >= hd).sum() + 1) / 201
        hits.append(hd)
        passed += int(hd >= 3 and p < 0.05)
    return {'planted_frac': frac, 'power': passed / reps, 'mean_hits': float(np.mean(hits))}


def main():
    rng = np.random.default_rng(seed('pe69-c1b'))
    T = tablets()
    res = {}
    for bo in (False, True):
        o, _ = run(T, 'non-joined', bo, rng)
        res['nonjoined_bare' if bo else 'nonjoined'] = o
        print(json.dumps(o), flush=True)
    TJ = [t for t in tablets(include_joined=True) if t['id'] in JOINED_IDS]
    o, (S, m, A, nc, disc) = run(TJ, 'joined', False, rng)
    o['disc_list'] = [{'id': S[i]['id'], 'line': S[i]['line'], 'sum': float(A['SUM_BEFORE'][i]), 'last': float(A['LAST'][i]),
                       'm': float(m[i])} for i in np.where(disc)[0]]
    res['joined'] = o
    print(json.dumps(o), flush=True)
    for f in (0.1, 0.25):
        r = planted(T, rng, f)
        res['planted_%d' % int(f * 100)] = r
        print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(CK, 'c1b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
