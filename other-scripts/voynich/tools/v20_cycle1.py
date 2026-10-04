"""v20 cycle 1: massive univariate dispersion search (about 3,900 features x 4 unit levels) against the
slot-preserving redeal null, family-wise corrected by the max-statistic over all features.
Corpora: Voynich ZL3b, Voynich IT2a, planted quotas inside ZL (cap per line, quota per page, balanced pair per
page; hard and soft), Markov-2 resynthesis (negative control), Latin (Isidore, Caesar), Italian (Manzoni prose,
Dante verse), English lipogram (Gadsby, no letter e).
Each job is checkpointed to data/results/v20/c1_<job>.npz. At most 2 worker processes."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L

REPS_V, REPS_R = 200, 100


def plant(C, names, T, kind, seed=11):
    """Return a modified tid array carrying a planted quota."""
    rng = np.random.default_rng(seed)
    tid = C.tid.copy()
    def cls(nm): return T[:, names.index(nm)] > 0  # type mask
    page = C.unit['page']; line = C.unit['line']; strat = C.unit['stratum']
    pools = {}
    def pool(s, mask):
        k = (s, mask.tobytes().__hash__())
        if k not in pools:
            sel = (strat == s) & (C.slot == 3)
            t = C.tid[sel]; t = t[mask[t]]
            if len(t) < 5:
                t = C.tid[C.slot == 3]; t = t[mask[t]]
            pools[k] = t
        return pools[k]
    if kind.startswith('cap'):
        cap = int(kind[3]); m = cls('g:t')
        for u in np.unique(line):
            pos = np.where((line == u) & m[tid])[0]
            if len(pos) > cap:
                for p in rng.choice(pos, len(pos) - cap, replace=False):
                    pl = pool(strat[p], ~m); tid[p] = pl[rng.integers(len(pl))]
    elif kind.startswith('quota'):
        frac = 1.0 if kind == 'quota_hard' else 0.5
        m = cls('ini:q')
        for s in np.unique(strat):
            rate = m[C.tid[strat == s]].mean()
            for u in np.unique(page[strat == s]):
                idx = np.where(page == u)[0]
                have = int(m[tid[idx]].sum()); target = int(round(rate * len(idx)))
                d = int(round((have - target) * frac))
                med = idx[C.slot[idx] == 3]
                if d > 0:
                    pos = med[m[tid[med]]]
                    for p in rng.choice(pos, min(d, len(pos)), replace=False):
                        pl = pool(s, ~m); tid[p] = pl[rng.integers(len(pl))]
                elif d < 0:
                    pos = med[~m[tid[med]]]
                    for p in rng.choice(pos, min(-d, len(pos)), replace=False):
                        pl = pool(s, m); tid[p] = pl[rng.integers(len(pl))]
    elif kind.startswith('balance'):
        a, b = cls('fin:l'), cls('ini:q')
        both = a & b
        a, b = a & ~both, b & ~both
        frac = 1.0 if kind == 'balance_hard' else 0.5
        for u in np.unique(page):
            idx = np.where(page == u)[0]; s = strat[idx[0]]
            med = idx[C.slot[idx] == 3]
            na, nb = int(a[tid[idx]].sum()), int(b[tid[idx]].sum())
            d = int(round((na - nb) / 2 * frac))
            src, dst = (a, b) if d > 0 else (b, a)
            pos = med[src[tid[med]]]
            for p in rng.choice(pos, min(abs(d), len(pos)), replace=False):
                pl = pool(s, dst); tid[p] = pl[rng.integers(len(pl))]
    return tid


def job(spec):
    name = spec['name']; out = os.path.join(L.CK, 'c1_%s.npz' % name)
    if os.path.exists(out): return name, 'cached'
    t0 = time.time()
    if spec['src'] == 'voy':
        C = L.voynich_corpus(spec.get('tr', 'ZL3b'))
    elif spec['src'] == 'ref':
        C = L.ref_corpus(spec['key'], verse=spec.get('verse', False))
    names, T, alph = L.build_features(C)
    tid = None
    if spec.get('plant'):
        tid = plant(C, names, T, spec['plant'])
    if spec.get('markov'):
        C = L.markov_resynth(C); names, T, alph = L.build_features(C)
    res = {}
    for lev in L.LEVELS:
        obs, nulls = L.run_level(C, lev, T, reps=spec['reps'], seed=5, tid=tid)
        Z = L.zscores(obs, nulls)
        res[lev] = dict(zD=Z[0][0], RD=Z[0][2], fwD_low=L.fw_p(Z[0][0], Z[0][1], 'low'),
                        fwD_high=L.fw_p(Z[0][0], Z[0][1], 'high'),
                        zMax=Z[1][0], fwMax_low=L.fw_p(Z[1][0], Z[1][1], 'low'),
                        zZero=Z[2][0], fwZero_low=L.fw_p(Z[2][0], Z[2][1], 'low'),
                        nullminz=Z[0][1].min(1))
    np.savez_compressed(out, names=np.array(names), **{lev + '__' + k: v for lev, d in res.items() for k, v in d.items()})
    return name, '%.0fs' % (time.time() - t0)


JOBS = [dict(name='ZL', src='voy', reps=REPS_V),
        dict(name='IT', src='voy', tr='IT2a', reps=REPS_V),
        dict(name='plant_cap2', src='voy', plant='cap2', reps=REPS_V),
        dict(name='plant_cap3', src='voy', plant='cap3', reps=REPS_V),
        dict(name='plant_quota_hard', src='voy', plant='quota_hard', reps=REPS_V),
        dict(name='plant_quota_soft', src='voy', plant='quota_soft', reps=REPS_V),
        dict(name='plant_balance_hard', src='voy', plant='balance_hard', reps=REPS_V),
        dict(name='markov', src='voy', markov=True, reps=REPS_V),
        dict(name='Latin-Isidore', src='ref', key='Latin-Isidore', reps=REPS_R),
        dict(name='Latin-Caesar', src='ref', key='Latin-Caesar', reps=REPS_R),
        dict(name='Italian-Manzoni', src='ref', key='Italian-Manzoni', reps=REPS_R),
        dict(name='Italian-Dante', src='ref', key='Italian-Dante', verse=True, reps=REPS_R),
        dict(name='Gadsby', src='ref', key='English-Gadsby-lipogram', reps=REPS_R)]

if __name__ == '__main__':
    only = sys.argv[1:]
    js = [j for j in JOBS if not only or j['name'] in only]
    with Pool(2) as P:
        for r in P.imap_unordered(job, js):
            print(*r, flush=True)
