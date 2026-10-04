"""pe32 cycle 1: IS THE STANDARD-RATION CLASS A CLASS?  The 14 signs of pe28 (C14) are tested for coherence
in every other respect: position in lines, left neighbour, number system and count size of the lines they end,
tablet header, system of the next line (an M288 successor is never used), herd tablets (pe20), pe15 office,
tablet co-occurrence.  All predecessor lines of any M288 entry are removed before features are computed.
Nulls: N1 = frequency-matched random 14-sets from all final signs (>= 3 final uses); N2 = frequency-matched
14-sets from the other signs that precede M288 entries (selection-matched).  Controls: pe15 GRAIN office
(positive), a planted class (14 random signs sharing a left neighbour on 30% of their final uses), and
calibration (40 random N1 sets run through the same test)."""
import os, sys, json, time, copy
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe32_common import *  # noqa

NR = int(os.environ.get('NR', 2000))
t0 = time.time()


_MEMO = {}


def run(T, cls, label, E, rng, nr=NR, fams=FAMS, pool2=None):
    if id(T) not in _MEMO:
        excl = {(e['tid'], e['line'] - 1) for e in E}
        F = sign_features(T, exclude=excl)
        univ0 = sorted(s for s, n in F['fin_n'].items() if n >= 3 and s not in ('x', '-'))
        _MEMO[id(T)] = (F, univ0, sim_matrix(F, univ0))
    F, univ, M = _MEMO[id(T)]
    freq = F['fin_n']
    cls = [s for s in cls if s in univ]
    p2 = sorted(set(pool2 or []) - set(cls))
    p2 = [s for s in p2 if s in univ]
    allS = univ
    ix = {s: i for i, s in enumerate(allS)}
    real = coherence(M, [ix[s] for s in cls])
    res = {'label': label, 'n': len(cls), 'cls': cls, 'real': real}
    for nm, pool in (('N1', [s for s in allS if s not in cls]), ('N2', p2)):
        if len(pool) < len(cls):
            continue
        N = {f: [] for f in fams}
        for r in range(nr):
            d = freq_matched(rng, cls, pool, freq)
            c = coherence(M, [ix[s] for s in d])
            for f in fams:
                N[f].append(c[f])
        row = {}
        zs = []
        for f in fams:
            a = np.array(N[f], float); a = a[np.isfinite(a)]
            if not np.isfinite(real[f]) or len(a) < 10:
                continue
            z = (real[f] - a.mean()) / (a.std() + 1e-12)
            p = (1 + (a >= real[f]).sum()) / (1 + len(a))
            row[f] = {'z': round(float(z), 2), 'p': round(float(p), 4)}
            zs.append(z)
        # composite: mean z over families, against the same statistic on null sets (leave-one-out style)
        Z = np.array([[(N[f][r] - np.nanmean(N[f])) / (np.nanstd(N[f]) + 1e-12) for f in fams if f in row]
                      for r in range(nr)])
        comp_null = np.nanmean(Z, 1)
        comp = float(np.mean(zs))
        row['composite'] = {'z_mean': round(comp, 2),
                            'p': round(float((1 + (comp_null >= comp).sum()) / (1 + nr)), 4)}
        res[nm] = row
    print(label, json.dumps({k: v for k, v in res.items() if k != 'real'}), round(time.time() - t0), flush=True)
    return res


def plant(T, members, rng, frac=0.3):
    T2 = copy.deepcopy(T)
    ms = set(members)
    for t in T2:
        for k, l in enumerate(t['L']):
            if k and l['sg'] and l['sg'][-1] in ms and rng.random() < frac:
                l['sg'] = l['sg'][:-1] + ['MPLANT', l['sg'][-1]]
    return T2


if __name__ == '__main__':
    rng = np.random.default_rng(32)
    T = table()
    E = m288_events(T)
    pool2 = sorted({e['pfin'] for e in E})
    out = {}
    out['C14'] = run(T, C14, 'C14 real', E, rng, pool2=pool2)
    out['GRAIN'] = run(T, OFFICE['GRAIN'], 'GRAIN office (positive)', E, rng, nr=1000,
                       pool2=pool2)
    F0 = sign_features(T)
    univ = sorted(s for s, n in F0['fin_n'].items() if n >= 3 and s not in ('x', '-') and s not in C14)
    pl = freq_matched(rng, C14, univ, F0['fin_n'])
    out['plant'] = run(plant(T, pl, rng), pl, 'PLANTED shared left neighbour 30%', E, rng, nr=1000, pool2=pool2)
    cal = []
    for c in range(int(os.environ.get('NCAL', 40))):
        rs = freq_matched(rng, C14, univ, F0['fin_n'])
        r = run(T, rs, f'calib {c}', E, rng, nr=300, pool2=pool2)
        cal.append(r)
    out['calib'] = {'n': len(cal), 'frac_p05_N1': float(np.mean([r['N1']['composite']['p'] <= 0.05 for r in cal])),
                    'frac_p05_N2': float(np.mean([r['N2']['composite']['p'] <= 0.05 for r in cal if 'N2' in r])),
                    'fam_frac_p05_N1': {f: float(np.mean([r['N1'].get(f, {'p': 1})['p'] <= 0.05 for r in cal]))
                                        for f in FAMS}}
    print('calib', out['calib'])
    json.dump(out, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
