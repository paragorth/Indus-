"""v5 cycle 2 -- meter fingerprints.

M1  symbol repeat rate P(s_i == s_{i+k}) inside a line, lags 1..16, minus a null that shuffles WORD order
    inside the line (keeps the line's make-up and every word intact; only inter-word order can make
    periodicity). 'Comb' score for period p = mean excess at lags that are multiples of p minus mean excess
    at other lags (3..16). A metre/beat would give a comb.
M2  word-length autocorrelation along the line at word lags 1..4 vs within-line word shuffle
    (long-short alternation -> negative lag 1, positive lag 2).
M3  line-length quantisation: Rayleigh statistic of (length mod u) for u = 2..10, on lines that are not the
    last of their unit, in symbols and in words; null = lengths drawn from the length histogram smoothed by a
    box of width u (destroys exactly the mod-u structure, keeps the overall spread).
Extra positive control: Dante with each letter reduced to C/V and syllables (vowel groups) as 'words'.
"""
import sys, os, math, random, re
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib

LAGS = list(range(1, 17))
R = 40


def lines_of(units):
    return [L for u in units for L in u['lines']]


def repeat_rates(lines):
    num = np.zeros(len(LAGS) + 1); den = np.zeros(len(LAGS) + 1)
    for L in lines:
        s = [x for w in L for x in w]
        n = len(s)
        a = np.array([hash(x) for x in s])
        for k in LAGS:
            if n > k:
                num[k] += np.sum(a[:-k] == a[k:]); den[k] += n - k
    return num[1:] / np.maximum(den[1:], 1)


def m1(units, seed=0):
    rng = random.Random(seed)
    lines = lines_of(units)
    obs = repeat_rates(lines)
    nulls = []
    for _ in range(R):
        sh = []
        for L in lines:
            ws = list(L); rng.shuffle(ws); sh.append(ws)
        nulls.append(repeat_rates(sh))
    nulls = np.array(nulls)
    ex = obs - nulls.mean(0); z = ex / (nulls.std(0) + 1e-12)
    comb = {}
    for p in range(3, 9):
        mult = [i for i, k in enumerate(LAGS) if k >= 3 and k % p == 0]
        other = [i for i, k in enumerate(LAGS) if k >= 3 and k % p != 0]
        cs = ex[mult].mean() - ex[other].mean()
        cn = [(nn - nulls.mean(0))[mult].mean() - (nn - nulls.mean(0))[other].mean() for nn in nulls]
        comb[p] = (float(cs), float(cs / (np.std(cn) + 1e-12)))
    return {'excess': ex.tolist(), 'z': z.tolist(), 'comb': comb}


def m2(units, seed=0):
    rng = random.Random(seed)
    lines = [L for L in lines_of(units) if len(L) >= 5]

    def ac(lines):
        out = []
        for k in range(1, 5):
            xs, ys = [], []
            for L in lines:
                l = [len(w) for w in L]; m = np.mean(l)
                for i in range(len(l) - k):
                    xs.append(l[i] - m); ys.append(l[i + k] - m)
            xs, ys = np.array(xs), np.array(ys)
            out.append(float((xs * ys).mean() / (np.sqrt((xs ** 2).mean() * (ys ** 2).mean()) + 1e-12)))
        return np.array(out)
    obs = ac(lines)
    nulls = np.array([ac([rng.sample(L, len(L)) for L in lines]) for _ in range(R)])
    return {'obs': obs.tolist(), 'null': nulls.mean(0).tolist(), 'z': ((obs - nulls.mean(0)) / (nulls.std(0) + 1e-12)).tolist()}


def rayleigh(ls, u):
    a = 2 * math.pi * np.asarray(ls) / u
    return math.hypot(np.cos(a).mean(), np.sin(a).mean())


def m3(units, seed=0, what='symbols'):
    rng = np.random.default_rng(seed)
    ls = []
    for u in units:
        for L in u['lines'][:-1]:
            ls.append(sum(len(w) for w in L) if what == 'symbols' else len(L))
    ls = np.array(ls)
    out = {}
    hist = np.bincount(ls).astype(float)
    for u in range(2, 11):
        sm = np.convolve(hist, np.ones(u) / u)  # box smoothing (shifts by u-1; mod-u structure removed)
        vals = np.arange(len(sm)) - (u - 1) // 2
        keep = vals > 0
        p = sm[keep] / sm[keep].sum(); v = vals[keep]
        obs = rayleigh(ls, u)
        nn = [rayleigh(rng.choice(v, size=len(ls), p=p), u) for _ in range(200)]
        out[u] = (obs, float(np.mean(nn)), float((obs - np.mean(nn)) / (np.std(nn) + 1e-12)))
    return {'n': len(ls), 'mean': float(ls.mean()), 'cv': float(ls.std() / ls.mean()), 'by_u': out}


def dante_cv(max_words=40000):
    """Positive metre control: Dante verse lines, letters -> C/V, words = syllables (vowel groups)."""
    lines = vlib.load_ref('Italian-Dante', max_words=max_words, skip_frac=0.05)
    units, cur = [], None
    for L in lines:
        if L['para_start'] or cur is None:
            cur = {'id': str(len(units)), 'stratum': '-', 'lines': []}; units.append(cur)
        s = ''.join(L['words'])
        syl = re.findall(r'[^aeiouàèéìíòóù]*[aeiouàèéìíòóù]+', s) or [s]
        cur['lines'].append([tuple('V' if c in 'aeiouàèéìíòóù' else 'C' for c in sy) for sy in syl])
    return units


if __name__ == '__main__':
    C = vc.all_corpora()
    C['Dante-CV-syllables'] = dante_cv()
    res = {}
    for k, units in C.items():
        r = {'m1': m1(units), 'm2': m2(units), 'm3_sym': m3(units, what='symbols'), 'm3_words': m3(units, what='words')}
        res[k] = r
        print(f"\n== {k}")
        print('  M1 excess x1000 by lag: ' + ' '.join(f"{e*1000:+.1f}" for e in r['m1']['excess']))
        print('  M1 z by lag:            ' + ' '.join(f"{z:+.0f}" for z in r['m1']['z']))
        print('  M1 comb (period: excess,z): ' + ' '.join(f"{p}:{v[0]*1000:+.2f},{v[1]:+.1f}" for p, v in r['m1']['comb'].items()))
        print('  M2 wordlen autocorr obs/null/z: ' + ' '.join(f"L{i+1}:{o:+.3f}/{n:+.3f}/{z:+.1f}" for i, (o, n, z) in enumerate(zip(r['m2']['obs'], r['m2']['null'], r['m2']['z']))))
        for w in ['m3_sym', 'm3_words']:
            m = r[w]
            print(f"  M3 {w}: n={m['n']} mean={m['mean']:.1f} cv={m['cv']:.2f} | " + ' '.join(f"u{u}:z{v[2]:+.1f}" for u, v in m['by_u'].items()))
    vlib.save('v5_cycle2', res)
