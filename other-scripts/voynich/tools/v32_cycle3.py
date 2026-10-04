"""v32 cycle 3: stranger forms of the calendar beat.

(a) COUNTER: a calendar entry often opens with its day number, which grows and resets each month (a sawtooth).
    For period P and the best phase (and best post-gap offset), correlation of a feature with the within-cycle
    position (0..P-1). Features: opening-word length, gallows in the opening word, opening-line length, number of
    words, edit distance of the opening word to the previous entry's opening word. Family-wise vs shuffles.
(b) STARS vs TEXT: the drawn stars carry their own beat (dotted / plain alternate on several pages, 7 or 8 points,
    dark or light). If the stars are calendar marks (e.g. odd/even days, day/night, lucky/unlucky), the paragraph
    text should co-vary with them. Test: per-attribute difference in every text feature, permutation of star
    attributes WITHIN pages (keeps page effects). Control: planted coupling (opening word chosen by dotted).
(c) DOSE-RESPONSE: planted calendars at P = 7 and 29.53 in the real star units at decreasing strength, scored by
    the cycle-1 family-wise test (fewer nulls): the weakest calendar the test can see, i.e. an upper bound.
"""
import os, sys, time, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import v32_lib as V
from v32_cycle2 import gap_index

R3 = int(os.environ.get('V32_R3', 200))
PER = np.arange(2, 41)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def counter_feats(units):
    W = [[w for l in u['lines'] for w in l] for u in units]
    f = {
        'w1_len': np.array([len(V._gl(w[0])) for w in W], float),
        'w1_gall': np.array([sum(c in V.GALL for c in w[0]) for w in W], float),
        'line1_len': np.array([len(u['lines'][0]) for u in units], float),
        'nwords': np.log([len(w) for w in W]),
        'w1_edit_prev': np.array([0.0] + [lev(W[i][0], W[i - 1][0]) / max(len(W[i][0]), len(W[i - 1][0])) for i in range(1, len(W))]),
    }
    return f


def ramp_stat(y, P, gap):
    """max over phase (and gap offset) of |corr(y, sawtooth position)|"""
    N = len(y); best = 0.0
    i = np.arange(N, dtype=float)
    offs = range(P) if gap > 0 else [0]
    ys = (y - y.mean()) / (y.std() + 1e-12)
    for off in offs:
        j = i.copy()
        if gap > 0: j[gap:] += off
        for ph in range(P):
            pos = (j + ph) % P
            pos = (pos - pos.mean()) / (pos.std() + 1e-12)
            best = max(best, abs((ys * pos).mean()))
    return best


def counter_scan(F, perm, gap):
    out = {}
    for k, x in F.items():
        y = V.detrend(np.asarray(x)[perm]) if k != 'w1_edit_prev' else np.asarray(x)[perm]
        out[('RAMP', k)] = np.array([ramp_stat(y, int(P), gap) for P in PER])
    return out


def part_a(args):
    name, units, seed = args
    ck = os.path.join(V.CK, f'c3a_{name}.pkl')
    if os.path.exists(ck): return pickle.load(open(ck, 'rb'))
    gap = gap_index(units)
    F = counter_feats(units)
    N = len(units)
    obs = counter_scan(F, np.arange(N), gap)
    rng = np.random.default_rng(seed)
    nulls = []
    for r in range(R3 // 2):
        p = rng.permutation(N)
        Fp = dict(F)
        # edit distance to the previous opening word must be recomputed in the shuffled order
        W = [units[i]['lines'][0][0] for i in p]
        Fp['w1_edit_prev'] = np.array([0.0] + [lev(W[i], W[i - 1]) / max(len(W[i]), len(W[i - 1])) for i in range(1, N)])
        nulls.append({k: v for k, v in counter_scan({k: (np.asarray(F[k])[p] if k != 'w1_edit_prev' else Fp[k]) for k in F},
                                                     np.arange(N), gap).items()})
    res = V.family_test(obs, nulls)
    out = {'name': name, 'N': N, 'zmax': res['zmax'], 'p': res['p_fw'], 'top': V.fmt_top(res, PER.astype(float))}
    pickle.dump(out, open(ck, 'wb'))
    print('A', name, N, f"zmax {res['zmax']:.2f} p {res['p_fw']:.3f}", out['top'][:180], flush=True)
    return out


def ado_counter_units():
    """Ado with the Roman day count written as a number word in front of the entry, body kept: the sawtooth control.
    Roman countdown numerals ('xviii kal') make the opening-word length a real sawtooth."""
    return V.ado_units(letter=False)


# ---------------------------------------------------------------- (b)
def star_text(units, R=1000, seed=0, plant=False):
    num, cat, vec = V.features(units, star=False)
    S = [u.get('star') for u in units]
    keep = np.array([s is not None and s.get('pts') in (7, 8) for s in S])
    U = [u for u, k in zip(units, keep) if k]
    num = {k: v[keep] for k, v in num.items()}
    pages = np.array([u['f'] for u in U])
    attrs = {'dotted': np.array([s['dotted'] for s in S if s is not None and s.get('pts') in (7, 8)]),
             'pts8': np.array([int(s['pts'] == 8) for s in S if s is not None and s.get('pts') in (7, 8)]),
             'dark': np.array([s['dark'] for s in S if s is not None and s.get('pts') in (7, 8)])}
    if plant:
        rng = random.Random(seed)
        for i, u in enumerate(U):
            if attrs['dotted'][i] and rng.random() < 0.4:
                num['w1_len'][i] = 2.0   # dotted entries open with a short word in 40% of cases
    rng = np.random.default_rng(seed)
    rows = []
    for a, lab in attrs.items():
        if lab.sum() < 10 or (1 - lab).sum() < 10: continue
        obs = {k: v[lab == 1].mean() - v[lab == 0].mean() for k, v in num.items()}
        nul = {k: [] for k in num}
        for r in range(R):
            l2 = lab.copy()
            for g in np.unique(pages):
                ix = np.where(pages == g)[0]; l2[ix] = rng.permutation(lab[ix])
            for k, v in num.items(): nul[k].append(v[l2 == 1].mean() - v[l2 == 0].mean())
        zs = {k: (obs[k] - np.mean(nul[k])) / (np.std(nul[k]) + 1e-12) for k in num}
        # family-wise: max |z| over features vs the null max (approximate with the same permutations)
        Nn = np.array([nul[k] for k in num])          # F x R
        Zn = (Nn - Nn.mean(1, keepdims=True)) / (Nn.std(1, keepdims=True) + 1e-12)
        nm = np.abs(Zn).max(0)
        zmax = max(abs(z) for z in zs.values())
        p = (1 + (nm >= zmax).sum()) / (R + 1)
        best = max(zs, key=lambda k: abs(zs[k]))
        rows.append((a, int(lab.sum()), int((1 - lab).sum()), best, round(zs[best], 2), round(p, 4)))
    return rows


# ---------------------------------------------------------------- (c)
def dose(args):
    P, mode, s, seed = args
    ck = os.path.join(V.CK, f'c3c_{P}_{mode}_{s}_{seed}.pkl')
    if os.path.exists(ck): return pickle.load(open(ck, 'rb'))
    q = V.load_q20()
    u = V.plant_calendar(q, P, s, seed, mode)
    num, cat, vec = V.features(u, star=False)
    obs = V.scan(num, cat, vec)
    nulls = V.perm_nulls(num, cat, vec, 60, seed + 99)
    res = V.family_test(obs, nulls)
    j = int(np.argmin(np.abs(V.PERIODS - P)))
    zP = float(res['Z'][:, max(0, j - 1):j + 2].max())
    out = (P, mode, s, seed, res['zmax'], res['p_fw'], zP, V.fmt_top(res, n=1))
    pickle.dump(out, open(ck, 'wb'))
    print('C', out, flush=True)
    return out


if __name__ == '__main__':
    q = V.load_q20()
    part = sys.argv[1] if len(sys.argv) > 1 else 'abc'
    if 'b' in part:
        print('B real', star_text(q, seed=1), flush=True)
        print('B planted', star_text(q, seed=2, plant=True), flush=True)
    jobs_a = [('VOY-star', q, 21), ('VOY-IT2a', V.load_q20('IT2a', mode='para'), 22),
              ('ADO-date-noletter', ado_counter_units(), 23), ('ADO-body', V.ado_units(letter=False, date=False), 24),
              ('SEC-bio-B', V.load_section('B', 'B'), 25)]
    jobs_c = [(P, m, s, sd) for P in (7, 29.53) for m, ss in (('opening', (0.1, 0.2, 0.3)), ('vocab', (0.01, 0.02, 0.04)),
                                                                 ('length', (0.1, 0.2)))
              for s in ss for sd in (31,)]
    with Pool(2) as p:
        if 'a' in part: p.map(part_a, jobs_a, chunksize=1)
        if 'c' in part: p.map(dose, jobs_c, chunksize=1)
    print('done')
