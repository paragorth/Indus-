"""pe81: the text sorts the sites, the weather judges.

Text side only (no environmental data is read here).  Random statistics over tablet features are scored per site; a
statistic 'survives' when its ordering of the sites repeats between disjoint tablet halves more often than in a corpus whose
tablets are re-dealt among the sites.  The survivors' consensus site axes are frozen for the outside test (pe81_outside.py).
"""
import sys, os, json, math, hashlib, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

CK = os.path.join(common.DATA, 'pe81_ckpt')
os.makedirs(CK, exist_ok=True)
SITES = ['Susa', 'Malyan', 'Yahya', 'Sialk', 'Sofalin']
PROV = {'Susa (mod. Shush)': 'Susa', 'Anšan (mod. Tell Malyan)': 'Malyan', 'uncertain (mod. Tepe Yahya)': 'Yahya',
        'uncertain (mod. Tepe Sialk)': 'Sialk', 'uncertain (mod. Tepe Sofalin)': 'Sofalin'}
SVAL = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}
SYSTEMS = ['C', 'C*', 'B', 'SDB', 'N23', 'S-frac', 'mod*']


def tablet_rows(min_signs=0):
    T = common.load()
    rows = []
    for t in T:
        s = PROV.get(t['provenience'])
        if s is None or not t['lines']:
            continue
        rows.append((s, t))
    return rows


def features(rows, top_signs=None):
    """Per-tablet feature matrix.  Features never use site information."""
    cnt = collections.Counter()
    for s, t in rows:
        seen = set()
        for l in t['lines']:
            for g in l['signs']:
                if common.is_sign(g):
                    seen.add(common.base(g))
        cnt.update(seen)
    if top_signs is None:
        top_signs = [g for g, c in cnt.most_common() if c >= 8][:120]
    codes = ['N01', 'N14', 'N34', 'N45', 'N48', 'N39B', 'N30C', 'N24', 'N39C', 'N51', 'N08', 'N02', 'N23', 'N30D', 'N28', 'N29B']
    names = ([f'sys_{x}' for x in SYSTEMS] + [f'code_{c}' for c in codes] +
             ['n_lines', 'n_entries', 'header', 'rev_numeric', 'frac_x', 'signs_per_entry', 'compound_share',
              'log_max_S', 'log_sum_S', 'n_numgroups', 'numeric_only_lines', 'multi_col', 'variant_share', 'n_distinct_signs'] +
             [f'sg_{g}' for g in top_signs])
    X = np.zeros((len(rows), len(names)))
    for i, (s, t) in enumerate(rows):
        f = {}
        sysc = collections.Counter(); codec = set(); vals = []; ngroups = 0; numonly = 0
        nsig = 0; nx = 0; ncomp = 0; nvar = 0; seen = set(); ent = []; cols = set()
        for l in t['lines']:
            cols.add(l.get('column', 1))
            if l['numerals']:
                ngroups += 1
                sysc[common.system_of(l['numerals'])] += 1
                for n, c in l['numerals']:
                    codec.add(c.split('@')[0])
                v = sum((n or 0) * SVAL.get(c, 0) for n, c in l["numerals"])
                if v:
                    vals.append(v)
            sg = [g for g in l['signs'] if common.is_sign(g) or g == 'x']
            if l['numerals'] and not sg:
                numonly += 1
            if l['numerals'] and sg:
                ent.append(len(sg))
            for g in sg:
                nsig += 1
                if g == 'x':
                    nx += 1; continue
                if g.startswith('|'):
                    ncomp += 1
                if '~' in g:
                    nvar += 1
                seen.add(common.base(g))
        tot = max(1, sum(sysc.values()))
        for x in SYSTEMS:
            f[f'sys_{x}'] = sysc.get(x, 0) / tot
        for c in codes:
            f[f'code_{c}'] = 1.0 if c in codec else 0.0
        f['n_lines'] = math.log1p(len(t['lines']))
        f['n_entries'] = math.log1p(len(ent))
        f['header'] = 1.0 if common.header(t) else 0.0
        f['rev_numeric'] = 1.0 if any(l['surface'] != 'obverse' and l['numerals'] for l in t['lines']) else 0.0
        f['frac_x'] = nx / max(1, nsig)
        f['signs_per_entry'] = float(np.mean(ent)) if ent else 0.0
        f['compound_share'] = ncomp / max(1, nsig - nx)
        f['log_max_S'] = math.log1p(max(vals)) if vals else 0.0
        f['log_sum_S'] = math.log1p(sum(vals)) if vals else 0.0
        f['n_numgroups'] = math.log1p(ngroups)
        f['numeric_only_lines'] = numonly / max(1, ngroups)
        f['multi_col'] = 1.0 if len(cols) > 1 else 0.0
        f['variant_share'] = nvar / max(1, nsig - nx)
        f['n_distinct_signs'] = math.log1p(len(seen))
        for g in top_signs:
            f[f'sg_{g}'] = 1.0 if g in seen else 0.0
        X[i] = [f[n] for n in names]
    return X, names, top_signs


def make_hyps(nfeat, n, rng):
    H = []
    for _ in range(n):
        k = rng.integers(1, 4)
        idx = rng.choice(nfeat, size=k, replace=False)
        w = rng.choice([-1.0, 1.0], size=k)
        H.append((idx, w))
    return H


def hyp_matrix(H, nfeat):
    W = np.zeros((nfeat, len(H)))
    for j, (idx, w) in enumerate(H):
        W[idx, j] = w
    return W


def site_scores(Z, site_idx, members):
    """Z: tablet x hyp scores; members: list per site of tablet index arrays -> site x hyp means."""
    return np.vstack([Z[m].mean(0) for m in members])


def kendall_rows(A, B):
    """Kendall tau between columns of A and B (sites x hyps), small n."""
    n = A.shape[0]
    num = np.zeros(A.shape[1]); den = 0
    for i in range(n):
        for j in range(i + 1, n):
            num += np.sign(A[i] - A[j]) * np.sign(B[i] - B[j])
            den += 1
    return num / den


def stability(Z, site_of, sites, rng, reps, susa_n):
    """Fraction of replicates where halves agree (tau >= 0.8 for 5 sites; tau == 1 for 4 sites)."""
    S = len(sites)
    thr = 0.8 if S == 5 else 0.99
    by = [np.where(site_of == k)[0] for k in range(S)]
    hits = np.zeros(Z.shape[1])
    for r in range(reps):
        A = []; B = []
        for k in range(S):
            m = rng.permutation(by[k])
            if sites[k] == 'Susa':
                m = m[:2 * susa_n]
            h = len(m) // 2
            A.append(m[:h]); B.append(m[h:2 * h])
        sa = site_scores(Z, None, A); sb = site_scores(Z, None, B)
        tau = kendall_rows(sa, sb)
        hits += (tau >= thr)
    return hits / reps


def run(X, site_of, sites, H, rng, reps=60, susa_n=12):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Zf = (X - mu) / sd
    W = hyp_matrix(H, X.shape[1])
    Z = Zf @ W
    st = stability(Z, site_of, sites, rng, reps, susa_n)
    return Z, st
