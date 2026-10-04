"""pe27 cycle 2: (a) split-half replication of the best ratios, (b) sign-tied ratios
(ratio x count-line sign x capacity-line sign), (c) random-notation null, (d) near-exact log grid,
(e) other capacity value sets / ambiguous lines dropped.  Same Ur III and planted controls."""
import json, sys, os, time
import numpy as np
from fractions import Fraction as Fr
from collections import Counter, defaultdict
from scipy.stats import poisson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import *  # noqa

NP = int(os.environ.get('NP', 200))
res = {}
t0 = time.time()


def sc(obs, mu):
    return -np.log10(np.maximum(poisson.sf(obs - 1, np.maximum(mu, 0.05)), 1e-300))


# ---------------------------------------------------------------- (a) split halves
def split_test(xs, ys, nsplit=40, k=5, seed=0, nperm=100):
    """Fit: top-k ratios by score on half A.  Test: hits of those k ratios on half B vs B's
    permutation null (sum over the k).  Returns mean excess and share of splits with p<0.05."""
    rng = np.random.default_rng(seed)
    n = len(xs)
    out = []
    for s in range(nsplit):
        idx = rng.permutation(n)
        A, B = idx[:n // 2], idx[n // 2:]
        RA = scan([xs[i] for i in A], [ys[i] for i in A], nperm=60, seed=s)
        top = [Fr(e['r']) for e in RA['top'][:k]]
        xb, yb = [xs[i] for i in B], [ys[i] for i in B]
        real = sum(H_of(xb, yb).get(r, 0) for r in top)
        nul = []
        for p in range(nperm):
            Hp = H_of(xb, yb, perm=rng.permutation(len(B)))
            nul.append(sum(Hp.get(r, 0) for r in top))
        nul = np.array(nul)
        out.append({'top': [str(r) for r in top], 'real': real, 'null_mean': float(nul.mean()),
                    'p': float((1 + (nul >= real).sum()) / (1 + nperm))})
    ex = np.mean([o['real'] - o['null_mean'] for o in out])
    sig = np.mean([o['p'] < 0.05 for o in out])
    cnt = Counter(r for o in out for r in o['top'])
    return {'excess_mean': float(ex), 'share_p05': float(sig), 'top_counts': cnt.most_common(10)}


# ---------------------------------------------------------------- (b) sign-tied scan
def Hsign(xs, ys, perm=None):
    H = Counter()
    for i in range(len(xs)):
        qy = ys[perm[i]] if perm is not None else ys[i]
        keys = set()
        for a in xs[i]:
            for b in qy:
                keys.add((b['v'] / a['v'], a['fin'], b['fin']))
        for kk in keys:
            H[kk] += 1
    return H


def sign_scan(xs, ys, nperm=200, seed=0, minH=3):
    """Null re-deals Y line-sets among tablets (signs travel with their numbers, so a sign that
    is always written with one value gets the same ratio under the null too)."""
    rng = np.random.default_rng(seed)
    Hr = Hsign(xs, ys)
    Hp = [Hsign(xs, ys, perm=rng.permutation(len(xs))) for _ in range(nperm)]
    keys = set(k for k, v in Hr.items() if v >= minH)
    for h in Hp:
        keys |= {k for k, v in h.items() if v >= minH}
    keys = list(keys)
    M = np.array([[h.get(k, 0) for k in keys] for h in Hp], float)
    tot = M.sum(0)
    real = np.array([Hr.get(k, 0) for k in keys], float)
    s_real = np.where(real >= minH, sc(real, tot / nperm), 0)
    mx = np.array([np.where(M[j] >= minH, sc(M[j], (tot - M[j]) / (nperm - 1)), 0).max() for j in range(nperm)])
    n_cand = len(set(Hr)) + sum(len(h) for h in Hp[:1])
    order = np.argsort(-s_real)[:15]
    top = [{'r': str(keys[j][0]), 'X': keys[j][1], 'Y': keys[j][2], 'H': int(real[j]),
            'mu': round(float(tot[j] / nperm), 2), 'score': round(float(s_real[j]), 2),
            'p_corr': float((1 + (mx >= s_real[j]).sum()) / (1 + nperm))} for j in order if s_real[j] > 0]
    return {'n_tab': len(xs), 'n_candidates_real': len(Hr), 'max_real': float(s_real.max()),
            'null_q95': float(np.quantile(mx, 0.95)), 'p_global': float((1 + (mx >= s_real.max()).sum()) / (1 + nperm)),
            'top': top}


# ---------------------------------------------------------------- (c) random notation
def digit_tables(tabs, sysn):
    D = defaultdict(list)
    for t in tabs:
        for q in t['Q']:
            if q['sys'] == sysn and 'nums' in q:
                for n, c in q['nums']:
                    D[c].append(n)
    return D


def randomize(tabs, sysn, rng, capset='apriori'):
    D = digit_tables(tabs, sysn)
    cv = CAPV[capset]
    out = []
    for t in tabs:
        Q = []
        for q in t['Q']:
            q = dict(q)
            if q['sys'] == sysn:
                v = 0
                for n, c in q['nums']:
                    d = D[c][rng.integers(len(D[c]))]
                    v += d * cv[c.split('@')[0]]
                q['v'] = Fr(v)
            Q.append(q)
        out.append({'id': t['id'], 'site': t['site'], 'Q': Q})
    return out


# ---------------------------------------------------------------- (d) near-exact log grid
GRID = np.exp(np.linspace(np.log(1 / 60), np.log(3000), 6000))


def Hnear(xs, ys, perm=None, tol=0.02, minx=3):
    H = np.zeros(len(GRID))
    lg = np.log(GRID)
    for i in range(len(xs)):
        qy = ys[perm[i]] if perm is not None else ys[i]
        hit = np.zeros(len(GRID), bool)
        for a in xs[i]:
            if a['v'] < minx:
                continue
            for b in qy:
                lr = math.log(b['v'] / a['v'])
                lo, hi = np.searchsorted(lg, lr - tol), np.searchsorted(lg, lr + tol)
                hit[lo:hi] = True
        H += hit
    return H


def near_scan(xs, ys, nperm=100, seed=0):
    rng = np.random.default_rng(seed)
    Hr = Hnear(xs, ys)
    M = np.array([Hnear(xs, ys, perm=rng.permutation(len(xs))) for _ in range(nperm)])
    tot = M.sum(0)
    s = sc(Hr, tot / nperm)
    mx = np.array([sc(M[j], (tot - M[j]) / (nperm - 1)).max() for j in range(nperm)])
    j = int(np.argmax(s))
    order = np.argsort(-s)
    picks, used = [], []
    for k in order:
        if all(abs(math.log(GRID[k] / u)) > 0.05 for u in used):
            used.append(GRID[k])
            picks.append({'r': round(float(GRID[k]), 4), 'H': int(Hr[k]), 'mu': round(float(tot[k] / nperm), 2),
                          'score': round(float(s[k]), 2), 'p_corr': float((1 + (mx >= s[k]).sum()) / (1 + nperm))})
        if len(picks) >= 8:
            break
    return {'max_real': float(s[j]), 'null_q95': float(np.quantile(mx, 0.95)),
            'p_global': float((1 + (mx >= s[j]).sum()) / (1 + nperm)), 'top': picks}


if __name__ == '__main__':
    U = ur3_tablets()
    ux, uy, _ = split_sys(U, 'CNT', 'CAP')
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ux), 257, replace=False)
    ux2, uy2 = [ux[i] for i in idx], [uy[i] for i in idx]
    T = pe_tablets()
    px, py, pids = split_sys(T, 'CNT', 'CAP')
    Tp = plant(T, 'CNT', 'CAP', 30, 0.10, np.random.default_rng(9))
    qx, qy, _ = split_sys(Tp, 'CNT', 'CAP')

    # (a)
    for name, a, b in (('ur3_257', ux2, uy2), ('plant30_10pc', qx, qy), ('pe', px, py)):
        res['split_' + name] = split_test(a, b, nsplit=int(os.environ.get('NS', 30)), nperm=60)
        print('SPLIT', name, res['split_' + name], round(time.time() - t0), flush=True)
    # (b)
    for name, a, b in (('ur3_257', ux2, uy2), ('pe', px, py)):
        res['sign_' + name] = sign_scan(a, b, nperm=NP)
        r = res['sign_' + name]
        print('SIGN', name, r['n_candidates_real'], r['max_real'], r['null_q95'], r['p_global'], flush=True)
        for e in r['top'][:8]:
            print('   ', e, flush=True)
    # sign-tied plant: M388-like plant: a specific count sign gets rate 30
    fins = Counter(q['fin'] for t in T for q in t['Q'] if q['sys'] == 'CNT')
    Tp2 = []
    rngp = np.random.default_rng(11)
    for t in T:
        Q = [dict(q) for q in t['Q']]
        qa = [q for q in Q if q['sys'] == 'CNT' and q['fin'] == 'M388']
        qb = [q for q in Q if q['sys'] == 'CAP']
        if qa and qb and rngp.random() < 0.5:
            qb[0]['v'] = qa[0]['v'] * 30
        Tp2.append({'id': t['id'], 'site': t['site'], 'Q': Q})
    a, b, _ = split_sys(Tp2, 'CNT', 'CAP')
    r = sign_scan(a, b, nperm=NP // 2)
    res['sign_plant_M388x30'] = {k: r[k] for k in ('max_real', 'null_q95', 'p_global')}
    res['sign_plant_M388x30']['top3'] = r['top'][:3]
    print('SIGN plant', res['sign_plant_M388x30'], flush=True)
    # (c) random notation: real PE pair structure vs Y values redrawn with the same notation
    rn = []
    for k in range(20):
        Tr = randomize(T, 'CAP', np.random.default_rng(100 + k))
        a, b, _ = split_sys(Tr, 'CNT', 'CAP')
        R = scan(a, b, nperm=60, seed=k)
        rn.append({'max': R['max_real'], 'q95': R['null_max_q95'], 'top1': R['top'][0]['r'] if R['top'] else None})
    res['random_notation'] = rn
    print('RANDNOT', [round(x['max'], 1) for x in rn], flush=True)
    # (d) near-exact
    for name, a, b in (('ur3_257', ux2, uy2), ('plant30_10pc', qx, qy), ('pe', px, py)):
        res['near_' + name] = near_scan(a, b, nperm=100)
        print('NEAR', name, res['near_' + name], flush=True)
    # (e) value sets
    for cs, amb in (('alt_d', 'sign'), ('alt_b', 'sign'), ('apriori', 'drop')):
        Ta = pe_tablets(capset=cs, amb=amb)
        a, b, _ = split_sys(Ta, 'CNT', 'CAP')
        R = scan(a, b, nperm=NP, seed=2)
        res[f'vs_{cs}_{amb}'] = {k: R[k] for k in ('n_tab', 'max_real', 'null_max_q95', 'p_global')}
        res[f'vs_{cs}_{amb}']['top'] = R['top'][:6]
        print('VS', cs, amb, res[f'vs_{cs}_{amb}'], flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1, default=str)
    print('done', time.time() - t0)
