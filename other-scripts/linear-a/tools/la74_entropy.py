#!/usr/bin/env python3
"""la74 cycle 2b: label-free pouring prediction. If amounts were poured with a vessel set and big
amounts were rounded to the big vessels, the fraction signs on big integers should be FEWER KINDS
(lower sign entropy, fewer compounds) than on fraction-only amounts. Null: integers permuted among
fraction quantities within site (and within site x support). Planted check: a pouring world with
relative tolerance written onto the real integers must show the effect; a no-vessel world must not."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
S = load_strings(); V = vocab([r['s'] for r in S])
st = [norm(r['s'], V) for r in S]; N = np.array([r['v'] for r in S]); site = np.array([r['site'] for r in S])
def H(strs):
    c = Counter(x for s in strs for x in s); n = sum(c.values()); p = np.array(list(c.values())) / n
    return -(p * np.log(p)).sum(), n
def stat(Nv, strs):
    big = [s for s, v in zip(strs, Nv) if v >= 5]; zero = [s for s, v in zip(strs, Nv) if v == 0]
    # entropy difference corrected for sample size by rarefying zero to len(big) tokens (mean of 50 draws)
    rng = np.random.default_rng(0)
    toks0 = [x for s in zero for x in s]; toksb = [x for s in big for x in s]; m = min(len(toks0), len(toksb))
    hb = np.mean([H([tuple(rng.choice(toksb, m, replace=False))])[0] for _ in range(10)]) if m < len(toksb) else H(big)[0]
    hz = np.mean([H([tuple(rng.choice(toks0, m, replace=False))])[0] for _ in range(30)])
    comp = np.mean([len(set(s)) > 1 for s in big]) - np.mean([len(set(s)) > 1 for s in zero])
    return hz - hb, comp
def perm_test(strs, Nv, nperm=2000, seed=1):
    rng = np.random.default_rng(seed); obs = stat(Nv, strs); null = []
    for _ in range(nperm):
        Np = Nv.copy()
        for g in set(site):
            idx = np.where(site == g)[0]; Np[idx] = rng.permutation(Nv[idx])
        null.append(stat(Np, strs))
    null = np.array(null)
    return dict(dH=float(obs[0]), p_dH=float((np.sum(null[:, 0] >= obs[0]) + 1) / (nperm + 1)),
                dcomp=float(obs[1]), p_comp=float((np.sum(null[:, 1] <= obs[1]) + 1) / (nperm + 1)),
                n_big=int((Nv >= 5).sum()), n_zero=int((Nv == 0).sum()))
def pour_world(seed, tol):
    """planted: a 4-vessel set; amount = N + R; pour fine-to-tolerance: stop when remainder < tol*(N+R)."""
    rng = np.random.default_rng(seed); sizes = [0.5, 0.25, 0.125, 1 / 16]; mp = list(rng.choice(V[:10], 4, replace=False))
    out = []
    for v in N:
        while True:
            R = rng.random(); rem = R; s = []
            for vs, sg in zip(sizes, mp):
                if rem < tol * (v + R) and s: break
                c = int(rem // vs); s += [sg] * c; rem -= c * vs
            if s: break
        out.append(tuple(s))
    return out
if __name__ == '__main__':
    print('REAL', json.dumps(perm_test(st, N)))
    nv = NoVessel(st, V); rng = np.random.default_rng(5)
    for i in range(3):
        print('NVWORLD', i, json.dumps(perm_test(nv_sample(nv, len(st), rng), N, 500, i)))
    for tol in (0.02, 0.05):
        for i in range(2):
            print('POUR tol', tol, i, json.dumps(perm_test(pour_world(100 + i, tol), N, 500, i)))
    # real, tablets only, read-only
    S2 = load_strings(('read',)); keep = [r['sup'] == 'Tablet' for r in S2]
    st2 = [norm(r['s'], V) for r, k in zip(S2, keep) if k]; N2 = np.array([r['v'] for r, k in zip(S2, keep) if k])
    site = np.array([r['site'] for r, k in zip(S2, keep) if k])
    print('REAL read-only tablets', json.dumps(perm_test(st2, N2)))
