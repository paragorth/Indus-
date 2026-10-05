"""pe49 cycle 2: let each tablet calibrate which habits are tics.

A tic belongs to the writer, so it should agree between the two halves of one
tablet (odd vs even text lines) more than between halves of different tablets of
the same content stratum (length x dominant unit). Weight per habit feature =
split-half reliability minus the same correlation computed between halves of
random same-stratum tablet pairs (content agreement). Then weighted consensus
clustering as in cycle 1. Usage: python3 pe49_cycle2.py {ur3|plant|pe} [n_runs]
"""
import os, sys, json, copy
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pe49_common as P
from pe49_cycle1 import plant


def halves(T):
    A, B = copy.deepcopy(T), copy.deepcopy(T)
    for a, b in zip(A, B):
        txt = [i for i, l in enumerate(a['lines']) if l['kind'] == 'text']
        odd = set(txt[0::2])
        a['lines'] = [l for i, l in enumerate(a['lines']) if i in odd]
        b['lines'] = [l for i, l in enumerate(b['lines']) if i not in odd]
        # tablet-level layout features are not split: blank them in half B so they do not inflate reliability
        b['dollars'], b['surfs'], b['ncol'] = [], set(), 1
        a['dollars'], a['surfs'], a['ncol'] = [], set(), 1
    return A, B


def reliability(T, rng):
    Fp = P.variant_families(T, 'pe')
    Fw = P.variant_families(T, 'word')
    A, B = halves(T)
    HA, names = P.habit_matrix(A, Fp, Fw)
    HB, _ = P.habit_matrix(B, Fp, Fw)
    C, strata = P.coarse_covariates(T)
    # same-stratum random partner
    partner = np.arange(len(T))
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            partner[idx] = np.roll(rng.permutation(idx), 1)
    w = {}
    for j, nm in enumerate(names):
        def corr(x, y):
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() < 15 or x[m].std() == 0 or y[m].std() == 0:
                return np.nan, int(m.sum())
            return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())
        r_self, n1 = corr(HA[:, j], HB[:, j])
        r_part, n2 = corr(HA[:, j], HB[partner, j])
        w[nm] = (r_self, r_part, n1)
    return w


def features_weighted(T, rng, mode):
    Fp = P.variant_families(T, 'pe')
    Fw = P.variant_families(T, 'word')
    H, names = P.habit_matrix(T, Fp, Fw)
    C, strata = P.coarse_covariates(T)
    R, keep = P.residualise(H, C)
    names = [names[k] for k in keep]
    rel = reliability(T, rng)
    wt = []
    for nm in names:
        rs, rp, n = rel.get(nm, (np.nan, np.nan, 0))
        if mode == 'flat':
            wt.append(1.0)
        else:
            d = (rs if not np.isnan(rs) else 0) - (rp if not np.isnan(rp) else 0)
            wt.append(max(d, 0.0) if not np.isnan(rs) else 0.3)  # tablet-level layout features: fixed weight
    wt = np.array(wt)
    return R * np.sqrt(wt), names, strata, rel, wt


def run(name, T, n_runs, seed, truth=None):
    rng = np.random.default_rng(seed)
    out = {'name': name}
    Xc, voc = P.content_matrix(T)
    for mode in ('flat', 'tic'):
        R, names, strata, rel, wt = features_weighted(T, rng, mode)
        use = wt > 0
        R = R[:, use]
        A = P.consensus(R, n_runs, rng, frac_feat=0.6)
        np.save(os.path.join(P.CK, f'c2_{name}_{mode}_A.npy'), A)
        An = [P.consensus(P.shuffle_habits(R, rng), max(100, n_runs // 4), rng, frac_feat=0.6) for _ in range(3)]
        res = {}
        for k in (6, 8, 12, 16):
            lab = P.spectral_cut(A, k)
            nl = [P.spectral_cut(a, k) for a in An]
            row = {'stab': P.stability(A, lab), 'stab_null': [P.stability(a, l) for a, l in zip(An, nl)],
                   'spec': P.specialty(Xc, lab)[0],
                   'spec_perm': float(np.mean([P.specialty(Xc, P.strata_perm(lab, strata, rng))[0] for _ in range(100)])),
                   'spec_shufhab': [P.specialty(Xc, l)[0] for l in nl]}
            if truth is not None:
                row['nmi'] = P.nmi(truth, lab)
                row['nmi_null'] = [P.nmi(truth, l) for l in nl]
            res[k] = row
            print(name, mode, k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b]) for a, b in row.items()}, flush=True)
        top = sorted([(nm, rel[nm]) for nm in names if nm in rel and not np.isnan(rel[nm][0])],
                     key=lambda x: -(x[1][0] - (x[1][1] if not np.isnan(x[1][1]) else 0)))[:15]
        out[mode] = {'by_k': res, 'n_feat': int(use.sum()), 'top_reliable': top}
        print(name, mode, 'n_feat', int(use.sum()), 'top reliable', [(a, round(b[0], 2), round(b[1], 2) if not np.isnan(b[1]) else None) for a, b in top[:8]], flush=True)
    json.dump(out, open(os.path.join(P.CK, f'c2_{name}.json'), 'w'), default=float)


if __name__ == '__main__':
    which = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    if which == 'ur3':
        T, ch = P.load_ur3()
        run('ur3', T, n_runs, 11, truth=[t['group'] for t in T])
    elif which == 'plant':
        T = P.load_pe()
        Tp, clerk, topic = plant(T, np.random.default_rng(5))
        run('plant', Tp, n_runs, 12, truth=clerk.tolist())
    elif which == 'plant0':
        T = P.load_pe()
        Tp, clerk, topic = plant(T, np.random.default_rng(6), follow=0.0)
        run('plant0', Tp, n_runs, 14, truth=clerk.tolist())
    else:
        run('pe', P.load_pe(), n_runs, 13)
