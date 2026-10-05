"""pe49 cycle 2b: make the tics content-blind, then look for job descriptions.

Cycle 2's no-specialty plant (clerks independent of content) still gave habit
groups that separate content as much as real PE does: habits leak content. Fix:
every habit is residualised on the WHOLE content bag (base signs + numeral units,
plus length and dominant unit) by 5-fold cross-fitted ridge, so a habit keeps only
what content cannot predict. Then the cycle-1 consensus. Decisive control: in the
no-specialty plant the hands must now show no content separation beyond the
strata-permuted null, while the specialty plant and Ur III must still show it.
Usage: python3 pe49_cycle2b.py {ur3|plant|plant0|pe} [n_runs]
"""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pe49_common as P
from pe49_cycle1 import plant
from sklearn.linear_model import Ridge


def resid_full(T, rng, alpha=10.0):
    Fp = P.variant_families(T, 'pe')
    Fw = P.variant_families(T, 'word')
    H, names = P.habit_matrix(T, Fp, Fw)
    C, strata = P.coarse_covariates(T)
    Xc, voc = P.content_matrix(T)
    Z = np.column_stack([C, Xc])
    R = np.full_like(H, np.nan)
    keep = []
    for j in range(H.shape[1]):
        m = np.where(~np.isnan(H[:, j]))[0]
        if len(m) < 30 or H[m, j].std() < 1e-9:
            continue
        fold = rng.integers(0, 5, len(m))
        r = np.zeros(len(m))
        for f in range(5):
            tr, te = m[fold != f], m[fold == f]
            if len(te) == 0:
                continue
            mdl = Ridge(alpha=alpha).fit(Z[tr], H[tr, j])
            r[fold == f] = H[te, j] - mdl.predict(Z[te])
        if r.std() < 1e-9:
            continue
        R[m, j] = r / r.std()
        keep.append(j)
    return R[:, keep], [names[k] for k in keep], strata, Xc


def run(name, T, n_runs, seed, truth=None):
    rng = np.random.default_rng(seed)
    R, names, strata, Xc = resid_full(T, rng)
    A = P.consensus(R, n_runs, rng)
    np.save(os.path.join(P.CK, f'c2b_{name}_A.npy'), A)
    An = [P.consensus(P.shuffle_habits(R, rng), max(100, n_runs // 4), rng) for _ in range(3)]
    out = {'name': name, 'n_feat': R.shape[1], 'by_k': {}}
    for k in (6, 8, 12, 16):
        lab = P.spectral_cut(A, k)
        nl = [P.spectral_cut(a, k) for a in An]
        perm = [P.specialty(Xc, P.strata_perm(lab, strata, rng))[0] for _ in range(200)]
        sp = P.specialty(Xc, lab)[0]
        row = {'stab': P.stability(A, lab), 'stab_null': [P.stability(a, l) for a, l in zip(An, nl)],
               'spec': sp, 'spec_perm': float(np.mean(perm)), 'spec_perm_p': float(np.mean(np.array(perm) >= sp)),
               'spec_shufhab': [P.specialty(Xc, l)[0] for l in nl]}
        if truth is not None:
            row['nmi'] = P.nmi(truth, lab)
            row['nmi_null'] = [P.nmi(truth, l) for l in nl]
        out['by_k'][k] = row
        print(name, k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b]) for a, b in row.items()}, flush=True)
    if truth is not None:
        tr = np.unique(truth, return_inverse=True)[1]
        sp = P.specialty(Xc, tr)[0]
        perm = [P.specialty(Xc, P.strata_perm(tr, strata, rng))[0] for _ in range(200)]
        out['true_spec'], out['true_spec_perm'] = sp, float(np.mean(perm))
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        out['ceiling_habit_acc'] = float(cross_val_score(LogisticRegression(C=0.3, max_iter=2000), np.nan_to_num(R), tr, cv=5).mean())
        print(name, 'true spec', round(sp, 4), 'perm', round(out['true_spec_perm'], 4), 'supervised habit acc', round(out['ceiling_habit_acc'], 3),
              'chance', round(np.bincount(tr).max() / len(tr), 3), flush=True)
    json.dump(out, open(os.path.join(P.CK, f'c2b_{name}.json'), 'w'), default=float)


if __name__ == '__main__':
    which = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    if which == 'ur3':
        T, ch = P.load_ur3()
        run('ur3', T, n_runs, 21, truth=[t['group'] for t in T])
    elif which in ('plant', 'plant0'):
        Tp, clerk, _ = plant(P.load_pe(), np.random.default_rng(5 if which == 'plant' else 6), follow=0.6 if which == 'plant' else 0.0)
        run(which, Tp, n_runs, 22, truth=clerk.tolist())
    else:
        run('pe', P.load_pe(), n_runs, 23)
