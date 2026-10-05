"""pe49 cycle 1: hands from tics. Usage: python3 pe49_cycle1.py {ur3|plant|pe} [n_runs]

ur3  : Ur III Umma, 12 dub-sar seal owners x 60 tablets, opaque signs. Must recover scribes.
plant: PE skeleton with 8 planted clerks (variant preferences, divider omission,
       blank-line habit) whose assignment follows content topics 60% of the time.
pe   : real Susa tablets (>= 4 numeric lines).
Every run: consensus over n_runs random habit subsets x seeds x k; nulls = habits
shuffled across tablets (3 replicas, same pipeline); specialty vs content-matched
(length x dominant unit strata) permutations of the hand labels.
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pe49_common as P


def features(T):
    Fp = P.variant_families(T, 'pe')
    Fw = P.variant_families(T, 'word')
    H, names = P.habit_matrix(T, Fp, Fw)
    C, strata = P.coarse_covariates(T)
    R, keep = P.residualise(H, C)
    return R, [names[k] for k in keep], strata


def plant(T, rng, K=8, follow=0.6):
    """Rewrite PE tablets as if written by K clerks with tics and specialties."""
    import copy
    T = copy.deepcopy(T)
    Xc, voc = P.content_matrix(T)
    from sklearn.cluster import KMeans
    topic = KMeans(K, n_init=4, random_state=0).fit_predict(Xc)
    clerk = np.where(rng.random(len(T)) < follow, topic, rng.integers(0, K, len(T)))
    fam = P.variant_families(T, 'pe')
    pref = {f: rng.dirichlet(np.ones(len(forms)) * 0.7, K) for f, forms in fam.items()}
    s2f = {s: f for f, forms in fam.items() for s in forms}
    comma = rng.choice([0.0, 0.01, 0.05, 0.2], K)
    blank = rng.choice([0, 0, 1, 2], K)
    for t, c in zip(T, clerk):
        for l in t['lines']:
            new = []
            for s in l['toks']:
                f = s2f.get(s)
                if f:
                    forms = sorted(fam[f])
                    s = forms[rng.choice(len(forms), p=pref[f][c])]
                new.append(s)
            l['toks'] = new
            l['words'] = [tuple(new)] if new else []
            if l['nums'] and new:
                l['divider'] = bool(rng.random() >= comma[c])
        t['dollars'] = [d for d in t['dollars'] if 'blank' not in d[1]] + [('reverse', 'blank space')] * int(blank[c])
        t['group'] = int(c)
    return T, clerk, topic


def run(name, T, n_runs, seed, truth=None, extra=None):
    rng = np.random.default_rng(seed)
    R, names, strata = features(T)
    Xc, voc = P.content_matrix(T)
    out = {'name': name, 'n': len(T), 'n_feat': R.shape[1], 'features': names}
    t0 = time.time()
    A = P.consensus(R, n_runs, rng)
    out['secs'] = round(time.time() - t0)
    np.save(os.path.join(P.CK, f'c1_{name}_A.npy'), A)
    nulls = []
    for r in range(3):
        An = P.consensus(P.shuffle_habits(R, rng), max(100, n_runs // 4), rng)
        nulls.append(An)
    res = {}
    for k in (4, 6, 8, 10, 12, 16):
        lab = P.spectral_cut(A, k)
        st = P.stability(A, lab)
        nst = [P.stability(An, P.spectral_cut(An, k)) for An in nulls]
        mi, _ = P.specialty(Xc, lab)
        perm = [P.specialty(Xc, P.strata_perm(lab, strata, rng))[0] for _ in range(200)]
        nlab = [P.spectral_cut(An, k) for An in nulls]
        nmi_null_spec = [P.specialty(Xc, nl)[0] for nl in nlab]
        row = {'stab': st, 'stab_null': nst, 'sizes': sorted(np.bincount(lab).tolist(), reverse=True),
               'spec': mi, 'spec_perm_mean': float(np.mean(perm)), 'spec_perm_p': float(np.mean(np.array(perm) >= mi)),
               'spec_shufhab': nmi_null_spec}
        if truth is not None:
            row['nmi'] = P.nmi(truth, lab)
            row['nmi_null'] = [P.nmi(truth, nl) for nl in nlab]
        res[k] = row
        print(name, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items() if a != 'sizes'}, flush=True)
    out['by_k'] = res
    if truth is not None:
        tr = np.unique(truth, return_inverse=True)[1]
        mi, _ = P.specialty(Xc, tr)
        perm = [P.specialty(Xc, P.strata_perm(tr, strata, rng))[0] for _ in range(200)]
        out['true_spec'] = mi
        out['true_spec_perm'] = float(np.mean(perm))
        out['true_spec_p'] = float(np.mean(np.array(perm) >= mi))
        # what content alone would give (reference, not a null)
        from sklearn.cluster import KMeans
        sq = (Xc ** 2).sum(1)
        Ec = P.embed(np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T, 0)), 8)
        out['content_only_nmi'] = P.nmi(tr, KMeans(len(set(tr)), n_init=4, random_state=0).fit_predict(Ec))
        # per-feature truth signal: which habits separate scribes (kruskal)
        from scipy.stats import kruskal
        fs = []
        for j, nm in enumerate(names):
            m = ~np.isnan(R[:, j])
            g = [R[m & (tr == k), j] for k in np.unique(tr)]
            g = [x for x in g if len(x) >= 3]
            if len(g) >= 3 and np.ptp(np.concatenate(g)) > 0:
                fs.append((nm, float(kruskal(*g).pvalue), int(m.sum())))
        fs.sort(key=lambda x: x[1])
        out['habit_vs_truth'] = fs[:15]
        out['habit_vs_truth_n_p01'] = sum(1 for x in fs if x[1] < 0.01)
        out['habit_vs_truth_n'] = len(fs)
        print('true specialty', mi, out['true_spec_perm'], out['true_spec_p'], 'content-only NMI', out['content_only_nmi'],
              'habits p<0.01', out['habit_vs_truth_n_p01'], '/', len(fs), flush=True)
    if extra:
        out.update(extra)
    json.dump(out, open(os.path.join(P.CK, f'c1_{name}.json'), 'w'), default=float)
    return out


if __name__ == '__main__':
    which = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    if which == 'ur3':
        T, ch = P.load_ur3()
        run('ur3', T, n_runs, 1, truth=[t['group'] for t in T], extra={'scribes': ch})
    elif which == 'plant':
        T = P.load_pe()
        rng = np.random.default_rng(5)
        Tp, clerk, topic = plant(T, rng)
        run('plant', Tp, n_runs, 2, truth=clerk.tolist(), extra={'topic_clerk_nmi': P.nmi(topic, clerk)})
    else:
        T = P.load_pe()
        out = run('pe', T, n_runs, 3)
