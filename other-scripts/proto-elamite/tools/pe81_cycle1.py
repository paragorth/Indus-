"""pe81 cycle 1: random statistics -> stable site orderings; nulls; planted power; freeze text-side site axes.
No environmental value is read in this script."""
import sys, os, json, hashlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E

NH = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
REPS = 60
rng = np.random.default_rng(81)
rows = E.tablet_rows()
X, names, top = E.features(rows)
X = X.astype(np.float32)
sites = E.SITES
site_of = np.array([sites.index(s) for s, _ in rows])
counts = {s: int((site_of == k).sum()) for k, s in enumerate(sites)}
print('tablets', counts, 'features', X.shape[1], flush=True)
H = E.make_hyps(X.shape[1], NH, rng)
out = {'counts': counts, 'n_features': X.shape[1], 'NH': NH, 'reps': REPS}


def susa_likeness(Z, site_of, idx):
    """per outpost: mean over hyps idx of |mean_k - mean_Susa| / sd(all)"""
    sd = Z[:, idx].std(0) + 1e-9
    ms = Z[site_of == 0][:, idx].mean(0)
    D = []
    for k in range(1, 5):
        mk = Z[site_of == k][:, idx].mean(0)
        D.append(float(np.mean(np.abs(mk - ms) / sd)))
    return D


def analyse(Xc, site_of, rng, tag):
    Z, st5 = E.run(Xc, site_of, sites, H, rng, reps=REPS, susa_n=12)
    m4 = site_of > 0
    st4 = E.stability(Z[m4], site_of[m4] - 1, sites[1:], rng, REPS, 0)
    return Z, st5, st4


t0 = time.time()
Z, st5, st4 = analyse(X, site_of, rng, 'real')
print('real done', round(time.time() - t0), 's', flush=True)

# null: re-deal tablets among sites (5-site) and among outposts (4-site)
NULLS = 20
null5 = []; null4 = []
for b in range(NULLS):
    perm = site_of.copy(); rng.shuffle(perm)
    _, a5, _ = analyse(X, perm, rng, 'null5')
    o = site_of.copy(); m4 = o > 0; v = o[m4]; rng.shuffle(v); o[m4] = v
    _, _, a4 = analyse(X, o, rng, 'null4')
    null5.append(a5); null4.append(a4)
    print('null', b, flush=True)
null5 = np.array(null5); null4 = np.array(null4)
q5 = float(np.quantile(null5, 0.99)); q4 = float(np.quantile(null4, 0.99))
surv5 = np.where(st5 > q5)[0]; surv4 = np.where(st4 > q4)[0]
n5null = [(a > q5).sum() for a in null5]; n4null = [(a > q4).sum() for a in null4]
out.update({'q5': q5, 'q4': q4, 'surv5': int(len(surv5)), 'surv4': int(len(surv4)),
            'surv5_null': [int(x) for x in n5null], 'surv4_null': [int(x) for x in n4null],
            'stab5_mean': float(st5.mean()), 'stab5_null_mean': float(null5.mean()),
            'stab4_mean': float(st4.mean()), 'stab4_null_mean': float(null4.mean())})
print(json.dumps({k: out[k] for k in ['q5', 'q4', 'surv5', 'surv4', 'surv5_null', 'surv4_null']}), flush=True)

# text-side frozen objects
USE = surv5 if len(surv5) >= 20 else np.arange(NH)
out['D_basis'] = 'survivors' if len(surv5) >= 20 else 'all hypotheses'
D = susa_likeness(Z, site_of, USE)
# stability of D across halves
Dh = []
for r in range(40):
    A = np.zeros(len(site_of), bool)
    for k in range(5):
        m = rng.permutation(np.where(site_of == k)[0]); A[m[:len(m) // 2]] = True
    for half in (A, ~A):
        Dh.append(susa_likeness(Z[half], site_of[half], USE))
Dh = np.array(Dh)
rank_agree = float(np.mean([np.all(np.argsort(Dh[2 * i]) == np.argsort(Dh[2 * i + 1])) for i in range(40)]))
# undirected axes: PC of survivor site scores (outposts-only survivors; outposts only)
def pcs(Z, site_of, idx, ks):
    S = np.vstack([Z[site_of == k][:, idx].mean(0) for k in ks])
    S = (S - S.mean(0)) / (S.std(0) + 1e-9)
    u, s, vt = np.linalg.svd(S, full_matrices=False)
    return u[:, :2] * s[:2], (s ** 2 / (s ** 2).sum())[:2]
pc4, ev4 = pcs(Z, site_of, surv4 if len(surv4) else np.arange(NH), [1, 2, 3, 4])
pc5, ev5 = pcs(Z, site_of, surv5 if len(surv5) else np.arange(NH), [0, 1, 2, 3, 4])
# top features among survivors
from collections import Counter
fc = Counter()
for j in surv4:
    for i in H[j][0]:
        fc[names[i]] += 1
fc5 = Counter()
for j in surv5:
    for i in H[j][0]:
        fc5[names[i]] += 1
out.update({'susa_likeness_D': dict(zip(sites[1:], D)), 'D_half_rank_agreement': rank_agree,
            'D_halves_mean': dict(zip(sites[1:], Dh.mean(0).tolist())), 'D_halves_sd': dict(zip(sites[1:], Dh.std(0).tolist())),
            'pc4': {s: pc4[i].tolist() for i, s in enumerate(sites[1:])}, 'pc4_ev': ev4.tolist(),
            'pc5': {s: pc5[i].tolist() for i, s in enumerate(sites)}, 'pc5_ev': ev5.tolist(),
            'top_features_surv4': fc.most_common(15), 'top_features_surv5': fc5.most_common(15)})

# planted power: outpost k gets a share f_k of its tablets replaced by random Susa tablets, f by hidden ruler rank
plant = []
for p in range(12):
    r = rng.permutation(4)                    # hidden ruler rank (0 = most Susa-like)
    f = np.array([0.45, 0.30, 0.15, 0.0])[r]  # rank 0 -> 45% Susa tablets mixed in
    Xp = X.copy()
    susa_idx = np.where(site_of == 0)[0]
    for k in range(4):
        m = np.where(site_of == k + 1)[0]
        nrep = int(round(f[k] * len(m)))
        if nrep:
            tgt = rng.choice(m, nrep, replace=False)
            Xp[tgt] = X[rng.choice(susa_idx, nrep, replace=False)]
    Zp, s5p, s4p = analyse(Xp, site_of, rng, 'plant')
    sv = np.where(s5p > q5)[0]
    Dp = susa_likeness(Zp, site_of, sv if len(sv) >= 20 else np.arange(NH))
    from scipy.stats import spearmanr
    rho = spearmanr(Dp, r).correlation   # D should rise with rank (rank 3 = least Susa-like)
    plant.append({'r': r.tolist(), 'D': Dp, 'rho': float(rho), 'surv5': int(len(sv))})
    print('plant', p, round(rho, 2), len(sv), flush=True)
out['planted'] = plant
out['planted_exact'] = float(np.mean([x['rho'] > 0.99 for x in plant]))
out['planted_rho_mean'] = float(np.mean([x['rho'] for x in plant]))

frozen = {'what': 'pe81 text-side site axes, frozen before any environmental value was fetched',
          'susa_likeness_D (lower = writes more like Susa)': out['susa_likeness_D'],
          'pc4': out['pc4'], 'pc5': out['pc5'], 'n_survivors': [out['surv5'], out['surv4']]}
b = json.dumps(frozen, sort_keys=True, indent=1).encode()
h = hashlib.sha256(b).hexdigest()
fp = os.path.join(E.common.DATA, 'pe81_frozen_axes.json')
open(fp, 'wb').write(b)
out['frozen_sha256'] = h
json.dump(out, open(os.path.join(E.CK, 'cycle1.json'), 'w'), indent=1, default=float)
print('FROZEN', h, flush=True)
print(json.dumps({k: out[k] for k in ['susa_likeness_D', 'D_half_rank_agreement', 'D_halves_sd', 'pc4', 'pc4_ev',
                                      'top_features_surv4', 'top_features_surv5', 'planted_exact', 'planted_rho_mean']}, default=float))
