"""pe76 cycle 1: span-in-careers ABC on variant markers; planted held-out worlds; two-seed stability;
variant-shuffle and curveball nulls; posterior-predictive distance check."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pe76_common as P, common
from pe76_bank import load_bank
from pe76_est import Est, vshuffle, curveball, calib

T = common.load(); n = len(T)
names, sets = P.variant_markers(T)
s_real = P.stats(P.to_matrix(sets, n))
res = {'n_markers': len(names), 'n_tokens': int(sum(len(s) for s in sets))}
for seed in [int(a) for a in sys.argv[1:]] or [1]:
    TH, ST = load_bank('var', seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(TH)); nh = len(TH) // 10
    h, tr = perm[:nh], perm[nh:]
    E = Est(TH[tr], ST[tr], seed=seed)
    r = {'n_sims': int(len(TH)), 'calib': calib(E, TH[h], ST[h])}
    o = E.rej(s_real); dall = o.pop('d_all')
    r['real'] = dict(o, rf=E.rf_pred(s_real), S_rf=float(np.exp(E.rf_pred(s_real))), S_adj=float(np.exp(o['adj_med'])))
    # how unusual is the real corpus for the model? distance to its 1% nearest vs the same for held-out sims
    dh = []
    for i in range(100):
        oo = E.rej(ST[h][i]); dh.append(oo['dist'])
    r['ppc'] = dict(real_dist=o['dist'], sim_dist_q50=float(np.median(dh)), sim_dist_q95=float(np.quantile(dh, .95)))
    # nulls
    nulls = {'VSHUF': [], 'CURVE': []}
    for k in range(5):
        rr = np.random.default_rng(1000 + k)
        sv = vshuffle(T, rr); sn = P.stats(P.to_matrix(sv, n))
        nulls['VSHUF'].append(dict(rf=E.rf_pred(sn), adj=E.rej(sn)['adj_med'], dist=E.rej(sn)['dist']))
        sc = curveball(sets, n, rr); sn = P.stats(P.to_matrix(sc, n))
        nulls['CURVE'].append(dict(rf=E.rf_pred(sn), adj=E.rej(sn)['adj_med'], dist=E.rej(sn)['dist']))
    r['nulls'] = nulls
    imp = sorted(zip(P.STAT_NAMES, E.rf.feature_importances_), key=lambda x: -x[1])[:6]
    r['rf_importance'] = [(a, round(float(b), 3)) for a, b in imp]
    # nuisance posteriors at the real corpus
    r['real_nuisance'] = {k: E.rej(s_real, col=j)['adj_med'] for j, k in enumerate(P.THETA_KEYS)}
    res['seed%d' % seed] = r
    print(json.dumps(r, indent=1, default=float), flush=True)
json.dump(res, open(os.path.join(P.CKPT, 'cycle1_%s.json' % '_'.join(sys.argv[1:] or ['1'])), 'w'), indent=1, default=float)
