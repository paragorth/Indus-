"""pe76 cycle 2: (a) planted worlds made through the real corpus with a DIFFERENT generator (soft fashions +
idiolects on the real variant tokens), estimated blind; (b) held-out signs: estimate on variant forms of
half A of base signs and of half B separately (own banks); (c) independent medium: numeral variants."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import spearmanr
import pe76_common as P, common
from pe76_bank import load_bank, marker_sets
from pe76_est import Est, vshuffle
from pe76_plant import plant

T = common.load(); n = len(T)
out = {}
part = sys.argv[1]
if part == 'a':
    TH, ST = load_bank('var', 1)
    E = Est(TH, ST, seed=1)
    rows = []
    for S in [0.15, 0.5, 1.0, 3.0, 10.0, 25.0]:
        for idio in [0.2, 0.6]:
            for M in [3.0, 12.0]:
                for k in range(2):
                    rng = np.random.default_rng(int(S * 100) * 1000 + int(idio * 10) * 100 + int(M) * 10 + k)
                    sets, _ = plant(T, S, rng, M=M, idio=idio)
                    s = P.stats(P.to_matrix(sets, n))
                    o = E.rej(s)
                    rows.append(dict(S=S, idio=idio, M=M, rf=E.rf_pred(s), adj=o['adj_med'], dist=o['dist']))
                    print(rows[-1], flush=True)
    y = np.log([r['S'] for r in rows])
    out['rows'] = rows
    out['spearman_rf'] = spearmanr(y, [r['rf'] for r in rows]).correlation
    out['spearman_adj'] = spearmanr(y, [r['adj'] for r in rows]).correlation
    for S in sorted({r['S'] for r in rows}):
        rr = [r for r in rows if r['S'] == S]
        out['S=%g' % S] = dict(rf_S=float(np.exp(np.mean([r['rf'] for r in rr]))), adj_S=float(np.exp(np.mean([r['adj'] for r in rr]))))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
else:
    which = part   # varA / varB / num
    TH, ST = load_bank(which, 1)
    rng = np.random.default_rng(5); perm = rng.permutation(len(TH)); nh = len(TH) // 10
    E = Est(TH[perm[nh:]], ST[perm[nh:]], seed=1)
    from pe76_est import calib, curveball
    _, sets = marker_sets(T, which)
    s = P.stats(P.to_matrix(sets, n)); o = E.rej(s); o.pop('d_all')
    out = dict(which=which, n_markers=len(sets), n_sims=len(TH), calib=calib(E, TH[perm[:nh]], ST[perm[:nh]]),
               real=dict(o, rf=E.rf_pred(s), S_rf=float(np.exp(E.rf_pred(s)))))
    nl = []
    for k in range(5):
        sc = curveball(sets, n, np.random.default_rng(2000 + k)); sn = P.stats(P.to_matrix(sc, n))
        nl.append(E.rf_pred(sn))
    out['curve_rf'] = nl
    print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(P.CKPT, 'cycle2_%s.json' % part), 'w'), indent=1, default=float)
