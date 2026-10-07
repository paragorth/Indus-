"""pe77: shuffle-calibrated unforgeable list for a corpus, split-half reliability, frozen with sha256."""
import sys, os, json
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe77_common as pc
from pe77_analyze import load, unforg


def main(real, shuf, out=None):
    R = load(real); S = load(shuf)
    U, As, Aw = unforg(R)
    Us, _, _ = unforg(S)
    nullv = np.array([abs(v[0]) for v in Us.values()])
    # scale null to the number of strong forgers in the real run (U se ~ 1/sqrt(n))
    ns = np.mean([v[2] for v in Us.values()]); nr = np.mean([v[2] for v in U.values()])
    thr = float(np.quantile(nullv, 0.99) * np.sqrt(ns / nr))
    surv = sorted([(f, v[0], v[1], v[3]) for f, v in U.items() if v[0] > thr and v[0] > 3 * v[3]], key=lambda x: -x[1])
    # split-half reliability (forgers split by parity of k)
    Ra = [r for r in R if r['k'] % 2 == 0]; Rb = [r for r in R if r['k'] % 2 == 1]
    Ua, _, _ = unforg(Ra, min_n=5); Ub, _, _ = unforg(Rb, min_n=5)
    com = sorted(set(Ua) & set(Ub))
    rho = spearmanr([Ua[f][0] for f in com], [Ub[f][0] for f in com])
    sa = {f for f, v in Ua.items() if v[0] > thr * np.sqrt(2)}; sb = {f for f, v in Ub.items() if v[0] > thr * np.sqrt(2)}
    res = {'real': real, 'shuf': shuf, 'n_forgers': len(R), 'A_strong': As, 'A_weak': Aw, 'thr': thr,
           'null_max': float(nullv.max()), 'split_half_rho': float(rho.correlation), 'split_half_n': len(com),
           'half_a_surv': sorted(sa), 'half_b_surv': sorted(sb), 'both_halves': sorted(sa & sb),
           'unforgeable': [{'f': f, 'U': round(u, 3), 'W': None if w is None else round(w, 3), 'se': round(se, 3),
                            'family': pc.family(f)} for f, u, w, se in surv]}
    if out:
        lst = [(d['f'], d['U']) for d in res['unforgeable']]
        res['sha256'] = pc.sha_list(lst)
        json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'unforgeable'}, default=str))
    for d in res['unforgeable']:
        print('  ', d)
    return res


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
