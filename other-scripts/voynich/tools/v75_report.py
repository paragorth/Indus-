"""v75 report: summarise run_<tag>.pkl. Usage: python3 v75_report.py TAG"""
import os, sys, pickle, json
import numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X


def rep(tag, quiet=False):
    R = pickle.load(open(os.path.join(X.CK, 'run_%s.pkl' % tag), 'rb'))
    A = R['A']; ka = np.array([a['kind_acc'] for a in A]); gg = np.array([a['gen_as_gen'] for a in A])
    s = R['surv']; B = R['B']
    out = dict(tag=tag, with_gen=R['with_gen'], n_models=len(A), kind_acc_A_median=float(np.median(ka)),
               kind_acc_A_q90=float(np.quantile(ka, .9)), kind_acc_A_max=float(ka.max()),
               perm_null_mean=float(np.mean(R['perm'])), perm_null_max=float(np.max(R['perm'])),
               top20_A_mean=float(np.mean(ka[R['top']])), n_surv=len(s),
               surv_A_kind_acc=float(np.mean(ka[s])), surv_B_kind_acc=float(np.mean([b['kind_acc'] for b in B])),
               surv_A_gen_as_gen=float(np.nanmean(gg[s])), surv_B_gen_as_gen=float(np.nanmean([b['gen_as_gen'] for b in B])),
               surv_B_gen_as_own=float(np.nanmean([b['gen_as_own'] for b in B])))
    pk = defaultdict(list)
    for b in B:
        for k, v in b['per_kind'].items(): pk[k].append(v)
    out['surv_B_per_kind'] = {k: round(float(np.mean(v)), 3) for k, v in sorted(pk.items())}
    lay = Counter(R['models'][i]['layout'] for i in s); typ = Counter(R['models'][i]['typ'] for i in s)
    out['surv_layout'] = dict(lay); out['surv_types'] = dict(typ)
    ro = {}
    for r in ('A', 'B'):
        for k, c in R['readout'][r]['agg'].items():
            tot = sum(c.values())
            top = sorted(c.items(), key=lambda x: -x[1])[:4]
            rk = R['readout'][r]['rank'][k]
            nn = sum(v for kk, v in rk.items() if kk.startswith('best_desig:'))
            bd = sorted([(kk.split(':')[1], v / nn) for kk, v in rk.items() if kk.startswith('best_desig:')], key=lambda x: -x[1])[:3]
            ro['%s %s' % (r, k)] = dict(top=[(a, round(b / tot, 3)) for a, b in top],
                                         best_desig=[(a, round(b, 3)) for a, b in bd],
                                         lang_before_desig=round(rk.get('lang_before_desig', 0) / nn, 3))
    out['readout'] = ro
    if not quiet: print(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    rep(sys.argv[1])
