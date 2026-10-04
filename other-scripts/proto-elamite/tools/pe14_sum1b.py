"""pe14 cycle 1b: COMB statistic. For feature f and period p (2-6):
  comb = mean relative excess over the adjacency null at lags that are multiples of p
         (p, 2p, ... <= 12) minus the mean at the other lags 2-12 (lag 1 is fitted).
A weave makes a comb (excess at p, 2p, 3p, deficit between); topic, sections and
adjacency do not. z against the same statistic on each of the 120 adjacency-null
surrogates; FWER from the surrogates' own search maxima; features with < 20
expected agreements at lag 2 are skipped. Half A searches, half B re-tests.
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK, LAGS  # noqa

PS = [2, 3, 4, 5, 6]


def comb(V, m, p):
    """V: (..., 12) counts; m: (12,) null mean -> comb statistic"""
    e = (V - m) / np.maximum(m, 1e-9)
    mult = [l - 1 for l in LAGS if l >= 2 and l % p == 0]
    oth = [l - 1 for l in LAGS if l >= 2 and l % p != 0]
    return e[..., mult].mean(-1) - e[..., oth].mean(-1)


def analyse(D, minexp=20):
    out = {}
    for k, r in D['res'].items():
        m = np.array(r['adj_m'])
        if m[1] < minexp:
            continue
        S = np.array(r['adj_sur'])
        o = np.array(r['obs'])
        for p in PS:
            cs = comb(S, m, p)
            co = comb(o, m, p)
            sd = cs.std() + 1e-12
            out[(k, p)] = {'z': (co - cs.mean()) / sd, 'zsur': (cs - cs.mean()) / sd,
                           'p': (1 + np.sum(cs >= co)) / (1 + len(cs)), 'comb': co}
    return out


def main(cfgs=None):
    cfgs = cfgs or sorted({os.path.basename(f).rsplit('_', 1)[0] for f in glob.glob(os.path.join(CK, 'c1', '*_A.json'))})
    summ = {}
    for cfg in cfgs:
        fa = os.path.join(CK, 'c1', cfg + '_A.json')
        fb = os.path.join(CK, 'c1', cfg + '_B.json')
        A = analyse(json.load(open(fa)))
        B = analyse(json.load(open(fb))) if os.path.exists(fb) else {}
        keys = list(A)
        Zs = np.array([A[k]['zsur'] for k in keys])
        thr = np.percentile(Zs.max(0), 95)
        rank = sorted(keys, key=lambda k: -A[k]['z'])
        surv = [k for k in rank if A[k]['z'] > thr]
        print('==', cfg, 'hypotheses', len(keys), 'FWER z %.2f' % thr, 'survivors', len(surv))
        rows = []
        for k in rank[:12]:
            b = B.get(k)
            s = '  %-12s p%d zA %5.2f comb %+.3f' % (k[0], k[1], A[k]['z'], A[k]['comb'])
            if b:
                s += ' | zB %5.2f pB %.3f comb %+.3f' % (b['z'], b['p'], b['comb'])
            print(s + (' *' if k in surv else ''))
            rows.append({'f': k[0], 'p': k[1], 'zA': A[k]['z'], 'surv': k in surv,
                         'zB': b['z'] if b else None, 'pB': b['p'] if b else None})
        summ[cfg] = {'thr': thr, 'n': len(keys), 'surv': [list(k) for k in surv], 'top': rows}
    json.dump(summ, open(os.path.join(CK, 'c1b_sum.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
