"""pe14 cycle 1 summary: FWER-corrected weave hits on half A, re-tested on half B."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK, LAGS  # noqa

P0 = 1  # index of lag 2


def load(cfg, half):
    f = os.path.join(CK, 'c1', '%s_%s.json' % (cfg, half))
    return json.load(open(f)) if os.path.exists(f) else None


def zmat(D):
    names = sorted(D['res'])
    Z = np.array([[(D['res'][k]['obs'][j] - D['res'][k]['adj_m'][j]) / max(D['res'][k]['adj_s'][j], 1e-9)
                   for j in range(len(LAGS))] for k in names])
    Zs = np.array([[(D['res'][k]['obs'][j] - D['res'][k]['shuf_m'][j]) / max(D['res'][k]['shuf_s'][j], 1e-9)
                    for j in range(len(LAGS))] for k in names])
    # FWER: each surrogate run through the same search
    S = np.array([D['res'][k]['adj_sur'] for k in names])  # F x nsur x L
    m = S.mean(1, keepdims=True)
    s = S.std(1, keepdims=True) + 1e-9
    Zsur = (S - m) / s
    mx = Zsur[:, :, P0:].max(axis=(0, 2))
    mn = Zsur[:, :, P0:].min(axis=(0, 2))
    return names, Z, Zs, np.percentile(mx, 95), np.percentile(mn, 5), S


def main():
    cfgs = sorted({os.path.basename(f).rsplit('_', 1)[0] for f in glob.glob(os.path.join(CK, 'c1', '*_A.json'))})
    out = {}
    for cfg in cfgs:
        A, B = load(cfg, 'A'), load(cfg, 'B')
        names, Z, Zs, thr, thrlo, _ = zmat(A)
        rows = []
        hit = [(Z[i, j], names[i], LAGS[j]) for i in range(len(names)) for j in range(P0, len(LAGS))]
        hit.sort(reverse=True)
        lo = sorted([(Z[i, j], names[i], LAGS[j]) for i in range(len(names)) for j in range(P0, len(LAGS))])[:5]
        print('==', cfg, 'ntab A', A['ntab'], 'search', len(hit), 'FWER z_hi %.2f z_lo %.2f' % (thr, thrlo))
        nsurv = sum(1 for h in hit if h[0] > thr)
        if B:
            nb, ZB, ZsB, _, _, SB = zmat(B)
            ib = {k: i for i, k in enumerate(nb)}
        for z, k, p in hit[:15]:
            line = '  %-14s lag %2d zA %5.2f (shufA %5.2f) rhoA %.2f' % (k, p, z, Zs[names.index(k), p - 1], A['res'][k]['rho'])
            rec = {'f': k, 'lag': p, 'zA': z, 'surv': bool(z > thr)}
            if B and k in ib:
                i = ib[k]
                ob = B['res'][k]['obs'][p - 1]
                pB = (1 + np.sum(SB[i, :, p - 1] >= ob)) / (1 + SB.shape[1])
                line += ' | zB %5.2f pB %.3f' % (ZB[i, p - 1], pB)
                rec.update(zB=ZB[i, p - 1], pB=pB)
            rows.append(rec)
            print(line + (' *' if z > thr else ''))
        print('  survivors on A:', nsurv, ' lowest:', ['%s@%d %.1f' % (k, p, z) for z, k, p in lo[:3]])
        out[cfg] = {'thr': thr, 'thrlo': thrlo, 'nsurv': nsurv, 'top': rows,
                    'lowest': [(k, p, z) for z, k, p in lo]}
    json.dump(out, open(os.path.join(CK, 'c1_sum.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
