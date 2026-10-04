"""pe14 cycle 3 summary: paired-record search (A -> B) and per-tablet weave counts."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK  # noqa


def L(cfg, h):
    f = os.path.join(CK, 'c3', f'{cfg}_{h}.npz')
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def pairz(D, minexp=8):
    C, CS = D['C'], D['CS']
    m, s = CS.mean(0), CS.std(0) + 1e-9
    Z = (C - m) / s
    Zs = (CS - m) / s
    ok = m >= minexp
    Z = np.where(ok, Z, -np.inf)
    mx = np.where(ok[None], np.abs(Zs), 0).reshape(len(CS), -1).max(1)
    return Z, m, np.percentile(mx, 95), ok


def per_tablet(D, ps=(2, 3, 4)):
    TL, TLS = D['TL'].astype(float), D['TLS'].astype(float)  # N x 6 (lags 1-6), S x N x 6
    idx = [p - 1 for p in ps]
    m = TLS.mean(0)
    sd = TLS.std(0)
    sd = np.where(sd < 1e-6, np.inf, sd)  # a lag the tablet cannot vary carries no evidence
    z = ((TL - m) / sd)[:, idx].max(1)
    zs = ((TLS - m) / sd)[:, :, idx].max(2)  # S x N
    p = (1 + (zs >= z[None]).sum(0)) / (1 + len(TLS))
    # false-positive rate of the same rule on surrogates (leave one out approx)
    fp = np.mean([(np.mean(zs >= zs[k][None], axis=0) <= 0.01).sum() for k in range(0, len(TLS), 10)])
    return z, p, fp


def main(cfgs):
    out = {}
    for cfg in cfgs:
        A, B = L(cfg, 'A'), L(cfg, 'B')
        if A is None:
            continue
        top = [str(x) for x in A['top']]
        Z, m, thr, ok = pairz(A)
        flat = [(Z[l, i, j], l + 1, top[i], top[j]) for l in range(2) for i in range(len(top)) for j in range(len(top)) if ok[l, i, j]]
        flat.sort(reverse=True)
        neg = sorted(flat)[:6]
        print('==', cfg, 'tablets', len(A['ids']), 'rho %.2f' % float(A['rho']), 'hypotheses', len(flat), 'FWER |z| %.2f' % thr)
        rows = []
        if B is not None:
            topB = [str(x) for x in B['top']]
            ZB, mB, _, okB = pairz(B)
        for z, lag, a, b in flat[:15] + neg:
            s = '  %-8s -> %-8s lag %d zA %6.2f obs %d exp %.1f' % (a, b, lag, z, A['C'][lag - 1, top.index(a), top.index(b)], m[lag - 1, top.index(a), top.index(b)])
            r = {'a': a, 'b': b, 'lag': lag, 'zA': float(z), 'surv': bool(abs(z) > thr)}
            if B is not None and a in topB and b in topB:
                i, j = topB.index(a), topB.index(b)
                cb = B['C'][lag - 1, i, j]
                CSb = B['CS'][:, lag - 1, i, j]
                pB = (1 + np.sum(CSb >= cb)) / (1 + len(CSb)) if z > 0 else (1 + np.sum(CSb <= cb)) / (1 + len(CSb))
                s += ' | zB %6.2f pB %.3f' % (ZB[lag - 1, i, j] if okB[lag - 1, i, j] else np.nan, pB)
                r.update(zB=float(ZB[lag - 1, i, j]), pB=float(pB))
            print(s + (' *' if abs(z) > thr else ''))
            rows.append(r)
        res = {'thr': float(thr), 'nhyp': len(flat), 'rows': rows}
        for h, D in (('A', A), ('B', B)):
            if D is None:
                continue
            z, p, fp = per_tablet(D)
            ids = [str(x) for x in D['ids']]
            n = D['n']
            sig = [(ids[k], int(n[k]), float(z[k])) for k in np.argsort(p) if p[k] <= 0.01]
            print('  per-tablet weave (lags 2-4 vs own DIP), half', h, ': tablets p<=0.01:', len(sig), 'of', len(ids),
                  'expected (surrogates) %.1f' % fp, sig[:12])
            res['tab_' + h] = {'nsig': len(sig), 'fp': float(fp), 'sig': sig}
            pl = os.path.join(CK, 'c1', cfg + '_planted.json')
            if os.path.exists(pl):
                P = set(json.load(open(pl)))
                inP = [k for k in range(len(ids)) if ids[k] in P]
                rec = np.mean([p[k] <= 0.01 for k in inP]) if inP else float('nan')
                print('   planted tablets recovered at p<=0.01: %.2f (%d planted)' % (rec, len(inP)))
                res['tab_' + h]['recovery'] = float(rec)
        out[cfg] = res
    json.dump(out, open(os.path.join(CK, 'c3_sum.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['PE_ENT', 'PE_NOPAIR', 'UR3_LIN', 'PL_P2LAST'])
