"""pe14 cycle 1c: the hardest non-periodic null -- a PER-TABLET adjacency model.
Each tablet gets its own rho (grid, chosen so the tablet's own expected lag-1
agreement matches its observed lag-1 agreement), so a tablet whose entries simply
avoid repeating their neighbour (or run in blocks) is reproduced tablet by tablet.
A weave must then show up as a comb (excess at even lags, deficit at odd lags) that
even this one-parameter-per-tablet first-order model cannot make.
Also: the same with the 12 longest tablets removed (does one big tablet carry it?).
usage: python3 pe14_cycle1c.py <config> [drop_long]
"""
import json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK, LAGS, features, lag_agree, _prep  # noqa
from pe14_cycle1 import corpus  # noqa
from pe14_sum1b import comb, PS  # noqa

GRID = np.exp(np.linspace(np.log(0.01), np.log(30), 13))
FEATS = ['last', 'first', 'lead', 'size', 'sys', 'len', 'sizeXlen', 'num',
         'sign:M288', 'sign:M346', 'sign:M387', 'sign:M218', 'sign:M297', 'sign:M066']


def sim(cnt0, counted, lens, rho_row, nsur, nr):
    """per-row rho; returns (nsur, N, 12) per-tablet lag agreements (original row order)."""
    order = np.argsort(-lens, kind='stable')
    inv = np.argsort(order)
    cnt0, lens, rr = cnt0[order], lens[order], rho_row[order]
    N, C = cnt0.shape
    T = lens.max()
    nact = np.array([(lens > t).sum() for t in range(T)])
    out = np.zeros((nsur, N, len(LAGS)), np.float32)
    for i in range(nsur):
        cnt = cnt0.copy()
        seq = np.full((N, T + 13), -1, int)
        prev = np.full(N, -1)
        for t in range(T):
            n = nact[t]
            w = cnt[:n].copy()
            pv = prev[:n]
            hp = np.nonzero(pv >= 0)[0]
            w[hp, pv[hp]] *= np.where(counted[pv[hp]] > 0, rr[:n][hp], 1.0)
            tot = w.sum(1)
            z = tot <= 0
            if z.any():
                w[z] = cnt[:n][z]
                tot = w.sum(1)
            u = nr.random_sample(n) * tot
            k = np.minimum((np.cumsum(w, 1) < u[:, None]).sum(1), C - 1)
            cnt[np.arange(n), k] -= 1
            seq[:n, t] = k
            prev[:n] = k
        cm = np.where(seq >= 0, np.where(counted[np.maximum(seq, 0)] > 0, seq, -1), -1)
        for j, p in enumerate(LAGS):
            x, y = cm[:, :-p], cm[:, p:]
            out[i, :, j] = np.count_nonzero((x == y) & (x >= 0), axis=1)
    return out[:, inv]


def per_tab_obs(seqs):
    useful = [s for s in seqs if sum(1 for x in s if x >= 0) >= 2]
    return np.array([lag_agree([s]) for s in useful]), useful


def main(cfg, drop_long=0):
    T = corpus(cfg)
    if drop_long:
        T = sorted(T, key=lambda t: len(t['u']))[:-drop_long]
    F = features(T, with_num=cfg.endswith('_LIN'))
    nr = np.random.RandomState(sum(map(ord, cfg)) + drop_long)
    res = {}
    for k in FEATS:
        if k not in F:
            continue
        O, useful = per_tab_obs(F[k])
        cnt0, counted, lens = _prep(useful)
        G = np.stack([sim(cnt0, counted, lens, np.full(len(lens), g), 12, nr).mean(0)[:, 0] for g in GRID], 1)
        # choose per tablet the rho whose expected lag-1 is closest to observed
        best = np.abs(G - O[:, :1]).argmin(1)
        rho_row = GRID[best]
        S = sim(cnt0, counted, lens, rho_row, 200, nr).sum(1)  # (200, 12)
        o = O.sum(0)
        m = S.mean(0)
        r = {'obs': o.tolist(), 'm': m.tolist(), 'lag1_ratio': float(o[0] / m[0])}
        for p in PS:
            cs = comb(S, m, p)
            co = comb(o, m, p)
            r['comb%d' % p] = float(co)
            r['z%d' % p] = float((co - cs.mean()) / (cs.std() + 1e-12))
            r['p%d' % p] = float((1 + np.sum(cs >= co)) / (1 + len(cs)))
        res[k] = r
        print(cfg, drop_long, '%-10s' % k, 'lag1 O/E %.2f' % r['lag1_ratio'],
              'O/E 2-8', ' '.join('%.2f' % x for x in (o / m)[1:8]),
              '| z2 %.1f z3 %.1f z4 %.1f p2 %.3f' % (r['z2'], r['z3'], r['z4'], r['p2']), flush=True)
    json.dump(res, open(os.path.join(CK, 'c1c_%s_%d.json' % (cfg, drop_long)), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
