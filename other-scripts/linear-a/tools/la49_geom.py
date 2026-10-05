#!/usr/bin/env python3
"""LA-49 cycle 3b: what the reproducible cycle-1 word geometry encodes. Consensus similarity of word-type
states (cycle-1 models); share explained by token kind (logogram vs word), log frequency and next-token
kind profile; clusters and stable neighbours of key words. Controls: shuffled LA, LB, UR, PLANT."""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
from la49_report import load, rsa, clusters
import la49_report as R
R.TAG = 'c1'


def profile(docs):
    """per type: log count, is-logogram, P(next is number), P(prev is number), P(first in doc)."""
    c = collections.Counter(); nn = collections.Counter(); pn = collections.Counter(); fi = collections.Counter()
    for d in docs:
        t = d['toks']
        for i, x in enumerate(t):
            if x[0] != 'T':
                continue
            w = x[1]; c[w] += 1
            nn[w] += i + 1 < len(t) and t[i + 1][0] == 'N'
            pn[w] += i > 0 and t[i - 1][0] == 'N'
            fi[w] += i == 0
    return {w: [np.log(c[w]), float(w.startswith('L:')), nn[w] / c[w], pn[w] / c[w], fi[w] / c[w]] for w in c}


for corp in ([] if (os.environ.get("LA49_GEOM2") or __name__ != "__main__") else sys.argv[1:]):
    ms = load(corp)
    keys, S, r = rsa(ms)
    P = profile(L.corpus(corp))
    X = np.array([P[k] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    y = S[iu]
    feats = [np.abs(X[:, j][:, None] - X[:, j][None, :])[iu] for j in range(X.shape[1])]
    A = np.column_stack([np.ones_like(y)] + feats)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.var(y - A @ beta) / np.var(y)
    # residual RSA: per-model residual similarity after regressing profile distances, cross-seed agreement
    res = []
    for m in ms:
        Xm = np.array([m['vec'][k] for k in keys]); Xm = Xm - Xm.mean(0)
        Xm /= np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-9
        s = (Xm @ Xm.T)[iu]
        b, *_ = np.linalg.lstsq(A, s, rcond=None)
        res.append(s - A @ b)
    rr = [np.corrcoef(res[i], res[j])[0, 1] for i in range(len(res)) for j in range(i + 1, len(res))]
    print('==', corp, 'models', len(ms), 'types', len(keys), 'RSA %.3f' % r, 'R2 by profile %.3f' % r2,
          'residual RSA %.3f' % np.mean(rr))
    if corp == 'LA':
        cl = clusters(keys, S, 6)
        grp = collections.defaultdict(list)
        for w in keys:
            grp[cl[w]].append(w)
        for g, ws in sorted(grp.items(), key=lambda x: -len(x[1])):
            ws = sorted(ws, key=lambda w: -P[w][0])
            print('  cluster', g, len(ws), 'logo %.2f' % np.mean([P[w][1] for w in ws]), 'nextNum %.2f' % np.mean([P[w][2] for w in ws]), ws[:16])
        # residual consensus neighbours
        Rm = np.zeros((len(keys), len(keys)))
        Rm[iu] = np.mean(res, 0); Rm = Rm + Rm.T
        for a in ('KU-RO', 'KI-RO', 'SA-RA₂', 'A-DU', 'KA-PA', 'DA-ME', 'KU-NI-SU', 'L:GRA', 'L:VIN', 'L:OLE'):
            if a in keys:
                i = keys.index(a); o = np.argsort(-Rm[i])
                print('  nn', a, [(keys[j], round(float(Rm[i, j]), 2)) for j in o[:6]])


def cooc(docs, keys):
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(keys), len(keys)))
    n = np.zeros(len(keys))
    for d in docs:
        s = sorted({idx[x[1]] for x in d['toks'] if x[0] == 'T' and x[1] in idx})
        for a in s:
            n[a] += 1
            for b in s:
                M[a, b] += 1
    return M / (n[:, None] + n[None, :] - M + 1e-9)


def cohesion(Rm, keys, group, rng, nperm=2000):
    g = [keys.index(w) for w in group if w in keys]
    if len(g) < 2:
        return None
    obs = np.mean([Rm[a, b] for i, a in enumerate(g) for b in g[i + 1:]])
    null = []
    for _ in range(nperm):
        h = rng.choice(len(keys), len(g), replace=False)
        null.append(np.mean([Rm[a, b] for i, a in enumerate(h) for b in h[i + 1:]]))
    return round(float(obs), 3), round(float(np.mean(np.array(null) >= obs)), 4), len(g)


if __name__ == '__main__' and os.environ.get('LA49_GEOM2'):
    import la45_common as C45
    rng = np.random.default_rng(0)
    print('--- part 2: co-occurrence and planted / truth cohesion')
    for corp in sys.argv[1:]:
        ms = load(corp)
        keys, S, r = rsa(ms)
        docs = L.corpus(corp)
        P = profile(docs)
        X = np.array([P[k] for k in keys])
        iu = np.triu_indices(len(keys), 1)
        Cm = cooc(docs, keys)
        feats = [np.abs(X[:, j][:, None] - X[:, j][None, :])[iu] for j in range(X.shape[1])] + [Cm[iu]]
        A = np.column_stack([np.ones(len(iu[0]))] + feats)
        res = []
        for m in ms:
            Xm = np.array([m['vec'][k] for k in keys]); Xm = Xm - Xm.mean(0)
            Xm /= np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-9
            s = (Xm @ Xm.T)[iu]
            b, *_ = np.linalg.lstsq(A, s, rcond=None)
            res.append(s - A @ b)
        rr = [np.corrcoef(res[i], res[j])[0, 1] for i in range(len(res)) for j in range(i + 1, len(res))]
        Rm = np.zeros((len(keys), len(keys))); Rm[iu] = np.mean(res, 0); Rm = Rm + Rm.T
        Sm = S.copy()
        out = {'residual_RSA_after_cooc': round(float(np.mean(rr)), 3)}
        if corp == 'PLANT':
            _, PL = L.planted()
            out['TOTAL_raw'] = cohesion(Sm, keys, PL['TOTAL'], rng); out['TOTAL_resid'] = cohesion(Rm, keys, PL['TOTAL'], rng)
            out['BIND_raw'] = cohesion(Sm, keys, PL['BIND'], rng); out['BIND_resid'] = cohesion(Rm, keys, PL['BIND'], rng)
        if corp in ('LB', 'UR'):
            tr = C45.lb_truth(docs) if corp == 'LB' else C45.ur_truth(docs)
            for role in sorted(set(tr.values())):
                grp = [w for w in keys if tr.get(w) == role]
                if len(grp) >= 3:
                    out[role + '_raw'] = cohesion(Sm, keys, grp, rng); out[role + '_resid'] = cohesion(Rm, keys, grp, rng)
        if corp == 'LA':
            groups = {'la45_commodity': ['L:VIN', 'L:OLE', 'L:OLIV', 'L:VIR', 'L:GRA', 'NI', 'KI', 'DI', 'RE', 'TI', 'MA'],
                      'la45_heading': ['*301', 'KA', 'KU', 'SI', 'RO', 'ZE'],
                      'totals': ['KU-RO', 'KI-RO', 'PO-TO-KU-RO']}
            for g, ws in groups.items():
                out[g + '_raw'] = cohesion(Sm, keys, ws, rng); out[g + '_resid'] = cohesion(Rm, keys, ws, rng)
            for a in ('KU-RO', 'KI-RO', 'SA-RA₂', 'KU-NI-SU', 'L:GRA'):
                if a in keys:
                    i = keys.index(a); o = np.argsort(-Rm[i])
                    out['nn_' + a] = [(keys[j], round(float(Rm[i, j]), 2)) for j in o[:6]]
        print('==', corp, out)
