"""pe29 cycle 1: do the independent weak evidences agree with each other (a necessary sign of shared sessions)?
For every pair of evidences (pool, hand, clay, mus, size): Pearson over tablet pairs measurable in both.
Null: one matrix relabelled inside format strata (museum prefix x numeral system x line-count bin; clay inside photo batch),
200 draws.  Ur III (Amar-Suen 5, Drehem, true day known): each evidence's AUC for same-day pairs.  Planted: 50 sessions of
4 PE tablets with weak shared pool, hand tic, museum run, size and colour pull; recovery AUC per evidence.
"""
import sys, json, collections, copy
import numpy as np
from pe29_common import *

rng = np.random.default_rng(29)
NP = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200


def nlbin(n):
    return 0 if n <= 3 else 1 if n <= 7 else 2 if n <= 15 else 3


def build(tabs, clay=None):
    S = {'pool': sim_pool(tabs), 'hand': sim_hand(tabs, ur='date' in tabs[0]), 'mus': sim_mus(tabs), 'size': sim_size(tabs)}
    if clay is not None:
        S['clay'] = sim_clay(*clay)
    return S


def strata(tabs, B=None):
    st = [f"{t['pre']}|{t.get('sys','')}|{nlbin(t['nl'])}" for t in tabs]
    return st, (B if B is not None else st)


def cross(S, st, stc, tag):
    rows = []
    iu = np.triu_indices(len(st), 1)
    evs = list(S)
    for a in range(len(evs)):
        for b in range(a + 1, len(evs)):
            ea, eb = evs[a], evs[b]
            x = S[ea][iu]; y = S[eb][iu]; ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < 200:
                continue
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            g = stc if eb == 'clay' else st
            null = []
            for _ in range(NP):
                P = permute(S[eb], g, rng)[iu]
                ok2 = ~np.isnan(x) & ~np.isnan(P)
                null.append(np.corrcoef(x[ok2], P[ok2])[0, 1])
            null = np.array(null)
            z = (r - null.mean()) / (null.std() + 1e-12)
            p = (1 + (null >= r).sum()) / (NP + 1)
            rows.append(dict(set=tag, a=ea, b=eb, r=r, null_mu=float(null.mean()), null_q95=float(np.quantile(null, .95)), z=float(z), p=float(p), npairs=int(ok.sum())))
            print(tag, ea, eb, 'r %.4f null %.4f q95 %.4f z %.1f p %.3f n %d' % (r, null.mean(), np.quantile(null, .95), z, p, ok.sum()), flush=True)
    return rows


def plant(tabs, V, B, ngrp=50, gsz=4):
    T = copy.deepcopy(tabs); V = V.copy()
    idx = rng.permutation(len(T))[:ngrp * gsz]
    lab = np.full(len(T), -1)
    for g in range(ngrp):
        mem = idx[g * gsz:(g + 1) * gsz]; lab[mem] = g
        var = 'abc'[rng.integers(3)]
        anchor = rng.integers(30000, 90000)
        pre = T[mem[0]]['pre'] or 'Sb'
        dims = {k: np.nanmean([T[i][k] or np.nan for i in mem]) for k in ('h', 'w', 't')}
        cl = np.nanmean(V[mem], 0) if not np.isnan(V[mem]).all() else None
        for i in mem:
            t = T[i]
            for s in (f'MPL{g}a', f'MPL{g}b'):
                if rng.random() < 0.6:
                    t['signs'].append(s); t['bases'].append(s)
            if rng.random() < 0.7:
                t['signs'].append('M999~' + var); t['bases'].append('M999')
            if rng.random() < 0.7:
                t['pre'] = pre; t['no'] = int(anchor + rng.integers(0, 40))
            for k in ('h', 'w', 't'):
                if t[k] and not np.isnan(dims[k]):
                    t[k] = 0.5 * t[k] + 0.5 * dims[k]
            if cl is not None and not np.isnan(V[i]).any():
                V[i] = 0.5 * V[i] + 0.5 * cl
    for i in range(len(T)):  # background users of the tic sign, random variant
        if lab[i] < 0 and rng.random() < 0.15:
            T[i]['signs'].append('M999~' + 'abc'[rng.integers(3)]); T[i]['bases'].append('M999')
    return T, V, lab


if __name__ == '__main__':
    out = {}
    T = pe_tablets(); V, B = pe_clay(T)
    stB = [b if b is not None else f'nob{i}' for i, b in enumerate(B)]
    S = build(T, (V, B))
    for k, M in S.items():
        print('PE', k, 'measurable pairs', int((~np.isnan(M[np.triu_indices(len(T), 1)])).sum()))
    st, _ = strata(T)
    out['PE'] = cross(S, st, stB, 'PE')
    # Ur III
    U = ur3_tablets()
    SU = build(U)
    for t in U:
        t['sys'] = ''
    stU, _ = strata(U)
    out['UR3_auc'] = {}
    for k, M in SU.items():
        iu = np.triu_indices(len(U), 1); ok = ~np.isnan(M[iu])
        lab = np.array([t['date'] for t in U]); y = (lab[:, None] == lab[None, :])[iu]
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(y[ok], M[iu][ok]) if y[ok].any() else float('nan')
        out['UR3_auc'][k] = (float(a), int(y[ok].sum()))
        print('UR3 same-day AUC', k, '%.3f' % a, 'true pairs measurable', int(y[ok].sum()), flush=True)
    out['UR3'] = cross(SU, stU, stU, 'UR3')
    # planted
    TP, VP, lab = plant(T, V, B)
    SP = build(TP, (VP, B))
    out['PLANT_auc'] = {}
    iu = np.triu_indices(len(TP), 1)
    y = ((lab[:, None] == lab[None, :]) & (lab[:, None] >= 0))[iu]
    from sklearn.metrics import roc_auc_score
    for k, M in SP.items():
        ok = ~np.isnan(M[iu])
        a = roc_auc_score(y[ok], M[iu][ok]) if y[ok].any() else float('nan')
        out['PLANT_auc'][k] = (float(a), int(y[ok].sum()))
        print('PLANT AUC', k, '%.3f' % a, int(y[ok].sum()), flush=True)
    stP, _ = strata(TP)
    out['PLANT'] = cross(SP, stP, stB, 'PLANT')
    np.save(os.path.join(CK, 'plant_lab.npy'), lab)
    json.dump(out, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1)
