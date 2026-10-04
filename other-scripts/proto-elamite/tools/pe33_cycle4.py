"""pe33 cycle 4: the picture predicts the SIZE of what is counted.
Cattle are kept in tens, sheep and goats in hundreds (Ur III: gu4 vs udu counts). If the seal picture says
what the tablet is about, tablets sealed with cattle-only scenes should count smaller herds than tablets
sealed with caprid-only scenes. Statistic: difference in mean log10(1 + largest bare count on the tablet)
(non-capacity lines, N01/N14/N34/N45/N48 sexagesimal-decimal values), CAPRID-only minus BOVID-only.
Null: motif vectors shuffled among seals within volume (seal-block, 5,000). Control: Ur III tablets with
gu4-only vs udu-only entries, subsampled to the PE group sizes (power)."""
import json, sys, os, collections
import numpy as np
from pe33_common import load, perm_seal, CK
VAL = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}


def maxcount(t, T):
    best = 0
    for l in T['lines']:
        nums = l['numerals']
        if not nums or any(c not in VAL for _, c in nums):
            continue
        v = sum(n * VAL[c] for n, c in nums if isinstance(n, int))
        best = max(best, v)
    return best


def stat(Y, mc, ib, ic):
    b = Y[:, ib].astype(bool); c = Y[:, ic].astype(bool)
    bo = b & ~c; co = c & ~b
    if bo.sum() < 2 or co.sum() < 2:
        return np.nan
    return mc[co].mean() - mc[bo].mean()


if __name__ == '__main__':
    scr = sys.argv[1]
    rows, T = load(True)
    corp = {t['id']: t for t in json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'pe_corpus.json')))}
    mc = np.array([np.log10(1 + maxcount(r, corp[r['id']])) for r in rows])
    mot = ['BOVID', 'CAPRID']
    Y = np.array([[m in r['motif'] for m in mot] for r in rows], dtype=np.int8)
    ib, ic = 0, 1
    real = stat(Y, mc, ib, ic)
    rng = np.random.default_rng(4)
    null = np.array([stat(perm_seal(Y, rows, rng), mc, ib, ic) for _ in range(5000)])
    null = null[~np.isnan(null)]
    p = float((null >= real).mean())
    bo = (Y[:, 0] == 1) & (Y[:, 1] == 0); co = (Y[:, 1] == 1) & (Y[:, 0] == 0)
    print('BOVID-only n', bo.sum(), 'seals', len({rows[i]['seal'] for i in np.where(bo)[0]}), 'median max count', np.median(10 ** mc[bo] - 1))
    print('CAPRID-only n', co.sum(), 'seals', len({rows[i]['seal'] for i in np.where(co)[0]}), 'median max count', np.median(10 ** mc[co] - 1))
    print('real diff (log10)', real, 'p', p)
    # all-motif version: mean log max count per motif vs rest
    allm = {}
    for m in ['BOVID', 'CAPRID', 'FELINE', 'ANTHRO', 'MONSTER', 'WATER', 'PREDATION']:
        Ym = np.array([[m in r['motif']] for r in rows], dtype=np.int8)
        f = lambda Yx: mc[Yx[:, 0] == 1].mean() - mc[Yx[:, 0] == 0].mean()
        rv = f(Ym); nv = np.array([f(perm_seal(Ym, rows, rng)) for _ in range(2000)])
        allm[m] = dict(diff=float(rv), p_two=float((np.abs(nv - nv.mean()) >= abs(rv - nv.mean())).mean()))
    print(allm)
    # Ur III control
    C = json.load(open(os.path.join(scr, 'x2', 'corpus_UR3.json')))
    g, u = [], []
    for t in C:
        coms = {e['com'] for e in t['entries'] if e['com']}
        qs = [e['q'] for e in t['entries'] if e['com'] in ('gu4', 'udu') and isinstance(e['q'], (int, float)) and not e['tot']]
        if not qs:
            continue
        if coms == {'gu4'}:
            g.append(np.log10(1 + max(qs)))
        elif coms == {'udu'}:
            u.append(np.log10(1 + max(qs)))
    g, u = np.array(g), np.array(u)
    print('Ur III gu4-only', len(g), 'median', np.median(10 ** g - 1), 'udu-only', len(u), 'median', np.median(10 ** u - 1))
    nb, nc = int(bo.sum()), int(co.sum())
    hits = 0
    for k in range(500):
        r2 = np.random.default_rng(k)
        a = r2.choice(g, nb); b = r2.choice(u, nc)
        d = b.mean() - a.mean()
        pool = np.concatenate([a, b])
        nd = []
        for _ in range(300):
            r2.shuffle(pool)
            nd.append(pool[nb:].mean() - pool[:nb].mean())
        hits += (np.array(nd) >= d).mean() < 0.05
    pw = hits / 500
    print('Ur III power at PE group sizes', pw)
    # planted: multiply CAPRID-only tablets' counts by 5
    mc2 = mc.copy(); mc2[co] = np.log10(1 + 5 * (10 ** mc[co] - 1))
    nl2 = np.array([stat(perm_seal(Y, rows, rng), mc2, ib, ic) for _ in range(2000)])
    p2 = float((nl2[~np.isnan(nl2)] >= stat(Y, mc2, ib, ic)).mean())
    print('planted x5 p', p2)
    json.dump(dict(n_bovid_only=nb, n_caprid_only=nc, real=float(real), p=p, all_motifs=allm,
                   ur3=dict(n_gu4=len(g), n_udu=len(u), med_gu4=float(np.median(10 ** g - 1)), med_udu=float(np.median(10 ** u - 1)), power=pw),
                   planted_x5_p=p2, med_bovid=float(np.median(10 ** mc[bo] - 1)), med_caprid=float(np.median(10 ** mc[co] - 1))),
              open(f'{CK}/cycle4.json', 'w'), indent=1)
