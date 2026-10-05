#!/usr/bin/env python3
"""LA-52 report: retention of excess structure along iterated-learning chains.
usage: la52_report.py TAG [suffix]   (suffix e.g. '_ngram' for single-learner chains)"""
import sys, os, json, glob, math, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la52_common as C

TAG = sys.argv[1]
SUF = sys.argv[2] if len(sys.argv) > 2 else ''
CK = C.CK


def load(cond, suf=SUF):
    fs = sorted(glob.glob(os.path.join(CK, f'{TAG}_{cond}{suf}_w*.npy')))
    fs = [f for f in fs if '.tmp' not in f]
    if not fs:
        return None, None
    arrs = [np.load(f) for f in fs]
    meta = json.load(open(fs[0].replace('.npy', '_meta.json')))
    return np.concatenate(arrs, 0), meta


def feats(base):
    m = json.load(open(os.path.join(CK, f'{TAG}_{base}_feats.json')))
    return [C.unkey(s) for s in m['feats']], np.array(m['c0'], float), np.array(m['b0'], float)


def retention(base, real, shuf, emin=3.0, boot=200, seed=0):
    F, c0, b0 = feats(base)
    e0 = c0 - b0
    R, meta = load(real)
    if R is None:
        return None
    eR = R[:, :, 0, :] - R[:, :, 1, :]          # chains x gens x F
    S, _ = load(shuf) if shuf else (None, None)
    if S is not None:
        Fs, _, _ = feats(base if shuf != 'LAS' or base == 'LA' else 'LA')
        if Fs != F:
            pos = {f: i for i, f in enumerate(Fs)}
            eS_full = S[:, :, 0, :] - S[:, :, 1, :]
            eS = np.zeros((eS_full.shape[0], eS_full.shape[1], len(F)), np.float32)
            for j, f in enumerate(F):
                if f in pos:
                    eS[:, :, j] = eS_full[:, :, pos[f]]
        else:
            eS = S[:, :, 0, :] - S[:, :, 1, :]
    else:
        eS = np.zeros_like(eR[:1])
    meas = meta['meas']
    mR = eR.mean(0); mS = eS.mean(0)
    den = e0 - mS[0]
    ok = (e0 >= emin) & (den > 1)
    d = {}
    for gi, g in enumerate(meas):
        d[g] = np.where(ok, (mR[gi] - mS[gi]) / np.where(ok, den, 1), np.nan)
    rng = np.random.default_rng(seed)
    G = meas[-1]
    bs = []
    for _ in range(boot):
        i = rng.integers(0, eR.shape[0], eR.shape[0])
        j = rng.integers(0, eS.shape[0], eS.shape[0])
        bs.append(np.where(ok, (eR[i, -1].mean(0) - eS[j, -1].mean(0)) / np.where(ok, den, 1), np.nan))
    bs = np.array(bs)
    auc = np.nanmean(np.array([d[g] for g in meas if g > 0]), 0)
    bsa = []
    for _ in range(boot):
        i = rng.integers(0, eR.shape[0], eR.shape[0])
        j = rng.integers(0, eS.shape[0], eS.shape[0])
        bsa.append(np.where(ok, ((eR[i, 1:].mean(0) - eS[j, 1:].mean(0)).mean(0)) / np.where(ok, den, 1), np.nan))
    d['auc'] = auc
    return dict(F=F, c0=c0, e0=e0, ok=ok, d=d, G=G, se_auc=np.nanstd(np.array(bsa), 0), se=np.nanstd(bs, 0), mS=mS, mR=mR, meas=meas,
                names=meta['names'], n=eR.shape[0], nS=eS.shape[0])


def fam_table(r):
    out = []
    fam = collections.defaultdict(list)
    for j, f in enumerate(r['F']):
        if r['ok'][j]:
            fam[f[0]].append(j)
    for k in sorted(fam, key=lambda k: np.nanmedian(r['d']['auc'][fam[k]])):
        js = fam[k]
        out.append((k, len(js), float(np.nanmedian(r['d'][1][js])), float(np.nanmedian(r['d'][r['G']][js])), float(np.nanmedian(r['d']['auc'][js]))))
    return out


def pct(r, j):
    v = r['d']['auc']
    okv = v[r['ok']]
    return float((okv < v[j]).mean())


def adjusted(r):
    """Learnability-adjusted anchoring: d residual within family x size decile (lower d = more
    anchored). Returns A (positive = decays faster than its family/size peers)."""
    v = r['d']['auc']
    A = np.full(len(v), np.nan)
    groups = collections.defaultdict(list)
    for j, f in enumerate(r['F']):
        if r['ok'][j]:
            groups[(f[0], min(4, int(math.log2(max(1, r['c0'][j])))))].append(j)
    for g, js in groups.items():
        if len(js) < 5:
            continue
        m = np.median(v[js]); s = 1.4826 * np.median(np.abs(v[js] - m)) + 1e-6
        for j in js:
            A[j] = -(v[j] - m) / s
    return A


def word_scores(r, A, minf=3, nperm=2000, seed=1):
    by = collections.defaultdict(list)
    for j, f in enumerate(r['F']):
        if not np.isnan(A[j]):
            for x in f[1:]:
                if x not in ('SUM', 'FIRST', 'LAST') and not x.isdigit() and x != 'f':
                    by[x].append(j)
    vals = A[~np.isnan(A)]
    rng = np.random.default_rng(seed)
    out = []
    for w, js in by.items():
        if len(js) < minf:
            continue
        m = float(np.mean(A[js]))
        null = rng.choice(vals, (nperm, len(js))).mean(1)
        p = float(((null >= m).sum() + 1) / (nperm + 1))
        out.append((w, len(js), m, p))
    out.sort(key=lambda x: -x[2])
    return out


def show(name, r, extra=()):
    print(f'== {name}: {r["n"]} chains (shuffled start {r["nS"]}), G={r["G"]}, features with excess>=3: {int(r["ok"].sum())}')
    print('   learners per generation:', dict(collections.Counter(x for nm in r['names'] for x in nm)))
    print('   family  n  median d(g1)  d(G)  AUC (mean d over generations)')
    for k, n, d1, dG, da in fam_table(r):
        print(f'   {k:4s} {n:4d}  {d1:6.2f}  {dG:6.2f}  {da:6.2f}  ({C46_DESC.get(k, "")})')
    for f in extra:
        if f in r['F']:
            j = r['F'].index(f)
            print(f'   {"/".join(f):40s} c0 {r["c0"][j]:.0f} e0 {r["e0"][j]:.1f} d1 {r["d"][1][j]:.2f} dG {r["d"][r["G"]][j]:.2f} AUC {r["d"]["auc"][j]:.2f} +- {r["se_auc"][j]:.2f} pct {pct(r, j):.2f}' if r['ok'][j] else f'   {"/".join(f)}: excess < 3')


C46_DESC = dict(C.C46.FAM_DESC, LO='logogram before another logogram (order)', WR='word -> logogram 3+ tokens later')

if __name__ == '__main__':
    res = {}
    for base, real, shuf in [('LA', 'LA', 'LAS'), ('LB', 'LB', 'LBS'), ('UR', 'UR', 'URS'), ('LAP', 'LAP', 'LAS'), ('LA', 'LAO', 'LAS')]:
        try:
            r = retention(base, real, shuf)
        except FileNotFoundError:
            r = None
        if r is None:
            continue
        res[real] = r
    if 'LAP' in res:
        show('LA + planted rules', res['LAP'], list(C.PLANT_FEATS))
    if 'LA' in res:
        r = res['LA']
        tot = [f for f in r['F'] if f[0] == 'TOT']
        sara = [f for f in r['F'] if 'SA-RA₂' in f]
        show('Linear A', r, tot + sara[:12])
        A = adjusted(r)
        json.dump({'F': [C.fkey(f) for f in r['F']], 'A': [None if np.isnan(x) else float(x) for x in A],
                   'd': [None if np.isnan(x) else float(x) for x in r['d']['auc']]},
                  open(os.path.join(CK, f'{TAG}{SUF}_LA_scores.json'), 'w'))
        print('   most anchored features (adjusted, top 25):')
        order = [j for j in np.argsort(-np.nan_to_num(A, nan=-99)) if not np.isnan(A[j])][:25]
        for j in order:
            print(f'     {"/".join(r["F"][j]):42s} c0 {r["c0"][j]:.0f} e0 {r["e0"][j]:.1f} AUC {r["d"]["auc"][j]:.2f} A {A[j]:.2f}')
        print('   most persistent features (adjusted, bottom 10):')
        for j in [j for j in np.argsort(np.nan_to_num(A, nan=99)) if not np.isnan(A[j])][:10]:
            print(f'     {"/".join(r["F"][j]):42s} c0 {r["c0"][j]:.0f} e0 {r["e0"][j]:.1f} AUC {r["d"]["auc"][j]:.2f} A {A[j]:.2f}')
        ws = word_scores(r, A)
        print('   words/signs ranked by anchoring (>=3 features): top 15 / bottom 5')
        for w in ws[:15] + [('...', 0, 0, 1)] + ws[-5:]:
            print(f'     {w[0]:14s} nf {w[1]:3d} meanA {w[2]:6.2f} p {w[3]:.4f}')
    if 'LAO' in res and 'LA' in res:
        tg = set(C.unkey(s) for s in json.load(open(os.path.join(CK, f'{TAG}_LAO_targets.json'))))
        r, r0 = res['LAO'], res['LA']
        js = [j for j, f in enumerate(r['F']) if f in tg and r['ok'][j]]
        ns = [j for j, f in enumerate(r['F']) if f not in tg and r['ok'][j]]
        G = r['G']
        print(f'== Arbitrary-preservation objective: {r["n"]} chains')
        for g in (1, G, 'auc'):
            print(f'   d[{g}] targets (n {len(js)}): median {np.nanmedian(r["d"][g][js]):.2f} (plain chains {np.nanmedian(r0["d"][g][js]):.2f}); others (n {len(ns)}): {np.nanmedian(r["d"][g][ns]):.2f} (plain {np.nanmedian(r0["d"][g][ns]):.2f})')
        A0 = adjusted(r0)
        print(f'   mean adjusted anchoring of targets: plain {np.nanmean(A0[js]):.2f}, with objective {np.nanmean(adjusted(r)[js]):.2f}')
    if 'LB' in res:
        r = res['LB']
        tot = [f for f in r['F'] if f[0] == 'TOT'][:8]
        show('Linear B (KN+PY at LA size)', r, tot)
        A = adjusted(r)
        ws = word_scores(r, A)
        print('   LB words ranked by anchoring: top 12')
        for w in ws[:12]:
            print(f'     {w[0]:16s} nf {w[1]:3d} meanA {w[2]:6.2f} p {w[3]:.4f}')
    if 'UR' in res:
        r = res['UR']
        tot = [f for f in r['F'] if f[0] == 'TOT'][:8]
        show('Ur III at LA size', r, tot)
        A = adjusted(r)
        ws = word_scores(r, A)
        print('   Ur III words ranked by anchoring: top 12')
        for w in ws[:12]:
            print(f'     {w[0]:16s} nf {w[1]:3d} meanA {w[2]:6.2f} p {w[3]:.4f}')
