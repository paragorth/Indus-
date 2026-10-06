#!/usr/bin/env python3
"""LA-69 cycle 2: thousands of random fingerprint definitions, leave-one-contrast-out, label-permuted
and random-sign nulls; placement of Linear A and of a lines-shuffled Linear A."""
import os, sys, pickle, random, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C
from la69_c1b import resid, fit_len, CAL

R = pickle.load(open(os.path.join(C.CK, 'c1_feats.pkl'), 'rb'))
F = C.FEATS
NF = len(F)
SG = np.array([C.SIGN[k] for k in F], float)
NFP = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def la_line_shuffle(rng):
    la = C.la_docs()
    lines = [ln for d in la for ln in C.lines_of(d)]
    out = []
    for d in la:
        k = len(C.lines_of(d))
        toks = []
        for i in range(k):
            if i:
                toks.append(('L',))
            toks.extend(lines[rng.randrange(len(lines))])
        nd = dict(d, toks=toks, id='SH:' + d['id'])
        if C.ok(nd):
            out.append(nd)
    return out


def get_rows():
    p = os.path.join(C.CK, 'c2_shuffled.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    rng = random.Random(692)
    la = C.la_docs()
    extra = []
    for rep in range(3):
        sh = la_line_shuffle(rng)
        fs = C.score_docs(sh, sh, rng, M=min(300, len(sh) - 1))
        extra += [('LA', 'LA_SHUF%d' % rep, {'id': d['id'], 'nT': sum(1 for x in d['toks'] if x[0] == 'T')}, f)
                  for d, f in zip(sh, fs)]
    pickle.dump(extra, open(p, 'wb'))
    return extra


def aucs(S, y):
    """S: (nfp, n) scores; y bool (n). AUC of y=True (FINAL) over y=False (NOTE) per fingerprint."""
    r = np.apply_along_axis(rankdata, 1, S)
    n1 = y.sum(); n0 = (~y).sum()
    return (r[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    rows = R + get_rows()
    B = fit_len([r for r in R if r[0] in CAL and r[1] in ('NOTE', 'FINAL')])
    rng = np.random.default_rng(6902)
    out = {}
    for mode in ('res', 'raw'):
        X = resid(rows, B) if mode == 'res' else np.array([[r[3][k] for k in F] for r in rows], float)
        cal = np.array([r[0] in CAL and r[1] in ('NOTE', 'FINAL') for r in rows])
        mu = np.nanmean(X[cal], 0); sd = np.nanstd(X[cal], 0) + 1e-9
        Z = np.nan_to_num((X - mu) / sd) * SG          # a priori sign: higher = reader-independent
        # random fingerprints: subset 2-6 features, Dirichlet weights; null family with random signs
        W = np.zeros((NFP, NF)); Wr = np.zeros((NFP, NF))
        for i in range(NFP):
            k = rng.integers(2, 7)
            idx = rng.choice(NF, k, replace=False)
            w = rng.dirichlet(np.ones(k))
            W[i, idx] = w
            Wr[i, idx] = w * rng.choice([-1, 1], k)
        S = Z @ W.T            # (n, nfp)
        Sr = Z @ Wr.T
        cid = np.array(['%s/%s' % (r[0], r[1]) for r in rows])
        contrasts = {}
        for c in CAL:
            contrasts[c] = ('%s/FINAL' % c, '%s/NOTE' % c)
            contrasts['PL:' + c] = ('%s/FINAL' % c, '%s/PLANT' % c)
        contrasts['PL:LA'] = ('LA/LA', 'LA/LA_PLANT')

        def evalA(Sm, perm=False):
            A = {}
            for name, (a, b) in contrasts.items():
                m = (cid == a) | (cid == b)
                y = (cid[m] == a)
                if perm and not name.startswith('PL:'):
                    y = rng.permutation(y)
                A[name] = aucs(Sm[m].T, y)
            return A
        A = evalA(S)
        Ar = evalA(Sr)
        TH = 0.55
        real = CAL
        plant = [k for k in contrasts if k.startswith('PL:')]

        def loco(A):
            res = {}
            for h in real:
                tr = [c for c in real if c != h]
                sel = np.all([A[c] > TH for c in tr + plant], axis=0)
                res[h] = (int(sel.sum()), float(np.mean(A[h][sel] > TH)) if sel.sum() else np.nan,
                          float(np.median(A[h][sel])) if sel.sum() else np.nan)
            return res
        L = loco(A); Lr = loco(Ar)
        Lp = [loco(evalA(S, perm=True)) for _ in range(NPERM)]
        allpass = np.all([A[c] > TH for c in real + plant], axis=0)
        ctrl = np.all([A[c] > TH for c in ['LB_shape', 'LB_draft', 'UR3'] + plant], axis=0)
        print('\n=== mode', mode, 'NFP', NFP)
        for name in contrasts:
            print('  %-12s median AUC %.3f  frac>%.2f %.3f   (random-sign median %.3f)' % (
                name, np.median(A[name]), TH, np.mean(A[name] > TH), np.median(Ar[name])))
        for h in real:
            pn = [x[h][0] for x in Lp]; ph = [x[h][1] for x in Lp if x[h][0] > 0]
            print('  LOCO held-out %-9s: selected %5d, held-out pass %.3f, median AUC %.3f | random-sign: %d, %.3f | '
                  'label-perm: selected median %d (max %d), held-out pass median %.3f' % (
                      h, L[h][0], L[h][1], L[h][2], Lr[h][0], Lr[h][1], np.median(pn), np.max(pn),
                      np.median(ph) if ph else np.nan))
        print('  pass ALL 4 real + 5 planted: %d / %d (random-sign %d)' % (
            allpass.sum(), NFP, np.all([Ar[c] > TH for c in real + plant], axis=0).sum()))
        print('  pass user controls (LB_shape, LB_draft, UR3 + planted): %d' % ctrl.sum())
        # placement
        sets = sorted(set(cid))
        M = {s: S[cid == s].mean(0) for s in sets}
        for label, sel in [('controls-pass', ctrl), ('all-pass', allpass)]:
            if sel.sum() == 0:
                print('  placement (%s): no fingerprint' % label)
                continue
            print('  placement (%s, n=%d): median set score (higher = reader-independent)' % (label, sel.sum()))
            order = sorted(sets, key=lambda s: np.median(M[s][sel]))
            for s in order:
                print('     %-18s %7.3f' % (s, np.median(M[s][sel])))
            notes = ['%s/NOTE' % c for c in CAL]; fins = ['%s/FINAL' % c for c in CAL]
            la = M['LA/LA'][sel]
            below_allF = np.mean(np.all([la < M[f][sel] for f in fins], 0))
            below_allN = np.mean(np.all([la < M[n][sel] for n in notes], 0))
            pos = np.median([np.median((la - M[n][sel]) / (M[f][sel] - M[n][sel] + 1e-12)) for n, f in zip(notes, fins)])
            shuf = np.mean([M['LA/LA_SHUF%d' % i][sel] for i in range(3)], 0)
            print('     LA below every FINAL: %.3f; below every NOTE: %.3f; LA position (0 = NOTE, 1 = FINAL), '
                  'median over corpora %.2f; LA > its line-shuffled self in %.3f' % (below_allF, below_allN, pos,
                                                                                     np.mean(la > shuf)))
            out[(mode, label)] = {s: float(np.median(M[s][sel])) for s in sets}
        out[(mode, 'loco')] = (L, Lr)
        # feature usage among control-passers
        if ctrl.sum():
            use = (W[ctrl] > 0).mean(0) / (W > 0).mean(0)
            print('  feature enrichment among control-passers:', ' '.join('%s %.2f' % (k, u) for k, u in zip(F, use)))
        np.save(os.path.join(C.CK, 'c2_W_%s.npy' % mode), W)
        np.save(os.path.join(C.CK, 'c2_ctrl_%s.npy' % mode), ctrl)
    pickle.dump(out, open(os.path.join(C.CK, 'c2_out.pkl'), 'wb'))


if __name__ == '__main__':
    main()
