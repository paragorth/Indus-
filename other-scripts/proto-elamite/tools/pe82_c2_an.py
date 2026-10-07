"""pe82 cycle 2 analysis: select on held-out half H1, re-test on half H2 (and the reverse), vs shuffled-novelty null."""
import json, sys, os, collections
import numpy as np
import pe82_common as pc


def analyse(fn, top=50):
    R = json.load(open(fn))
    info = [r[1] for r in R if r[0] == 'info']
    out = []
    for si in sorted({r[0] for r in R if r[0] != 'info'}):
        H = [r for r in R if r[0] == si]
        base = [r for r in H if r[1] == ['elen', 'epos'] and r[2] is None]
        if not base:
            base = [r for r in H if sorted(r[1]) == ['elen', 'epos'] and r[2] is None]
        b = base[0]
        def gains(get):
            g1 = np.array([get(r)[0] - get(b)[0] for r in H]); g2 = np.array([get(r)[1] - get(b)[1] for r in H])
            return g1, g2
        real = gains(lambda r: (r[4], r[5]))
        nulls = [gains(lambda r, k=k: tuple(r[7][k])) for k in range(len(b[7]))]
        def sel(g1, g2):
            ok = ~np.isnan(g1) & ~np.isnan(g2)
            g1 = np.where(ok, g1, -9); g2 = np.where(ok, g2, -9)
            a = np.argsort(-g1)[:top]; c = np.argsort(-g2)[:top]
            return float(np.mean(g2[a])), float(np.mean(g1[c])), a, c
        rs = sel(*real)
        ns = [sel(*n)[:2] for n in nulls]
        best = [(H[i][1], H[i][2], round(real[0][i], 3), round(real[1][i], 3)) for i in rs[2][:8]]
        feat_freq = collections.Counter(f for i in list(rs[2]) + list(rs[3]) for f in H[i][1])
        out.append(dict(si=si, base_auc=(b[4], b[5]), base_auc_null=[tuple(x) for x in b[7]],
                        real_retest=(round(rs[0], 4), round(rs[1], 4)),
                        null_retest=[(round(x, 4), round(y, 4)) for x, y in ns],
                        best=best, feat_freq=feat_freq.most_common(), n=len(H)))
    return info, out


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        info, out = analyse(fn)
        print(os.path.basename(fn), info)
        for o in out:
            print(' split', o['si'], 'base AUC', o['base_auc'], 'null base', o['base_auc_null'])
            print('   top-50 by H1 -> mean gain on H2, and reverse: real', o['real_retest'], 'null', o['null_retest'])
            print('   feats among selected', o['feat_freq'][:8])
            print('   best', o['best'][:5])
