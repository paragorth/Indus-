"""pe51 cycle 2: real PE against its nulls (signs shuffled within lines; signs re-dealt across lines),
and the controls with the models available. Prints the comparison used in loops/pe51_cycle2.txt."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_sum as S
import pe51_lib as L


def summary(R):
    C = R['C']
    n_cls = sum(c != 'SILENT' for c in C.values())
    n_stab = sum(c != 'SILENT' and R['stable'][w] for w, c in C.items())
    ex = {g: np.array([S.mean_excess(R, w, g) for w in C if S.mean_excess(R, w, g) == S.mean_excess(R, w, g)]) for g in S.GROUPS}
    big = {g: int((v >= S.FLOOR).sum()) for g, v in ex.items()}
    tot = {g: float(np.clip(v, 0, None).sum()) for g, v in ex.items()}
    return n_cls, n_stab, big, tot


def spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation


if __name__ == '__main__':
    models = list(range(6))
    out = {}
    for name in ['pe', 'pe_nw', 'pe_nx']:
        R = S.analyse(name, models=models)   # same model indices (same archs, folds, seeds)
        out[name] = R
        n_cls, n_stab, big, tot = summary(R)
        print('%-6s models %s classed %d stable %d; signs with excess >= %.2f: %s; summed positive excess %s' % (
            name, R['models'], n_cls, n_stab, S.FLOOR, big, {k: round(v, 2) for k, v in tot.items()}))
        print('   classed:', {w: c for w, c in R['C'].items() if c != 'SILENT'})
    for g in S.GROUPS:
        for nm in ['pe_nw', 'pe_nx']:
            ws = [w for w in out['pe']['G'] if w in out[nm]['G']]
            a = [S.mean_excess(out['pe'], w, g) for w in ws]
            b = [S.mean_excess(out[nm], w, g) for w in ws]
            ok = [i for i in range(len(ws)) if a[i] == a[i] and b[i] == b[i]]
            print('rank corr real vs %s on %s excess: %.2f (n %d)' % (nm, g, spearman([a[i] for i in ok], [b[i] for i in ok]), len(ok)))
    for w in ['M297', 'M376', 'M032', 'M288', 'M317', 'M036', 'M367']:
        print(w, ' | '.join('%s %s NUM-L %+.3f t%.1f' % (nm, out[nm]['C'].get(w), S.mean_excess(out[nm], w, 'NUM-L'),
                                                          out[nm]['G'][w]['NUM-L']) for nm in out if w in out[nm]['G']))
