"""v29 report helper: summarise a cycle-1 style result pickle."""
import sys, os, pickle
import numpy as np
import v29_lib as L


def load(name, cyc='c1'):
    return pickle.load(open(os.path.join(L.CK, f'{cyc}_{name}.pkl'), 'rb'))


def summary(name, cyc='c1', top=4, stats=('tier1', 'tier2', 'tier3', 'raw3', 'junc')):
    r = load(name, cyc)
    z, ex = L.zscores(r)
    thr = np.nanmax(L.maxnull(r), 0)
    kinds = np.array([f['kind'] for f in r['feats']])
    st = np.isin(kinds, ['stroke', 'rstroke'])
    lines = [f"== {name}: {len(kinds)} features, {r['ntok']} glyphs, {r['nwords']} words"]
    for s in stats:
        i = L.STATS.index(s)
        zz = z[:, i]
        ok = np.isfinite(zz)
        n_st = int(((zz > thr[i]) & st & ok).sum()); n_cl = int(((zz > thr[i]) & ~st & ok).sum())
        n_neg = int(((zz < -thr[i]) & ok).sum())
        o = np.argsort(-np.nan_to_num(zz, nan=-99))[:top]
        lines.append(f"  {s}: maxnull thr z {thr[i]:.1f}; above: stroke {n_st}/{int((st&ok).sum())}, class {n_cl}/{int((~st&ok).sum())}; below -thr {n_neg}; "
                     f"median z stroke {np.nanmedian(zz[st]):+.2f} class {np.nanmedian(zz[~st]):+.2f}")
        for k in o:
            lines.append(f"     z {zz[k]:+6.1f} phi {r['obs'][k, i]:+.3f} (null {r['obs'][k, i]-ex[k, i]:+.3f}) dir {ex[k, 7]:+.3f}  {L.fname(r['feats'][k], r['alph'])}")
    return '\n'.join(lines), r, z, ex, thr


if __name__ == '__main__':
    for n in sys.argv[1:]:
        print(summary(n)[0])
