"""pe19 scoring: frozen orders vs the hidden chronology.

usage: python3 pe19_score.py <prefix> [out.json]
Scores every data/pe19_ckpt/frozen_<prefix>*.json, plus baselines built here:
  ELEN  order by mean entry length only (the orientation rule alone)
  LINES order by number of lines only (tablet size)
  NUMSYS CA on numeral codes only (number system ~ commodity / genre)
  SITE  Susa first or last, random within (site alone)
and the NULLSHUF orders (trait columns shuffled before CA).
Permutation p: share of 4,000 random orders of the same tablets with stat >= observed.
"""
import glob, json, os, sys
import numpy as np
from pe19_common import *


def pvals(order, H, obs, reps=4000):
    nul = perm_null(order, H, reps=reps, seed=1)
    out = {}
    for k in ('rho_susa', 'auc_early', 'auc_malyan', 'comb'):
        v = np.array([d[k] for d in nul], float)
        v = v[~np.isnan(v)]
        out['p_' + k] = float((1 + (v >= obs[k]).sum()) / (1 + len(v))) if not np.isnan(obs[k]) else None
    return out


def baselines(H):
    T = load_pe()
    ids, X, names, elen = build_matrix(T, 'BVNF')
    nl = {t['id']: len(t['lines']) for t in T}
    rng = np.random.default_rng(3)
    B = {}
    B['ELEN'] = [ids[i] for i in np.lexsort((rng.uniform(size=len(ids)), elen))]
    L = np.array([nl[p] for p in ids])
    B['LINES'] = [ids[i] for i in np.lexsort((rng.uniform(size=len(ids)), L))]
    nidx = [j for j, n in enumerate(names) if n.startswith('N_')]
    Xn = X[:, nidx]; ok = Xn.sum(1) > 0
    sc = orient(ca_scores(Xn[ok]), elen[ok])
    B['NUMSYS'] = [ids[i] for i in np.where(ok)[0][np.argsort(sc)]]
    prov = {t['id']: t['provenience'] for t in T}
    susa = [p for p in ids if prov[p].startswith('Susa')]
    other = [p for p in ids if not prov[p].startswith('Susa')]
    B['SITE_susa_first'] = list(rng.permutation(susa)) + list(rng.permutation(other))
    B['SITE_susa_last'] = list(rng.permutation(other)) + list(rng.permutation(susa))
    return B


def main(prefix, outp=None):
    H = load_hidden()
    res = {}
    for f in sorted(glob.glob(os.path.join(CK, 'frozen_%s*.json' % prefix))):
        d = json.load(open(f))
        name = os.path.basename(f)[7:-5]
        s = score_order(d['order'], H)
        if not name.startswith('NULLSHUF'):
            s.update(pvals(d['order'], H, s))
        s['hash'] = d['hash']
        res[name] = s
    for k, o in baselines(H).items():
        s = score_order(o, H); s.update(pvals(o, H, s, reps=1000)); res['BASE_' + k] = s
    sh = [v for k, v in res.items() if k.startswith('NULLSHUF')]
    if sh:
        res['NULLSHUF_summary'] = {k: [float(np.nanmean([d[k] for d in sh])), float(np.nanmax([d[k] for d in sh]))]
                                   for k in ('rho_susa', 'auc_early', 'auc_malyan', 'comb')}
    for k, v in res.items():
        if k.startswith('NULLSHUF_') and k != 'NULLSHUF_summary':
            continue
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    if outp:
        json.dump(res, open(outp, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '', sys.argv[2] if len(sys.argv) > 2 else None)
