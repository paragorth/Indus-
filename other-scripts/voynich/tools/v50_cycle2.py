"""v50 cycle 2: held-out test, second transcription, language battery -- with the known margin chain removed:
the real pages are REAL_FW (line-initial words permuted within each page; cycle 1 showed every survivor was the
v6 left-margin succession), and the second transcription is IT_FW.
(a) Discovery on EVEN pages only (full 1.67M det search on REAL and its 4 line-shuffled replicates, even mask);
    the top 300 paths by z_even are re-scored on ODD pages (REAL vs REAL_LS1-3, odd mask). Null: the same
    discovery/test pipeline with REAL_LS4 playing the real pages (top 300 by its pseudo-z on even pages, tested on
    odd pages against LS1-3).
(b) Paths that hold out are re-scored on IT2a (IT vs IT_LS1-3).
(c) Language battery: P(language) of each surviving path's stream (glyph or word classifier trained on letter /
    word streams of 8 / 5 languages vs Voynich-texture null streams), for REAL and for the same path on LS1."""
import os, sys, json, pickle, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_lib as L

NTOP = 300


def sub_z(setname, idx, sd, tag, mask):
    x = L.run_sub(setname, idx, f'{tag}_{setname}', mask=mask)
    reps = [L.run_sub(f'{setname}_LS{k}', idx, f'{tag}_{setname}_LS{k}', mask=mask) for k in (1, 2, 3)]
    ref = np.mean([r['D'] for r in reps], axis=0)
    z = (x['D'] - ref) / (sd * math.sqrt(1 + 1 / 3)); z[x['n'] < 150] = np.nan
    return z, x


def sub_z_pseudo(idx, sd, tag, mask):
    x = L.run_sub('REAL_FW_LS4', idx, f'{tag}_ps', mask=mask)
    reps = [L.run_sub(f'REAL_FW_LS{k}', idx, f'{tag}_ps_LS{k}', mask=mask) for k in (1, 2, 3)]
    ref = np.mean([r['D'] for r in reps], axis=0)
    z = (x['D'] - ref) / (sd * math.sqrt(1 + 1 / 3)); z[x['n'] < 150] = np.nan
    return z


def main():
    unit, sel, start, m = L.spec_cols(); groups = unit * 100 + sel * 10 + start
    S = L.specs(); res = {}
    xe = L.load('REAL_FW', 'even'); re_ = [L.load(f'REAL_FW_LS{k}', 'even') for k in (1, 2, 3, 4)]
    ze, ze0, sde = L.zscores(xe, re_, groups, minn=150)
    # noise sd for odd pages: same pooled model (odd half has the same size)
    top = np.argsort(-np.nan_to_num(ze, nan=-1e9))[:NTOP]
    top0 = np.argsort(-np.nan_to_num(ze0, nan=-1e9))[:NTOP]
    zo, xo = sub_z('REAL_FW', top, sde[top], 'c2o', 'odd')
    zo0 = sub_z_pseudo(top0, sde[top0], 'c2o0', 'odd')
    res['even_max_z'] = float(np.nanmax(ze)); res['even_max_z_null'] = float(np.nanmax(ze0))
    res['odd_z_of_top'] = {'median': float(np.nanmedian(zo)), 'n_gt3': int(np.sum(zo > 3)), 'max': float(np.nanmax(zo))}
    res['odd_z_of_null_top'] = {'median': float(np.nanmedian(zo0)), 'n_gt3': int(np.sum(zo0 > 3)), 'max': float(np.nanmax(zo0))}
    T_hold = max(3.0, float(np.nanmax(zo0)))
    res['T_hold'] = T_hold
    hold = [int(i) for i, z in zip(top, zo) if z > T_hold]
    res['n_hold'] = len(hold)
    # IT2a replication of the held-out survivors (all pages)
    sdf = L.zscores(L.load('REAL'), [L.load(f'REAL_LS{k}') for k in (1, 2, 3, 4)], groups)[2]
    rows = []
    res['battery_on'] = 'held-out survivors' if hold else 'top 10 by odd-page z (no survivor)'
    cand = hold if hold else [int(top[j]) for j in np.argsort(-np.nan_to_num(zo, nan=-1e9))[:10]]
    if cand:
        hi = np.array(cand)
        zit, xit = sub_z('IT_FW', hi, sdf[hi], 'c2it', 'all')
        o = L.train_clf(null_sets=('MK', 'REAL_LS1', 'REAL_FW'), nneg=450)
        clf = {k: v[0] for k, v in o.items()}; res['clf_cv_acc'] = {k: v[1] for k, v in o.items()}
        pickle.dump(clf, open(os.path.join(L.CK, 'clf.pkl'), 'wb'))
        stR = L.streams('REAL_FW', hi, tag='c2s'); stN = L.streams('REAL_FW_LS1', hi, tag='c2sn')
        for j, i in enumerate(hi):
            typ = 'word' if (unit[i] == 0 and sel[i] >= 4) else 'glyph'
            bR = L.battery(stR[i]); bN = L.battery(stN[i])
            pR = float(clf[typ].predict_proba([L.featvec(bR)])[0, 1]); pN = float(clf[typ].predict_proba([L.featvec(bN)])[0, 1])
            rows.append({'i': int(i), 'spec': L.spec_str(i, S), 'z_even': float(ze[i]), 'z_odd': float(zo[list(top).index(i)]),
                         'z_IT': float(zit[j]), 'n': int(xit['n'][j]), 'P_lang_REAL': pR, 'P_lang_LS1': pN,
                         'rmi1': bR['rmi1'], 'rmi1_LS': bN['rmi1']})
    res['survivors'] = rows
    json.dump(res, open(os.path.join(L.CK, 'c2.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'survivors'}, indent=1))
    for r in sorted(rows, key=lambda r: -r['z_odd'])[:60]:
        print(f"{r['z_even']:.1f} {r['z_odd']:.1f} {r['z_IT']:.1f} P={r['P_lang_REAL']:.2f}/{r['P_lang_LS1']:.2f} rmi1={r['rmi1']:.4f}/{r['rmi1_LS']:.4f} {r['spec']}")


if __name__ == '__main__':
    main()
