"""v50 cycle 1: 1.67M deterministic reading paths over the page grids (REAL ZL3b, Markov filler MK, 8 planted
controls). Per-path z = (D_set - mean D over line-shuffled replicates LS1-3) / pooled noise sd; search-size null =
max z of LS4 used as pseudo-real, and of the message-free Markov set MK. Planted controls must be found with their
path above that threshold."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_lib as L

PLANT_PATH = {'PA': 1104, 'PN': 361, 'PD': 218808, 'PC': 871562}


def main():
    unit, sel, start, m = L.spec_cols(); groups = unit * 100 + sel * 10 + start
    res = {}; S = L.specs()
    base = {}
    for b in ('REAL', 'MK'):
        x = L.load(b); reps = [L.load(f'{b}_LS{k}') for k in (1, 2, 3, 4)]
        z, z0, sd = L.zscores(x, reps, groups)
        base[b] = (x, reps, z, z0, sd)
        res[b] = {'max_z': float(np.nanmax(z)), 'max_z0_pseudo': float(np.nanmax(z0)),
                  'p9999_z0': float(np.nanpercentile(z0, 99.99)), 'n_valid': int(np.sum(~np.isnan(z)))}
    T = max(res['REAL']['max_z0_pseudo'], res['MK']['max_z'], res['MK']['max_z0_pseudo'])
    res['threshold'] = T
    x, reps, z, z0, sd = base['REAL']
    order = np.argsort(-np.nan_to_num(z, nan=-1e9))
    res['REAL_top'] = [{'i': int(i), 'z': float(z[i]), 'D': float(x['D'][i]), 'Dls': float(np.mean([r['D'][i] for r in reps[:3]])),
                        'n': int(x['n'][i]), 'spec': L.spec_str(i, S)} for i in order[:40]]
    res['REAL_n_above_T'] = int(np.sum(z > T))
    from collections import Counter
    surv = np.where(z > T)[0]
    res['REAL_above_T_by_family'] = [[f'unit{a} sel{b} start{c}', n] for (a, b, c), n in
                                     Counter((int(unit[i]), int(sel[i]), int(start[i])) for i in surv).most_common()]
    # known-effect control: line-initial words permuted within each page (kills the v6 margin chain)
    if len(surv):
        fw = L.run_sub('REAL_FW', surv, 'c1_fw')
        fwr = [L.run_sub(f'REAL_FW_LS{k}', surv, f'c1_fw_LS{k}') for k in (1, 2, 3)]
        zfw = (fw['D'] - np.mean([r['D'] for r in fwr], axis=0)) / (sd[surv] * np.sqrt(1 + 1 / 3))
        res['survivors'] = [{'i': int(i), 'z': float(z[i]), 'z_FW': float(a), 'D': float(x['D'][i]), 'n': int(x['n'][i]),
                             'spec': L.spec_str(int(i), S)} for i, a in sorted(zip(surv, zfw), key=lambda t: -z[t[0]])]
        res['n_surv_FW_above_T'] = int(np.sum(zfw > T)); res['n_surv_FW_above_3'] = int(np.sum(zfw > 3))
    xm, rm, zm, z0m, sdm = base['MK']
    om = np.argsort(-np.nan_to_num(zm, nan=-1e9))
    res['MK_top'] = [{'i': int(i), 'z': float(zm[i]), 'spec': L.spec_str(i, S)} for i in om[:5]]
    # planted controls
    res['plants'] = {}
    for kind, pi in PLANT_PATH.items():
        for b in ('MK', 'REAL'):
            name = f'{kind}_{b}'
            fn = os.path.join(L.OUT, f'{name}.det.bin')
            if not os.path.exists(fn) or os.path.getsize(fn) < 1000: continue
            xp = L.load(name); bx, breps, bz, bz0, bsd = base[b]
            s1 = (xp['D'] - np.mean([r['D'] for r in breps[:3]], axis=0)) / bsd
            s1[xp['n'] < 300] = np.nan
            top = list(np.argsort(-np.nan_to_num(s1, nan=-1e9))[:3000])
            if pi not in top: top.append(pi)
            top = np.array(top)
            preps = [L.run_sub(f'{name}_LS{k}', top, f'c1_{name}_LS{k}') for k in (1, 2, 3)]
            ref = np.mean([r['D'] for r in preps], axis=0)
            zz = (xp['D'][top] - ref) / (bsd[top] * np.sqrt(1 + 1 / 3))
            zz[xp['n'][top] < 300] = np.nan
            o = np.argsort(-np.nan_to_num(zz, nan=-1e9))
            pos = int(np.where(top[o] == pi)[0][0])
            res['plants'][name] = {'planted_path': L.spec_str(pi, S), 'z_planted': float(zz[top == pi][0]),
                                   'rank_planted': pos + 1, 'found_above_T': bool(zz[top == pi][0] > T),
                                   'top1': L.spec_str(int(top[o[0]]), S), 'z_top1': float(zz[o[0]]),
                                   'top5': [[L.spec_str(int(top[j]), S), float(zz[j])] for j in o[:5]]}
            print(name, res['plants'][name]['z_planted'], pos + 1, flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c1.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('REAL_top',)}, indent=1)[:4000])
    for t in res['REAL_top'][:40]: print(round(t['z'], 1), round(t['D'], 4), round(t['Dls'], 4), t['n'], t['spec'])


if __name__ == '__main__':
    main()
