"""v50 cycle 3: 600,000 RANDOM multi-move reading paths (3-6 moves per cycle, random starts, edge modes,
selectors; tools/v50_search.c build_rand, seed 50). Discovery on EVEN pages (REAL vs LS1-3, pooled noise from
LS1-4), test of the top 300 on ODD pages; null = LS4 as pseudo-real through the same pipeline. Positive controls:
Latin letters (PR1) / German code words (PR2) hidden along a random path of the same family, in Markov filler and
in the real pages; each must be top-ranked on even pages and hold out on odd pages. Second score: cross-page
trigram recurrence (R3) excess, same z machinery. Real pages = REAL_FW (margin chain removed, see cycle 1)."""
import os, sys, json, gzip, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_lib as L

KIND = 'subrand:600000:50'


def rspecs():
    return [l.split() for l in gzip.open(os.path.join(L.CK, 'rand_specs.txt.gz'), 'rt')]


def zsub(setname, reps, idx, sd, tag, mask, key='D'):
    x = L.run_sub(setname, idx, f'{tag}_{setname}', kind=KIND, mask=mask)
    rr = [L.run_sub(r, idx, f'{tag}_{r}', kind=KIND, mask=mask) for r in reps]
    ref = np.mean([r[key] for r in rr], axis=0)
    z = (x[key] - ref) / (sd * math.sqrt(1 + 1 / len(rr))); z[x['n'] < 150] = np.nan
    return z


def zfull(x, reps, groups, key):
    Ds = [r[key] for r in reps]
    V = np.var(np.stack(Ds), axis=0, ddof=1)
    b = np.clip(np.log2(np.maximum(x['n'], 1)).astype(int), 0, 30); kk = groups * 32 + b
    sd = np.zeros_like(V)
    for k in np.unique(kk):
        ix = kk == k; sd[ix] = math.sqrt(max(np.mean(V[ix]), 1e-12))
    sd = np.maximum(sd, np.sqrt(np.var(np.stack(Ds[:-1]), axis=0, ddof=1)))
    ref = np.mean(Ds[:-1], axis=0); c = math.sqrt(1 + 1 / (len(Ds) - 1))
    z = (x[key] - ref) / (sd * c); z0 = (Ds[-1] - ref) / (sd * c)
    bad = (x['n'] < 150) | (reps[0]['n'] < 150); z[bad] = np.nan; z0[bad] = np.nan
    return z, z0, sd


def main():
    R = rspecs()
    unit = np.array([int(r[1][5:]) for r in R]); sel = np.array([int(r[2][4:]) for r in R]); start = np.array([int(r[3][6:]) for r in R])
    groups = unit * 100 + sel * 10 + start
    sp = lambda i: ' '.join(R[i][1:])
    plants = json.load(open(os.path.join(L.CK, 'c3_plants.json')))
    res = {}
    x = L.load('REAL_FW', 'reven'); reps = [L.load(f'REAL_FW_LS{k}', 'reven') for k in (1, 2, 3, 4)]
    LSr = [f'REAL_FW_LS{k}' for k in (1, 2, 3)]
    for key in ('D', 'r3'):
        z, z0, sd = zfull(x, reps, groups, key)
        top = np.argsort(-np.nan_to_num(z, nan=-1e9))[:300]; top0 = np.argsort(-np.nan_to_num(z0, nan=-1e9))[:300]
        zo = zsub('REAL_FW', LSr, top, sd[top], f'c3o{key}', 'odd', key)
        zo0 = zsub('REAL_FW_LS4', LSr, top0, sd[top0], f'c3o0{key}', 'odd', key)
        Th = max(3.0, float(np.nanmax(zo0)))
        hold = [(int(i), float(z[i]), float(a)) for i, a in zip(top, zo) if a > Th]
        res[key] = {'even_max_z': float(np.nanmax(z)), 'even_max_z_null': float(np.nanmax(z0)),
                    'odd_median_top': float(np.nanmedian(zo)), 'odd_median_null': float(np.nanmedian(zo0)),
                    'odd_n_gt3_top': int(np.sum(zo > 3)), 'odd_n_gt3_null': int(np.sum(zo0 > 3)), 'T_hold': Th,
                    'n_hold': len(hold), 'hold': [[i, a, b, sp(i)] for i, a, b in sorted(hold, key=lambda t: -t[2])[:40]],
                    'top10_even': [[int(i), float(z[i]), float(zo[j]), sp(int(i))] for j, i in enumerate(top[:10])]}
        print(key, json.dumps({k: v for k, v in res[key].items() if k not in ('hold', 'top10_even')}), flush=True)
        for h in res[key]['hold'][:25]: print('  hold', h)
        if key == 'D':
            # positive controls
            res['plants'] = {}
            for kind in ('PR1', 'PR2'):
                pi = plants[kind]['path']
                for b in ('MK', 'REAL'):
                    name = f'{kind}_{b}'
                    xp = L.load(name, 'reven')
                    s1 = (xp['D'] - np.mean([r['D'] for r in reps[:3]], axis=0)) / sd; s1[xp['n'] < 150] = np.nan
                    cand = list(np.argsort(-np.nan_to_num(s1, nan=-1e9))[:1000])
                    if pi not in cand: cand.append(pi)
                    cand = np.array(cand)
                    ze = zsub(name, [f'{name}_LS{k}' for k in (1, 2, 3)], cand, sd[cand], f'c3p{name}', 'even')
                    o = np.argsort(-np.nan_to_num(ze, nan=-1e9)); rank = int(np.where(cand[o] == pi)[0][0]) + 1
                    zodd = zsub(name, [f'{name}_LS{k}' for k in (1, 2, 3)], np.array([pi]), sd[[pi]], f'c3q{name}', 'odd')
                    res['plants'][name] = {'path': sp(pi), 'z_even': float(ze[cand == pi][0]), 'rank_even': rank,
                                           'z_odd': float(zodd[0]), 'found': bool(rank <= 3 and ze[cand == pi][0] > res['D']['even_max_z_null'] and zodd[0] > Th),
                                           'top1_even': [sp(int(cand[o[0]])), float(ze[o[0]])]}
                    print(name, res['plants'][name], flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
