"""v41 cycle 2: is there a drift clock inside each scribal hand and Currier language?
Groups: hand x language (x section, or demeaned by section). Traits: the 10 fixed traits
plus a stem-held-fixed variant trait (share of B endings among words whose stem occurs in
the corpus with both an A and a B ending: the spelling choice with the word held fixed).
Run on ZL3b and IT2a. Also a 'trait-label shuffle': orientation signs randomised (does the
A->B orientation matter, or would any sign pattern cohere as well?).
"""
import sys, os, json, random
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from v41_lib import *

res = {}
rows = []


def run_group(pages, T, o, strata, seed, nperm=300):
    (X1, X2), names = rate_matrix(pages, T)
    coh = coherence_null(X1, X2, o, strata, nperm=nperm, seed=seed)
    D1 = demean(X1, strata); D2 = demean(X2, strata)
    order, f, orders = seriate(D1, restarts=10, iters=12000, seed=seed)
    pos = position_of(order)
    z = lambda D: ((D - D.mean(0)) / (D.std(0) + 1e-12)) * o
    if spearmanr(pos, z(D1).mean(1))[0] < 0:
        pos = pos.max() - pos
    held = spearmanr(pos, z(D2).mean(1))[0]
    sg = [spearmanr(pos, D2[:, j])[0] * o[j] for j in range(len(names))]
    stab = np.mean([abs(spearmanr(position_of(orders[i]), position_of(orders[j]))[0])
                    for i in range(len(orders)) for j in range(i + 1, len(orders))])
    # random orientation null: how coherent would a random sign pattern be?
    rng = np.random.default_rng(seed)
    rs = [coherence(X1, X2, rng.choice([-1., 1.], len(o)), strata)['S'] for _ in range(300)]
    # held-out trajectory null: traits permuted independently (both halves together), order by
    # first principal component of half 1 (the SA optimum is close to it), same held-out score
    def pc_held(Y1, Y2):
        E1 = demean(Y1, strata); E2 = demean(Y2, strata)
        Zs = (E1 - E1.mean(0)) / (E1.std(0) + 1e-12)
        u = np.linalg.svd(Zs, full_matrices=False)[0][:, 0]
        if spearmanr(u, (Zs * o).mean(1))[0] < 0:
            u = -u
        return spearmanr(u, z(E2).mean(1))[0]
    held_pc = pc_held(X1, X2)
    hn = []
    for _ in range(200):
        Y1 = X1.copy(); Y2 = X2.copy()
        for j in range(X1.shape[1]):
            pi = perm_within(len(pages), strata, rng); Y1[:, j] = X1[pi, j]; Y2[:, j] = X2[pi, j]
        hn.append(pc_held(Y1, Y2))
    hn = np.array(hn)
    Dall = demean((X1 + X2) / 2, strata)
    clock = z(Dall).mean(1)
    return dict(n=len(pages), R=coh['R'], S=coh['S'], ratio=coh['ratio'], z=coh['z'], p=coh['p'],
                pos_pairs=coh['pos_pairs'], held=float(held), signs_ok=int(sum(s > 0 for s in sg)),
                sign_detail=dict(zip(names, [round(float(s), 2) for s in sg])),
                stab=float(stab), held_pc=float(held_pc), held_null=float(hn.mean()), held_null_sd=float(hn.std()),
                held_z=float((held_pc - hn.mean()) / (hn.std() + 1e-12)), randsign_S_q=float(np.mean(np.array(rs) >= coh['S'])),
                C=np.round(coh['C'], 3).tolist(), names=names,
                pages=[p['id'] for p in pages], clock=clock.tolist(), pos=pos.tolist())


for name in ('ZL3b', 'IT2a'):
    P = vpages(name)
    T, nshared = make_traits(P)
    A = [p for p in P if p['lang'] == 'A']; B = [p for p in P if p['lang'] == 'B']
    o = orient_from(A, B, T)
    res[name + '_orient'] = dict(zip(T, o.tolist()))
    groups = {
        'H1A_herbal': ([p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'H'], 'none'),
        'H1A_pharma': ([p for p in P if p['hand'] == '1' and p['lang'] == 'A' and p['illus'] == 'P'], 'none'),
        'H1A_all': ([p for p in P if p['hand'] == '1' and p['lang'] == 'A'], 'illus'),
        'H2B_herbal': ([p for p in P if p['hand'] == '2' and p['lang'] == 'B' and p['illus'] == 'H'], 'none'),
        'H2B_bio': ([p for p in P if p['hand'] == '2' and p['lang'] == 'B' and p['illus'] == 'B'], 'none'),
        'H2B_all': ([p for p in P if p['hand'] == '2' and p['lang'] == 'B'], 'illus'),
        'H3B_stars': ([p for p in P if p['hand'] == '3' and p['lang'] == 'B' and p['illus'] == 'S'], 'none'),
        'H3B_all': ([p for p in P if p['hand'] == '3' and p['lang'] == 'B'], 'illus'),
        'B_all': ([p for p in P if p['lang'] == 'B' and (p['hand'] or '') in ('2', '3', '5')], 'hand_illus'),
    }
    for g, (pg, sm) in groups.items():
        strata = None if sm == 'none' else [p['illus'] if sm == 'illus' else p['hand'] + p['illus'] for p in pg]
        if strata is None:
            strata = ['x'] * len(pg)
        r = run_group(pg, T, o, strata, seed=sum(map(ord, g)))
        res[f'{name}_{g}'] = r
        print(name, g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                        if k not in ('C', 'pages', 'clock', 'pos', 'names')}, flush=True)
        rows.append(f"| V-41.2.{len(rows)+1} | {name} {g}: 11 oriented traits (10 fixed + stem-held-fixed B-ending share; {nshared} shared stems), odd/even line halves, strata={sm}; S vs independent-trait permutation (300x); SA seriation 10 restarts x 12k swaps; held-out even-line composite; random-orientation null (300 sign patterns) | n {r['n']}, R {r['R']:.3f}, S {r['S']:+.3f} (z {r['z']:+.1f}, p {r['p']:.3f}), S/R {r['ratio']:.2f}, oriented pairs >0 {r['pos_pairs']:.2f}, held-out rho {r['held']:+.2f} (PC order {r['held_pc']:+.2f} vs shuffled-trait null {r['held_null']:+.2f}+-{r['held_null_sd']:.2f}, z {r['held_z']:+.1f}), signs ok {r['signs_ok']}/11, restart stability {r['stab']:.2f}, random-sign S >= obs {r['randsign_S_q']:.3f} | see cycle verdict |")
json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), default=float)
write_rows(os.path.join(CK, 'c2_rows.txt'), rows)
print('\n'.join(rows))
