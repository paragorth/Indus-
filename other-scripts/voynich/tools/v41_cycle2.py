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


def stem_split(w):
    for e in sorted(A_END + B_END, key=len, reverse=True):
        if w.endswith(e) and len(w) > len(e):
            return w[:-len(e)], e
    return None, None


def make_traits(P):
    st = defaultdict(set)
    for p in P:
        for w in p['all']:
            s, e = stem_split(w)
            if s:
                st[s].add('B' if e in B_END else 'A')
    shared = {s for s, v in st.items() if len(v) == 2}
    T = dict(VTRAITS)
    T['stem_Bshare'] = (lambda w: stem_split(w)[1] in B_END,
                        lambda w: stem_split(w)[0] in shared)
    return T, len(shared)


def run_group(pages, T, o, strata, seed, nperm=400):
    (X1, X2), names = rate_matrix(pages, T)
    coh = coherence_null(X1, X2, o, strata, nperm=nperm, seed=seed)
    D1 = demean(X1, strata); D2 = demean(X2, strata)
    order, f, orders = seriate(D1, restarts=16, iters=15000, seed=seed)
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
    Dall = demean((X1 + X2) / 2, strata)
    clock = z(Dall).mean(1)
    return dict(n=len(pages), R=coh['R'], S=coh['S'], ratio=coh['ratio'], z=coh['z'], p=coh['p'],
                pos_pairs=coh['pos_pairs'], held=float(held), signs_ok=int(sum(s > 0 for s in sg)),
                sign_detail=dict(zip(names, [round(float(s), 2) for s in sg])),
                stab=float(stab), randsign_S_q=float(np.mean(np.array(rs) >= coh['S'])),
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
        'B_all': ([p for p in P if p['lang'] == 'B' and p['hand'] in '235'], 'hand_illus'),
    }
    for g, (pg, sm) in groups.items():
        strata = None if sm == 'none' else [p['illus'] if sm == 'illus' else p['hand'] + p['illus'] for p in pg]
        if strata is None:
            strata = ['x'] * len(pg)
        r = run_group(pg, T, o, strata, seed=hash(g) % 1000)
        res[f'{name}_{g}'] = r
        print(name, g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                        if k not in ('C', 'pages', 'clock', 'pos', 'names')}, flush=True)
        rows.append(f"| V-41.2.{len(rows)+1} | {name} {g}: 11 oriented traits (10 fixed + stem-held-fixed B-ending share; {nshared} shared stems), odd/even line halves, strata={sm}; S vs independent-trait permutation (400x); SA seriation 16 restarts; held-out even-line composite; random-orientation null (300 sign patterns) | n {r['n']}, R {r['R']:.3f}, S {r['S']:+.3f} (z {r['z']:+.1f}, p {r['p']:.3f}), S/R {r['ratio']:.2f}, oriented pairs >0 {r['pos_pairs']:.2f}, held-out rho {r['held']:+.2f}, signs ok {r['signs_ok']}/11, restart stability {r['stab']:.2f}, random-sign S >= obs {r['randsign_S_q']:.3f} | see cycle verdict |")
json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), default=float)
write_rows(os.path.join(CK, 'c2_rows.txt'), rows)
print('\n'.join(rows))
