#!/usr/bin/env python3
"""S-DARK-60 cycle 4: pairwise functional distinctness of the heads, done properly.
Cycle 3 (dark_loop60.py 3) ran 400 permutations per pair, so its smallest attainable p (1/401 = 0.0025) could never pass
the Bonferroni alpha (0.05/55 = 0.0009) and every pair came out 'not distinct' by construction. This script redoes the test:
  * 5,000 permutations per pair (vectorised), labels shuffled between the two heads within length-bin;
  * two profiles: FULL (object type, site group + grammar) and GRAMMAR-ONLY (opener, connective, suffix, numeral epithet,
    any count, fish, quantity phrase, middle-length bin, second unit), so medium and city cannot carry a pair alone;
  * definitional dimensions dropped where a pseudo-head is involved ('none', W400-only, W90-only have no epithet; W400/W90
    are defined by their suffix);
  * effect size: leave-one-out naive-Bayes balanced accuracy (BA) of telling the two heads apart, with its own permutation
    null (300x) -> BA excess = BA - null mean;
  * controls: (a) split-half pseudo-pairs drawn from one head (jar, arrow, W700, none) at the sizes of small heads -> must be
    indistinct; (b) allograph pairs W154 vs W158 (S262 seal/tablet forms of one sign) and W526 vs W527 (strong merge) ->
    expected indistinct on grammar;
  * number of functionally distinct heads = connected components after joining every pair that is not distinct
    (p >= Bonferroni alpha OR BA excess < the 95th percentile of the split-half controls).
Wells at seq_raw / seq_strong / seq_all; IM77 through the bridge (M15 = W154/156/158 lumped; M245, M66 proposed entries).
Usage: python3 tools/dark_loop60_pairs.py <seq_raw|seq_strong|seq_all|im77> [nperm]
"""
import sys, collections, itertools
import numpy as np
LV = sys.argv[1]; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
sys.argv = ['x', '3', LV if LV != 'im77' else 'seq_raw', '10']
sys.path.insert(0, '/home/user/Indus-/tools')
import dark_loop60 as D

rng = np.random.default_rng(6004)
OUT = D.OUT
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

PSEUDO = {'none', 'W400', 'W90/91'}
SITES_W = ('Mohenjo-daro', 'Harappa', 'Dholavira', 'Kalibangan', 'Lothal', 'Chanhu-daro')
SITES_I = ('Mohenjodaro', 'Harappa', 'Lothal', 'Kalibangan', 'Chanhudaro')
def sitegrp(t):
    if t['foreign']: return 'foreign'
    return t['site'] if t['site'] in SITES_W + SITES_I else 'other_home'
DIMS_GRAM = {'opener': lambda t: t['opener'], 'conn': lambda t: t['conn'], 'suffix': lambda t: t['suffix'],
             'epithet': lambda t: t['epithet'], 'numeral': lambda t: t['numeral'], 'fish': lambda t: t['fish'],
             'quantity': lambda t: t['quantity'], 'midlen': lambda t: t['midbin'], 'second': lambda t: t['second_unit']}
DIMS_MED = {'type': lambda t: t['tf'], 'site': sitegrp}

def dims_for(a, b, profile):
    d = dict(DIMS_GRAM)
    if a in PSEUDO or b in PSEUDO:
        d.pop('suffix'); d.pop('epithet')
    if profile == 'full': d = {**DIMS_MED, **d}
    return d

def encode(pool, dims):
    enc = []
    for name, fn in dims.items():
        v = [fn(t) for t in pool]; cats = sorted(set(map(str, v)))
        ci = {c: k for k, c in enumerate(cats)}
        enc.append((name, np.array([ci[str(x)] for x in v]), len(cats)))
    return enc

def chi_many(L, enc):
    """L: (P, N) 0/1 label matrix (1 = head A). Returns (P,) summed chi-square and (P, ndim) per dim."""
    N = L.shape[1]; na = L.sum(1, keepdims=True).astype(float); nb = N - na
    per = []
    for name, codes, K in enc:
        oh = np.zeros((N, K)); oh[np.arange(N), codes] = 1
        tot = oh.sum(0)[None, :]
        ca = L.astype(float) @ oh; cb = tot - ca
        ea = tot * na / N; eb = tot * nb / N
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.where(ea > 0, (ca - ea) ** 2 / ea, 0) + np.where(eb > 0, (cb - eb) ** 2 / eb, 0)
        per.append(x.sum(1))
    per = np.array(per).T
    return per.sum(1), per

def nb_ba(L, enc, alpha=0.5):
    """leave-one-out naive Bayes balanced accuracy for each row of L (P, N)."""
    Pn, N = L.shape; Lf = L.astype(float)
    na = Lf.sum(1, keepdims=True); nb = N - na
    lp_a = np.zeros((Pn, N)); lp_b = np.zeros((Pn, N))
    for name, codes, K in enc:
        oh = np.zeros((N, K)); oh[np.arange(N), codes] = 1
        ca = Lf @ oh; cb = oh.sum(0)[None, :] - ca          # (P, K)
        ca_i = ca[:, codes] - Lf; cb_i = cb[:, codes] - (1 - Lf)   # leave own text out
        lp_a += np.log((ca_i + alpha) / (na - Lf + alpha * K))
        lp_b += np.log((cb_i + alpha) / (nb - (1 - Lf) + alpha * K))
    pred = (lp_a - lp_b) > 0           # equal priors -> balanced accuracy target
    tpr = (pred & (L == 1)).sum(1) / na[:, 0]; tnr = (~pred & (L == 0)).sum(1) / nb[:, 0]
    return (tpr + tnr) / 2

def perm_labels(lab, groups, nperm):
    L = np.tile(lab, (nperm, 1))
    for g in groups:
        L[:, g] = rng.permuted(L[:, g], axis=1)
    return L

def test_pair(A, B, dims, nperm, nperm_ba=300):
    pool = A + B; lab = np.array([1] * len(A) + [0] * len(B))
    enc = encode(pool, dims)
    gd = collections.defaultdict(list)
    for i, t in enumerate(pool): gd[D.lenbin(t['n'])].append(i)
    groups = [np.array(v) for v in gd.values() if len(v) > 1]
    obs, per = chi_many(lab[None, :], enc)
    null, _ = chi_many(perm_labels(lab, groups, nperm), enc)
    p = (np.sum(null >= obs[0] - 1e-9) + 1) / (nperm + 1)
    ba = nb_ba(lab[None, :], enc)[0]
    ba_null = nb_ba(perm_labels(lab, groups, nperm_ba), enc)
    drivers = sorted(zip(dims.keys(), per[0]), key=lambda kv: -kv[1])[:3]
    bn = ba_null.mean()
    # normalised excess: share of the room above the length-preserving null that the real labels use ((BA - null) / (1 - null))
    return dict(chi=obs[0], chi_null=null.mean(), p=p, ba=ba, ba_null=bn, ba_x=(ba - bn) / max(1 - bn, 1e-6), drivers=drivers)

def max_cliques(nodes, edges):
    """all maximum cliques of the DISTINCT graph (brute force; <= 15 nodes)."""
    E = set(edges) | {(b, a) for a, b in edges}; best = []; bs = 0
    for r in range(len(nodes), 0, -1):
        for c in itertools.combinations(nodes, r):
            if all((a, b) in E for a, b in itertools.combinations(c, 2)):
                best.append(c)
        if best: return r, best
    return 0, []

def components(nodes, joins):
    parent = {h: h for h in nodes}
    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    for a, b in joins: parent[find(a)] = find(b)
    comps = collections.defaultdict(list)
    for h in nodes: comps[find(h)].append(h)
    return list(comps.values())

def run(T, heads, label, min_n=10):
    T2 = [t for t in T if t['n'] >= 2]
    byh = {h: [t for t in T2 if t['head'] == h] for h in heads}
    use = [h for h in heads if len(byh[h]) >= min_n]
    P(f'\n######## {label}: heads with >= {min_n} texts of >= 2 signs: ' + ', '.join(f'{h} {len(byh[h])}' for h in use))
    P(f'excluded (< {min_n}): ' + ', '.join(f'{h} {len(byh[h])}' for h in heads if h not in use))
    out = {}
    for profile in ('full', 'grammar'):
        # ---- controls: split-half pseudo-pairs ----
        ctrl = []
        sizes = [15, 25, 40, 60]
        for src in [h for h in ('jar', 'arrow', 'W700', 'none') if h in use]:
            for sz in sizes:
                if len(byh[src]) < 2 * sz: continue
                for rep in range(5):
                    idx = rng.permutation(len(byh[src]))
                    A = [byh[src][i] for i in idx[:sz]]; B = [byh[src][i] for i in idx[sz:2 * sz]]
                    r = test_pair(A, B, dims_for(src, src, profile), 1000, 200)
                    ctrl.append((src, sz, r))
        ctrl_p = np.array([r['p'] for _, _, r in ctrl]); ctrl_bx = np.array([r['ba_x'] for _, _, r in ctrl])
        bx95 = float(np.percentile(ctrl_bx, 95))
        P(f'\n=== {label} / {profile} profile ===')
        P(f'  split-half controls ({len(ctrl)} pseudo-pairs from jar/arrow/W700/none at n = 15-60 a side): '
          f'p < 0.05 in {np.mean(ctrl_p < 0.05):.2f} (nominal 0.05); normalised BA excess (BA-null)/(1-null) mean {ctrl_bx.mean():.3f}, 95th pct {bx95:.3f}')
        # ---- allograph controls (Wells only) ----
        if label.startswith('Wells'):
            for a, b in [(154, 158), (526, 527)]:
                A = [t for t in T2 if t['head_sign'] == a]; B = [t for t in T2 if t['head_sign'] == b]
                if len(A) >= 8 and len(B) >= 8:
                    r = test_pair(A, B, dims_for('x', 'y', profile), NP)
                    P(f'  allograph control W{a} vs W{b} [n {len(A)}/{len(B)}]: chi {r["chi"]:.0f} (null {r["chi_null"]:.0f}), p {r["p"]:.4f}, '
                      f'BA {r["ba"]:.2f} (null {r["ba_null"]:.2f}, excess {r["ba_x"]:.2f}); drivers ' + ', '.join(f'{d} {v:.0f}' for d, v in r['drivers']))
        # ---- all pairs ----
        pairs = {}
        for a, b in itertools.combinations(use, 2):
            pairs[(a, b)] = test_pair(byh[a], byh[b], dims_for(a, b, profile), NP)
        alpha = 0.05 / len(pairs)
        P(f'  pairs {len(pairs)}; Bonferroni alpha {alpha:.5f}; min attainable p {1 / (NP + 1):.5f}; practical bar: normalised BA excess >= {bx95:.3f} (95th pct of split-half controls)')
        joins_stat = []; joins_both = []
        for (a, b), r in sorted(pairs.items(), key=lambda kv: kv[1]['ba_x']):
            st = r['p'] < alpha; pr = r['ba_x'] >= bx95
            tag = 'DISTINCT' if st and pr else 'stat-only' if st else 'size-only' if pr else 'INDISTINCT'
            if not st: joins_stat.append((a, b))
            if not (st and pr): joins_both.append((a, b))
            P(f'  {a} vs {b} [n {len(byh[a])}/{len(byh[b])}]: chi {r["chi"]:.0f} (null {r["chi_null"]:.0f}), p {r["p"]:.4f}, BA {r["ba"]:.2f} '
              f'(null {r["ba_null"]:.2f}, excess {r["ba_x"]:+.2f}) -> {tag}; drivers ' + ', '.join(f'{d} {v:.0f}' for d, v in r['drivers']))
        dist = [(a, b) for (a, b), r in pairs.items() if r['p'] < alpha and r['ba_x'] >= bx95]
        k, cl = max_cliques(use, dist)
        P(f'  DISTINCT pairs {len(dist)} of {len(pairs)}. Largest set of MUTUALLY distinct heads: {k} of {len(use)}; {len(cl)} such set(s): ' +
          ' | '.join('{' + ', '.join(c) + '}' for c in cl[:8]))
        always = sorted(set(use) - set().union(*[set(c) for c in cl])) if cl else use
        P('  heads in no maximum set: ' + (', '.join(always) if always else 'none'))
        for h in use:
            nd = [b if a == h else a for (a, b) in pairs if h in (a, b) and (a, b) not in dist]
            P(f'    {h}: not distinct from {len(nd)}: ' + (', '.join(nd) if nd else '-'))
        big = [h for h in use if len(byh[h]) >= 40]
        k2, cl2 = max_cliques(big, [(a, b) for a, b in dist if a in big and b in big])
        P(f'  heads with >= 40 texts ({len(big)}): largest mutually distinct set {k2}: ' + ' | '.join('{' + ', '.join(c) + '}' for c in cl2[:8]))
        cs = components(use, joins_stat)
        P(f'  (single-linkage groups, cycle-3 method, for comparison only: {len(cs)} of {len(use)})')
        out[profile] = (pairs, dist, k, cl, k2, cl2, bx95)
    return out

if __name__ == '__main__':
    if LV == 'im77':
        T = D.load_im77()
        heads = ['jar', 'mjar', 'arrow', 'W151', 'W156+154/158', 'box527', 'W226', 'W615/617', 'W236', 'W700', 'W595', 'W400', 'W90/91', 'none']
        run(T, heads, 'IM77 (M numbers through the bridge)')
    else:
        T = D.load_wells(LV, complete_only=True)
        run(T, D.HEADS_ORDER, f'Wells {LV}')
    open(OUT + f'loop60_c4_{LV}.txt', 'w').write('\n'.join(LOG) + '\n')
