"""pe20 cycle 2: re-test the search's survivors outside the search set.

  A held-out herd records (every PE herd record except MDP 17,096+ and its copy
    MDP 17,191): MAP and top-500 assignments scored blind; nulls = the same
    assignment on held-out numbers shuffled within sign (500) and 20,000 random
    assignments; does search-set rank predict held-out rank?
  B chain: herd size implied by an assignment (sum of non-X categories) in the
    17,096+ herd blocks vs the same herds' first-column counts in the herd lists
    MDP 17,085 and 17,097 (pe5 link); MAP vs 20,000 random assignments.
  C random rulers: 300 nonsense prior boxes (young per female 0.05-8, males per
    female 0.005-3, log-uniform box edges) vs the pastoral box, max score on the
    search set at equal search effort; and the same on Ur III PE-sized subsamples.
Usage: python3 pe20_cycle2.py AB | C
"""
import sys
from scipy.stats import spearmanr
from pe20_common import *

# 17,096+ block -> first-column counts in 17,085 / 17,097 (matched by herd-name compound)
CHAIN = {'o3': [6, None], 'o4': [5, 5], 'o5': [4, 2], 'r2': [5, 5], 'r3': [2, None],
         'r4': [None, 3], 'r5': [7, 11], 'r6': [3, 3]}


def load_c1():
    c1 = json.load(open(os.path.join(DATA, 'pe20_cycle1_pe.json')))
    inv = {state_name(k): k for k in range(NSTATE)}
    amap = [inv[c1['max_assign'][s]] for s in PE_SIGNS]
    top = json.load(open(os.path.join(CK, 'c1_top_assign.json')))['top']
    return c1, amap, [(v, a) for v, a in top]


def herd_size(V, a):
    m = np.array([x != 0 for x in a])
    W = np.where(np.isnan(V), np.nan, V)[:, m]
    return np.nansum(W, 1), np.isnan(W).sum(1)


def chain_rho(Vmain, labs, a):
    hs, nmiss = herd_size(Vmain, a)
    xs, ys = [], []
    for i, lab in enumerate(labs):
        if lab in CHAIN and nmiss[i] == 0:
            for c in CHAIN[lab]:
                if c is not None:
                    xs.append(hs[i])
                    ys.append(c)
    if len(xs) < 5 or len(set(xs)) < 3:
        return np.nan, len(xs)
    return spearmanr(xs, ys)[0], len(xs)


def part_AB():
    rng = np.random.default_rng(21)
    c1, amap, top = load_c1()
    allr = pe_records()
    ho = [r for r in allr if r[0] not in (MAIN, CHAIN_COPY)]
    main = [r for r in allr if r[0] == MAIN]
    Vho, Vm = to_matrix(ho, PE_SIGNS), to_matrix(main, PE_SIGNS)
    lg = Scorer(Vm).lg
    sho = Scorer(Vho, lg=lg)
    out = {'n_heldout_records': len(ho), 'n_heldout_tablets': len({r[0] for r in ho}),
           'map': [state_name(x) for x in amap]}
    s_map = sho.score(amap)
    nul = [Scorer(shuffle_within_sign(Vho, rng), lg=lg).score(amap) for _ in range(500)]
    nul_all = [Scorer(shuffle_all(Vho, rng), lg=lg).score(amap) for _ in range(500)]
    rnd = []
    for _ in range(20000):
        rnd.append(sho.score(random_assignment(8, rng)))
    rnd = np.array(rnd)
    out['A'] = {'map_heldout': s_map,
                'p_within_shuffle': float((1 + np.sum(np.array(nul) >= s_map)) / 501),
                'p_all_shuffle': float((1 + np.sum(np.array(nul_all) >= s_map)) / 501),
                'null_within_mean': float(np.mean(nul)), 'null_all_mean': float(np.mean(nul_all)),
                'pct_among_random': float(np.mean(rnd < s_map)),
                'random_pct': np.percentile(rnd, [50, 90, 99]).tolist()}
    tops = [sho.score(a) for v, a in top]
    ss = [v for v, a in top]
    out['A']['top500_heldout_mean'] = float(np.mean(tops))
    out['A']['top500_pct_among_random_mean'] = float(np.mean([np.mean(rnd < t) for t in tops]))
    out['A']['rho_search_vs_heldout_top500'] = float(spearmanr(ss, tops)[0])
    # random assignments: search vs held-out rank correlation (does fit transfer at all?)
    smain = Scorer(Vm)
    pairs = []
    for _ in range(3000):
        a = random_assignment(8, rng)
        pairs.append((smain.score(a), sho.score(a)))
    pairs = np.array(pairs)
    out['A']['rho_search_vs_heldout_random3000'] = float(spearmanr(pairs[:, 0], pairs[:, 1])[0])
    # per-record contributions of the MAP on held-out (which records carry it)
    contrib = []
    for i, r in enumerate(ho):
        s1 = Scorer(Vho[i:i + 1], lg=lg).score(amap)
        contrib.append((r[0], r[1], round(s1, 2)))
    out['A']['per_record'] = sorted(contrib, key=lambda x: -abs(x[2]))[:15]
    print('A', json.dumps({k: v for k, v in out['A'].items() if k != 'per_record'}), flush=True)
    # B chain
    labs = [r[1] for r in main]
    rho_map, n_map = chain_rho(Vm, labs, amap)
    rr = []
    for _ in range(20000):
        rho, n = chain_rho(Vm, labs, random_assignment(8, rng))
        if not np.isnan(rho):
            rr.append(rho)
    rr = np.array(rr)
    top_rho = [chain_rho(Vm, labs, a)[0] for v, a in top[:100]]
    single = {}
    for j, s in enumerate(PE_SIGNS):
        a = [0] * 8
        a[j] = 1
        single[s] = chain_rho(Vm, labs, a)
    out['B'] = {'map_rho': rho_map, 'map_n': n_map,
                'pct_among_random': float(np.mean(rr < rho_map)) if not np.isnan(rho_map) else None,
                'random_mean': float(rr.mean()), 'random_pct': np.percentile(rr, [50, 90, 99]).tolist(),
                'top100_mean_rho': float(np.nanmean(top_rho)), 'single_sign': single}
    print('B', json.dumps(out['B']), flush=True)
    dump(out, os.path.join(DATA, 'pe20_cycle2_AB.json'))


def part_C(n_rulers=300):
    rng = np.random.default_rng(22)
    main = [r for r in pe_records() if r[0] == MAIN]
    Vm = to_matrix(main, PE_SIGNS)
    real = [maximise(Scorer(Vm), rng, restarts=3)[0] for _ in range(3)]
    rul = []
    for k in range(n_rulers):
        lo, hi = sorted(np.exp(rng.uniform(np.log(0.05), np.log(8), 2)))
        rlo, rhi = sorted(np.exp(rng.uniform(np.log(0.005), np.log(3), 2)))
        fg = np.geomspace(lo, max(hi, lo * 1.05), 12)
        rg = np.geomspace(rlo, max(rhi, rlo * 1.05), 10)
        v = maximise(Scorer(Vm, fg=fg, rg=rg), rng, restarts=3)[0]
        rul.append({'f': [lo, hi], 'rho': [rlo, rhi], 'max': v})
        if k % 25 == 24:
            print(k, np.mean([x['max'] >= np.mean(real) for x in rul]), flush=True)
    U = ur_records()
    Vu = to_matrix(U, UR_SIGNS)
    rich = [i for i in range(len(U)) if np.sum(~np.isnan(Vu[i])) >= 4]
    ur = []
    for rep in range(6):
        idx = rng.choice(rich, 10, replace=False)
        rv = maximise(Scorer(Vu[idx]), rng, restarts=3)[0]
        rs = []
        for k in range(40):
            lo, hi = sorted(np.exp(rng.uniform(np.log(0.05), np.log(8), 2)))
            rlo, rhi = sorted(np.exp(rng.uniform(np.log(0.005), np.log(3), 2)))
            rs.append(maximise(Scorer(Vu[idx], fg=np.geomspace(lo, max(hi, lo * 1.05), 12),
                                      rg=np.geomspace(rlo, max(rhi, rlo * 1.05), 10)),
                               rng, restarts=3)[0])
        ur.append({'real': rv, 'rank_frac_rulers_beating': float(np.mean(np.array(rs) >= rv))})
        print('ur', ur[-1], flush=True)
    out = {'pe_real_max': real, 'rulers': rul,
           'pe_frac_rulers_beating': float(np.mean([x['max'] >= np.mean(real) for x in rul])),
           'pe_rank': int(1 + sum(x['max'] > np.mean(real) for x in rul)),
           'ur3': ur}
    dump(out, os.path.join(DATA, 'pe20_cycle2_C.json'))
    print('C pe frac rulers beating', out['pe_frac_rulers_beating'], 'rank', out['pe_rank'])


if __name__ == '__main__':
    if sys.argv[1] == 'AB':
        part_AB()
    else:
        part_C()
