"""v17 cycle 2: the full exemplar-layout search on the real text, with search-size correction.

Part A (brute-force periodic grid): for every section (HA SB BB HB PA TB other + pooled ALL), every
channel (ghost = init+fin+rep, and each alone), both units (glyph+space width, words) and both phase
models (anchored at paragraph start, free common phase), scan all exemplar widths on the grid.
Nulls: 150 line-order shuffles, 150 within-line word shuffles, 150 Markov resyntheses (composed
directly). Per-combination p from the null max; family-wise p = rank of the observed min-p among
the null replicates' min-p (each null replicate scored against the other replicates).

Part B (flexible grid, exact DP = the 'annealing' search done exactly): exemplar line widths free
within +-15% of L, first line anchored at paragraph start; best boundary chain picked per paragraph.
Statistic z(L) = (DP(L) - mean null DP(L)) / sd, max over L; nulls 40 line-order shuffles; family
correction by max-z among held-out null replicates. Calibrated on two planted copies.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v17_lib import *
from multiprocessing import Pool

NREP = int(os.environ.get('NREP', 150)); NDP = int(os.environ.get('NDP', 40))
CH = ('ghost', 'init', 'fin', 'rep')
V = voynich_paras(); TV = crossfit_tables(V); MK = Markov(V)

def full_stats(paras):
    gt = gap_table(paras, TV); W = residual(gt)
    out = {}
    for unit in ('g', 'w'):
        for ch in CH:
            grid, res = scan(gt, W, ch, unit)
            for sec, d in summarize(grid, res).items():
                out['%s|%s|%s|A' % (unit, ch, sec)] = (d['Amax'], d['LA'])
                if 'Rmax' in d: out['%s|%s|%s|R' % (unit, ch, sec)] = (d['Rmax'], d['LR'])
    return out

def jobA(args):
    kind, seed = args; rng = random.Random(seed)
    p = V if kind == 'obs' else null_lshuf(V, rng) if kind == 'lshuf' else null_wshuf(V, rng) if kind == 'wshuf' else MK.gen(V, rng)
    return kind, seed, full_stats(p)

def dp_all(paras, tabs):
    gt = gap_table(paras, tabs); W = residual(gt)
    grid, res = dp_scan(gt, W, 'ghost', 'g')
    return grid, {k: v.tolist() for k, v in res.items()}

def jobB(args):
    name, seed = args
    paras, tabs = DPSETS[name]
    p = paras if seed < 0 else null_lshuf(paras, random.Random(seed))
    return name, seed, dp_all(p, tabs)

init_pool = [ws[0] for p in V for ws in p['lines'][1:]]
fin_pool = [ws[-1] for p in V for ws in p['lines'][:-1]]
DPSETS = {'VOYNICH': (V, TV),
          'PC_voy_ex_w1.30': (copy_from_exemplar(V, random.Random(1), width_factor=1.3, p_ditto=0.05, p_skip=0.05)[0], TV),
          'PC_voy_fixed32': (copy_from_exemplar(V, random.Random(2), width_factor=1.4, fixed_width=32, plant_init=init_pool,
                                                plant_fin=fin_pool, p_ditto=0.03, p_skip=0.03)[0], TV)}

def main():
    t0 = time.time()
    ck = load('cycle2A.json')
    if ck is None:
        jobs = [('obs', 0)] + [(k, 5000 + i) for k in ('lshuf', 'wshuf', 'markov') for i in range(NREP)]
        with Pool(2) as pool: R = pool.map(jobA, jobs, chunksize=4)
        ck = {'obs': None, 'lshuf': [], 'wshuf': [], 'markov': []}
        for kind, seed, s in R:
            if kind == 'obs': ck['obs'] = s
            else: ck[kind].append(s)
        save('cycle2A.json', ck)
    print('A done %.0fs' % (time.time() - t0), flush=True)
    obs = ck['obs']; keys = sorted(obs)
    summ = {}
    for kind in ('lshuf', 'wshuf', 'markov'):
        N = ck[kind]
        nv = {k: np.array([n[k][0] for n in N if k in n]) for k in keys}
        pk = {k: (1 + (nv[k] >= obs[k][0]).sum()) / (1 + len(nv[k])) for k in keys}
        # family-wise: null replicate i's min p against the other replicates
        mins = []
        for i, n in enumerate(N):
            ps = []
            for k in keys:
                if k not in n: continue
                v = nv[k]; ps.append((1 + (np.delete(v, i) >= n[k][0]).sum()) / len(v))
            mins.append(min(ps))
        omin = min(pk.values()); fw = (1 + sum(m <= omin for m in mins)) / (1 + len(mins))
        best = sorted(keys, key=lambda k: pk[k])[:8]
        summ[kind] = {'min_p': omin, 'familywise_p': fw, 'n_combos': len(keys),
                      'best': [(k, obs[k][0], obs[k][1], pk[k]) for k in best],
                      'ALL_ghost': {k: (obs[k][0], obs[k][1], pk[k]) for k in keys if '|ghost|ALL|' in k}}
        print(kind, json.dumps(summ[kind]), flush=True)
    out = {'A': summ}
    # ---- Part B ----
    ckb = load('cycle2B.json')
    if ckb is None:
        jobs = [(nm, -1) for nm in DPSETS] + [(nm, 7000 + i) for nm in DPSETS for i in range(NDP)]
        with Pool(2) as pool: R = pool.map(jobB, jobs, chunksize=2)
        ckb = defaultdict(lambda: {'obs': None, 'null': []})
        for nm, seed, (grid, res) in R:
            ckb[nm]['grid'] = list(map(float, grid))
            if seed < 0: ckb[nm]['obs'] = res
            else: ckb[nm]['null'].append(res)
        ckb = dict(ckb); save('cycle2B.json', ckb)
    print('B done %.0fs' % (time.time() - t0), flush=True)
    outB = {}
    for nm, d in ckb.items():
        grid = np.array(d['grid']); r = {}
        for sec in d['obs']:
            N = np.array([n[sec] for n in d['null']]); mu = N.mean(0); sd = N.std(0) + 1e-9
            z = (np.array(d['obs'][sec]) - mu) / sd; zn = ((N - mu) / sd).max(1)
            i = int(np.argmax(z))
            r[sec] = {'zmax': float(z[i]), 'L': float(grid[i]), 'p_fw': float((1 + (zn >= z[i]).sum()) / (1 + len(zn))),
                      'top3': [(float(grid[j]), float(z[j])) for j in np.argsort(-z)[:3]]}
        outB[nm] = r
        print('DP', nm, json.dumps(r), flush=True)
    out['B'] = outB
    save('cycle2.json', out)
    print('done %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()
