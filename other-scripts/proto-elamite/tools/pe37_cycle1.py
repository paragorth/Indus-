"""pe37 cycle 1: exhaustive bone-ruler search on herd-sign totals.
  A Ur III oracle control (7 signs with known classes; target = their own true
    composition, with the PE bone-target widths) + biased-oracle variant
  B planted PE-shaped totals with known classes
  C PE: 8 herd signs (pe20 set), Malyan Banesh target, all 7^8 assignments
  D null: random faunal profiles (same widths) -> how special is the real fit?
"""
import time
from pe37_common import *
import pe20_common as P20

rng = np.random.default_rng(37)
TG = json.load(open(TARGETS))['banesh_all']
SD = {'sheep': TG['sheep_share_nisp'][1], 'young': TG['living_young_share'][1],
      'adF': TG['living_adultF_share'][1]}
REAL = {'sheep': TG['sheep_share_nisp'][0], 'young': TG['living_young_share'][0],
        'adF': TG['living_adultF_share'][0]}


def target_of(means, sd=SD):
    return {k: beta_from(means[k], sd[k]) for k in means}


def truth_props(t, truth):
    S = class_sums(np.array([truth]), np.asarray(t, float))
    p = proportions(S)
    return {'sheep': float(p[0][0]), 'young': float(p[1][0]), 'adF': float(p[2][0])}


def recover(t, truth, target, A=None):
    A, sc = enumerate_scores(t, target, A=A)
    K = len(t)
    M, w = posterior_marginals(A, sc, K)
    best = A[np.nanargmax(sc)]
    truth = np.array(truth)
    exact = int((best == truth).sum())
    post_exact = float(sum(M[j, truth[j]] for j in range(K)))
    st = sc[np.all(A == truth, 1)][0]
    rank = float(np.mean(sc[np.isfinite(sc)] > st))
    near = int(np.sum(sc > np.nanmax(sc) - 2))
    return {'map': best.tolist(), 'map_exact': exact, 'post_exact': post_exact,
            'truth_rank_frac': rank, 'n_within_2': near, 'n_valid': int(np.isfinite(sc).sum()),
            'max': float(np.nanmax(sc[np.isfinite(sc)])), 'truth_score': float(st)}, A, M


def chance_exact(K, truth, n=20000):
    A = rng.integers(0, 7, size=(n, K))
    return float(np.mean((A == np.array(truth)).sum(1)))


out = {}
# ---- A: Ur III oracle
ur = P20.ur_herd_records()
USIG = [s for s in P20.UR_SIGNS if s != 'asz2-gar3']
V = P20.to_matrix(ur, USIG)
tU = np.nansum(V, 0)
TRU = [1, 2, 3, 3, 4, 5, 6]
tp = truth_props(tU, TRU)
print('Ur III totals', dict(zip(USIG, tU.tolist())), 'true props', tp, flush=True)
AU = all_assignments(7, range(7))
r, _, M = recover(tU, TRU, target_of(tp), A=AU)
r['chance_exact'] = chance_exact(7, TRU)
r['marg_true'] = {s: float(M[j, TRU[j]]) for j, s in enumerate(USIG)}
r['map_named'] = {s: CLASSES[c] for s, c in zip(USIG, r['map'])}
out['A_ur3_oracle'] = r
print('A oracle', r, flush=True)
# biased oracle: true means shifted by one target sd in random directions (20 draws)
bo = []
for i in range(20):
    m = {k: float(np.clip(tp[k] + rng.normal(0, SD[k]), 0.02, 0.98)) for k in tp}
    rr, _, _ = recover(tU, TRU, target_of(m), A=AU)
    bo.append(rr['map_exact'])
out['A_ur3_biased_map_exact'] = bo
print('A biased', bo, np.mean(bo), flush=True)
# Ur III subsamples of PE size (24 tablets)
pids = sorted({x[0] for x in ur})
sub = []
for i in range(20):
    keep = set(rng.choice(pids, 24, replace=False))
    tt = np.nansum(P20.to_matrix([x for x in ur if x[0] in keep], USIG), 0)
    rr, _, _ = recover(tt, TRU, target_of(tp), A=AU)
    sub.append(rr['map_exact'])
out['A_ur3_sub24_map_exact'] = sub
print('A sub24', sub, np.mean(sub), flush=True)

# ---- C: PE
pe = P20.pe_records()
tP = np.nansum(P20.to_matrix(pe, P20.PE_SIGNS), 0)
print('PE totals', dict(zip(P20.PE_SIGNS, tP.tolist())), flush=True)
t0 = time.time()
AP = all_assignments(8, range(7))
AP, scP = enumerate_scores(tP, target_of(REAL), A=AP)
MP, wP = posterior_marginals(AP, scP, 8)
order = np.argsort(-np.where(np.isfinite(scP), scP, -1e9))[:15]
out['C_pe'] = {'max': float(scP[order[0]]), 'top': [([CLASSES[c] for c in AP[i]], float(scP[i])) for i in order],
               'n_within_2': int(np.sum(scP > scP[order[0]] - 2)), 'n_valid': int(np.isfinite(scP).sum()),
               'marginals': {s: {CLASSES[c]: float(MP[j, c]) for c in range(7)} for j, s in enumerate(P20.PE_SIGNS)},
               'map_post_mass': float(wP.max())}
print('C pe max %.2f within2 %d (%.0fs)' % (scP[order[0]], out['C_pe']['n_within_2'], time.time() - t0), flush=True)
for s, d in out['C_pe']['marginals'].items():
    print('  ', s, ' '.join('%s %.2f' % (k, v) for k, v in d.items()))

# ---- B: planted PE-shaped totals
pl = []
for i in range(20):
    truth = rng.integers(1, 7, size=8)
    truth[rng.choice(8, 2, replace=False)] = 0
    # draw totals: class composition near target means, split among signs of the class
    cap = 1500.0
    sh = REAL['sheep']; yg = REAL['young']; af = REAL['adF']
    ccomp = {1: sh * (1 - yg) * af, 2: sh * (1 - yg) * (1 - af), 3: sh * yg,
             4: (1 - sh) * (1 - yg) * af, 5: (1 - sh) * (1 - yg) * (1 - af), 6: (1 - sh) * yg}
    t = np.zeros(8)
    for c, v in ccomp.items():
        js = np.where(truth == c)[0]
        if len(js):
            t[js] = rng.multinomial(int(cap * v), rng.dirichlet(np.ones(len(js))))
    t[truth == 0] = rng.integers(5, 300, size=(truth == 0).sum())
    rr, _, _ = recover(t, truth.tolist(), target_of(REAL), A=AP)
    rr['chance'] = chance_exact(8, truth.tolist(), 5000)
    pl.append(rr)
out['B_plant'] = pl
print('B plant map_exact', [x['map_exact'] for x in pl], 'mean %.2f; post_exact mean %.2f; chance %.2f; truth rank mean %.3f'
      % (np.mean([x['map_exact'] for x in pl]), np.mean([x['post_exact'] for x in pl]),
         np.mean([x['chance'] for x in pl]), np.mean([x['truth_rank_frac'] for x in pl])), flush=True)

# ---- D: random faunal profiles
nul = []
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    m = {'sheep': rng.uniform(0.05, 0.95), 'young': rng.uniform(0.1, 0.5), 'adF': rng.uniform(0.4, 0.95)}
    _, s = enumerate_scores(tP, target_of(m), A=AP)
    Mx, w = posterior_marginals(AP, s, 8)
    nul.append({'max': float(np.nanmax(s[np.isfinite(s)])), 'mapmass': float(w.max()),
                'marg_max_mean': float(Mx.max(1).mean())})
real_mm = float(MP.max(1).mean())
out['D_null'] = {'n': len(nul), 'p_max_ge_real': float(np.mean([x['max'] >= out['C_pe']['max'] for x in nul])),
                 'null_max_median': float(np.median([x['max'] for x in nul])),
                 'real_marg_conc': real_mm,
                 'p_conc_ge_real': float(np.mean([x['marg_max_mean'] >= real_mm for x in nul]))}
print('D null', out['D_null'], flush=True)
dump(out, os.path.join(DATA, 'pe37_cycle1.json'))
