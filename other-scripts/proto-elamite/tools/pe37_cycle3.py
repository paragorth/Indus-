"""pe37 cycle 3: a same-period outside control, and the bones as a test of the pe20 guesses.
  R  bone-predicted ratios (living herd): young per adult female, adult males per adult
     female, from herd-demography ABC on Malyan Banesh (for PE) and Malyan Kaftari
     (c. 2400-1600 BCE, contemporary with Ur III; for the Ur III control).
  T  every ordered triple (F, Y, M) of herd signs scored by the bone ratio densities
     (pooled over records where all three are written); is the true Ur III triple
     (u8, sila4, udu-nita2) / (ud5, masz2, masz2-nita2) near the top with the
     Kaftari ruler? Where do PE triples fall, and pe20's (M362, M367, M006)?
  K  Ur III totals, full 7^7 search with the Kaftari ruler (sheep, young, adF):
     a real outside-fauna version of the cycle 1 oracle control.
"""
import itertools
from pe37_common import *
import pe20_common as P20

rng = np.random.default_rng(3737)
rows = malyan_rows()
out = {}
dens = {}
for lab, ed in (('banesh', '-3400.0'), ('kaftari', '-2400.0')):
    bc = bone_counts(rows, ed)
    r = demography_abc(bc, n=int(sys.argv[1]) if len(sys.argv) > 1 else 150000, seed=11)
    y, a, w = r['young'], r['adultF'], r['w']
    ypf = y / (a * (1 - y))
    mpf = (1 - a) / a
    # resample by weight, fit log-normal densities to the ratios
    idx = rng.choice(len(w), 20000, p=w)
    ly, lm = np.log(ypf[idx]), np.log(mpf[idx])
    dens[lab] = {'ly': (float(ly.mean()), float(ly.std())), 'lm': (float(lm.mean()), float(lm.std())),
                 'young_per_F_q': np.quantile(ypf[idx], [0.05, 0.5, 0.95]).tolist(),
                 'males_per_F_q': np.quantile(mpf[idx], [0.05, 0.5, 0.95]).tolist()}
    print(lab, dens[lab], flush=True)
out['ratios'] = dens


def lnorm(x, mu, sd):
    return stats.norm.logpdf(np.log(x), mu, sd)


def triples(recs, signs, d):
    V = P20.to_matrix(recs, signs)
    res = []
    for f, yv, m in itertools.permutations(range(len(signs)), 3):
        w = ~np.isnan(V[:, f]) & ~np.isnan(V[:, yv]) & ~np.isnan(V[:, m])
        if w.sum() < 3:
            continue
        F, Y, M = V[w, f].sum(), V[w, yv].sum(), V[w, m].sum()
        if F <= 0 or Y <= 0 or M <= 0:
            continue
        s = lnorm(Y / F, *d['ly']) + lnorm(M / F, *d['lm'])
        res.append((float(s), signs[f], signs[yv], signs[m], float(Y / F), float(M / F), int(w.sum())))
    res.sort(key=lambda x: -x[0])
    return res


ur = P20.ur_herd_records()
US = [s for s in P20.UR_SIGNS if s != 'asz2-gar3']
tU = triples(ur, US, dens['kaftari'])
truth = {('u8', 'sila4', 'udu-nita2'), ('u8', 'kir11', 'udu-nita2'), ('ud5', 'masz2', 'masz2-nita2')}
ranks = {t: next((i for i, x in enumerate(tU) if (x[1], x[2], x[3]) == t), None) for t in truth}
out['T_ur3'] = {'n': len(tU), 'top10': tU[:10], 'truth_ranks': {'|'.join(k): v for k, v in ranks.items()},
                'truth_rows': [x for x in tU if (x[1], x[2], x[3]) in truth]}
print('T ur3 n', len(tU), 'truth ranks', ranks, flush=True)
for x in tU[:8]:
    print('   ', x)
for x in out['T_ur3']['truth_rows']:
    print('  truth', x)
pe = P20.pe_records()
tP = triples(pe, P20.PE_SIGNS, dens['banesh'])
k = next((i for i, x in enumerate(tP) if (x[1], x[2], x[3]) == ('M362', 'M367', 'M006')), None)
out['T_pe'] = {'n': len(tP), 'top10': tP[:10], 'pe20_rank': k,
               'pe20_row': tP[k] if k is not None else None}
print('T pe n', len(tP), 'pe20 triple rank', k, tP[k] if k is not None else None, flush=True)
for x in tP[:8]:
    print('   ', x)
# predictive check of pe20 ratios against bone 90% intervals
out['pe20_vs_bone'] = {'young_per_F_bone90': dens['banesh']['young_per_F_q'],
                       'males_per_F_bone90': dens['banesh']['males_per_F_q']}
out['ur3_true_vs_kaftari'] = {'young_per_F_bone90': dens['kaftari']['young_per_F_q'],
                              'males_per_F_bone90': dens['kaftari']['males_per_F_q']}

# K: Ur III totals with the Kaftari ruler
J = json.load(open(TARGETS))['kaftari_all']
tg = {'sheep': beta_from(*J['sheep_share_nisp']), 'young': beta_from(*J['living_young_share']),
      'adF': beta_from(*J['living_adultF_share'])}
TRU = np.array([1, 2, 3, 3, 4, 5, 6])
t = np.nansum(P20.to_matrix(ur, US), 0)
A, sc = enumerate_scores(t, tg, A=all_assignments(7, range(7)))
M, w = posterior_marginals(A, sc, 7)
best = A[np.nanargmax(np.where(np.isfinite(sc), sc, -1e18))]
# age/sex only (species share dropped: Mesopotamian flocks need not match Fars bones)
A2, sc2 = enumerate_scores(t, tg, A=A, use=('young', 'adF'))
best2 = A2[np.nanargmax(np.where(np.isfinite(sc2), sc2, -1e18))]
M2, _ = posterior_marginals(A2, sc2, 7)
out['K_ur3_kaftari'] = {'map': [CLASSES[c] for c in best], 'map_exact': int((best == TRU).sum()),
                        'post_exact': float(sum(M[j, TRU[j]] for j in range(7))),
                        'map_agesex': [CLASSES[c] for c in best2], 'map_agesex_exact': int((best2 == TRU).sum()),
                        'post_agesex_exact': float(sum(M2[j, TRU[j]] for j in range(7))),
                        'n_within2': int(np.sum(sc > np.nanmax(sc[np.isfinite(sc)]) - 2)),
                        'chance_exact': 1.0}
print('K', out['K_ur3_kaftari'], flush=True)
dump(out, os.path.join(DATA, 'pe37_cycle3.json'))
