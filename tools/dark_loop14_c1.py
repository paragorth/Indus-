"""S-DARK-14.1: physical-only workshop clusters (boss, material, colour, shape, cross-section,
log h/v/th; NO emblem, NO text) within Mohenjo-daro, Harappa and the pooled other sites.
Tests (a) vocabulary G, closer MI, length Kruskal; (d) find-area MI. Null: cluster labels permuted
within site x object type (so cluster sizes and type mix are kept). k = 4, 6, 8. Levels raw/strong/all.
Holm correction over all arrows fired (city x k x statistic), levels treated as robustness, not arrows."""
import sys, collections, time
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop14 import *

t0 = time.time()
seals, nseal = load()
groups = city_groups(seals)
lines = [f'LOOP 14 (workshop signatures) cycle1  seals joined {len(seals)}/{nseal}  nperm={NPERM}',
         'Clusters: Gower distance on boss/material/colour/shape/cross-section/log(h,v,th), Ward linkage, '
         'seals with >=3 known features. Null: labels permuted within site x object type.']
rows = []   # (city, k, stat, level, obs, nullmean, z, p)
for city in ['Mohenjo-daro', 'Harappa', 'OTHER']:
    ix = groups[city]
    cs = [seals[i] for i in ix]
    for k in (4, 6, 8):
        keep, lab = cluster_city(cs, k)
        if keep is None:
            lines.append(f'{city}: too few seals'); continue
        sub = [cs[i] for i in keep]
        strata = [r['site'] + '|' + r['type'] for r in sub]
        # second null: also hold the coarse emblem class fixed (emblem-text links are known, S326/S329)
        strata_e = [r['site'] + '|' + r['type'] + '|' + (r['symbol'].split(':')[0] if r['symbol'] not in ('-', '') else 'none') for r in sub]
        if k == 6:
            lines.append(f'\n{city} k={k} n={len(sub)} (of {len(cs)})\n' + describe_clusters(cs, keep, lab))
        # (d) find area, text independent, one per k
        area = [r['area-section'] for r in sub]
        known = np.array([a != '--' for a in area])
        if known.sum() >= 30 and len(set(np.array(area)[known])) > 1:
            la = np.asarray(lab)[known]; ar = np.array(area)[known]; st = np.array(strata)[known]
            obs, p, z, nm = perm_p(lambda L: mi(list(L), list(ar)), la, st, NPERM)
            rows.append((city, k, 'area-MI', '-', obs, nm, z, p))
        for level in LEVELS:
          for nullname, strat in (('', strata), ('/E', strata_e)):
            seqs = [tuple(r['_c'][level]) for r in sub]
            ok = np.array([len(s) >= 1 and 0 not in s for s in seqs])
            L0 = np.asarray(lab)[ok]; S0 = [s for s, o in zip(seqs, ok) if o]; st = np.array(strat)[ok]
            sets = [set(s) for s in S0]
            freq = collections.Counter(x for s in sets for x in s)
            signs = [s for s, c in freq.most_common(40) if c >= 5]
            obs, p, z, nm = perm_p(lambda L: gstat_vocab(L, sets, signs), L0, st, NPERM)
            rows.append((city, k, 'vocab-G' + nullname, level, obs, nm, z, p))
            # closer = last sign of complete texts (R/L reading already applied in canonical)
            comp = np.array([r['complete'] == 'Y' for r, o in zip(sub, ok) if o])
            if comp.sum() >= 30:
                clo = [s[-1] for s, c in zip(S0, comp) if c]
                obs, p, z, nm = perm_p(lambda L: mi(list(L), clo), L0[comp], st[comp], NPERM)
                rows.append((city, k, 'closer-MI' + nullname, level, obs, nm, z, p))
                ln = [len(s) for s, c in zip(S0, comp) if c]
                obs, p, z, nm = perm_p(lambda L: kruskal(L, ln), L0[comp], st[comp], NPERM)
                rows.append((city, k, 'length-KW' + nullname, level, obs, nm, z, p))

lines.append('\nRESULTS (obs, null mean, z, raw P; levels raw/strong/all shown separately):')
for r in rows:
    lines.append(f'  {r[0]:13s} k={r[1]} {r[2]:12s} {r[3]:10s} obs={r[4]:.3f} null={r[5]:.3f} z={r[6]:+.2f} P={r[7]:.4f}')
# arrows: city x k x statistic; P for an arrow = max over levels (conservative: must hold at all levels)
arrows = collections.defaultdict(list)
for r in rows:
    arrows[(r[0], r[1], r[2])].append(r[7])
keys = sorted(arrows); ps = [max(arrows[k]) for k in keys]
adj = holm(ps)
lines.append(f'\nARROWS FIRED: {len(keys)} (city x k x statistic); P per arrow = worst level; Holm-adjusted:')
for k, p, a in zip(keys, ps, adj):
    lines.append(f'  {k[0]:13s} k={k[1]} {k[2]:12s} P={p:.4f}  Holm={a:.4f} {"**" if a < 0.05 else ""}')
# replication across cities per statistic (k=6)
lines.append('\nHELD-OUT REPLICATION (k=6, worst level P): MD -> Harappa -> other sites')
for stat in ['vocab-G', 'closer-MI', 'length-KW', 'vocab-G/E', 'closer-MI/E', 'length-KW/E', 'area-MI']:
    s = '  ' + stat + ': ' + '  '.join(f'{c}={max(arrows.get((c, 6, stat), [float("nan")])):.3f}' for c in ['Mohenjo-daro', 'Harappa', 'OTHER'])
    lines.append(s)
lines.append(f'\n{time.time()-t0:.0f}s')
open(OUT + 'loop14_cycle1.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
