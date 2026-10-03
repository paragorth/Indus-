"""S-DARK-14.2: do allograph VARIANTS track workshops?
Variants = raw Wells form vs the form it merges into (sign_allographs_levels.json, strong+probable);
a seal 'votes' for a variant when its seq_raw contains one of the forms. By construction this uses
seq_raw (seq_strong/seq_all erase the distinction). Statistic: sum over variant sets of the G statistic
of variant x grouping. Groupings: (1) physical workshop cluster (cycle-1 method, k=6, no emblem),
null = cluster labels permuted within site x type (and /E: also within coarse emblem class);
(2) emblem style code (Bull1:W/J/S/I/L/U..., the letter after the colon), null = style permuted within
site x type; (3) site, null = site permuted within type. Replication: MD / Harappa / other sites."""
import sys, collections, time
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop14 import *

t0 = time.time()
seals, nseal = load()
merges = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']
vsets = collections.defaultdict(set)
for x in merges:
    if x['level'] in ('strong', 'probable'):
        vsets[x['into']].add(x['form']); vsets[x['into']].add(x['into'])
vsets = {k: sorted(v) for k, v in vsets.items()}


def votes(r):
    s = set(r['_c']['seq_raw'])
    out = {}
    for into, forms in vsets.items():
        hit = [f for f in forms if f in s]
        if len(hit) == 1:
            out[into] = hit[0]
    return out


def gsum(lab, vote_list, min_n=10):
    """sum of G statistics (variant x label) over variant sets with >= min_n voters and >1 variant."""
    lab = np.asarray(lab); total = 0.0; used = 0
    for into in vsets:
        ix = [i for i, v in enumerate(vote_list) if into in v]
        if len(ix) < min_n:
            continue
        vv = [vote_list[i][into] for i in ix]
        if len(set(vv)) < 2:
            continue
        ll = lab[ix]
        n = len(ix); ca = collections.Counter(ll); cb = collections.Counter(vv); cab = collections.Counter(zip(ll, vv))
        total += sum(2 * c * math.log(c * n / (ca[a] * cb[b])) for (a, b), c in cab.items())
        used += 1
    return total


lines = [f'LOOP 14 (workshop signatures) cycle2  seals joined {len(seals)}/{nseal}  nperm={NPERM}  level=seq_raw (variants only exist there)',
         f'variant sets (strong+probable merges): {len(vsets)}; statistic = sum of G(variant x grouping) over sets with >=10 voters']
rows = []
groups = city_groups(seals)
for city in ['Mohenjo-daro', 'Harappa', 'OTHER']:
    cs = [seals[i] for i in groups[city]]
    keep, lab = cluster_city(cs, 6)
    sub = [cs[i] for i in keep]
    vl = [votes(r) for r in sub]
    has = np.array([len(v) > 0 for v in vl])
    sub = [r for r, h in zip(sub, has) if h]; vl = [v for v, h in zip(vl, has) if h]; lab = np.asarray(lab)[has]
    strata = np.array([r['site'] + '|' + r['type'] for r in sub])
    strata_e = np.array([r['site'] + '|' + r['type'] + '|' + (r['symbol'].split(':')[0] if r['symbol'] not in ('-', '') else 'none') for r in sub])
    nsets = sum(1 for into in vsets if sum(into in v for v in vl) >= 10 and len({v[into] for v in vl if into in v}) > 1)
    lines.append(f'\n{city}: {len(sub)} seals carry a variant-bearing sign; {nsets} variant sets testable; clusters sizes {sorted(collections.Counter(lab).values(), reverse=True)}')
    # per-set table for the record (k=6 cluster x variant)
    for into in vsets:
        ix = [i for i, v in enumerate(vl) if into in v]
        if len(ix) >= 10 and len({vl[i][into] for i in ix}) > 1:
            tab = collections.Counter((int(lab[i]), vl[i][into]) for i in ix)
            lines.append(f'   W{into}: ' + '  '.join(f'c{c}:' + '/'.join(f'{f}x{tab[(c, f)]}' for f in vsets[into] if tab[(c, f)]) for c in sorted({c for c, _ in tab})))
    obs, p, z, nm = perm_p(lambda L: gsum(L, vl), lab, strata, NPERM); rows.append((city, 'variant~cluster', obs, nm, z, p))
    obs, p, z, nm = perm_p(lambda L: gsum(L, vl), lab, strata_e, NPERM); rows.append((city, 'variant~cluster/E', obs, nm, z, p))
    # (2) emblem style code
    style = np.array([r['symbol'].split(':')[1] if ':' in r['symbol'] else ('none' if r['symbol'] in ('None', '-', '') else 'plain') for r in sub])
    stl = np.array([s not in ('none',) for s in style])
    if stl.sum() >= 30:
        obs, p, z, nm = perm_p(lambda L: gsum(L, [vl[i] for i in np.where(stl)[0]]), style[stl], strata[stl], NPERM)
        rows.append((city, 'variant~style', obs, nm, z, p))
        tab = collections.Counter()
        for i in np.where(stl)[0]:
            for into, f in vl[i].items():
                if into in (390, 705, 803, 861):
                    tab[(into, style[i], f)] += 1
        for into in (390, 705, 803, 861):
            lines.append(f'   style x W{into}: ' + '  '.join(f'{s}:' + '/'.join(f'{f}x{tab[(into, s, f)]}' for f in vsets[into] if tab[(into, s, f)]) for s in sorted({s for (i, s, f) in tab if i == into})))
    # (4) find area
    area = np.array([r['area-section'] for r in sub]); ka = area != '--'
    if ka.sum() >= 30 and len(set(area[ka])) > 1:
        obs, p, z, nm = perm_p(lambda L: gsum(L, [vl[i] for i in np.where(ka)[0]]), area[ka], strata[ka], NPERM)
        rows.append((city, 'variant~area', obs, nm, z, p))

# (3) site effect, all seals
vl = [votes(r) for r in seals]
has = [i for i, v in enumerate(vl) if v]
site = np.array([seals[i]['site'] if seals[i]['site'] in BIG else 'OTHER' for i in has])
typ = np.array([seals[i]['type'] for i in has])
obs, p, z, nm = perm_p(lambda L: gsum(L, [vl[i] for i in has]), site, typ, NPERM); rows.append(('ALL', 'variant~site', obs, nm, z, p))
# site effect within physical clusters? (is site signal just the cluster signal) -- site permuted within type x emblem class
emb = np.array([typ[j] + '|' + (seals[i]['symbol'].split(':')[0] if seals[i]['symbol'] not in ('-', '') else 'none') for j, i in enumerate(has)])
obs, p, z, nm = perm_p(lambda L: gsum(L, [vl[i] for i in has]), site, emb, NPERM); rows.append(('ALL', 'variant~site/E', obs, nm, z, p))

lines.append('\nRESULTS (G sum obs, null mean, z, raw P):')
for r in rows:
    lines.append(f'  {r[0]:13s} {r[1]:18s} obs={r[2]:.1f} null={r[3]:.1f} z={r[4]:+.2f} P={r[5]:.4f}')
ps = [r[5] for r in rows]; adj = holm(ps)
lines.append(f'\nARROWS FIRED: {len(rows)}; Holm-adjusted:')
for r, a in zip(rows, adj):
    lines.append(f'  {r[0]:13s} {r[1]:18s} P={r[5]:.4f} Holm={a:.4f} {"**" if a < 0.05 else ""}')
lines.append(f'\n{time.time()-t0:.0f}s')
open(OUT + 'loop14_cycle2.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
