"""S-DARK-14.3: (c) do a workshop's clients share text? Within-site pairs of seals (>=3 signs, complete):
mean Jaccard of sign sets and share of pairs with identical middles (signs between first and last),
within-group minus across-group. Groups: physical clusters (cycle-1 method, k=6) and emblem STYLE
codes (letter after the colon in 'symbol', e.g. Bull1:W/J/S/I/L/U; unicorn seals only).
Also: do STYLE codes carry their own vocabulary / closer / length (null: style permuted within
site x type x coarse emblem, so only the style letter moves) and do they map onto find areas?
Levels raw/strong/all. Replication MD / Harappa / other."""
import sys, collections, time
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop14 import *

t0 = time.time()
seals, nseal = load()
groups = city_groups(seals)
lines = [f'LOOP 14 (workshop signatures) cycle3  seals joined {len(seals)}/{nseal}  nperm={NPERM}']
rows = []


def share_stats(lab, sets, mids):
    """(within - across) for mean Jaccard and identical-middle share."""
    lab = np.asarray(lab); n = len(lab)
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(n, 1)
    w = same[iu]
    jac = JAC[iu]; eq = EQ[iu]
    if w.sum() == 0 or (~w).sum() == 0:
        return 0.0, 0.0
    return jac[w].mean() - jac[~w].mean(), eq[w].mean() - eq[~w].mean()


for city in ['Mohenjo-daro', 'Harappa', 'OTHER']:
    cs = [seals[i] for i in groups[city]]
    keep, lab = cluster_city(cs, 6)
    sub = [cs[i] for i in keep]; lab = np.asarray(lab)
    for level in LEVELS:
        seqs = [tuple(r['_c'][level]) for r in sub]
        ok = np.array([len(s) >= 3 and 0 not in s and r['complete'] == 'Y' for s, r in zip(seqs, sub)])
        S = [s for s, o in zip(seqs, ok) if o]; R = [r for r, o in zip(sub, ok) if o]; L = lab[ok]
        n = len(S)
        sets = [set(s) for s in S]; mids = [middles(s) for s in S]
        JAC = np.zeros((n, n)); EQ = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(i + 1, n):
                JAC[i, j] = JAC[j, i] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                EQ[i, j] = EQ[j, i] = mids[i] == mids[j] and len(mids[i]) > 0
        strata = np.array([r['site'] + '|' + r['type'] for r in R])
        strata_e = np.array([r['site'] + '|' + r['type'] + '|' + (r['symbol'].split(':')[0] if r['symbol'] not in ('-', '') else 'none') for r in R])
        for nm_, st in (('', strata), ('/E', strata_e)):
            obs, p, z, nmean = perm_p(lambda x: share_stats(x, sets, mids)[0], L, st, NPERM); rows.append((city, level, 'cluster: Jaccard within-across' + nm_, obs, nmean, z, p))
            obs, p, z, nmean = perm_p(lambda x: share_stats(x, sets, mids)[1], L, st, NPERM); rows.append((city, level, 'cluster: same-middle within-across' + nm_, obs, nmean, z, p))
        # style codes on unicorn (Bull1) seals
        uni = np.array([r['symbol'].startswith('Bull1:') for r in R])
        if uni.sum() >= 40:
            style = np.array([r['symbol'].split(':')[1] if ':' in r['symbol'] else '' for r in R])[uni]
            st = strata[uni]
            idx = np.where(uni)[0]
            JACu = JAC[np.ix_(idx, idx)]; EQu = EQ[np.ix_(idx, idx)]
            def ss(x, which):
                x = np.asarray(x); same = x[:, None] == x[None, :]; iu = np.triu_indices(len(x), 1); w = same[iu]
                m = (JACu if which == 0 else EQu)[iu]
                return m[w].mean() - m[~w].mean() if w.sum() and (~w).sum() else 0.0
            obs, p, z, nmean = perm_p(lambda x: ss(x, 0), style, st, NPERM); rows.append((city, level, 'style: Jaccard within-across', obs, nmean, z, p))
            obs, p, z, nmean = perm_p(lambda x: ss(x, 1), style, st, NPERM); rows.append((city, level, 'style: same-middle within-across', obs, nmean, z, p))
            setsu = [sets[i] for i in idx]
            freq = collections.Counter(x for s in setsu for x in s); signs = [s for s, c in freq.most_common(40) if c >= 5]
            obs, p, z, nmean = perm_p(lambda x: gstat_vocab(x, setsu, signs), style, st, NPERM); rows.append((city, level, 'style: vocab-G', obs, nmean, z, p))
            clo = [S[i][-1] for i in idx]
            obs, p, z, nmean = perm_p(lambda x: mi(list(x), clo), style, st, NPERM); rows.append((city, level, 'style: closer-MI', obs, nmean, z, p))
            ln = [len(S[i]) for i in idx]
            obs, p, z, nmean = perm_p(lambda x: kruskal(x, ln), style, st, NPERM); rows.append((city, level, 'style: length-KW', obs, nmean, z, p))
            if level == 'seq_raw':
                lines.append(f'\n{city} unicorn seals with style code, complete >=3 signs: {len(idx)}; styles {dict(collections.Counter(style).most_common(8))}')
                area = np.array([R[i]['area-section'] for i in idx]); ka = area != '--'
                if ka.sum() >= 30:
                    obs, p, z, nmean = perm_p(lambda x: mi(list(x), list(area[ka])), style[ka], st[ka], NPERM); rows.append((city, '-', 'style: area-MI', obs, nmean, z, p))
                # style x closer table for the record
                tab = collections.Counter(zip(style, clo))
                for s_ in sorted(set(style)):
                    cc = collections.Counter({c: tab[(s_, c)] for c in set(clo) if tab[(s_, c)]})
                    lines.append(f'   style {s_} n={sum(cc.values())} closers ' + ' '.join(f'{c}x{k}' for c, k in cc.most_common(5)))

lines.append('\nRESULTS (obs, null mean, z, raw P):')
for r in rows:
    lines.append(f'  {r[0]:13s} {r[1]:10s} {r[2]:36s} obs={r[3]:.4f} null={r[4]:.4f} z={r[5]:+.2f} P={r[6]:.4f}')
arrows = collections.defaultdict(list)
for r in rows:
    arrows[(r[0], r[2])].append(r[6])
keys = sorted(arrows); ps = [max(arrows[k]) for k in keys]; adj = holm(ps)
lines.append(f'\nARROWS FIRED: {len(keys)} (city x statistic; P = worst level); Holm-adjusted:')
for k, p, a in zip(keys, ps, adj):
    lines.append(f'  {k[0]:13s} {k[1]:36s} P={p:.4f} Holm={a:.4f} {"**" if a < 0.05 else ""}')
lines.append(f'\n{time.time()-t0:.0f}s')
open(OUT + 'loop14_cycle3.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
