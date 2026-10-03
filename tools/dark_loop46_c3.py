"""Loop 46 cycle 3: were identical texts cut by one hand? Variant agreement among copies of a stock text,
and between seals and the tags/tablets that repeat their text.

Copy = another object with the same merged text (seq_all, complete texts, >= 2 signs) that carries >= 1 usable class.
Statistic: share of copy pairs that agree on the form of every shared class (and per-class agreement).
Null: variant labels permuted among texts within site x type (1,000x) -> agreement expected if every copy were cut
by a random carver of that city and medium; also a within-stock-group null is impossible (that is the hypothesis).
Splits: same site vs different site; SEAL-SEAL, TAB-TAB, SEAL-TAG, SEAL-TAB; >= 3 copies (S-DARK-12 stock texts);
plus the 'Harappa package' lead of cycle 1 (405+806 seals at Mohenjo-daro).
Usage: python3 tools/dark_loop46_c3.py [level]
"""
import sys, json, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop46_common import *

LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'all'
NPERM = 1000
rng = np.random.default_rng(463)
corpus = load_corpus(); N = len(corpus)
classes = load_classes(LEVEL); use = usable_classes(corpus, classes)
labels, _, _ = text_labels(corpus, classes, use)
heads = sorted(use)
lines = [f'# loop 46 cycle 3: stock texts and seal/tag copies, same variants?  level={LEVEL}  classes={heads}  nperm={NPERM}']
coarse = lambda i: corpus[i]['type'].split(':')[0]

groups = collections.defaultdict(list)
for i, r in enumerate(corpus):
    if r['complete'] == 'Y' and len(r['seq_all']) >= 2 and labels[i]:
        groups[tuple(r['seq_all'])].append(i)
groups = {k: v for k, v in groups.items() if len(v) >= 2}
lines.append(f'stock groups (identical complete seq_all, >= 2 objects, >= 1 variant class): {len(groups)}; objects {sum(len(v) for v in groups.values())}; '
             f'groups with >= 3 copies: {sum(1 for v in groups.values() if len(v) >= 3)}')

# all copy pairs with categories
pairs = []
for k, v in groups.items():
    for a, b in itertools.combinations(v, 2):
        shared = sorted(set(labels[a]) & set(labels[b]))
        if not shared:
            continue
        cat = '-'.join(sorted([coarse(a), coarse(b)]))
        pairs.append(dict(a=a, b=b, shared=shared, cat=cat, same_site=corpus[a]['site'] == corpus[b]['site'],
                          same_area=corpus[a]['site'] == corpus[b]['site'] and corpus[a]['area-section'] == corpus[b]['area-section'] and corpus[a]['area-section'] not in ('-', '--'),
                          ncopies=len(v)))
lines.append(f'copy pairs sharing >= 1 class: {len(pairs)}; by category: {dict(collections.Counter(p["cat"] for p in pairs))}; same site {sum(p["same_site"] for p in pairs)}')

st = strata(corpus, ['site', 'type'])
cls_idx = {h: np.array([i for i in range(N) if h in labels[i]]) for h in heads}
cls_lab = {h: np.array([labels[i][h] for i in cls_idx[h]]) for h in heads}
cls_pos = {h: {t: k for k, t in enumerate(cls_idx[h].tolist())} for h in heads}
st_codes = {h: np.array([hash(st[i]) for i in cls_idx[h]]) for h in heads}


def agree_rate(lab, sel):
    if not sel:
        return float('nan'), 0
    ok = 0
    for p in sel:
        ok += all(lab[h][cls_pos[h][p['a']]] == lab[h][cls_pos[h][p['b']]] for h in p['shared'])
    return ok / len(sel), len(sel)


def test(name, sel):
    obs, n = agree_rate(cls_lab, sel)
    if n < 5:
        lines.append(f'  {name}: n={n} too few'); return
    null = np.array([agree_rate({h: permute_within(cls_lab[h], st_codes[h], rng) for h in heads}, sel)[0] for _ in range(NPERM)])
    P_hi = (np.sum(null >= obs) + 1) / (NPERM + 1); P_lo = (np.sum(null <= obs) + 1) / (NPERM + 1)
    lines.append(f'  {name}: pairs {n}, all shared forms agree {obs:.3f} vs null {null.mean():.3f}+/-{null.std():.3f}  P(>=)={P_hi:.3f} P(<=)={P_lo:.3f}')
    return dict(n=n, obs=obs, null=float(null.mean()), sd=float(null.std()), P_hi=float(P_hi), P_lo=float(P_lo))


summary = {}
lines.append('Agreement of variant forms among copies of one text (null: forms permuted within site x type):')
summary['all'] = test('all copy pairs', pairs)
summary['same_site'] = test('same site', [p for p in pairs if p['same_site']])
summary['diff_site'] = test('different site', [p for p in pairs if not p['same_site']])
summary['same_area'] = test('same site and find area', [p for p in pairs if p['same_area']])
summary['same_site_diff_area'] = test('same site, different/unknown area', [p for p in pairs if p['same_site'] and not p['same_area']])
summary['ge3'] = test('stock texts with >= 3 copies', [p for p in pairs if p['ncopies'] >= 3])
for cat in sorted(set(p['cat'] for p in pairs)):
    summary[cat] = test(cat, [p for p in pairs if p['cat'] == cat])
summary['SEAL-TAG same site'] = test('SEAL-TAG same site', [p for p in pairs if p['cat'] == 'SEAL-TAG' and p['same_site']])
summary['SEAL-TAG diff site'] = test('SEAL-TAG different site', [p for p in pairs if p['cat'] == 'SEAL-TAG' and not p['same_site']])
summary['TAB-TAB same site'] = test('TAB-TAB same site', [p for p in pairs if p['cat'] == 'TAB-TAB' and p['same_site']])
summary['SEAL-SEAL same site'] = test('SEAL-SEAL same site', [p for p in pairs if p['cat'] == 'SEAL-SEAL' and p['same_site']])
summary['SEAL-SEAL diff site'] = test('SEAL-SEAL different site', [p for p in pairs if p['cat'] == 'SEAL-SEAL' and not p['same_site']])
summary['TAB-TAB Harappa moulded'] = test('TAB-TAB Harappa both TAB:B (moulded)', [p for p in pairs if p['cat'] == 'TAB-TAB' and corpus[p['a']]['site'] == 'Harappa' and corpus[p['a']]['type'] == 'TAB:B' == corpus[p['b']]['type']])
summary['TAB-TAB Harappa incised'] = test('TAB-TAB Harappa both TAB:I (incised)', [p for p in pairs if p['cat'] == 'TAB-TAB' and corpus[p['a']]['site'] == 'Harappa' and corpus[p['a']]['type'] == 'TAB:I' == corpus[p['b']]['type']])

# per class agreement, same site
lines.append('Per class, same-site copy pairs: agree / n (null expectation = 1 - 2 r (1-r) with r the site x type non-head rate)')
for h in heads:
    sel = [p for p in pairs if p['same_site'] and h in p['shared']]
    if len(sel) < 5:
        continue
    ag = sum(cls_lab[h][cls_pos[h][p['a']]] == cls_lab[h][cls_pos[h][p['b']]] for p in sel)
    null = np.array([sum(l[cls_pos[h][p['a']]] == l[cls_pos[h][p['b']]] for p in sel) for l in (permute_within(cls_lab[h], st_codes[h], rng) for _ in range(300))])
    lines.append(f'  W{h}: {ag}/{len(sel)} agree vs null {null.mean():.1f}+/-{null.std():.1f}')

# disagreeing pairs listed
dis = [p for p in pairs if not all(cls_lab[h][cls_pos[h][p['a']]] == cls_lab[h][cls_pos[h][p['b']]] for h in p['shared'])]
lines.append(f'Disagreeing copy pairs ({len(dis)}): ' + '; '.join(
    f"{corpus[p['a']]['cisi']}({corpus[p['a']]['type']})~{corpus[p['b']]['cisi']}({corpus[p['b']]['type']}) text {'-'.join(map(str, corpus[p['a']]['seq_raw']))} vs {'-'.join(map(str, corpus[p['b']]['seq_raw']))}" for p in dis[:40]))

# tag texts: how many tag texts have a seal copy at all
tags = [i for i, r in enumerate(corpus) if coarse(i) == 'TAG' and r['complete'] == 'Y' and len(r['seq_all']) >= 2]
sealtexts = collections.defaultdict(list)
for i, r in enumerate(corpus):
    if coarse(i) == 'SEAL' and r['complete'] == 'Y':
        sealtexts[tuple(r['seq_all'])].append(i)
m = [(i, sealtexts[tuple(corpus[i]['seq_all'])]) for i in tags if tuple(corpus[i]['seq_all']) in sealtexts]
mv = [(i, s) for i, s in m if labels[i]]
lines.append(f'complete tag texts >= 2 signs: {len(tags)}; with an identical seal text: {len(m)}; of these carrying a variant class: {len(mv)}; '
             + '; '.join(f"{corpus[i]['cisi']}({corpus[i]['site']}) {'-'.join(map(str, corpus[i]['seq_raw']))} ~ " + ','.join(f"{corpus[j]['cisi']}({corpus[j]['site']}) {'-'.join(map(str, corpus[j]['seq_raw']))}" for j in s[:4]) for i, s in mv))

# the cycle-1 lead: Mohenjo-daro seals with the Harappa package 405+806
lines.append('Cycle-1 lead: Mohenjo-daro objects carrying both 405 and 806 (Harappa forms):')
for i, r in enumerate(corpus):
    if r['site'] == 'Mohenjo-daro' and 405 in r['seq_raw'] and 806 in r['seq_raw']:
        lines.append(f"  {r['cisi']} {r['type']} {r['material']} area {r['area-section']} grid {r['room-grid']} period {r['period']} emblem {r['symbol']} text {'-'.join(map(str, r['seq_raw']))}")
lines.append('Mohenjo-daro objects carrying 390 and 803 (home forms), for comparison:')
for i, r in enumerate(corpus):
    if r['site'] == 'Mohenjo-daro' and 390 in r['seq_raw'] and 803 in r['seq_raw']:
        lines.append(f"  {r['cisi']} {r['type']} {r['material']} area {r['area-section']} grid {r['room-grid']} period {r['period']} emblem {r['symbol']} text {'-'.join(map(str, r['seq_raw']))}")
# is the 405+806 package at MD a stock text shared with Harappa?
htexts = {tuple(r['seq_all']) for r in corpus if r['site'] == 'Harappa'}
for i, r in enumerate(corpus):
    if r['site'] == 'Mohenjo-daro' and 405 in r['seq_raw'] and 806 in r['seq_raw']:
        lines.append(f"  {r['cisi']}: same merged text at Harappa? {tuple(r['seq_all']) in htexts}")

json.dump(summary, open(OUT + f'loop46_cycle3_{LEVEL}.json', 'w'), indent=1, default=float)
open(OUT + f'loop46_cycle3_{LEVEL}_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
