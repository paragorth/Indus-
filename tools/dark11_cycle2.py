#!/usr/bin/env python3
"""Loop 11 cycle 2: (i) site/area clustering with the Harappa-tablet confound removed,
(ii) held-out-site replication (fit on Mohenjo-daro + Harappa, test on all other sites),
(iii) missed-allograph scan: non-merged one-slot substitutions with high glyph similarity.
Usage: python3 tools/dark11_cycle2.py LEVEL [NPERM] [SEED]
"""
import json, sys, random, collections, itertools, math
import numpy as np

ROOT = '/home/user/Indus-'
LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 22
rng = random.Random(SEED)

corpus = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
bridge = json.load(open(f'{ROOT}/data/derived/bridge_extended.json'))
allo = json.load(open(f'{ROOT}/data/derived/sign_allographs_levels.json'))
variants = json.load(open(f'{ROOT}/data/derived/sign_variant_classes.json'))
glyph_signs = json.load(open(f'{ROOT}/data/derived/glyph_sim_signs.json'))
glyph_sim = np.load(f'{ROOT}/data/derived/glyph_sim.npy')
gidx = {s: i for i, s in enumerate(glyph_signs)}
head = {}
for tier in ('tierA', 'tierB'):
    for f, into in variants[tier].items():
        head[int(f)] = into
for m in allo['merges']:
    head.setdefault(m['form'], m['into'])
def merged(a, b):
    return head.get(a, a) == head.get(b, b)
SLOT = {}
for s in (861, 817, 820): SLOT[s] = 'opener'
for s in (2, 60): SLOT[s] = 'marker'
for s in (235, 31): SLOT[s] = 'mid-initial'
for s in (740, 390, 405, 406, 407, 156, 151, 527, 526, 520, 595): SLOT[s] = 'closer'
for s in (400, 90): SLOT[s] = 'suffix'
for s in range(3, 60): SLOT.setdefault(s, 'numeral')
def slot_agree(a, b):
    sa, sb = SLOT.get(a), SLOT.get(b)
    return None if sa is None or sb is None else sa == sb

def load(filter_fn):
    objs = []
    for r in corpus:
        if r['complete'] != 'Y': continue
        seq = tuple(r[LEVEL])
        if len(seq) < 3: continue
        o = dict(cisi=r['cisi'], site=r['site'], tc=r['type'].split(':')[0], area=r['area-section'], seq=seq)
        if filter_fn(o): objs.append(o)
    texts = collections.defaultdict(list)
    for o in objs: texts[o['seq']].append(o)
    return objs, texts

def find_subs(texts, minfreq):
    freq = {t for t, l in texts.items() if len(l) >= minfreq}
    by_del = collections.defaultdict(list)
    for T in texts:
        for p in range(len(T)):
            by_del[T[:p] + T[p + 1:]].append((T, p))
    seen, subs = set(), []
    for key, lst in by_del.items():
        for (T1, p1), (T2, p2) in itertools.combinations(lst, 2):
            if p1 != p2 or T1 == T2: continue
            if T1 not in freq and T2 not in freq: continue
            k = frozenset((T1, T2))
            if k in seen: continue
            seen.add(k); subs.append((T1, T2, p1, T1[p1], T2[p1]))
    return subs

def posdists(objs):
    pd = collections.defaultdict(collections.Counter); pool = collections.defaultdict(collections.Counter)
    for o in objs:
        L = len(o['seq']); sk = (o['site'] if o['site'] in ('Harappa', 'Mohenjo-daro') else 'other', o['tc'])
        for p, s in enumerate(o['seq']):
            pd[(sk, L, p)][s] += 1; pool[(L, p)][s] += 1
    return pd, pool

def draw(pd, pool, o, L, p, a):
    sk = (o['site'] if o['site'] in ('Harappa', 'Mohenjo-daro') else 'other', o['tc'])
    dist = pd[(sk, L, p)]
    if sum(dist.values()) - dist.get(a, 0) < 3: dist = pool[(L, p)]
    items = [(s, n) for s, n in dist.items() if s != a]
    if not items: return a
    tot = sum(n for _, n in items); x = rng.random() * tot
    for s, n in items:
        x -= n
        if x <= 0: return s
    return items[-1][0]

def pval(obs, nulls, side='ge'):
    nulls = [x for x in nulls if not (isinstance(x, float) and math.isnan(x))]
    if not nulls: return float('nan')
    return (1 + sum(1 for x in nulls if (x >= obs if side == 'ge' else x <= obs))) / (len(nulls) + 1)

out = []; P = out.append
P(f'LOOP 11 cycle 2  level={LEVEL} nperm={NPERM} seed={SEED}')

# ---------- (i) site / area clustering, confound checks ----------
objs, texts = load(lambda o: True)
subs = find_subs(texts, 3)
def share(subs, texts, label, exclude=None):
    same = tot = 0
    for (T1, T2, p, a, b) in subs:
        for o1 in texts[T1]:
            for o2 in texts[T2]:
                if exclude and (exclude(o1) or exclude(o2)): continue
                if label == 'area' and (o1['area'] == '--' or o2['area'] == '--' or o1['site'] != o2['site']): continue
                tot += 1; same += (o1[label] == o2[label])
    return (same / tot if tot else float('nan')), tot

def perm_labels(objs, strat_fn, nrep, stat_fn):
    orig = {o['cisi']: (o['site'], o['area']) for o in objs}
    strata = collections.defaultdict(list)
    for o in objs: strata[strat_fn(o)].append(o)
    res = []
    for it in range(nrep):
        for lst in strata.values():
            labs = [(o['site'], o['area']) for o in lst]; rng.shuffle(labs)
            for o, (s, a) in zip(lst, labs): o['site'], o['area'] = s, a
        res.append(stat_fn())
    for o in objs: o['site'], o['area'] = orig[o['cisi']]
    return res

P('(i) SITE/AREA CLUSTERING of confusion-pair objects')
# i-a: full, strata (type, length) as cycle 1
# i-b: exclude Harappa tablets (TAB) objects
# i-c: area clustering with strata (site, type, length) i.e. permute areas within site and type
# i-d: seals only
variants_i = [
    ('all objects, strata (type,len)', None, lambda o: (o['tc'], len(o['seq']))),
    ('excluding Harappa TAB objects', lambda o: o['site'] == 'Harappa' and o['tc'] == 'TAB', lambda o: (o['tc'], len(o['seq']))),
    ('SEAL objects only', lambda o: o['tc'] != 'SEAL', lambda o: (o['tc'], len(o['seq']))),
]
for name, excl, strat in variants_i:
    os_, n_s = share(subs, texts, 'site', excl)
    oa, n_a = share(subs, texts, 'area', excl)
    nulls = perm_labels(objs, strat, min(NPERM, 400), lambda: (share(subs, texts, 'site', excl)[0], share(subs, texts, 'area', excl)[0]))
    ns = [x[0] for x in nulls]; na = [x[1] for x in nulls]
    P(f'  {name}: same site {os_:.3f} (n {n_s}) null {np.nanmean(ns):.3f} P={pval(os_, ns):.4f} | same area {oa:.3f} (n {n_a}) null {np.nanmean(na):.3f} P={pval(oa, na):.4f}')
# i-e: area permuted within (site,type,len) strata -- the strict area test
oa, n_a = share(subs, texts, 'area')
nulls = perm_labels(objs, lambda o: (o['site'], o['tc'], len(o['seq'])), min(NPERM, 400), lambda: share(subs, texts, 'area')[0])
P(f'  area, labels permuted within (site,type,len): obs {oa:.3f} (n {n_a}) null {np.nanmean(nulls):.3f} (95% {np.nanpercentile(nulls,2.5):.3f}..{np.nanpercentile(nulls,97.5):.3f}) P={pval(oa, nulls):.4f}')
for site in ('Harappa', 'Mohenjo-daro'):
    excl = lambda o, s=site: o['site'] != s
    oa, n_a = share(subs, texts, 'area', excl)
    nulls = perm_labels(objs, lambda o: (o['site'], o['tc'], len(o['seq'])), min(NPERM, 400), lambda: share(subs, texts, 'area', excl)[0])
    P(f'    {site} only: same area {oa:.3f} (n {n_a}) null {np.nanmean(nulls):.3f} P={pval(oa, nulls):.4f}')
# i-f: compare with the baseline of identical-text duplicates (same text, two objects): how often same area?
same = tot = 0
for T, lst in texts.items():
    for o1, o2 in itertools.combinations(lst, 2):
        if o1['area'] == '--' or o2['area'] == '--' or o1['site'] != o2['site']: continue
        tot += 1; same += (o1['area'] == o2['area'])
P(f'  reference: identical-text duplicates share find area {same/tot if tot else float("nan"):.3f} (n {tot})')

# ---------- (ii) held-out replication ----------
P('(ii) HELD-OUT REPLICATION: fit Mohenjo-daro + Harappa, test on all other sites')
train_objs, train_texts = load(lambda o: o['site'] in ('Harappa', 'Mohenjo-daro'))
test_objs, test_texts = load(lambda o: o['site'] not in ('Harappa', 'Mohenjo-daro', 'Unknown'))
train_subs = find_subs(train_texts, 3)
train_pairs = collections.Counter((min(a, b), max(a, b)) for (_, _, _, a, b) in train_subs)
P(f'  train: {len(train_objs)} objects, {len(train_subs)} substitution frames, {len(train_pairs)} distinct (a,b)')
for mf in (2, 1):
    test_subs = find_subs(test_texts, mf)
    pd, pool = posdists(test_objs)
    def test_stats(sl):
        hit = sum(1 for (_, _, _, a, b) in sl if (min(a, b), max(a, b)) in train_pairs)
        mg = sum(1 for (_, _, _, a, b) in sl if merged(a, b))
        ag = [slot_agree(a, b) for (_, _, _, a, b) in sl]; ag = [x for x in ag if x is not None]
        gl = [glyph_sim[gidx[a], gidx[b]] for (_, _, _, a, b) in sl if a in gidx and b in gidx]
        return hit, mg, (sum(ag) / len(ag) if ag else float('nan')), len(ag), (float(np.mean(gl)) if gl else float('nan'))
    obs = test_stats(test_subs)
    nulls = []
    for it in range(NPERM):
        fake = [(T1, T2, p, a, draw(pd, pool, test_texts[T1][0], len(T1), p, a)) for (T1, T2, p, a, b) in test_subs]
        nulls.append(test_stats(fake))
    sites = collections.Counter(o['site'] for (T1, T2, p, a, b) in test_subs for o in test_texts[T1] + test_texts[T2])
    P(f'  test (minfreq {mf}): {len(test_objs)} objects, {len(test_subs)} substitution frames; object sites {dict(sites.most_common(6))}')
    P(f'    frames whose (a,b) was seen in train: obs {obs[0]} null {np.mean([x[0] for x in nulls]):.1f} P={pval(obs[0], [x[0] for x in nulls]):.4f}')
    P(f'    already-merged pairs: obs {obs[1]} null {np.mean([x[1] for x in nulls]):.1f} P={pval(obs[1], [x[1] for x in nulls]):.4f}')
    P(f'    slot agreement: obs {obs[2]:.3f} of {obs[3]} null {np.nanmean([x[2] for x in nulls]):.3f} P={pval(obs[2], [x[2] for x in nulls]):.4f}')
    P(f'    glyph similarity: obs {obs[4]:.3f} null {np.nanmean([x[4] for x in nulls]):.3f} P={pval(obs[4], [x[4] for x in nulls]):.4f}')
    if mf == 1:
        P('    held-out substitutions seen in train: ' + ', '.join(f'{a}<->{b}' for (_, _, _, a, b) in test_subs if (min(a, b), max(a, b)) in train_pairs))
        P('    held-out recurrent pairs (>=2 frames): ' + ', '.join(f'{a}<->{b} x{n}' for (a, b), n in collections.Counter((min(a, b), max(a, b)) for (_, _, _, a, b) in test_subs).most_common() if n >= 2))

# ---------- (iii) missed-allograph scan ----------
P('(iii) MISSED-ALLOGRAPH SCAN: non-merged substitution pairs (all sites, minfreq 2), glyph similarity and slot')
subs2 = find_subs(texts, 2)
pd, pool = posdists(objs)
nonm = [(T1, T2, p, a, b) for (T1, T2, p, a, b) in subs2 if not merged(a, b)]
gl_obs = [glyph_sim[gidx[a], gidx[b]] for (_, _, _, a, b) in nonm if a in gidx and b in gidx]
hi_obs = sum(1 for g in gl_obs if g >= 0.85)
nulls_g, nulls_hi = [], []
for it in range(NPERM):
    fake = [(T1, T2, p, a, draw(pd, pool, texts[T1][0], len(T1), p, a)) for (T1, T2, p, a, b) in subs2]
    fk = [(a, b) for (_, _, _, a, b) in fake if not merged(a, b) and a in gidx and b in gidx]
    g = [glyph_sim[gidx[a], gidx[b]] for a, b in fk]
    nulls_g.append(float(np.mean(g))); nulls_hi.append(sum(1 for x in g if x >= 0.85))
P(f'  non-merged frames {len(nonm)} of {len(subs2)}; mean glyph similarity obs {np.mean(gl_obs):.3f} null {np.mean(nulls_g):.3f} (95% {np.percentile(nulls_g,2.5):.3f}..{np.percentile(nulls_g,97.5):.3f}) P={pval(float(np.mean(gl_obs)), nulls_g):.4f}')
P(f'  non-merged frames with glyph similarity >= 0.85: obs {hi_obs} null {np.mean(nulls_hi):.1f} (max {max(nulls_hi)}) P={pval(hi_obs, nulls_hi):.4f}')
P('  candidates (non-merged, glyph >= 0.80), with frames, slot agreement, token counts, M numbers:')
cnt = collections.Counter(); ex = {}
for (T1, T2, p, a, b) in nonm:
    if a in gidx and b in gidx and glyph_sim[gidx[a], gidx[b]] >= 0.80:
        k = (min(a, b), max(a, b)); cnt[k] += 1; ex.setdefault(k, (T1, T2, p))
sc = collections.Counter(s for o in objs for s in o['seq'])
for (a, b), n in cnt.most_common():
    T1, T2, p = ex[(a, b)]
    P(f'    {a}<->{b} frames {n} glyph {glyph_sim[gidx[a], gidx[b]]:.2f} slot {SLOT.get(a,"-")}/{SLOT.get(b,"-")} tokens {sc[a]}/{sc[b]} M{bridge.get(str(a),"?")}/M{bridge.get(str(b),"?")} e.g. {"-".join(map(str,T1))} vs {"-".join(map(str,T2))}')
print('\n'.join(out))
