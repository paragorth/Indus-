#!/usr/bin/env python3
"""Loop 11: scribal 'confusions' as plaintext.

Pairs of distinct texts at edit distance 1 (one substitution, insertion or
deletion), at least one member a frequent text (>= MINFREQ objects).  The
differing sign pair (a, b) is a candidate confusion.

Tests (see loop11_*.txt):
 (a) recurrence of specific (a,b) confusions across independent frames vs a
     random-partner null stratified by site x object class x position;
 (b) slot agreement of a and b vs the same null;
 (c) confusion rate per sign vs sign frequency;
 (d) clustering of confusion frames by site / find area vs a label-permutation
     null stratified by object class x length;
 (e) insertions/deletions: which signs are optional.

Usage: python3 tools/dark11_confusions.py LEVEL [MINFREQ] [NPERM] [SEED] [COMPLETE]
  LEVEL in seq_raw | seq_strong | seq_all
  COMPLETE: Y (default, only complete texts) or YQ (complete + '?')
"""
import json, sys, random, collections, math, itertools
import numpy as np

ROOT = '/home/user/Indus-'
LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
MINFREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 11
COMPLETE = sys.argv[5] if len(sys.argv) > 5 else 'Y'
rng = random.Random(SEED)

corpus = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
bridge = json.load(open(f'{ROOT}/data/derived/bridge_extended.json'))
allo = json.load(open(f'{ROOT}/data/derived/sign_allographs_levels.json'))
variants = json.load(open(f'{ROOT}/data/derived/sign_variant_classes.json'))
glyph_signs = json.load(open(f'{ROOT}/data/derived/glyph_sim_signs.json'))
glyph_sim = np.load(f'{ROOT}/data/derived/glyph_sim.npy')
gidx = {s: i for i, s in enumerate(glyph_signs)}

MERGED = {}
for m in allo['merges']:
    MERGED[frozenset((m['form'], m['into']))] = 'allograph-' + m['level']
for tier in ('tierA', 'tierB'):
    for f, into in variants[tier].items():
        MERGED.setdefault(frozenset((int(f), into)), 'variantclass-' + tier)
# transitive: two forms merged into the same head
head = {}
for tier in ('tierA', 'tierB'):
    for f, into in variants[tier].items():
        head[int(f)] = into
for m in allo['merges']:
    head.setdefault(m['form'], m['into'])

def same_class(a, b):
    if frozenset((a, b)) in MERGED:
        return MERGED[frozenset((a, b))]
    if head.get(a, a) == head.get(b, b):
        return 'same-head-' + str(head.get(a, a))
    return ''

# GRAMMAR.md slot classes in Wells numbers (via bridge_extended)
SLOT = {}
for s in (861, 817, 820):
    SLOT[s] = 'opener'
for s in (2, 60):
    SLOT[s] = 'marker'
for s in (235, 31):
    SLOT[s] = 'mid-initial'
for s in (740, 390, 405, 406, 407, 156, 151, 527, 526, 520, 595):
    SLOT[s] = 'closer'
for s in (400, 90):
    SLOT[s] = 'suffix'
for s in range(3, 60):
    SLOT.setdefault(s, 'numeral')

def typeclass(t):
    return t.split(':')[0]

accept = {'Y'} if COMPLETE == 'Y' else {'Y', '?'}
objs = []
for r in corpus:
    if r['complete'] not in accept:
        continue
    seq = tuple(r[LEVEL])
    if len(seq) < 3:
        continue
    objs.append(dict(cisi=r['cisi'], site=r['site'], tc=typeclass(r['type']),
                     area=r['area-section'], seq=seq))

texts = collections.defaultdict(list)
for o in objs:
    texts[o['seq']].append(o)
freq_texts = {t for t, os_ in texts.items() if len(os_) >= MINFREQ}
sign_count = collections.Counter(s for o in objs for s in o['seq'])
total_signs = sum(sign_count.values())

# positional sign distribution: key (stratum, L, p) -> Counter
def strat_key(o):
    return (o['site'] if o['site'] in ('Harappa', 'Mohenjo-daro') else 'other', o['tc'])
posdist = collections.defaultdict(collections.Counter)
posdist_pool = collections.defaultdict(collections.Counter)   # (L,p) only
for o in objs:
    L = len(o['seq'])
    for p, s in enumerate(o['seq']):
        posdist[(strat_key(o), L, p)][s] += 1
        posdist_pool[(L, p)][s] += 1

# ---- find edit-distance-1 pairs between distinct text types -----------------
# index by (L, position-deleted) -> key for substitution and indel lookup
subs = []    # (T1, T2, pos, a, b)
indels = []  # (long, short, pos, deleted_sign)
by_del = collections.defaultdict(list)   # key: tuple with one position removed -> (T, pos)
all_texts = list(texts)
for T in all_texts:
    L = len(T)
    for p in range(L):
        by_del[T[:p] + T[p + 1:]].append((T, p))
seen = set()
for key, lst in by_del.items():
    # substitutions: two different texts of the same length agree off position p
    for (T1, p1), (T2, p2) in itertools.combinations(lst, 2):
        if p1 != p2 or T1 == T2:
            continue
        if T1 not in freq_texts and T2 not in freq_texts:
            continue
        k = frozenset((T1, T2))
        if k in seen:
            continue
        seen.add(k)
        subs.append((T1, T2, p1, T1[p1], T2[p1]))
    # indels: key itself is a text
    if key in texts and len(key) >= 3:
        for (T, p) in lst:
            if T not in freq_texts and key not in freq_texts:
                continue
            k = (T, key)
            if k in seen:
                continue
            seen.add(k)
            indels.append((T, key, p, T[p]))

def conf_key(a, b):
    return (a, b) if a < b else (b, a)

def summarise_subs(sublist):
    """frames per unordered confusion pair"""
    c = collections.Counter(conf_key(a, b) for (_, _, _, a, b) in sublist)
    return c

obs = summarise_subs(subs)
n_pairs = len(subs)

# ---- null (a),(b): random partner, stratified ---------------------------------
def draw_partner(T1, p, a, stratum_objs):
    """Draw b != a from the positional distribution of signs at (stratum, L, p)."""
    L = len(T1)
    o = stratum_objs
    dist = posdist[(strat_key(o), L, p)]
    if sum(dist.values()) - dist.get(a, 0) < 3:
        dist = posdist_pool[(L, p)]
    items = [(s, n) for s, n in dist.items() if s != a]
    tot = sum(n for _, n in items)
    x = rng.random() * tot
    for s, n in items:
        x -= n
        if x <= 0:
            return s
    return items[-1][0]

def slot_agree(a, b):
    sa, sb = SLOT.get(a), SLOT.get(b)
    if sa is None or sb is None:
        return None
    return sa == sb

def posprofile_corr(a, b):
    """cosine between relative-position profiles (initial/medial/final) of a and b"""
    def prof(s):
        v = np.zeros(3)
        for o in objs:
            L = len(o['seq'])
            for p, x in enumerate(o['seq']):
                if x == s:
                    v[0 if p == 0 else 2 if p == L - 1 else 1] += 1
        return v
    pa, pb = prof(a), prof(b)
    if pa.sum() == 0 or pb.sum() == 0:
        return np.nan
    return float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb)))

def stats_for(sublist):
    c = summarise_subs(sublist)
    rec2 = sum(1 for v in c.values() if v >= 2)
    rec3 = sum(1 for v in c.values() if v >= 3)
    mx = max(c.values()) if c else 0
    agree = [slot_agree(a, b) for (_, _, _, a, b) in sublist]
    known = [x for x in agree if x is not None]
    agree_rate = sum(known) / len(known) if known else np.nan
    merged = sum(1 for (_, _, _, a, b) in sublist if same_class(a, b))
    gl = [glyph_sim[gidx[a], gidx[b]] for (_, _, _, a, b) in sublist if a in gidx and b in gidx]
    return dict(rec2=rec2, rec3=rec3, mx=mx, agree=agree_rate, nknown=len(known),
                merged=merged, glyph=float(np.mean(gl)) if gl else np.nan, nd=len(c))

obs_stats = stats_for(subs)
null_stats = collections.defaultdict(list)
for it in range(NPERM):
    fake = []
    for (T1, T2, p, a, b) in subs:
        o = texts[T1][0]
        # keep a (the frequent text's sign where possible), randomise partner
        fake.append((T1, T2, p, a, draw_partner(T1, p, a, o)))
    st = stats_for(fake)
    for k, v in st.items():
        null_stats[k].append(v)

def pval(obs, nulls, side='ge'):
    nulls = [x for x in nulls if not (isinstance(x, float) and math.isnan(x))]
    if not nulls:
        return float('nan')
    if side == 'ge':
        return (1 + sum(1 for x in nulls if x >= obs)) / (len(nulls) + 1)
    return (1 + sum(1 for x in nulls if x <= obs)) / (len(nulls) + 1)

# ---- (c) confusion rate per sign vs frequency ---------------------------------
conf_events = collections.Counter()
for (_, _, _, a, b) in subs:
    conf_events[a] += 1
    conf_events[b] += 1
signs_c = [s for s in sign_count if sign_count[s] >= 10]
rates = np.array([conf_events[s] / sign_count[s] for s in signs_c])
freqs = np.array([sign_count[s] for s in signs_c])
from scipy.stats import spearmanr
rho_c, p_c = spearmanr(np.log(freqs), rates)
# null for (c): same but partners randomised (keep a)
rho_null = []
for it in range(min(NPERM, 300)):
    ce = collections.Counter()
    for (T1, T2, p, a, b) in subs:
        ce[a] += 1
        ce[draw_partner(T1, p, a, texts[T1][0])] += 1
    r = np.array([ce[s] / sign_count[s] for s in signs_c])
    rho_null.append(spearmanr(np.log(freqs), r)[0])

# ---- (d) site / area clustering of confusion frames ---------------------------
def site_share(sublist, label):
    same = tot = 0
    for (T1, T2, p, a, b) in sublist:
        for o1 in texts[T1]:
            for o2 in texts[T2]:
                l1, l2 = o1[label], o2[label]
                if label == 'area' and (l1 == '--' or l2 == '--'):
                    continue
                if label == 'area' and o1['site'] != o2['site']:
                    continue
                tot += 1
                same += (l1 == l2)
    return same / tot if tot else np.nan, tot

obs_site, n_site = site_share(subs, 'site')
obs_area, n_area = site_share(subs, 'area')
# null: permute site (area) labels among objects within (object class, length) strata
strata = collections.defaultdict(list)
for o in objs:
    strata[(o['tc'], len(o['seq']))].append(o)
null_site, null_area = [], []
orig = {o['cisi']: (o['site'], o['area']) for o in objs}
for it in range(min(NPERM, 300)):
    for key, lst in strata.items():
        labels = [(o['site'], o['area']) for o in lst]
        rng.shuffle(labels)
        for o, (s, a) in zip(lst, labels):
            o['site'], o['area'] = s, a
    null_site.append(site_share(subs, 'site')[0])
    null_area.append(site_share(subs, 'area')[0])
for o in objs:
    o['site'], o['area'] = orig[o['cisi']]

# ---- (e) insertions / deletions -----------------------------------------------
del_count = collections.Counter(d for (_, _, _, d) in indels)
del_frames = collections.Counter()
for (T, S, p, d) in indels:
    del_frames[d] += 1
# expected deletions per sign under a null where the deleted position is random
# among positions of the long text (frequency-proportional); compute expected count
exp_del = collections.Counter()
for (T, S, p, d) in indels:
    for s in T:
        exp_del[s] += 1 / len(T)
del_pos = collections.defaultdict(collections.Counter)
for (T, S, p, d) in indels:
    L = len(T)
    del_pos[d]['initial' if p == 0 else 'final' if p == L - 1 else 'medial'] += 1

# ---- report -------------------------------------------------------------------
out = []
P = out.append
P(f'LOOP 11 (scribal confusions)  level={LEVEL} minfreq={MINFREQ} complete={COMPLETE} nperm={NPERM} seed={SEED}')
P(f'objects (complete, len>=3): {len(objs)}  distinct texts: {len(texts)}  frequent texts (>= {MINFREQ} objects): {len(freq_texts)}')
P(f'edit-distance-1 pairs with a frequent member: substitutions {n_pairs} (distinct text pairs), indels {len(indels)}')
P('')
P('(a) RECURRENCE OF CONFUSION PAIRS (frames = distinct text pairs sharing the same unordered (a,b))')
for k, lab in (('nd', 'distinct (a,b) pairs'), ('rec2', 'pairs recurring in >=2 frames'), ('rec3', '>=3 frames'), ('mx', 'max frames for one pair')):
    nl = null_stats[k]
    side = 'le' if k == 'nd' else 'ge'
    P(f'  {lab}: obs {obs_stats[k]}  null mean {np.mean(nl):.1f} (95% {np.percentile(nl,2.5):.0f}-{np.percentile(nl,97.5):.0f}, max {max(nl)})  P={pval(obs_stats[k], nl, side):.4f}')
P('(b) SLOT AGREEMENT (both signs in a GRAMMAR.md slot class and the same class)')
P(f'  obs {obs_stats["agree"]:.3f} of {obs_stats["nknown"]} classified pairs; null mean {np.nanmean(null_stats["agree"]):.3f}  P={pval(obs_stats["agree"], null_stats["agree"]):.4f}')
P(f'  already merged/variant-class pairs among substitutions: obs {obs_stats["merged"]} null mean {np.mean(null_stats["merged"]):.1f} P={pval(obs_stats["merged"], null_stats["merged"]):.4f}')
P(f'  glyph similarity of substituting pairs: obs {obs_stats["glyph"]:.3f} null mean {np.nanmean(null_stats["glyph"]):.3f} P={pval(obs_stats["glyph"], null_stats["glyph"]):.4f}')
P('(c) CONFUSION RATE PER SIGN vs log frequency (signs with >=10 tokens)')
P(f'  Spearman rho obs {rho_c:.3f} (p {p_c:.3g}); random-partner null rho mean {np.mean(rho_null):.3f} (95% {np.percentile(rho_null,2.5):.3f}..{np.percentile(rho_null,97.5):.3f})  P(obs<=null)={pval(rho_c, rho_null, "le"):.4f}')
P('(d) SITE / AREA CLUSTERING of the two objects in a confusion pair')
P(f'  same site: obs {obs_site:.3f} (n object pairs {n_site}) null mean {np.mean(null_site):.3f} (95% {np.percentile(null_site,2.5):.3f}..{np.percentile(null_site,97.5):.3f}) P={pval(obs_site, null_site):.4f}')
P(f'  same find area (within site, both known): obs {obs_area:.3f} (n {n_area}) null mean {np.nanmean(null_area):.3f} (95% {np.nanpercentile(null_area,2.5):.3f}..{np.nanpercentile(null_area,97.5):.3f}) P={pval(obs_area, null_area):.4f}')
P('')
P('TOP RECURRENT CONFUSIONS (frames = independent text pairs; objs = object pairs; M = Mahadevan via bridge; status = merge status; glyph = shape similarity; sites)')
rows = []
for (a, b), n in obs.most_common():
    if n < 2:
        break
    objpairs = 0
    sites = collections.Counter()
    slots = collections.Counter()
    for (T1, T2, p, x, y) in subs:
        if conf_key(x, y) == (a, b):
            objpairs += len(texts[T1]) * len(texts[T2])
            for o in texts[T1] + texts[T2]:
                sites[o['site'][:4]] += 1
            L = len(T1)
            slots['ini' if p == 0 else 'fin' if p == L - 1 else 'med'] += 1
    st = same_class(a, b) or 'NEW'
    g = glyph_sim[gidx[a], gidx[b]] if a in gidx and b in gidx else float('nan')
    rows.append((n, a, b, objpairs, st, g, dict(sites.most_common(3)), dict(slots)))
    P(f'  {a:>4}<->{b:<4} frames {n:2d} objs {objpairs:3d}  M{bridge.get(str(a),"?")}/M{bridge.get(str(b),"?")}  {st:22s} glyph {g:.2f}  freq {sign_count[a]}/{sign_count[b]}  pos {dict(slots)}  sites {dict(sites.most_common(3))}')
P('')
P('(e) OPTIONAL SIGNS (deleted/inserted without changing the rest; frames = distinct text pairs; exp = expected if the deleted position were random)')
P('  sign  frames  exp   ratio  freq   pos(deleted)         M')
for d, n in del_frames.most_common(30):
    P(f'  {d:>4}  {n:5d}  {exp_del[d]:5.1f}  {n/exp_del[d] if exp_del[d] else float("inf"):5.2f}  {sign_count[d]:5d}  {dict(del_pos[d])!s:22s} M{bridge.get(str(d),"?")}')
# optional-sign enrichment test: do the GRAMMAR deletable signs (235,31,400,90, 2, 60) and qualifiers carry more deletions than expected?
known_opt = {235, 31, 400, 90, 2, 60, 1, 100, 705, 706, 33}
ko = sum(del_frames[s] for s in known_opt)
ke = sum(exp_del[s] for s in known_opt)
P(f'  known deletable/qualifier set {sorted(known_opt)}: deletions obs {ko} vs exp {ke:.1f} (ratio {ko/ke if ke else float("nan"):.2f}) of {len(indels)} indel frames')
# binomial-ish permutation: random position deletion
sim = []
for it in range(2000):
    s = 0
    for (T, S, p, d) in indels:
        s += (rng.choice(T) in known_opt)
    sim.append(s)
P(f'  random-position null: mean {np.mean(sim):.1f} max {max(sim)}  P={pval(ko, sim):.4f}')
json.dump(dict(level=LEVEL, minfreq=MINFREQ, complete=COMPLETE, n_subs=n_pairs, n_indels=len(indels),
               obs=obs_stats, null_means={k: float(np.nanmean(v)) for k, v in null_stats.items()},
               top=[(int(a), int(b), int(n), int(op), st) for (n, a, b, op, st, g, si, sl) in rows],
               site=dict(obs=obs_site, null=float(np.mean(null_site))), area=dict(obs=obs_area, null=float(np.nanmean(null_area))),
               rho=dict(obs=float(rho_c), null=float(np.mean(rho_null))),
               deletions={str(d): [int(n), float(exp_del[d])] for d, n in del_frames.most_common(40)}),
          open(f'{ROOT}/data/derived/dark/loop11_{LEVEL}_f{MINFREQ}_{COMPLETE}.json', 'w'), indent=1)
print('\n'.join(out))
