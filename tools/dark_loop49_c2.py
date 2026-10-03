"""S-DARK-49 cycle 2: TIME-STAMP test. A year-name (or any dating element) recurs site-wide within one stratum and then disappears.
For every sign with >= MINN dated objects in a city (Mohenjo-daro depth in ft below datum; Harappa depth in ft and HARP fine period),
measure how concentrated its objects are in stratigraphic time: (i) IQR of depth, (ii) peak share = share of its objects inside its modal
4-ft band (Harappa fine period: modal period share), (iii) for the Mohenjo-daro Marshall period labels (Early / Intermediate / Late) the
modal share. Null: depth / period labels shuffled among objects within site x object type (and, second null, site x type x area-section),
NPERM x; one-sided P for 'more concentrated than chance'; BH over elements per city.  Cross-city agreement: for elements dated in both cities,
Spearman rho between the element's mean within-city depth percentile at Mohenjo-daro and at Harappa (a year-name, if the two cities shared
one calendar and comparable stratigraphy, should agree), with the same permutation null.
Objects = data/raw/inscriptions.csv rows grouped by id n.k (all faces pooled; tools/dark_loop37.load_faces, three merge levels).
Usage: python3 tools/dark_loop49_c2.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, csv, re, json, math, collections, random, statistics
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import load_faces, OPEN, CL, FISH, NUM, pval
ROOT = '/home/user/Indus-/'
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
MINN = 8
rnd = random.Random(492)
OUT = open(ROOT + f'data/derived/dark/loop49_c2_{LV}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

FINE = {'Period 2': 0, 'Period 3B-1': 1, 'Period 3B-2': 2, 'Period 3B': 2, 'Period 3B/C': 3, 'Period 3C-1': 4, 'Period 3C-2': 5, 'Period 3C-3': 6,
        'Period 3C-4': 7, 'Period 3C': 6, 'Period 4': 8, 'Period 5A': 9}
FINE_STRICT = {'Period 2': 0, 'Period 3B-1': 1, 'Period 3B-2': 2, 'Period 3C-1': 4, 'Period 3C-2': 5, 'Period 3C-3': 6, 'Period 3C-4': 7, 'Period 4': 8, 'Period 5A': 9}
def depth_ft(s):
    m = re.match(r'^-([\d.]+)\s*ft', s or '', re.I)
    if not m: return None
    try: return float(m.group(1).replace('..', '.'))
    except ValueError: return None
def md_period(p):
    p = (p or '').strip()
    for k, v in (('Early', 0), ('Intermediate', 1), ('Late', 2)):
        if p.startswith(k): return v
    return None

# metadata per object id from the csv (first face row)
meta = {}
for r in csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')):
    oid = r['id'].split('.')[0]
    if oid in meta: continue
    meta[oid] = dict(depth=depth_ft(r['depth']), fine=FINE_STRICT.get(r['time']), mdper=md_period(r['period']), area=r['area-section'] or '--',
                     type2=r['type'].split(':')[0])
objs = load_faces(LV)
OBJ = []
for oid, o in objs.items():
    if o['site'] not in ('Mohenjo-daro', 'Harappa'): continue
    signs = set(w for f in o['faces'] for w in f['seq'])
    if not signs: continue
    m = meta[oid]
    OBJ.append(dict(oid=oid, site=o['site'], type2=m['type2'], area=m['area'], signs=signs, depth=m['depth'], fine=m['fine'], mdper=m['mdper']))
say(f'# S-DARK-49 cycle 2, level {LV}, nperm {NPERM}: time-stamp test. Objects: MD {sum(o["site"]=="Mohenjo-daro" for o in OBJ)}, HP {sum(o["site"]=="Harappa" for o in OBJ)}; '
    f'with depth MD {sum(o["site"]=="Mohenjo-daro" and o["depth"] is not None for o in OBJ)}, HP {sum(o["site"]=="Harappa" and o["depth"] is not None for o in OBJ)}; '
    f'HP fine period {sum(o["site"]=="Harappa" and o["fine"] is not None for o in OBJ)}; MD Marshall period {sum(o["site"]=="Mohenjo-daro" and o["mdper"] is not None for o in OBJ)}')

def iqr(v):
    v = sorted(v); n = len(v)
    return v[int(0.75 * (n - 1))] - v[int(0.25 * (n - 1))]
def peak_share(v, width):
    c = collections.Counter(int(x // width) if width else x for x in v); return c.most_common(1)[0][1] / len(v)

def run(site, key, width, label):
    """key: 'depth' | 'fine' | 'mdper'; width: band width for depth (4 ft) or None for ordinal labels"""
    D = [o for o in OBJ if o['site'] == site and o[key] is not None]
    if len(D) < 30: say(f'-- {site} {label}: only {len(D)} dated objects, skipped'); return {}
    vals = [o[key] for o in D]
    cnt = collections.Counter(w for o in D for w in o['signs'])
    elems = [w for w, n in cnt.items() if n >= MINN]
    obs = {}
    for w in elems:
        v = [o[key] for o in D if w in o['signs']]
        obs[w] = (iqr(v), peak_share(v, width), statistics.mean(v), len(v), collections.Counter(int(x // width) if width else x for x in v).most_common(1)[0][0])
    # nulls
    def shuffled(strata_fn):
        groups = collections.defaultdict(list)
        for i, o in enumerate(D): groups[strata_fn(o)].append(i)
        newv = list(vals)
        for g in groups.values():
            vs = [vals[i] for i in g]; rnd.shuffle(vs)
            for i, x in zip(g, vs): newv[i] = x
        return newv
    nulls = {w: ([], []) for w in elems}; nulls2 = {w: ([], []) for w in elems}
    idx = {w: [i for i, o in enumerate(D) if w in o['signs']] for w in elems}
    for it in range(NPERM):
        for nulld, fn in ((nulls, lambda o: o['type2']), (nulls2, lambda o: (o['type2'], o['area']))):
            nv = shuffled(fn)
            for w in elems:
                v = [nv[i] for i in idx[w]]
                nulld[w][0].append(iqr(v)); nulld[w][1].append(peak_share(v, width))
    rows = []
    for w in elems:
        o = obs[w]
        p_iqr = pval(-o[0], [-x for x in nulls[w][0]]); p_peak = pval(o[1], nulls[w][1])
        p_iqr2 = pval(-o[0], [-x for x in nulls2[w][0]]); p_peak2 = pval(o[1], nulls2[w][1])
        rows.append(dict(w=w, n=o[3], iqr=o[0], peak=o[1], mean=o[2], modal=o[4], p_iqr=p_iqr, p_peak=p_peak, p_iqr2=p_iqr2, p_peak2=p_peak2,
                         null_peak=sum(nulls[w][1]) / NPERM, null_iqr=sum(nulls[w][0]) / NPERM))
    # BH on the peak-share P (type null) and on the area-stratified null
    def bh(ps, q=0.05):
        m = len(ps); srt = sorted(range(m), key=lambda i: ps[i]); thr = 0
        for k, i in enumerate(srt):
            if ps[i] <= q * (k + 1) / m: thr = k + 1
        return set(srt[:thr])
    sig1 = bh([r['p_peak'] for r in rows]); sig2 = bh([r['p_peak2'] for r in rows])
    sig1i = bh([r['p_iqr'] for r in rows]); sig2i = bh([r['p_iqr2'] for r in rows])
    say(f'-- {site} {label}: {len(D)} dated objects, {len(elems)} elements with >= {MINN}; ' f'BH-significant concentration (peak share): type-null {len(sig1)}, type x area null {len(sig2)}; (IQR): {len(sig1i)}, {len(sig2i)}; '
        f'expected false positives at P<0.05: {0.05*len(elems):.1f}; raw P<0.05 peak: {sum(r["p_peak"]<0.05 for r in rows)} / area-null {sum(r["p_peak2"]<0.05 for r in rows)}')
    rows.sort(key=lambda r: r['p_peak2'])
    for r in rows[:12]:
        say(f"   W{r['w']:<4d} n={r['n']:3d} mean={r['mean']:6.2f} modal band={r['modal']} peak={r['peak']:.2f} (null {r['null_peak']:.2f}) P_type={r['p_peak']:.3f} P_area={r['p_peak2']:.3f} | IQR={r['iqr']:.1f} (null {r['null_iqr']:.1f}) P_type={r['p_iqr']:.3f} P_area={r['p_iqr2']:.3f}"
            + ('  <BH type' if rows.index(r) in sig1 else '') + ('  <BH area' if rows.index(r) in sig2 else ''))
    # distribution of the dated objects themselves (the stratigraphic sample)
    say(f"   sample bands: {sorted(collections.Counter(int(x // width) if width else x for x in vals).items())}")
    return {r['w']: r for r in rows}

R_md = run('Mohenjo-daro', 'depth', 4.0, 'depth (ft, 4-ft bands)')
R_hp = run('Harappa', 'depth', 4.0, 'depth (ft, 4-ft bands)')
R_hf = run('Harappa', 'fine', None, 'HARP fine period (ordinal 0-9)')
R_mp = run('Mohenjo-daro', 'mdper', None, 'Marshall period (Early 0 / Intermediate 1 / Late 2)')

# cross-city agreement: mean within-city percentile of depth for shared elements
def percentiles(site, key):
    D = [o for o in OBJ if o['site'] == site and o[key] is not None]
    vals = sorted(o[key] for o in D)
    def pct(x): return (sum(1 for v in vals if v < x) + 0.5 * sum(1 for v in vals if v == x)) / len(vals)
    for o in D: o['pct_' + key] = pct(o[key])
    return D
for (sa, ka), (sb, kb), lab in [(('Mohenjo-daro', 'depth'), ('Harappa', 'depth'), 'MD depth vs HP depth'),
                                (('Mohenjo-daro', 'depth'), ('Harappa', 'fine'), 'MD depth (deeper = older) vs HP fine period (higher = later; sign flipped)'),
                                (('Mohenjo-daro', 'mdper'), ('Harappa', 'fine'), 'MD Marshall period vs HP fine period (both: higher = later)')]:
    Da = percentiles(sa, ka); Db = percentiles(sb, kb)
    ca = collections.Counter(w for o in Da for w in o['signs']); cb = collections.Counter(w for o in Db for w in o['signs'])
    shared = [w for w in ca if ca[w] >= MINN and cb.get(w, 0) >= MINN]
    if len(shared) < 5: say(f'-- cross-city {lab}: only {len(shared)} shared elements'); continue
    ma = {w: statistics.mean(o['pct_' + ka] for o in Da if w in o['signs']) for w in shared}
    mb = {w: statistics.mean(o['pct_' + kb] for o in Db if w in o['signs']) for w in shared}
    sign = -1 if (ka == 'depth') != (kb == 'depth') else 1
    def spearman(x, y):
        def rk(v):
            s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
            for k, i in enumerate(s): r[i] = k
            return r
        rx, ry = rk(x), rk(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
        return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    rho = spearman([ma[w] for w in shared], [sign * mb[w] for w in shared])
    # null: shuffle the dating labels within type at both sites, recompute
    null = []
    for it in range(min(NPERM, 500)):
        for D, k in ((Da, ka), (Db, kb)):
            groups = collections.defaultdict(list)
            for o in D: groups[o['type2']].append(o)
            for g in groups.values():
                vs = [o['pct_' + k] for o in g]; rnd.shuffle(vs)
                for o, x in zip(g, vs): o['tmp'] = x
        na = {w: statistics.mean(o['tmp'] for o in Da if w in o['signs']) for w in shared}
        nb = {w: statistics.mean(o['tmp'] for o in Db if w in o['signs']) for w in shared}
        null.append(spearman([na[w] for w in shared], [sign * nb[w] for w in shared]))
    null_s = sorted(null)
    say(f'-- cross-city {lab}: {len(shared)} shared elements (>= {MINN} dated objects each side); Spearman rho of mean within-city time percentile = {rho:+.3f}; '
        f'null (labels shuffled within type) mean {sum(null)/len(null):+.3f} [{null_s[int(0.025*len(null))]:+.3f}, {null_s[int(0.975*len(null))-1]:+.3f}], P_hi = {pval(rho, null):.3f}')
    # the elements most concentrated in both cities: do their bands agree?
    both = [w for w in shared if w in R_md and w in R_hp and R_md[w]['p_peak2'] < 0.1 and R_hp[w]['p_peak2'] < 0.1] if ka == kb == 'depth' else []
    if ka == kb == 'depth':
        say(f'   elements with P_area < 0.10 in BOTH cities (depth): {[(w, R_md[w]["modal"], R_hp[w]["modal"]) for w in both]}' if both else '   elements concentrated (P_area < 0.10) in both cities: none')
json.dump(dict(md=R_md, hp=R_hp, hf=R_hf, mp=R_mp), open(ROOT + f'data/derived/dark/loop49_c2_{LV}.json', 'w'), indent=1, default=str)
