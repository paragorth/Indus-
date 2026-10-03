"""S-DARK-49 cycle 5: the cross-city time agreement of cycle 2 (Spearman rho +0.27 to +0.32 between a sign's mean depth percentile at
Mohenjo-daro and at Harappa, P 0.002-0.022 against depth shuffled within site x coarse object type) - is it a shared time signal (as a
year-name field would give) or object-mix / sub-type / area leakage?  Stricter nulls and splits:
 (a) depth shuffled within site x FULL object type string (TAB:B vs TAB:I, SEAL:S vs SEAL:R ...) x area-section;
 (b) seals only; tablets only; non-frame signs only (openers, markers, closers, suffixes, numerals removed);
 (c) leave-one-sign-out influence and the signs that carry the agreement;
 (d) the same on IM77's own levels with object_type x locus strata (independent transcription of largely the same objects);
 (e) the direction: do the agreeing signs sit DEEP (older) in both cities or SHALLOW in both, and is it the tablet / seal split?
Usage: python3 tools/dark_loop49_c5.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, csv, re, json, math, collections, random, statistics
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import load_faces, OPEN, MARK, SUF, CL, NUM, pval
ROOT = '/home/user/Indus-/'
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
MINN = 8
rnd = random.Random(495)
OUT = open(ROOT + f'data/derived/dark/loop49_c5_{LV}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()
FRAME = set(OPEN) | set(MARK) | set(SUF) | set(CL) | set(NUM) | {741, 742, 745}

def depth_ft(s):
    m = re.match(r'^-([\d.]+)\s*ft', s or '', re.I)
    if not m: return None
    try: return float(m.group(1).replace('..', '.'))
    except ValueError: return None
meta = {}
for r in csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')):
    oid = r['id'].split('.')[0]
    if oid not in meta: meta[oid] = dict(depth=depth_ft(r['depth']), area=r['area-section'] or '--', ftype=r['type'], type2=r['type'].split(':')[0])
objs = load_faces(LV)
OBJ = []
for oid, o in objs.items():
    if o['site'] not in ('Mohenjo-daro', 'Harappa'): continue
    signs = set(w for f in o['faces'] for w in f['seq'])
    m = meta[oid]
    if signs and m['depth'] is not None: OBJ.append(dict(site=o['site'], signs=signs, depth=m['depth'], area=m['area'], ftype=m['ftype'], type2=m['type2']))
say(f'# S-DARK-49 cycle 5, level {LV}, nperm {NPERM}: cross-city time agreement under stricter nulls. Dated objects MD {sum(o["site"]=="Mohenjo-daro" for o in OBJ)}, HP {sum(o["site"]=="Harappa" for o in OBJ)}')
say(f'  Harappa full types among dated: {collections.Counter(o["ftype"] for o in OBJ if o["site"]=="Harappa").most_common(8)}')
say(f'  Mohenjo-daro full types among dated: {collections.Counter(o["ftype"] for o in OBJ if o["site"]=="Mohenjo-daro").most_common(8)}')

def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
            for k in range(i, j + 1): r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy) if sxx and syy else 0.0

def agreement(D_md, D_hp, strata_fn, label, signs_filter=lambda w: True, nperm=NPERM, show=True):
    out = {}
    for site, D in (('MD', D_md), ('HP', D_hp)):
        vals = [o['depth'] for o in D]; srt = sorted(vals); n = len(srt)
        import bisect
        pct = {}
        for o in D:
            lo = bisect.bisect_left(srt, o['depth']); hi = bisect.bisect_right(srt, o['depth']); o['pct'] = (lo + 0.5 * (hi - lo)) / n
        cnt = collections.Counter(w for o in D for w in o['signs'])
        out[site] = (D, cnt)
    shared = [w for w in out['MD'][1] if out['MD'][1][w] >= MINN and out['HP'][1].get(w, 0) >= MINN and signs_filter(w)]
    if len(shared) < 6:
        say(f'-- {label}: only {len(shared)} shared signs'); return None
    def means(D, key):
        acc = collections.defaultdict(list)
        for o in D:
            for w in o['signs']:
                if w in sset: acc[w].append(o[key])
        return {w: statistics.mean(acc[w]) for w in shared}
    sset = set(shared)
    ma = means(D_md, 'pct'); mb = means(D_hp, 'pct')
    rho = spearman([ma[w] for w in shared], [mb[w] for w in shared])
    null = []
    for it in range(nperm):
        for D in (D_md, D_hp):
            groups = collections.defaultdict(list)
            for o in D: groups[strata_fn(o)].append(o)
            for g in groups.values():
                vs = [o['pct'] for o in g]; rnd.shuffle(vs)
                for o, x in zip(g, vs): o['tmp'] = x
        na = means(D_md, 'tmp'); nb = means(D_hp, 'tmp')
        null.append(spearman([na[w] for w in shared], [nb[w] for w in shared]))
    ns = sorted(null)
    if show:
        say(f'-- {label}: {len(shared)} shared signs; rho = {rho:+.3f}; null mean {sum(null)/len(null):+.3f} [{ns[int(0.025*len(ns))]:+.3f}, {ns[int(0.975*len(ns))-1]:+.3f}] P_hi = {pval(rho, null):.3f}')
    return rho, null, shared, ma, mb

MD = [o for o in OBJ if o['site'] == 'Mohenjo-daro']; HP = [o for o in OBJ if o['site'] == 'Harappa']
say('\n== (a) nulls of increasing strictness, all dated objects, all signs')
r0 = agreement(MD, HP, lambda o: o['type2'], 'coarse type (cycle 2 null)')
r1 = agreement(MD, HP, lambda o: o['ftype'], 'FULL object type')
r2 = agreement(MD, HP, lambda o: (o['ftype'], o['area']), 'FULL object type x area-section')
say('\n== (b) splits (full type x area null)')
agreement([o for o in MD if o['type2'] == 'SEAL'], [o for o in HP if o['type2'] == 'SEAL'], lambda o: (o['ftype'], o['area']), 'SEALS only')
agreement([o for o in MD if o['type2'] == 'TAB'], [o for o in HP if o['type2'] == 'TAB'], lambda o: (o['ftype'], o['area']), 'TABLETS only')
agreement(MD, HP, lambda o: (o['ftype'], o['area']), 'all objects, NON-FRAME signs only (no opener/marker/closer/suffix/numeral/marked jar)', signs_filter=lambda w: w not in FRAME)
agreement(MD, HP, lambda o: (o['ftype'], o['area']), 'all objects, FRAME signs only', signs_filter=lambda w: w in FRAME)
say('\n== (c) which signs carry the agreement (full type x area null): leave-one-out drop in rho, and the per-sign percentiles')
if r2:
    rho, null, shared, ma, mb = r2
    infl = []
    for w in shared:
        rest = [v for v in shared if v != w]
        infl.append((rho - spearman([ma[v] for v in rest], [mb[v] for v in rest]), w))
    infl.sort(reverse=True)
    say('   signs whose removal lowers rho most (drop, sign, MD pct, HP pct): ' + ', '.join(f'{d:+.3f} W{w} {ma[w]:.2f}/{mb[w]:.2f}' for d, w in infl[:10]))
    say('   signs whose removal raises rho most: ' + ', '.join(f'{d:+.3f} W{w} {ma[w]:.2f}/{mb[w]:.2f}' for d, w in infl[-6:]))
    deep = [w for w in shared if ma[w] > 0.55 and mb[w] > 0.55]; shallow = [w for w in shared if ma[w] < 0.45 and mb[w] < 0.45]
    say(f'   signs deep (older, pct > 0.55) in BOTH cities: {[(w, round(ma[w],2), round(mb[w],2)) for w in deep]}')
    say(f'   signs shallow (later, pct < 0.45) in BOTH cities: {[(w, round(ma[w],2), round(mb[w],2)) for w in shallow]}')
    say(f'   mean |pct - 0.5| over shared signs: MD {statistics.mean(abs(ma[w]-0.5) for w in shared):.3f}, HP {statistics.mean(abs(mb[w]-0.5) for w in shared):.3f} (a year-name would sit far from 0.5 in both)')

# (d) IM77 levels, object_type x locus strata
say('\n== (d) IM77 levels (ft below datum): same agreement test, strata object_type / object_type x locus')
rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
TX = {}
for r in rows:
    toks = [int(x) for x in r['signs_reading_order'].split() if x.isdigit() and int(x) != 0]
    if not toks: continue
    t = TX.setdefault(r['text_no'], dict(site=r['site'], ftype=r['object_type'], type2=r['object_type'], area=r['locus'], signs=set(), depth=None))
    t['signs'].update(toks)
    try:
        lv = float(r['level'])
        if lv < 0: t['depth'] = -lv
    except ValueError: pass
IMD = [t for t in TX.values() if t['site'] == 'Mohenjodaro' and t['depth']]; IHP = [t for t in TX.values() if t['site'] == 'Harappa' and t['depth']]
say(f'   IM77 dated texts MD {len(IMD)}, HP {len(IHP)}; HP object types {collections.Counter(t["ftype"] for t in IHP).most_common(5)}')
agreement(IMD, IHP, lambda o: o['ftype'], 'IM77, object_type null')
ri = agreement(IMD, IHP, lambda o: (o['ftype'], o['area']), 'IM77, object_type x locus null')
agreement([t for t in IMD if t['ftype'] == 'seal'], [t for t in IHP if t['ftype'] == 'seal'], lambda o: (o['ftype'], o['area']), 'IM77 SEALS only, locus null')
M_FRAME = {267, 391, 293, 150, 99, 100, 123, 176, 1, 342, 162, 169, 15, 254, 12, 211, 60, 328, 343, 345, 86, 87, 89, 95, 97, 98, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 114, 121}
agreement(IMD, IHP, lambda o: (o['ftype'], o['area']), 'IM77 all texts, NON-FRAME M signs only', signs_filter=lambda w: w not in M_FRAME)
if ri:
    rho, null, shared, ma, mb = ri
    deep = [w for w in shared if ma[w] > 0.55 and mb[w] > 0.55]; shallow = [w for w in shared if ma[w] < 0.45 and mb[w] < 0.45]
    say(f'   IM77 signs deep in both: {[(w, round(ma[w],2), round(mb[w],2)) for w in deep]}; shallow in both: {[(w, round(ma[w],2), round(mb[w],2)) for w in shallow]}')
