#!/usr/bin/env python3
"""Loop 23: pots as a Rosetta for the numerals.
Usage: python3 loop23_engine.py <cycle 1|2|3|4> [seq_raw|seq_strong|seq_all]
Writes nothing itself; print to stdout and redirect.
Data: data/raw/inscriptions.csv (pot physical fields) + data/derived/sign_allographs_levels.json (merges).
"""
import csv, json, sys, re, random, math, collections, statistics as st
from itertools import combinations

ROOT = '/home/user/Indus-'
random.seed(23)
NPERM = 2000

SHORT = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10}
TALL = {31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7}
VAL = {**SHORT, **TALL}
JAR = {740}; JARFAM = {740, 741, 742, 743, 744, 745}
TREE = {390, 405, 407}; FISH = {220}
OPENERS = {817, 861, 820, 920, 692}
CLOSERS = JARFAM | {520, 162, 169, 15, 254, 12}  # W-side closers used in GRAMMAR (jar family + arrow); others are M-numbers kept for safety

level = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
L = json.load(open(f'{ROOT}/data/derived/sign_allographs_levels.json'))
MERGE = {}
for m in L['merges']:
    if level == 'seq_strong' and m['level'] == 'strong':
        MERGE[m['form']] = m['into']
    elif level == 'seq_all' and m['level'] in ('strong', 'probable'):
        MERGE[m['form']] = m['into']

def canon(s):
    seen = set()
    while s in MERGE and s not in seen:
        seen.add(s); s = MERGE[s]
    return s

def parse_text(t):
    """'+740-900/003-999+' -> list of ints (first alternative of a slash), None for 000 (lost)."""
    if not t: return None, False, False
    lead_lost = t.startswith(']'); trail_lost = t.endswith('[')
    body = t.strip('+[]')
    out = []
    for tok in body.split('-'):
        tok = tok.split('/')[0]
        if not re.match(r'^\d+$', tok): continue
        v = int(tok)
        out.append(None if v == 0 else canon(v))
    return out, lead_lost, trail_lost

def num(x):
    """'6 - 9' -> 7.5 ; '2.5cm' -> 25 ; '14mm' -> 14 ; '0'/'' -> None."""
    if x is None: return None
    x = x.strip()
    if x in ('', '0', '-'): return None
    m = re.match(r'^([\d.]+)\s*-\s*([\d.]+)$', x)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r'^([\d.]+)\s*cm$', x)
    if m: return float(m.group(1)) * 10
    m = re.match(r'^([\d.]+)\s*mm$', x)
    if m: return float(m.group(1))
    try: return float(x)
    except ValueError: return None

rows = list(csv.DictReader(open(f'{ROOT}/data/raw/inscriptions.csv')))
POTS = []
for r in rows:
    if not r['type'].startswith('POT'): continue
    if not r['id'].endswith('.1'): continue   # one record per object
    seq, ll, tl = parse_text(r['text'])
    if seq is None: continue
    H = num(r['horizontal(mm)']); V = num(r['vertical(mm)']); T = num(r['thickness(mm)'])
    # fall back on h/v/th; plain numbers there are mm when > 20 (Kalibangan sherds) else inches/cm ambiguity -> skip
    if H is None:
        h = num(r['h']);  H = h if (h is not None and (r['h'].endswith('cm') or r['h'].endswith('mm') or h > 20)) else None
    if V is None:
        v = num(r['v']);  V = v if (v is not None and (r['v'].endswith('cm') or r['v'].endswith('mm') or v > 20)) else None
    if T is None:
        T = num(r['th'])
    sub = r['type'].split(':')[-1].lower() if ':' in r['type'] else r['type']
    POTS.append(dict(id=r['id'], site=r['site'], sub=sub, type=r['type'], area=r['area-section'], period=r['period'],
                     pres=r['preservation'], complete=r['complete'], cond=r['condition'].capitalize(), mat=r['material'],
                     col=r['color'], seq=seq, ll=ll, tl=tl, H=H, V=V, T=T, text=r['text'], idno=r['excavation-idno']))

def whole(p): return not p['ll'] and not p['tl'] and None not in p['seq']
def region(p): return p['site'] if p['site'] in ('Harappa', 'Kalibangan') else 'rest'

def spearman(x, y):
    n = len(x)
    if n < 3: return float('nan')
    def rank(a):
        order = sorted(range(n), key=lambda i: a[i]); r = [0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')

def strat_perm_spearman(pairs, strata, nperm=NPERM):
    """pairs: list of (x,y); strata: list of keys. Null: permute y within stratum."""
    x = [a for a, b in pairs]; y = [b for a, b in pairs]
    obs = spearman(x, y)
    if math.isnan(obs): return obs, float('nan'), 0
    idx = collections.defaultdict(list)
    for i, s in enumerate(strata): idx[s].append(i)
    cnt = 0
    for _ in range(nperm):
        yp = y[:]
        for s, ii in idx.items():
            vals = [y[i] for i in ii]; random.shuffle(vals)
            for i, v in zip(ii, vals): yp[i] = v
        r = spearman(x, yp)
        if not math.isnan(r) and abs(r) >= abs(obs) - 1e-12: cnt += 1
    return obs, (cnt + 1) / (nperm + 1), len(pairs)

def strat_perm_groupdiff(vals, flag, strata, nperm=NPERM):
    """median(vals|flag) - median(vals|not flag), flag permuted within strata (two-sided)."""
    def stat(f):
        a = [v for v, g in zip(vals, f) if g]; b = [v for v, g in zip(vals, f) if not g]
        if len(a) < 2 or len(b) < 2: return float('nan')
        return st.median(a) - st.median(b)
    obs = stat(flag)
    if math.isnan(obs): return obs, float('nan'), sum(flag), len(flag) - sum(flag)
    idx = collections.defaultdict(list)
    for i, s in enumerate(strata): idx[s].append(i)
    cnt = 0
    for _ in range(nperm):
        fp = flag[:]
        for s, ii in idx.items():
            vals_ = [flag[i] for i in ii]; random.shuffle(vals_)
            for i, v in zip(ii, vals_): fp[i] = v
        s_ = stat(fp)
        if not math.isnan(s_) and abs(s_) >= abs(obs) - 1e-12: cnt += 1
    return obs, (cnt + 1) / (nperm + 1), sum(flag), len(flag) - sum(flag)

def P(*a): print(*a); sys.stdout.flush()

# ------------------------------------------------------------------ cycle 1
def cycle1():
    P(f'== LOOP 23 CYCLE 1  ({level})  pots as capacity labels: numeral value vs pot size ==')
    P(f'pots (one record per object): {len(POTS)}')
    P('\n-- (e) DIMENSION COVERAGE, honestly --')
    anyd = [p for p in POTS if p['H'] or p['V'] or p['T']]
    P(f'pots with any dimension (H, V or thickness): {len(anyd)} of {len(POTS)}')
    c = collections.Counter((p['sub'], p['pres']) for p in anyd)
    for k, v in sorted(c.items()): P(f'  {k}: {v}')
    P('  by site:', dict(collections.Counter(p["site"] for p in anyd)))
    P('  sub=s (seal-stamped) dimensions are the STAMP impression (10-35 mm), i.e. the seal, not the vessel.')
    comp = [p for p in anyd if p['pres'] == 'complete' and p['sub'] == 'g']
    P(f'  complete graffito vessels with a dimension: {len(comp)} -> ' + '; '.join(f"{p['id']} {p['site']} H{p['H']} V{p['V']} T{p['T']} {p['text']}" for p in comp))
    thick = [p for p in anyd if p['T'] and p['sub'] == 'g']
    P(f'  graffito sherds with wall thickness (a proxy for vessel size): {len(thick)}, sites {dict(collections.Counter(p["site"] for p in thick))}')
    P('  Conclusion on coverage: H and V of fragments are SHERD sizes, not vessel sizes. Only wall thickness is a vessel-size proxy. No shape class is recorded for 612 of 635 pots.')

    def lone_numeral(p):
        return whole(p) and len(p['seq']) == 1 and p['seq'][0] in VAL
    def numeral_first(p):
        s = [z for z in p['seq'] if z is not None]
        return bool(s) and s[0] in VAL and not p['ll']

    for label, dimkey in (('wall thickness', 'T'), ('sherd H (max dimension)', 'H'), ('sherd V', 'V')):
        P(f'\n-- (a) numeral value vs {label} --')
        for sel_name, sel in (('lone numeral', lone_numeral), ('numeral-first text', numeral_first)):
            for ser_name, ser in (('tall', TALL), ('short', SHORT), ('both', VAL)):
                pr, strata = [], []
                for p in POTS:
                    if p[dimkey] and sel(p):
                        n = [z for z in p['seq'] if z is not None][0]
                        if n in ser:
                            pr.append((ser[n], p[dimkey])); strata.append((p['site'], p['sub']))
                if len(pr) >= 4:
                    rho, pv, n = strat_perm_spearman(pr, strata)
                    P(f'  {sel_name:18s} {ser_name:5s} n={n:3d} rho={rho:+.2f} p_perm(site x subtype)={pv:.3f}  pairs={sorted(pr)}')
                else:
                    P(f'  {sel_name:18s} {ser_name:5s} n={len(pr)} (too few)  pairs={sorted(pr)}')

    P('\n-- (a2) is a lone sign a size class? dims of pots carrying X alone vs other pots (median diff, label permuted within site x subtype) --')
    for dimkey in ('T', 'H', 'V'):
        for name, S in (('jar W740 alone', JAR), ('tree alone', TREE), ('fish alone', FISH), ('any numeral alone', set(VAL)), ('jar anywhere', JAR)):
            vals, flag, strata = [], [], []
            for p in POTS:
                if not p[dimkey]: continue
                if name.endswith('alone'):
                    f = whole(p) and len(p['seq']) == 1 and p['seq'][0] in S
                else:
                    f = any(z in S for z in p['seq'] if z is not None)
                vals.append(p[dimkey]); flag.append(f); strata.append((p['site'], p['sub']))
            d, pv, na, nb = strat_perm_groupdiff(vals, flag, strata)
            P(f'  {dimkey} {name:18s} n_X={na:3d} n_other={nb:3d} median diff={d if isinstance(d,float) and math.isnan(d) else round(d,1)} p={pv:.3f}')

    P('\n-- (a3) within-text consistency check: two numerals on one pot (e.g. 34-999-34): same value? (needs no dimensions) --')
    same = diff = 0; ex = []
    for p in POTS:
        ns = [VAL[z] for z in p['seq'] if z in VAL]
        if len(ns) >= 2:
            if len(set(ns)) == 1: same += 1
            else: diff += 1
            ex.append(p['text'])
    P(f'  pots with >=2 numerals: same value {same}, different {diff}; texts: {ex}')

# ------------------------------------------------------------------ cycle 2
def cycle2():
    P(f'== LOOP 23 CYCLE 2  ({level})  pre-firing stamps and marks: one workshop product line? ==')
    stamped = [p for p in POTS if p['sub'] == 's']
    pre = [p for p in POTS if p['sub'] == 'p']
    P(f'seal-stamped pots: {len(stamped)}; pre-firing incised: {len(pre)}')
    def key(p): return '-'.join(str(z) if z else '0' for z in p['seq'])
    P('\n-- stamped texts (whole) by count, site, find area, period, stamp size --')
    groups = collections.defaultdict(list)
    for p in stamped:
        if whole(p): groups[key(p)].append(p)
    for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(g) < 2: continue
        sizes = [(p['H'], p['V']) for p in g if p['H'] or p['V']]
        P(f'  {k:22s} n={len(g):2d} sites={dict(collections.Counter(p["site"] for p in g))} areas={dict(collections.Counter(p["area"] for p in g))} periods={dict(collections.Counter(p["period"] for p in g))} pres={dict(collections.Counter(p["pres"] for p in g))} stamp mm={sizes}')
    P(f'  singletons: {sum(1 for g in groups.values() if len(g)==1)} texts')

    P('\n-- (b1) W440: concentration in one find area at Harappa vs other Harappa stamped pots (text labels permuted among Harappa stamped pots with a known area) --')
    H = [p for p in stamped if p['site'] == 'Harappa' and p['area'] != '--' and whole(p)]
    P(f'  Harappa stamped pots with an area: {len(H)}; areas: {dict(collections.Counter(p["area"] for p in H))}')
    def maxshare(labels, areas, target):
        A = [a for l, a in zip(labels, areas) if l == target]
        if not A: return float('nan')
        return collections.Counter(A).most_common(1)[0][1] / len(A)
    labels = [key(p) for p in H]; areas = [p['area'] for p in H]
    for target in [k for k, g in groups.items() if len(g) >= 3 and any(p['site'] == 'Harappa' for p in g)]:
        obs = maxshare(labels, areas, target)
        if math.isnan(obs): continue
        cnt = 0
        for _ in range(NPERM):
            lp = labels[:]; random.shuffle(lp)
            if maxshare(lp, areas, target) >= obs - 1e-12: cnt += 1
        n_t = labels.count(target)
        P(f'  text {target}: n={n_t}, largest-area share={obs:.2f} (areas {dict(collections.Counter(a for l,a in zip(labels,areas) if l==target))}), p_perm={(cnt+1)/(NPERM+1):.3f}')

    P('\n-- (b2) stamp impression size: are the W440 stamps one die? spread of H/V among W440 vs among other stamped texts --')
    for k, g in groups.items():
        dims = [d for p in g for d in (p['H'], p['V']) if d]
        if len(dims) >= 3:
            P(f'  {k:12s} n_dims={len(dims)} mean={st.mean(dims):.1f} sd={st.pstdev(dims):.1f} range={min(dims)}-{max(dims)} mm')
    alld = [d for p in stamped for d in (p['H'], p['V']) if d and 5 < d < 40]
    P(f'  all stamped (5-40 mm): n={len(alld)} mean={st.mean(alld):.1f} sd={st.pstdev(alld):.1f}')
    # Permutation: is W440 sd smaller than random same-size draws of stamp dims?
    w = [d for p in groups.get('440', []) for d in (p['H'], p['V']) if d]
    if len(w) >= 3:
        obs = st.pstdev(w); cnt = 0
        for _ in range(NPERM):
            s = random.sample(alld, len(w))
            if st.pstdev(s) <= obs + 1e-12: cnt += 1
        P(f'  W440 sd={obs:.2f} vs random draws of {len(w)} stamp dims: p(sd <= obs)={(cnt+1)/(NPERM+1):.3f}')

    P('\n-- (b3) pre-firing incised marks: text, site, preservation, colour --')
    for p in pre:
        P(f"  {p['id']:7s} {p['site']:14s} {p['text']:22s} pres={p['pres']} complete={p['complete']} mat={p['mat']} col={p['col']} period={p['period']} area={p['area']}")
    c = collections.Counter(z for p in pre for z in p['seq'] if z)
    P('  pre-firing sign counts:', c.most_common())
    op = sum(1 for p in pre if any(z in OPENERS for z in p['seq'] if z)); jf = sum(1 for p in pre if any(z in JARFAM for z in p['seq'] if z))
    P(f'  pre-firing pots with an opener sign: {op}/{len(pre)}; with a jar-family sign: {jf}/{len(pre)}; numeral: {sum(1 for p in pre if any(z in VAL for z in p["seq"] if z))}/{len(pre)}')
    gra = [p for p in POTS if p['sub'] == 'g']
    op_g = sum(1 for p in gra if any(z in OPENERS for z in p['seq'] if z)); jf_g = sum(1 for p in gra if any(z in JARFAM for z in p['seq'] if z)); nu_g = sum(1 for p in gra if any(z in VAL for z in p['seq'] if z))
    P(f'  graffiti for comparison: opener {op_g}/{len(gra)} ({op_g/len(gra):.0%}), jar-family {jf_g}/{len(gra)} ({jf_g/len(gra):.0%}), numeral {nu_g}/{len(gra)} ({nu_g/len(gra):.0%})')
    # Fisher-ish permutation: opener share pre vs graffiti, labels permuted within site
    allp = pre + gra; flag = [1] * len(pre) + [0] * len(gra); strata = [p['site'] for p in allp]
    for name, S in (('opener', OPENERS), ('numeral', set(VAL)), ('jar-family', JARFAM)):
        feat = [1 if any(z in S for z in p['seq'] if z) else 0 for p in allp]
        def stat(f):
            a = [x for x, g in zip(feat, f) if g]; b = [x for x, g in zip(feat, f) if not g]
            return sum(a) / len(a) - sum(b) / len(b)
        obs = stat(flag); cnt = 0
        idx = collections.defaultdict(list)
        for i, s in enumerate(strata): idx[s].append(i)
        for _ in range(NPERM):
            fp = flag[:]
            for s, ii in idx.items():
                v = [flag[i] for i in ii]; random.shuffle(v)
                for i, x in zip(ii, v): fp[i] = x
            if abs(stat(fp)) >= abs(obs) - 1e-12: cnt += 1
        P(f'  pre-firing minus graffiti share of {name}: {obs:+.2f}, p_perm(within site)={(cnt+1)/(NPERM+1):.3f}')

# ------------------------------------------------------------------ cycle 3
def cycle3():
    P(f'== LOOP 23 CYCLE 3  ({level})  one-sign labels vs multi-sign pot texts: do multi-sign pots carry the seal frame, on different pots? ==')
    seals = []
    for r in rows:
        if r['type'].startswith('SEAL') and r['id'].endswith('.1'):
            seq, ll, tl = parse_text(r['text'])
            if seq and not ll and not tl and None not in seq:
                seals.append(dict(seq=seq, site=r['site']))
    P(f'whole seal texts: {len(seals)}')
    def opener_first(s): return s[0] in OPENERS
    def jar_last(s): return s[-1] in JARFAM
    def arrow_last(s): return s[-1] == 520
    def framed(s): return opener_first(s) or jar_last(s)
    W = [p for p in POTS if whole(p)]
    one = [p for p in W if len(p['seq']) == 1]; multi = [p for p in W if len(p['seq']) >= 2]; long_ = [p for p in W if len(p['seq']) >= 3]
    P(f'whole pot texts: {len(W)}; one-sign {len(one)}; >=2 signs {len(multi)}; >=3 signs {len(long_)}')
    P('\n-- frame share, pots (>=2 and >=3 signs) vs seals length-matched, by region --')
    for reg in ('Harappa', 'Kalibangan', 'rest', 'ALL'):
        for mn in (2, 3):
            pp = [p['seq'] for p in W if len(p['seq']) >= mn and (reg == 'ALL' or region(p) == reg)]
            if not pp: continue
            lens = collections.Counter(len(s) for s in pp)
            ss = [s['seq'] for s in seals if len(s['seq']) >= mn and (reg == 'ALL' or (s['site'] if s['site'] in ('Harappa', 'Kalibangan') else 'rest') == reg)]
            # length-matched seal rate: weight seal texts by pot length distribution
            def rate(fn, texts, weights):
                num = den = 0
                for L_, w in weights.items():
                    sub = [t for t in texts if len(t) == L_]
                    if sub: num += w * sum(fn(t) for t in sub) / len(sub); den += w
                return num / den if den else float('nan')
            po = sum(opener_first(s) for s in pp) / len(pp); pj = sum(jar_last(s) for s in pp) / len(pp); pf = sum(framed(s) for s in pp) / len(pp)
            so = rate(opener_first, ss, lens); sj = rate(jar_last, ss, lens); sf = rate(framed, ss, lens)
            P(f'  {reg:10s} >= {mn} signs: pots n={len(pp):3d} opener-first {po:.2f} jar-last {pj:.2f} framed {pf:.2f} | seals length-matched (n={len(ss)}) opener {so:.2f} jar {sj:.2f} framed {sf:.2f}')
    # pot one-sign vocabulary: is it the closer-slot vocabulary or the numeral vocabulary?
    P('\n-- one-sign pot labels by region --')
    for reg in ('Harappa', 'Kalibangan', 'rest'):
        c = collections.Counter(p['seq'][0] for p in one if region(p) == reg)
        P(f'  {reg:10s} n={sum(c.values())} top: {c.most_common(10)}  numeral share {sum(v for k,v in c.items() if k in VAL)/max(1,sum(c.values())):.2f} jar {c[740]/max(1,sum(c.values())):.2f}')
    P('\n-- do multi-sign pot texts come on different pots? (features: subtype, preservation=complete, condition, period, colour) multi flag permuted within site --')
    feats = {
        'seal-stamped (sub s)': lambda p: p['sub'] == 's',
        'pre-firing (sub p)': lambda p: p['sub'] == 'p',
        'vessel complete': lambda p: p['pres'] == 'complete',
        'condition Good/Fine': lambda p: p['cond'] in ('Good', 'Fine'),
        'Harappa period 3': lambda p: p['period'] == '3',
        'Harappa period 2': lambda p: p['period'].startswith('2'),
        'has thickness record': lambda p: bool(p['T']),
    }
    flag = [1 if len(p['seq']) >= 2 else 0 for p in W]; strata = [p['site'] for p in W]
    idx = collections.defaultdict(list)
    for i, s in enumerate(strata): idx[s].append(i)
    for name, fn in feats.items():
        feat = [1 if fn(p) else 0 for p in W]
        def stat(f):
            a = [x for x, g in zip(feat, f) if g]; b = [x for x, g in zip(feat, f) if not g]
            return sum(a) / len(a) - sum(b) / len(b)
        obs = stat(flag); cnt = 0
        for _ in range(NPERM):
            fp = flag[:]
            for s, ii in idx.items():
                v = [flag[i] for i in ii]; random.shuffle(v)
                for i, x in zip(ii, v): fp[i] = x
            if abs(stat(fp)) >= abs(obs) - 1e-12: cnt += 1
        a = sum(x for x, g in zip(feat, flag) if g); b = sum(x for x, g in zip(feat, flag) if not g)
        P(f'  {name:22s} multi {a}/{sum(flag)} vs one-sign {b}/{len(flag)-sum(flag)}  diff={obs:+.2f}  p_perm(within site)={(cnt+1)/(NPERM+1):.3f}')
    # thickness: multi vs one-sign, within site x subtype
    vals, fl, strt = [], [], []
    for p in W:
        if p['T']: vals.append(p['T']); fl.append(len(p['seq']) >= 2); strt.append((p['site'], p['sub']))
    d, pv, na, nb = strat_perm_groupdiff(vals, fl, strt)
    P(f'  wall thickness: multi-sign n={na} vs one-sign n={nb}, median diff={d:+.1f} mm, p={pv:.3f}')
    vals, fl, strt = [], [], []
    for p in W:
        if p['H']: vals.append(p['H']); fl.append(len(p['seq']) >= 2); strt.append((p['site'], p['sub']))
    d, pv, na, nb = strat_perm_groupdiff(vals, fl, strt)
    P(f'  sherd H: multi-sign n={na} vs one-sign n={nb}, median diff={d:+.1f} mm, p={pv:.3f}')
    # Do multi-sign pot texts ever contain a numeral + good (count) as on tablets?
    P('\n-- numeral use inside multi-sign pot texts: numeral followed by what? --')
    foll = collections.Counter(); prec = collections.Counter()
    for p in W:
        s = p['seq']
        for i, z in enumerate(s):
            if z in VAL:
                foll[s[i + 1] if i + 1 < len(s) else 'END'] += 1
                prec[s[i - 1] if i > 0 else 'START'] += 1
    P('  sign after a numeral:', foll.most_common(12))
    P('  sign before a numeral:', prec.most_common(12))

# ------------------------------------------------------------------ cycle 4
def cycle4():
    P(f'== LOOP 23 CYCLE 4  ({level})  the lone pot numerals themselves: series, value hump, minimum-lot rule, held-out by region ==')
    W = [p for p in POTS if whole(p)]
    lone = [p for p in W if len(p['seq']) == 1 and p['seq'][0] in VAL]
    P(f'lone-numeral pots: {len(lone)}')
    for reg in ('Harappa', 'Kalibangan', 'rest', 'ALL'):
        sel = [p for p in lone if reg == 'ALL' or region(p) == reg]
        tall = collections.Counter(TALL[p['seq'][0]] for p in sel if p['seq'][0] in TALL)
        short = collections.Counter(SHORT[p['seq'][0]] for p in sel if p['seq'][0] in SHORT)
        P(f'  {reg:10s} n={len(sel):3d} tall values {dict(sorted(tall.items()))}  short values {dict(sorted(short.items()))}  subtypes {dict(collections.Counter(p["sub"] for p in sel))}')
    # tablet count faces for the hump: tall numeral + W700 on TAB
    tab = collections.Counter()
    for r in rows:
        if r['type'].startswith('TAB') and r['id'].endswith('.1'):
            seq, ll, tl = parse_text(r['text'])
            if seq and len(seq) == 2 and seq[0] in TALL and seq[1] == 700: tab[TALL[seq[0]]] += 1
            if seq and len(seq) == 1 and seq[0] in TALL: tab[TALL[seq[0]]] += 1
    P(f'  tablet tall-count faces (tall n + W700, or tall n alone): {dict(sorted(tab.items()))}')
    # chi2 2/3/4 pots vs tablets, with permutation (labels pot/tablet shuffled)
    pt = collections.Counter(TALL[p['seq'][0]] for p in lone if p['seq'][0] in TALL)
    vals = [2, 3, 4]
    a = [pt.get(v, 0) for v in vals]; b = [tab.get(v, 0) for v in vals]
    def chi2(a, b):
        tot = sum(a) + sum(b); s = 0
        for i in range(len(a)):
            col = a[i] + b[i]
            for x, rs in ((a[i], sum(a)), (b[i], sum(b))):
                e = rs * col / tot
                if e > 0: s += (x - e) ** 2 / e
        return s
    obs = chi2(a, b)
    pool = [v for v, k in zip(vals, a) for _ in range(k)] + [v for v, k in zip(vals, b) for _ in range(k)]
    na = sum(a); cnt = 0
    for _ in range(NPERM):
        random.shuffle(pool)
        aa = collections.Counter(pool[:na]); bb = collections.Counter(pool[na:])
        if chi2([aa.get(v, 0) for v in vals], [bb.get(v, 0) for v in vals]) >= obs - 1e-12: cnt += 1
    P(f'  pots tall 2/3/4 = {a} vs tablets {b}: chi2={obs:.2f} p_perm={(cnt+1)/(NPERM+1):.3f}')
    # minimum-lot rule: lone short 1/2 on pots?
    s12 = [p for p in lone if p['seq'][0] in (1, 2, 12)]
    P(f'  lone SHORT 1 or 2 (W1, W2, W12) on pots: {len(s12)} -> {[(p["site"], p["text"]) for p in s12]}')
    t12 = [p for p in lone if p['seq'][0] in (31, 32)]
    P(f'  lone TALL 1 or 2 on pots: {len(t12)} -> {dict(collections.Counter(p["site"] for p in t12))}')
    # numeral + good on pots (count-like): numeral directly before a tree/jar/fish/700
    P('\n-- numeral + item pairs on pots (whole texts), which series --')
    pairs = collections.Counter()
    for p in W:
        s = p['seq']
        for i in range(len(s) - 1):
            if s[i] in VAL: pairs[('T' if s[i] in TALL else 'S', VAL[s[i]], s[i + 1])] += 1
    P('  ', pairs.most_common(25))
    # Value distribution: are pot lone numerals a smooth count distribution (like trees 3-8) or peaked like tablets?
    P('\n-- lone pot numerals: value frequencies all series, by site --')
    for site in ('Harappa', 'Kalibangan', 'Mohenjo-daro', 'Lothal', 'Nausharo'):
        c = collections.Counter((('T' if p['seq'][0] in TALL else 'S'), VAL[p['seq'][0]]) for p in lone if p['site'] == site)
        P(f'  {site:13s} {sorted(c.items())}')

if __name__ == '__main__':
    {'1': cycle1, '2': cycle2, '3': cycle3, '4': cycle4}[sys.argv[1]]()
