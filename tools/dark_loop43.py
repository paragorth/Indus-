"""S-DARK-43: was the code born complete or did it grow?
Chronology series (site x dating scheme), per-phase frame statistics, trend tests with phase labels permuted within
site x object-type strata, born-complete test on the earliest well-sampled phase, growth simulation for detectability,
sign complexity and allograph drift, and the pottery marks. Run on seq_raw / seq_strong / seq_all.
Usage: python3 tools/dark_loop43.py <cycle 1|2|3|4> [nperm]
"""
import csv, json, collections, re, sys, random, math
import numpy as np

CYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(43)
random.seed(43)
ROOT = '/home/user/Indus-/'
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
RAWROWS = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
raw = {r['cisi']: r for r in RAWROWS if r['cisi'] and r['cisi'] != '-'}
bridge = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
CPLX = {int(k): v for k, v in json.load(open(ROOT + 'data/derived/sign-complexity.json')).items()}
LEV = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
VARIANT_FORMS = {m['form']: m['into'] for m in LEV['merges']}          # non-canonical glyph forms (all levels)
VARIANT_SETS = collections.defaultdict(set)
for f, i in VARIANT_FORMS.items(): VARIANT_SETS[i] |= {f, i}
M2W = collections.defaultdict(set)
for w, ms in bridge.items():
    for m in ms: M2W[m].add(int(w))
LEVELS = ('seq_raw', 'seq_strong', 'seq_all')
OPEN = {817, 861, 820, 920}
MARK = {2, 60}
CLOSER_M = {342, 162, 169, 15, 254, 12, 211}
CLOSERS = set().union(*(M2W[m] for m in CLOSER_M)) | {740, 390, 405, 520, 154, 158, 527, 156, 151, 226, 617, 236, 700, 595}
SUFFIX = set(M2W[176]) | set(M2W[1])
NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34}
OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)

def depth_ft(s):
    m = re.match(r'^-([\d.]+)\s*ft', s or '', re.I)
    if not m: return None
    try: return float(m.group(1).replace('..', '.'))
    except ValueError: return None

# ---------------- chronology ----------------
def hp_harp(o):
    if o['site'] != 'Harappa' or o['period'].strip() != '3': return None
    return {'B': 0, 'B?': 0, 'B/C': 1, 'C': 2}.get(o['phase'].strip())
def hp_vats(o):
    ph = o['phase'].strip()
    if o['site'] != 'Harappa' or not ph.startswith('Stratum'): return None
    return {'VII': 0, 'VI': 0, 'V': 1, 'IV': 2, 'III': 3, 'II': 4, 'I': 5}.get(ph.split()[1])
def hp_fine(o):
    if o['site'] != 'Harappa': return None
    t = o['time']
    if t in ('Period 3B', 'Period 3B-1', 'Period 3B-2'): return 0
    if t == 'Period 3B/C': return 1
    if t in ('Period 3C', 'Period 3C-1', 'Period 3C-2'): return 2
    if t in ('Period 3C-3', 'Period 3C-4'): return 3
    if t in ('Period 4', 'Period 5A'): return 4
    return None
def md7(o):
    if o['site'] != 'Mohenjo-daro': return None
    per, ph = o['period'].strip(), o['phase'].strip()
    if per == 'Early': return 0
    if per == 'Intermediate': return {'III': 1, 'II': 2, 'I': 3}.get(ph)
    if per == 'Late': return {'III': 4, 'II': 5, 'I': 6, 'IA': 6, 'IB': 6}.get(ph)
    return None
def md_depth(o):
    if o['site'] != 'Mohenjo-daro': return None
    d = depth_ft(raw.get(o['cisi'], {}).get('depth', ''))
    if d is None: return None
    return 3 - min(int(d // 5), 3)          # 15+ ft = 0 (oldest) ... 0-5 ft = 3 (latest)
def hp_depth(o):
    if o['site'] != 'Harappa': return None
    d = depth_ft(raw.get(o['cisi'], {}).get('depth', ''))
    if d is None: return None
    return 3 - min(int(d // 5), 3)
def dhol(o):
    if o['site'] != 'Dholavira': return None
    return {'4': 0, '4/5': 1, '5': 2, '6': 3}.get(o['period'].strip())
def kali(o):
    if o['site'] != 'Kalibangan': return None
    return {'Early': 0, 'Middle': 1, 'Late': 2}.get(o['phase'].strip())
def lothal(o):
    if o['site'] != 'Lothal': return None
    m = re.match(r'Layer (\d+)', o['period'].strip())
    if not m: return None
    L = int(m.group(1)); return 0 if L >= 6 else 1 if L >= 4 else 2 if L == 3 else 3   # deeper layer = older
SERIES = [
    ('Harappa HARP phases 3B|3B/C|3C', hp_harp, ['3B', '3B/C', '3C']),
    ('Harappa Vats strata VI+VII..I', hp_vats, ['VI+VII', 'V', 'IV', 'III', 'II', 'I']),
    ('Harappa fine period 3B|3B/C|3C1-2|3C3-4|4-5', hp_fine, ['3B', '3B/C', '3C-1/2', '3C-3/4', '4/5']),
    ('Mohenjo-daro 7 levels', md7, ['Early', 'IntIII', 'IntII', 'IntI', 'LateIII', 'LateII', 'LateI']),
    ('Mohenjo-daro depth bands 15+|10-15|5-10|0-5 ft', md_depth, ['15+', '10-15', '5-10', '0-5']),
    ('Harappa depth bands 15+|10-15|5-10|0-5 ft', hp_depth, ['15+', '10-15', '5-10', '0-5']),
    ('Dholavira stages 4|4/5|5|6', dhol, ['4', '4/5', '5', '6']),
    ('Kalibangan Early|Middle|Late', kali, ['Early', 'Middle', 'Late']),
    ('Lothal layers 6+|4-5|3|2', lothal, ['L6+', 'L4-5', 'L3', 'L2']),
]
def type2(o):
    p = o['type'].split(':'); return p[0] if p[0] != 'TAB' else ':'.join(p[:2])
def tclass(o): return o['type'].split(':')[0]

# ---------------- per-text features ----------------
def opener_first(s): return int(s[0] in OPEN or (len(s) > 1 and s[0] == 692 and s[1] == 60))
def closer_final(s): return int(s[-1] in CLOSERS or (len(s) > 1 and s[-1] in SUFFIX and s[-2] in CLOSERS))
def jar_final(s): return int(s[-1] == 740 or (len(s) > 1 and s[-1] in SUFFIX and s[-2] == 740))
def w2_second(s):   # W2/W60 in slot 2 among texts that contain the marker
    if not (set(s) & MARK): return None
    return int(len(s) > 1 and s[1] in MARK)
def has_repeat(s):
    if len(s) < 3: return None
    return int(len(set(s)) < len(s))
def mean_cplx(s):
    v = [CPLX[x] for x in s if x in CPLX]; return float(np.mean(v)) if v else None
def variant_share(s_raw):   # share of tokens in a variant set that are the non-canonical form (seq_raw)
    toks = [x for x in s_raw if x in VARIANT_FORMS or x in VARIANT_SETS]
    if not toks: return None
    return float(np.mean([x in VARIANT_FORMS for x in toks]))

# fixed-pair partial order fitted on ALL dated-or-undated texts of the two big cities (S-DARK-19 recipe, light)
def fit_order(level):
    ab = collections.Counter(); seen = collections.Counter()
    for o in C:
        if o['site'] not in ('Mohenjo-daro', 'Harappa'): continue
        s = o[level]
        if len(s) < 2: continue
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if s[i] != s[j]:
                    ab[(s[i], s[j])] += 1
    fixed = {}
    for (a, b), n in ab.items():
        if a < b:
            m = ab.get((b, a), 0)
            if n + m >= 5:
                k = min(n, m); tot = n + m
                p = sum(math.comb(tot, i) for i in range(k + 1)) / 2 ** tot * 2
                if p < 0.05: fixed[(a, b)] = 1 if n > m else -1
    return fixed
def order_agree(s, fixed):
    ok = tot = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = s[i], s[j]
            if a == b: continue
            key = (min(a, b), max(a, b))
            if key in fixed:
                tot += 1; ok += int((fixed[key] == 1) == (a < b))
    return ok / tot if tot else None

def chao1(counter):
    f1 = sum(1 for v in counter.values() if v == 1); f2 = sum(1 for v in counter.values() if v == 2)
    S = len(counter)
    return S + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
def rarefy(tokens, n, reps=50):
    if len(tokens) < n: return None
    return float(np.mean([len(set(rng.choice(tokens, n, replace=False))) for _ in range(reps)]))
def js(p, q):
    keys = set(p) | set(q); P = np.array([p.get(k, 0) for k in keys], float); Q = np.array([q.get(k, 0) for k in keys], float)
    P /= P.sum(); Q /= Q.sum(); M = (P + Q) / 2
    def kl(a, b): return float(np.sum([x * math.log2(x / y) for x, y in zip(a, b) if x > 0]))
    return (kl(P, M) + kl(Q, M)) / 2

def build(level, fixed):
    """rows: dict(series index, phase, stratum, type class, features)"""
    rows = []
    for o in C:
        s = o[level]
        if len(s) < 2: continue
        for si, (name, f, labels) in enumerate(SERIES):
            ph = f(o)
            if ph is None: continue
            strat = (o['site'], type2(o)) if 'depth' not in name else (o['site'], o.get('area-section') or '--', type2(o))
            rows.append(dict(si=si, ph=ph, strat=strat, tc=tclass(o), seq=s, raw=o['seq_raw'],
                             opener=opener_first(s), closer=closer_final(s), jar=jar_final(s), w2=w2_second(s),
                             rep=has_repeat(s), length=len(s), cplx=mean_cplx(s), var=variant_share(o['seq_raw']),
                             order=order_agree(s, fixed), suffix=int(s[-1] in SUFFIX)))
    return rows

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0: return float('nan')
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

def perm_labels(ph, strat):
    """phase labels permuted within strata"""
    out = ph.copy()
    for st in set(strat):
        idx = [i for i, s in enumerate(strat) if s == st]
        if len(idx) > 1: out[idx] = rng.permutation(ph[idx])
    return out

FEATS = ['opener', 'closer', 'jar', 'w2', 'rep', 'length', 'order', 'suffix', 'cplx', 'var']

def trend_tests(rows, name, labels, nperm, tcs=('SEAL', 'TAB', None), feats=FEATS):
    """per series x type class: per-phase means, Spearman(phase, feature) and permutation P (two-sided), earliest vs rest"""
    res = []
    for tc in tcs:
        R = [r for r in rows if tc is None or r['tc'] == tc]
        n_ph = collections.Counter(r['ph'] for r in R)
        if len([p for p in n_ph if n_ph[p] >= 10]) < 2: continue
        ph = np.array([r['ph'] for r in R]); strat = [r['strat'] for r in R]
        tag = tc or 'ALL'
        say(f'  [{name} | {tag}] n per phase: ' + ', '.join(f'{labels[p]}={n_ph[p]}' for p in sorted(n_ph)))
        for ft in feats:
            vals = np.array([np.nan if r[ft] is None else r[ft] for r in R], float)
            ok = ~np.isnan(vals)
            if ok.sum() < 20: continue
            means = {p: float(np.nanmean(vals[(ph == p) & ok])) for p in sorted(n_ph) if ((ph == p) & ok).sum() >= 5}
            rho = spearman(ph[ok], vals[ok])
            # earliest well-sampled phase (n>=20 with feature) vs all phases
            early = min([p for p in sorted(n_ph) if ((ph == p) & ok).sum() >= 20], default=None)
            d_early = float(np.nanmean(vals[(ph == early) & ok]) - np.nanmean(vals[ok])) if early is not None else float('nan')
            null_r, null_d = [], []
            for _ in range(nperm):
                pp = perm_labels(ph, strat)
                null_r.append(spearman(pp[ok], vals[ok]))
                if early is not None:
                    null_d.append(float(np.nanmean(vals[(pp == early) & ok]) - np.nanmean(vals[ok])))
            null_r = np.array(null_r); P_r = float(np.mean(np.abs(null_r) >= abs(rho) - 1e-12)) if not np.isnan(rho) else float('nan')
            P_d = float(np.mean(np.abs(np.array(null_d)) >= abs(d_early) - 1e-12)) if early is not None else float('nan')
            sd_d = float(np.std(null_d)) if null_d else float('nan')
            res.append(dict(series=name, tc=tag, feat=ft, rho=rho, P=P_r, means=means, early=early, d_early=d_early, P_early=P_d, sd_early=sd_d, n=int(ok.sum())))
            say(f'    {ft:7s} rho={rho:+.3f} P={P_r:.3f} | ' + ' '.join(f'{labels[p]}:{m:.2f}' for p, m in means.items()) +
                (f' | earliest {labels[early]} - all = {d_early:+.3f} (null sd {sd_d:.3f}, P={P_d:.2f})' if early is not None else ''))
    return res

def phase_level(rows, name, labels, nperm, tc):
    """inventory (distinct, Chao1, rarefied), closer menu JS to pooled, repeat O/E, with permutation null for the trend"""
    R = [r for r in rows if tc is None or r['tc'] == tc]
    n_ph = collections.Counter(r['ph'] for r in R)
    phases = [p for p in sorted(n_ph) if n_ph[p] >= 15]
    if len(phases) < 2: return []
    ph = np.array([r['ph'] for r in R]); strat = [r['strat'] for r in R]
    pooled_menu = collections.Counter(r['seq'][-1] for r in R if r['closer'])
    nmin = min(sum(len(r['seq']) for r in R if r['ph'] == p) for p in phases)
    def stats(lab):
        out = {}
        for p in phases:
            S = [r for r, l in zip(R, lab) if l == p]
            toks = [x for r in S for x in r['seq']]; cnt = collections.Counter(toks)
            menu = collections.Counter(r['seq'][-1] for r in S if r['closer'])
            # repeat O/E: observed repeat share vs texts with signs drawn from the phase's token pool
            long = [r['seq'] for r in S if len(r['seq']) >= 3]
            if long:
                obs = np.mean([len(set(s)) < len(s) for s in long])
                pool = np.array(toks)
                exp = np.mean([len(set(rng.choice(pool, len(s)))) < len(s) for s in long for _ in range(3)])
                rep_oe = obs / exp if exp > 0 else float('nan')
            else: rep_oe = float('nan')
            out[p] = dict(distinct=len(cnt), chao1=chao1(cnt), rare=rarefy(np.array(toks), nmin, 20), tokens=len(toks),
                          menu_js=js(menu, pooled_menu) if menu else float('nan'), n_closer=sum(menu.values()),
                          menu_top=menu.most_common(4), rep_oe=rep_oe, n=len(S))
        return out
    obs = stats(ph)
    keys = ['distinct', 'chao1', 'rare', 'menu_js', 'rep_oe']
    rho_obs = {k: spearman(phases, [obs[p][k] for p in phases]) for k in keys}
    null = {k: [] for k in keys}
    for _ in range(min(nperm, 200)):
        st = stats(perm_labels(ph, strat))
        for k in keys: null[k].append(spearman(phases, [st[p][k] for p in phases]))
    tag = tc or 'ALL'
    say(f'  [{name} | {tag}] phase-level (rarefied to {nmin} tokens):')
    for p in phases:
        o = obs[p]
        say(f'    {labels[p]:8s} n={o["n"]:4d} tokens={o["tokens"]:5d} distinct={o["distinct"]:3d} Chao1={o["chao1"]:6.0f} rarefied={o["rare"] if o["rare"] is None else round(o["rare"],1)} '
            f'closer-menu JS={o["menu_js"]:.3f} (n={o["n_closer"]}, top {o["menu_top"]}) repeat O/E={o["rep_oe"]:.2f}')
    res = []
    for k in keys:
        nr = np.array([x for x in null[k] if not np.isnan(x)])
        P = float(np.mean(np.abs(nr) >= abs(rho_obs[k]) - 1e-12)) if len(nr) and not np.isnan(rho_obs[k]) else float('nan')
        say(f'    trend {k:8s} rho={rho_obs[k]:+.2f} P={P:.2f}')
        res.append(dict(series=name, tc=tag, feat=k, rho=rho_obs[k], P=P))
    return res

def write(cycle, extra=''):
    open(ROOT + f'data/derived/dark/loop43_cycle{cycle}_detail.txt', 'w').write('\n'.join(OUT) + '\n')
    if extra: open(ROOT + f'data/derived/dark/loop43_cycle{cycle}.txt', 'w').write(extra + '\n')

# =====================================================================================================
if CYCLE == 1:
    ALL = {}
    for level in LEVELS:
        say(f'\n===== level {level} =====')
        fixed = fit_order(level)
        say(f'fixed pairs fitted on MD+HP all phases: {len(fixed)}')
        rows = build(level, fixed)
        for si, (name, f, labels) in enumerate(SERIES):
            R = [r for r in rows if r['si'] == si]
            if len(R) < 30: say(f'[{name}] n={len(R)} too small'); continue
            say(f'\n[{name}] n={len(R)}; by type: {dict(collections.Counter(r["tc"] for r in R))}')
            ALL[(level, name)] = trend_tests(R, name, labels, NPERM)
    json.dump({f'{k[0]}|{k[1]}': v for k, v in ALL.items()}, open(ROOT + 'data/derived/dark/loop43_cycle1_trends.json', 'w'), default=float)
    # summary: count of arrows, survivors (P<0.05), sign agreement across series for same feature, both-city
    say('\n===== SUMMARY =====')
    for level in LEVELS:
        arrows = [a for (lv, nm), v in ALL.items() if lv == level for a in v if not np.isnan(a['rho'])]
        hits = [a for a in arrows if a['P'] < 0.05]
        say(f'{level}: {len(arrows)} trend arrows, {len(hits)} with P<0.05 (expected {0.05*len(arrows):.1f}): ' +
            '; '.join(f"{a['series'].split()[0]}/{a['tc']}/{a['feat']} rho={a['rho']:+.2f} P={a['P']:.3f}" for a in hits))
        # frame features: direction votes
        for ft in ['opener', 'closer', 'jar', 'w2', 'rep', 'order', 'length']:
            A = [a for a in arrows if a['feat'] == ft]
            pos = sum(1 for a in A if a['rho'] > 0); neg = len(A) - pos
            say(f'  {ft}: {len(A)} series x type arrows, rho>0 (rule stronger later) {pos}, rho<0 {neg}; mean rho {np.mean([a["rho"] for a in A]):+.3f}')
        early = [a for a in arrows if a['early'] is not None and a['feat'] in ('opener', 'closer', 'jar', 'w2', 'rep', 'order')]
        say(f'  earliest-phase vs all: {len(early)} arrows, {sum(1 for a in early if a["P_early"] < 0.05)} with P<0.05; '
            f'mean |d| {np.mean([abs(a["d_early"]) for a in early]):.3f}; largest: ' +
            '; '.join(f"{a['series'].split()[0]}/{a['tc']}/{a['feat']} d={a['d_early']:+.2f} P={a['P_early']:.2f}" for a in sorted(early, key=lambda a: -abs(a['d_early']))[:5]))
    write(1)

elif CYCLE == 2:
    # growth simulation on real series: break frame rules in early phases with a ramp, re-run the trend test, report power
    for level in ('seq_all',):
        fixed = fit_order(level); rows = build(level, fixed)
        for si in (3, 1, 0, 4):
            name, f, labels = SERIES[si]
            for tc in ('SEAL', 'TAB'):
                R = [r for r in rows if r['si'] == si and r['tc'] == tc]
                if len(R) < 60: continue
                ph = np.array([r['ph'] for r in R]); strat = [r['strat'] for r in R]
                K = max(ph) + 1
                say(f'\n[{name} | {tc}] n={len(R)}, {K} phases; growth simulation: rule applied with prob g(phase) = 1 - ramp*(K-1-phase)/(K-1)')
                for ft in ('opener', 'closer', 'order', 'rep'):
                    base = np.array([np.nan if r[ft] is None else r[ft] for r in R], float); ok = ~np.isnan(base)
                    if ok.sum() < 30: continue
                    for ramp in (0.0, 0.1, 0.2, 0.3, 0.5):
                        det = 0; NS = 40
                        for s in range(NS):
                            g = 1 - ramp * (K - 1 - ph) / (K - 1)
                            v = base.copy()
                            if ft in ('opener', 'closer'):
                                kill = (rng.random(len(v)) > g) & (v == 1); v[kill] = 0
                            elif ft == 'order':
                                kill = rng.random(len(v)) > g; v[kill & ok] = np.minimum(v[kill & ok], rng.uniform(0.3, 0.7, int((kill & ok).sum())))
                            else:  # repeat ban: rule off -> repeats appear at the language-like rate 0.7x chance ~ 0.35
                                kill = (rng.random(len(v)) > g) & (v == 0); v[kill & ok] = (rng.random(int((kill & ok).sum())) < 0.35).astype(float)
                            rho = spearman(ph[ok], v[ok])
                            null = [spearman(perm_labels(ph, strat)[ok], v[ok]) for _ in range(200)]
                            P = np.mean(np.abs(null) >= abs(rho) - 1e-12)
                            det += int(P < 0.05 and (rho > 0) == (ft != 'rep'))
                        say(f'    {ft:7s} ramp={ramp:.1f} (earliest-phase rule strength {1-ramp:.1f}x): detected {det}/{NS} runs')
    write(2)

elif CYCLE == 3:
    ALL = {}
    for level in LEVELS:
        say(f'\n===== level {level} =====')
        fixed = fit_order(level); rows = build(level, fixed)
        for si, (name, f, labels) in enumerate(SERIES):
            R = [r for r in rows if r['si'] == si]
            if len(R) < 30: continue
            say(f'\n[{name}] n={len(R)}')
            for tc in ('SEAL', 'TAB', None):
                ALL.setdefault(level, []).extend(phase_level(R, name, labels, NPERM, tc))
    say('\n===== SUMMARY =====')
    for level in LEVELS:
        A = [a for a in ALL[level] if not np.isnan(a['rho'])]
        say(f'{level}: {len(A)} phase-level trend arrows, {sum(1 for a in A if a["P"] < 0.05)} with P<0.05: ' +
            '; '.join(f"{a['series'].split()[0]}/{a['tc']}/{a['feat']} rho={a['rho']:+.2f} P={a['P']:.2f}" for a in A if a['P'] < 0.05))
        for k in ['distinct', 'chao1', 'rare', 'menu_js', 'rep_oe']:
            B = [a for a in A if a['feat'] == k]
            say(f'  {k}: {len(B)} arrows, rho>0 {sum(1 for a in B if a["rho"] > 0)}, mean rho {np.mean([a["rho"] for a in B]):+.2f}')
    write(3)

elif CYCLE == 4:
    # pottery marks from inscriptions.csv (every row, direction-free statistics), seq_raw forms with merges applied by level
    MERGE = {lv: {m['form']: m['into'] for m in LEV['merges'] if (lv == 'seq_all' and m['level'] in ('strong', 'probable')) or (lv == 'seq_strong' and m['level'] == 'strong')} for lv in LEVELS}
    def parse(t):
        return [int(x) for x in re.split(r'[-/]', t.replace(']', '').replace('[', '').replace('+', '')) if x.isdigit() and int(x) > 0]
    P = json.load(open(ROOT + 'data/derived/parsed_texts.json'))
    slotc = collections.defaultdict(collections.Counter)
    for o in P:
        if o['type'].split(':')[0] == 'POT': continue
        for s, l in zip(o['seq'], o['slots']): slotc[s][l] += 1
    SLOT = {s: c.most_common(1)[0][0] for s, c in slotc.items()}
    def pot_phase(r):
        if r['type'].split(':')[0] != 'POT': return None
        t = r['time']; per = r['period'].strip(); site = r['site']
        if t in ('Period 1', 'Period 2', 'Period 2A', 'Period 2B') or per in ('1', '2', 'IIa', 'IIb') and site in ('Harappa', 'Kanmer'): return 'early (pre-3)'
        if site == 'Harappa' and (t.startswith('Period 3') or per == '3'): return 'Harappa 3'
        if site == 'Harappa' and per == '2/3': return 'Harappa 2/3'
        if r['type'] == 'POT:T:p': return 'pre-firing (any)'
        return 'other dated/undated'
    seals_first = collections.Counter(); seals_last = collections.Counter(); seal_tok = collections.Counter()
    for o in C:
        if tclass(o) != 'SEAL' or len(o['seq_raw']) < 2: continue
        seals_first[o['seq_raw'][0]] += 1; seals_last[o['seq_raw'][-1]] += 1
        for x in o['seq_raw']: seal_tok[x] += 1
    for lv in LEVELS:
        say(f'\n===== level {lv} =====')
        mp = MERGE[lv]
        groups = collections.defaultdict(list)
        for r in RAWROWS:
            g = pot_phase(r)
            if g is None: continue
            s = [mp.get(x, x) for x in parse(r['text'])]
            if r['dir.'].strip() in ('R/L', 'L/R'): s = s[::-1]      # canonical seq_raw order = reversed text field for directed objects
            if not s: continue
            groups[g].append((s, r))
        def cls(x):
            if x in OPEN: return 'opener'
            if x in NUM: return 'numeral'
            if x in CLOSERS: return 'closer'
            if x in MARK: return 'marker'
            return 'other'
        for g in ['early (pre-3)', 'Harappa 2/3', 'Harappa 3', 'pre-firing (any)', 'other dated/undated']:
            S = groups.get(g, [])
            if not S: continue
            toks = [x for s, r in S for x in s]
            cc = collections.Counter(cls(x) for x in toks)
            sites = collections.Counter(r['site'] for s, r in S)
            say(f'[{g}] pots={len(S)} tokens={len(toks)} sites={dict(sites)} class shares: ' + ', '.join(f'{k}={v/len(toks):.2f} ({v})' for k, v in cc.most_common()))
            say('   signs: ' + ', '.join(f'W{x}x{n}[{cls(x)}/{SLOT.get(x, "?")}]' for x, n in collections.Counter(toks).most_common(25)))
            if g == 'early (pre-3)':
                for s, r in S: say(f'     {r["site"]:10s} {r["time"]:10s} per={r["period"]:3s} {r["type"]:8s} {r["text"]:22s} complete={r["complete"]} dir={r["dir."]}')
        # do early marks become closers / openers later? modal slot of each early-mark sign among non-pot texts; seal first/last share
        early = groups.get('early (pre-3)', [])
        toks = collections.Counter(x for s, r in early for x in s)
        say(f'\nEarly marks (n={sum(toks.values())} tokens, {len(toks)} signs): later behaviour of each sign on seals (seq_raw numbers):')
        slot_hist = collections.Counter()
        for x, n in toks.most_common():
            st = SLOT.get(x, 'unattested'); slot_hist[st] += n
            tf = seals_first[x] / seal_tok[x] if seal_tok[x] else float('nan'); tl = seals_last[x] / seal_tok[x] if seal_tok[x] else float('nan')
            say(f'   W{x:<4d} x{n:<3d} class={cls(x):8s} modal slot (non-pot texts)={st:8s} seal tokens={seal_tok[x]:4d} first-share={tf:.2f} last-share={tl:.2f}')
        say('   early-mark tokens by later modal slot: ' + str(dict(slot_hist)))
        # null: slot composition of early marks vs random pot tokens from Harappa Period 3 (label permutation within POT)
        hp3 = [x for s, r in groups.get('Harappa 3', []) for x in s]
        pool = np.array(hp3 + [x for s, r in early for x in s]); ne = sum(toks.values())
        def comp(arr):
            c = collections.Counter(cls(x) for x in arr); return {k: c[k] / len(arr) for k in ('opener', 'numeral', 'closer', 'marker', 'other')}
        obs = comp([x for s, r in early for x in s]); nulls = [comp(rng.choice(pool, ne, replace=False)) for _ in range(NPERM)]
        for k in obs:
            nv = np.array([n[k] for n in nulls])
            say(f'   early vs Harappa-3 pots, {k:8s}: {obs[k]:.2f} vs null {nv.mean():.2f} (P two-sided {np.mean(np.abs(nv - nv.mean()) >= abs(obs[k] - nv.mean()) - 1e-12):.2f}); upper bound for zero = {3/ne:.2f}')
        # closers on pots per group
        for g in ['early (pre-3)', 'Harappa 3', 'pre-firing (any)', 'other dated/undated']:
            S = groups.get(g, [])
            if not S: continue
            comp_dir = [s for s, r in S if r['complete'] == 'Y' and r['dir.'].strip() in ('R/L', 'L/R') and len(s) >= 2]
            if comp_dir:
                say(f'   [{g}] complete directed pots >=2 signs: {len(comp_dir)}; opener-first {np.mean([opener_first(s) for s in comp_dir]):.2f}, closer-final {np.mean([closer_final(s) for s in comp_dir]):.2f}, jar-final {np.mean([jar_final(s) for s in comp_dir]):.2f}')
    write(4)
