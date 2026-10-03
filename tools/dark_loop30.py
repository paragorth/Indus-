#!/usr/bin/env python3
"""Loop 30: display seals vs working seals.

A stamp seal is cut mirrored so that the impression reads correctly. A seal
whose recorded direction is anomalous (impression would read L/R, boustrophedon,
top/bottom) is a candidate 'display' object. Here we flag those seals and ask
whether their texts differ from ordinary R/L seals, whether ordinary seals with an
attested impression/copy ('working', S-DARK-16.3) differ from 'dead' seals, and
which side the anomalous seals fall on.

Data: data/raw/inscriptions.csv (5,680 rows; one row per inscribed line).
Sign sequences: CSV 'text' parsed and reversed = canonical seq_raw (verified
against data/derived/merged-corpus-canonical.json); merge levels from
data/derived/sign_allographs_levels.json (seq_strong = strong merges only,
seq_all = strong + probable).

Usage: python3 tools/dark_loop30.py [cycle ...]   (default: 1 2 3 4)
Outputs: data/derived/dark/loop30_cycle<N>.txt
"""
import csv, json, re, sys, collections, random
import numpy as np

ROOT = '/home/user/Indus-'
OUT = ROOT + '/data/derived/dark/'
CSV = ROOT + '/data/raw/inscriptions.csv'
rng = np.random.default_rng(30)

# ---------- frame vocabulary (Wells numbers; GRAMMAR.md, bridge_extended.json) ----------
OPENERS = {817, 861, 820}                     # M267/M391 opener family
JAR = {740}                                   # M342
ARROW = {520}                                 # M211
OTHER_CLOSERS = {156, 527, 617, 226, 390, 405, 154, 158, 15, 254, 12}  # S289 closer paradigm (final members)
CLOSERS = JAR | ARROW | OTHER_CLOSERS
NUMERALS = {1, 2, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}  # short 1-8, tall 1-4, 12, 24
ANOM = {'L/R', 'BUS', 'T/B', 'SYM'}
ORD = {'R/L'}

def parse(t):
    t = (t or '').strip()
    lost_edge = t.startswith(']') or t.endswith('[')
    toks = [s for s in re.split(r'[-/]', t.strip('+[] ')) if s.strip()]
    seq = [int(s) for s in toks if s.strip().isdigit()]
    return list(reversed(seq)), lost_edge  # reading order, first-read first (= canonical)

def merge_maps():
    a = json.load(open(ROOT + '/data/derived/sign_allographs_levels.json'))
    strong, alll = {}, {}
    for m in a['merges']:
        if m['level'] == 'strong':
            strong[m['form']] = m['into']; alll[m['form']] = m['into']
        elif m['level'] == 'probable':
            alll[m['form']] = m['into']
    def close(mp):
        for k in list(mp):
            v = mp[k]; n = 0
            while v in mp and n < 10: v = mp[v]; n += 1
            mp[k] = v
        return mp
    return {'seq_raw': {}, 'seq_strong': close(strong), 'seq_all': close(alll)}

MAPS = merge_maps()

def norm_dir(d):
    d = (d or '').strip().upper()
    return 'R/L' if d == 'R/L' else d

def size_cm2(x):
    def f(v):
        try: v = float(str(v).split('(')[0])
        except: return None
        return v / 10 if v > 10 else v   # Dholavira rows are in mm
    h, v = f(x['h']), f(x['v'])
    return h * v if h and v else None

def load(variant):
    rows = list(csv.DictReader(open(CSV)))
    mp = MAPS[variant]
    recs = []
    for x in rows:
        seq0, lost_edge = parse(x['text'])
        has_lost = 0 in seq0 or lost_edge
        seq = tuple(mp.get(s, s) for s in seq0 if s != 0)
        b = (x['boss'] or '-').strip()
        recs.append(dict(
            id=x['id'], obj=x['id'].split('.')[0], cisi=x['cisi'], site=x['site'], type=x['type'],
            tclass=x['type'].split(':')[0], dir=norm_dir(x['dir.']), boss=b,
            bossed=('B' in b and b != '-'), perforated=b.startswith('P'),
            material=(x['material'] or '-').strip().capitalize(), shape=x['shape'], sides=x['sides'],
            complete=(x['complete'] == 'Y' and not has_lost), has_lost=has_lost,
            area=size_cm2(x), seq=seq, L=len(seq),
            phase=x['phase'], period=x['period'], cond=x['condition'].strip().capitalize()))
    # stock texts: complete text on >= 3 distinct objects (any type)
    cnt = collections.defaultdict(set)
    for r in recs:
        if r['complete'] and r['L'] >= 1: cnt[r['seq']].add(r['obj'])
    stock = {k for k, v in cnt.items() if len(v) >= 3}
    tag_texts = collections.Counter(r['seq'] for r in recs if r['tclass'] == 'TAG' and r['L'] >= 1)
    tab_texts = collections.Counter(r['seq'] for r in recs if r['tclass'] == 'TAB' and r['L'] >= 1)
    for r in recs:
        s = r['seq']
        r['stock'] = s in stock
        r['tag_match'] = tag_texts.get(s, 0)
        r['tab_match'] = tab_texts.get(s, 0)
        r['opener'] = bool(s) and s[0] in OPENERS
        r['final'] = s[-1] if s else None
        r['closer_class'] = ('jar' if s and s[-1] in JAR else 'arrow' if s and s[-1] in ARROW
                             else 'other' if s and s[-1] in OTHER_CLOSERS else 'none')
        r['n_closers'] = sum(1 for t in s if t in CLOSERS)
        r['n_jar'] = sum(1 for t in s if t in JAR)
        r['numeral'] = any(t in NUMERALS for t in s)
        r['nested'] = r['opener'] and r['n_closers'] >= 1 and r['L'] >= 6   # long credential text
        r['frame_only'] = (not r['opener']) and r['closer_class'] != 'none' and r['L'] <= 3
    return recs

# ---------- generic stratified permutation test ----------
def feat_vec(group, f):
    return np.array([f(r) for r in group], dtype=float)

def strat_perm(a, b, f, key, nperm=1000):
    """Difference in mean of f between group a and group b; null = labels permuted within strata key()."""
    allr = a + b
    lab = np.array([1] * len(a) + [0] * len(b))
    v = np.array([f(r) for r in allr], dtype=float)
    ok = ~np.isnan(v)
    allr = [r for r, o in zip(allr, ok) if o]; lab = lab[ok]; v = v[ok]
    if lab.sum() == 0 or (1 - lab).sum() == 0: return None
    strata = collections.defaultdict(list)
    for i, r in enumerate(allr): strata[key(r)].append(i)
    obs = v[lab == 1].mean() - v[lab == 0].mean()
    diffs = np.empty(nperm)
    idx_groups = [np.array(ix) for ix in strata.values()]
    for p in range(nperm):
        pl = lab.copy()
        for ix in idx_groups:
            pl[ix] = pl[rng.permutation(ix)]
        diffs[p] = v[pl == 1].mean() - v[pl == 0].mean()
    pval = (np.sum(np.abs(diffs) >= abs(obs) - 1e-12) + 1) / (nperm + 1)
    return dict(obs=obs, exp=diffs.mean(), p=pval, n1=int(lab.sum()), n0=int((1 - lab).sum()),
                m1=v[lab == 1].mean(), m0=v[lab == 0].mean())

FEATURES = [
    ('length (signs)', lambda r: r['L']),
    ('opener initial', lambda r: float(r['opener'])),
    ('jar-final', lambda r: float(r['closer_class'] == 'jar')),
    ('arrow-final', lambda r: float(r['closer_class'] == 'arrow')),
    ('other-closer-final', lambda r: float(r['closer_class'] == 'other')),
    ('no closer', lambda r: float(r['closer_class'] == 'none')),
    ('n closer-class signs', lambda r: r['n_closers']),
    ('>=2 closer-class signs', lambda r: float(r['n_closers'] >= 2)),
    ('numeral present', lambda r: float(r['numeral'])),
    ('stock text (>=3 objects)', lambda r: float(r['stock'])),
    ('nested credential (opener+closer, L>=6)', lambda r: float(r['nested'])),
    ('frame-only short (closer, no opener, L<=3)', lambda r: float(r['frame_only'])),
    ('area cm2', lambda r: r['area'] if r['area'] else np.nan),
    ('bossed', lambda r: float(r['bossed']) if r['boss'] != '-' else np.nan),
]
KEY_SM = lambda r: (r['site'], r['material'])
KEY_SML = lambda r: (r['site'], r['material'], min(r['L'], 8))

def fmt(name, s):
    if s is None: return f'  {name}: n/a'
    return (f'  {name:45s} A={s["m1"]:.3f} (n={s["n1"]})  B={s["m0"]:.3f} (n={s["n0"]})  '
            f'diff={s["obs"]:+.3f} null={s["exp"]:+.3f}  p={s["p"]:.3f}')

def seals(recs, minL=2):
    return [r for r in recs if r['tclass'] == 'SEAL' and r['L'] >= minL]

# ======================= cycle 1 =======================
def cycle1():
    out = []
    recs = load('seq_raw')
    S = [r for r in recs if r['tclass'] == 'SEAL']
    out.append('LOOP 30 cycle 1: the flag set (data/raw/inscriptions.csv, 5,680 lines; SEAL rows only)')
    out.append('Direction codes on SEAL lines: ' + str(collections.Counter(r['dir'] for r in S).most_common()))
    out.append("NR = single-sign texts (58 of 59 have 1 sign): direction not applicable. '-' = unrecorded (149 lines; "
               "S-DARK-8.4: ~25% of these are probably stored backwards, so they are kept out of the anomalous set).")
    out.append("Convention assumed: Wells' dir. is the reading direction of the IMPRESSION (normalised); R/L is the norm "
               "(2,021 seal lines). A seal recorded L/R is one whose impression reads L/R, i.e. the seal face itself "
               "reads R/L = carved un-mirrored. BUS and T/B are also non-standard. This is an inference from the "
               "field, not a photo check; CISI photo check pending for every flagged object.")
    A = [r for r in S if r['dir'] in ANOM]
    objs = {r['obj'] for r in A}
    out.append(f'\nFLAG SET: {len(A)} seal lines on {len(objs)} objects: ' +
               str(collections.Counter(r['dir'] for r in A).most_common()))
    out.append('  by site: ' + str(collections.Counter(r['site'] for r in A).most_common()))
    out.append('  by material: ' + str(collections.Counter(r['material'] for r in A).most_common()))
    out.append('  by seal type: ' + str(collections.Counter(r['type'] for r in A).most_common()))
    out.append('  by shape: ' + str(collections.Counter(r['shape'] for r in A).most_common()))
    out.append('  boss code: ' + str(collections.Counter(r['boss'] for r in A).most_common()))
    kb = [r for r in A if r['boss'] != '-']
    out.append(f'  bossed (code contains B) {sum(r["bossed"] for r in kb)} of {len(kb)} with a boss record; '
               f'perforated (P*) {sum(r["perforated"] for r in kb)} of {len(kb)}')
    O = [r for r in S if r['dir'] in ORD]
    kbo = [r for r in O if r['boss'] != '-']
    out.append(f'  ordinary R/L seals for comparison: bossed {sum(r["bossed"] for r in kbo)} of {len(kbo)} '
               f'({100*sum(r["bossed"] for r in kbo)/len(kbo):.0f}%) vs flagged {100*sum(r["bossed"] for r in kb)/max(1,len(kb)):.0f}%')
    out.append('  text length: ' + str(sorted(collections.Counter(r['L'] for r in A).items())))
    out.append('  complete texts: %d of %d' % (sum(r['complete'] for r in A), len(A)))
    # foreign format (S46): round / cylinder / abroad
    abroad = {'Qala\'at al-Bahrain', 'Janabiyah', 'Altyn Depe', 'Failaka', 'Ur', 'Saar', 'Kish', 'Gonur Depe', 'Tello', 'Nippur'}
    ff = [r for r in A if r['type'] in ('SEAL:C', 'SEAL:CY') or r['site'] in abroad]
    out.append(f'  foreign-format or found abroad (S46 community): {len(ff)} lines ({[r["cisi"] or r["site"] for r in ff]})')
    # matching impressions
    for lab, k in (('TAG sealing', 'tag_match'), ('TAB tablet', 'tab_match')):
        m = [r for r in A if r['L'] >= 2 and r[k] > 0]
        n = sum(1 for r in A if r['L'] >= 2)
        out.append(f'  flagged seals (>=2 signs, n={n}) with an exact {lab} match anywhere: {len(m)} '
                   f'-> upper bound < {3/n:.3f}' + ('' if not m else '  matches: ' + str([(r['cisi'], r['seq'], r[k]) for r in m])))
    mo = [r for r in O if r['L'] >= 2]
    out.append(f'  ordinary R/L seals (>=2 signs, n={len(mo)}): TAG match {sum(1 for r in mo if r["tag_match"])} '
               f'({100*sum(1 for r in mo if r["tag_match"])/len(mo):.1f}%), TAB match {sum(1 for r in mo if r["tab_match"])} '
               f'({100*sum(1 for r in mo if r["tab_match"])/len(mo):.1f}%), either {sum(1 for r in mo if r["tag_match"] or r["tab_match"])} '
               f'({100*sum(1 for r in mo if r["tag_match"] or r["tab_match"])/len(mo):.1f}%)')
    # binomial: expected matches among flagged at ordinary rate
    p_any = sum(1 for r in mo if r['tag_match'] or r['tab_match']) / len(mo)
    n2 = sum(1 for r in A if r['L'] >= 2); k2 = sum(1 for r in A if r['L'] >= 2 and (r['tag_match'] or r['tab_match']))
    from math import comb
    pbin = sum(comb(n2, j) * p_any**j * (1 - p_any)**(n2 - j) for j in range(0, k2 + 1))
    out.append(f'  expected matches at the ordinary rate: {n2*p_any:.1f}; observed {k2}; P(<= observed) = {pbin:.3f}')
    # multi-side objects where only one side is anomalous
    bys = collections.defaultdict(list)
    for r in S: bys[r['obj']].append(r['dir'])
    mixed = [o for o, ds in bys.items() if len(ds) > 1 and any(d in ANOM for d in ds) and any(d in ORD for d in ds)]
    out.append(f'  multi-sided seals mixing an anomalous and an R/L side: {len(mixed)} {mixed}')
    # undirected '-' seals (caveat set)
    U = [r for r in S if r['dir'] == '-']
    out.append(f"\nCAVEAT SET: {len(U)} seal lines with dir '-' (unrecorded), {sum(1 for r in U if r['L']>=2)} with >= 2 signs; "
               f"sites {collections.Counter(r['site'] for r in U).most_common(6)}; these are excluded from both groups.")
    out.append('\nFULL LIST (id, cisi, site, type, dir, boss, material, shape, L, reading-order seq):')
    for r in sorted(A, key=lambda r: (r['site'], r['id'])):
        out.append(f"  {r['id']:8s} {r['cisi'] or '-':8s} {r['site']:18s} {r['type']:8s} {r['dir']:4s} {r['boss']:5s} "
                   f"{r['material']:11s} {r['shape']:12s} L={r['L']} {list(r['seq'])}"
                   + (' TAG' if r['tag_match'] else '') + (' TAB' if r['tab_match'] else ''))
    txt = '\n'.join(out)
    open(OUT + 'loop30_cycle1.txt', 'w').write(txt + '\n')
    print(txt)

# ======================= cycle 2 =======================
def cycle2(nperm=1000):
    out = ['LOOP 30 cycle 2: anomalous-direction seals (A) vs ordinary R/L seals (B), seal lines >= 2 signs.',
           'Control: A/B label permuted within site x material strata (and within site x material x length for '
           f'non-length features), {nperm}x; two-sided p. Source: data/raw/inscriptions.csv.']
    for variant in ('seq_raw', 'seq_strong', 'seq_all'):
        recs = load(variant)
        S = seals(recs)
        A = [r for r in S if r['dir'] in ANOM]; B = [r for r in S if r['dir'] in ORD]
        out.append(f'\n--- {variant}: A n={len(A)}, B n={len(B)} ---')
        out.append('Closer class A: ' + str(collections.Counter(r['closer_class'] for r in A).most_common()) +
                   '   B: ' + str({k: round(v/len(B), 3) for k, v in collections.Counter(r['closer_class'] for r in B).most_common()}))
        out.append('[site x material strata]')
        for name, f in FEATURES:
            out.append(fmt(name, strat_perm(A, B, f, KEY_SM, nperm)))
        out.append('[site x material x length strata]')
        for name, f in FEATURES:
            if name.startswith('length'): continue
            out.append(fmt(name, strat_perm(A, B, f, KEY_SML, nperm)))
        # Mohenjo-daro only / excluding
        for lab, sel in (('Mohenjo-daro only', lambda r: r['site'] == 'Mohenjo-daro'), ('excluding Mohenjo-daro', lambda r: r['site'] != 'Mohenjo-daro')):
            A2 = [r for r in A if sel(r)]; B2 = [r for r in B if sel(r)]
            out.append(f'[{lab}: A n={len(A2)}, B n={len(B2)}; site x material strata]')
            for name, f in FEATURES[:1] + FEATURES[2:3] + FEATURES[9:12]:
                out.append(fmt(name, strat_perm(A2, B2, f, KEY_SM, nperm)))
        # excluding the S46 foreign-format community (round/cylinder/abroad)
        abroad = {'Qala\'at al-Bahrain', 'Janabiyah', 'Altyn Depe'}
        A3 = [r for r in A if r['type'] not in ('SEAL:C', 'SEAL:CY') and r['site'] not in abroad]
        B3 = [r for r in B if r['type'] not in ('SEAL:C', 'SEAL:CY') and r['site'] not in abroad]
        out.append(f'[home square/rectangular only: A n={len(A3)}, B n={len(B3)}; site x material strata]')
        for name, f in FEATURES[:3] + FEATURES[8:12]:
            out.append(fmt(name, strat_perm(A3, B3, f, KEY_SM, nperm)))
    txt = '\n'.join(out)
    open(OUT + 'loop30_cycle2.txt', 'w').write(txt + '\n')
    print(txt)

# ======================= cycle 3 =======================
def cycle3(nperm=1000):
    out = ['LOOP 30 cycle 3: ordinary R/L seals with an attested copy (working: exact match on a TAG sealing or TAB tablet) '
           'vs none (dead, S-DARK-16.3), seal lines >= 2 signs; then where the anomalous seals sit.',
           f'Control: working label permuted within site x material (and x length), {nperm}x. Source: data/raw/inscriptions.csv.']
    for variant in ('seq_raw', 'seq_strong', 'seq_all'):
        recs = load(variant)
        S = seals(recs)
        B = [r for r in S if r['dir'] in ORD]; A = [r for r in S if r['dir'] in ANOM]
        for mlab, mk in (('TAG or TAB match', lambda r: r['tag_match'] or r['tab_match']), ('TAG sealing match only', lambda r: r['tag_match'])):
            W = [r for r in B if mk(r)]; D = [r for r in B if not mk(r)]
            out.append(f'\n--- {variant}, working = {mlab}: working n={len(W)}, dead n={len(D)}, anomalous n={len(A)} ---')
            out.append('[site x material strata]  (A = working, B = dead)')
            res_sm = {}
            for name, f in FEATURES:
                s = strat_perm(W, D, f, KEY_SM, nperm); res_sm[name] = s; out.append(fmt(name, s))
            out.append('[site x material x length strata]')
            for name, f in FEATURES:
                if name.startswith('length'): continue
                out.append(fmt(name, strat_perm(W, D, f, KEY_SML, nperm)))
            # where do the anomalous seals sit? standardized position between dead (0) and working (1) means
            out.append('[anomalous seals: position of their mean on the dead(0) -> working(1) axis, per feature; '
                       'bootstrap 95% CI over anomalous seals]')
            for name, f in FEATURES:
                va = np.array([f(r) for r in A], float); va = va[~np.isnan(va)]
                vw = np.array([f(r) for r in W], float); vw = vw[~np.isnan(vw)]
                vd = np.array([f(r) for r in D], float); vd = vd[~np.isnan(vd)]
                if len(va) < 5 or abs(vw.mean() - vd.mean()) < 1e-9: continue
                pos = lambda m: (m - vd.mean()) / (vw.mean() - vd.mean())
                bs = [pos(rng.choice(va, len(va)).mean()) for _ in range(1000)]
                out.append(f'  {name:45s} dead={vd.mean():.3f} working={vw.mean():.3f} anomalous={va.mean():.3f}  '
                           f'pos={pos(va.mean()):+.2f} (CI {np.percentile(bs,2.5):+.2f}..{np.percentile(bs,97.5):+.2f})')
            # classifier-free summary: Euclidean distance of standardized anomalous mean to dead vs working means
            names = [n for n, _ in FEATURES if n not in ('area cm2', 'bossed')]
            M = {}
            for g, grp in (('A', A), ('W', W), ('D', D)):
                M[g] = np.array([np.nanmean([f(r) for r in grp]) for n, f in FEATURES if n in names])
            sd = np.array([np.nanstd([f(r) for r in B]) for n, f in FEATURES if n in names]) + 1e-9
            dW = np.linalg.norm((M['A'] - M['W']) / sd); dD = np.linalg.norm((M['A'] - M['D']) / sd)
            out.append(f'  standardized distance of anomalous mean to working {dW:.2f} vs dead {dD:.2f} '
                       f'-> nearer {"WORKING" if dW < dD else "DEAD"}')
    txt = '\n'.join(out)
    open(OUT + 'loop30_cycle3.txt', 'w').write(txt + '\n')
    print(txt)

# ======================= cycle 4 =======================
def cycle4(nperm=1000):
    out = ['LOOP 30 cycle 4: credential model. Prediction A: display/dead seals carry the long nested credential texts '
           '(opener + closer, >= 6 signs) and working seals the short frame-only texts (closer, no opener, <= 3 signs). '
           'Prediction B (ledger): the opposite. Tested at three merge levels, by site, with working label permuted '
           f'within site x material ({nperm}x) and with length held (site x material x length).',
           'Source: data/raw/inscriptions.csv; R/L seal lines >= 2 signs; working = exact TAG or TAB match.']
    verdict = collections.Counter()
    for variant in ('seq_raw', 'seq_strong', 'seq_all'):
        recs = load(variant)
        S = seals(recs)
        B = [r for r in S if r['dir'] in ORD]; A = [r for r in S if r['dir'] in ANOM]
        W = [r for r in B if r['tag_match'] or r['tab_match']]; D = [r for r in B if not (r['tag_match'] or r['tab_match'])]
        out.append(f'\n=== {variant}: working {len(W)}, dead {len(D)}, anomalous {len(A)} ===')
        for lab, sel in (('all sites', lambda r: True), ('Mohenjo-daro', lambda r: r['site'] == 'Mohenjo-daro'),
                         ('Harappa', lambda r: r['site'] == 'Harappa'),
                         ('other sites', lambda r: r['site'] not in ('Mohenjo-daro', 'Harappa'))):
            W2 = [r for r in W if sel(r)]; D2 = [r for r in D if sel(r)]; A2 = [r for r in A if sel(r)]
            out.append(f'[{lab}: working {len(W2)}, dead {len(D2)}, anomalous {len(A2)}]')
            for name, f in (FEATURES[10], FEATURES[11], FEATURES[0], FEATURES[7]):
                s = strat_perm(W2, D2, f, KEY_SM, nperm)
                out.append(fmt(name + ' [W vs D]', s))
                if s and lab == 'all sites' and name.startswith('nested'):
                    verdict[(variant, 'nested', 'B-ledger' if s['obs'] > 0 else 'A-display')] += 1
                if s and lab == 'all sites' and name.startswith('frame-only'):
                    verdict[(variant, 'frame-only', 'B-ledger' if s['obs'] < 0 else 'A-display')] += 1
            if len(A2) >= 5:
                for name, f in (FEATURES[10], FEATURES[11]):
                    out.append(fmt(name + ' [anomalous vs dead]', strat_perm(A2, D2, f, KEY_SM, nperm)))
                    out.append(fmt(name + ' [anomalous vs working]', strat_perm(A2, W2, f, KEY_SM, nperm)))
            # length-held version of the nested test
            for name, f in (FEATURES[10], FEATURES[11]):
                out.append(fmt(name + ' [W vs D, length held]', strat_perm(W2, D2, f, KEY_SML, nperm)))
        # distribution of length among working vs dead
        out.append('  length distribution working: ' + str(sorted(collections.Counter(min(r['L'], 10) for r in W).items())))
        out.append('  length distribution dead:    ' + str(sorted(collections.Counter(min(r['L'], 10) for r in D).items())))
        out.append('  length distribution anomalous: ' + str(sorted(collections.Counter(min(r['L'], 10) for r in A).items())))
        # how much of 'use' is tablets rather than sealings
        out.append(f'  working seals: TAG match {sum(1 for r in W if r["tag_match"])}, TAB match {sum(1 for r in W if r["tab_match"])}, '
                   f'both {sum(1 for r in W if r["tag_match"] and r["tab_match"])}')
        # nested among the longest only (L>=6): does opener/closer still separate?
        W6 = [r for r in W if r['L'] >= 6]; D6 = [r for r in D if r['L'] >= 6]
        out.append(f'  among L>=6 texts (working {len(W6)}, dead {len(D6)}): ' + fmt('opener initial', strat_perm(W6, D6, FEATURES[1][1], KEY_SM, nperm)).strip())
    out.append('\nDIRECTION TALLY (all sites, sign of W-D difference): ' + str(dict(verdict)))
    txt = '\n'.join(out)
    open(OUT + 'loop30_cycle4.txt', 'w').write(txt + '\n')
    print(txt)

if __name__ == '__main__':
    cyc = [int(c) for c in sys.argv[1:]] or [1, 2, 3, 4]
    for c in cyc:
        {1: cycle1, 2: cycle2, 3: cycle3, 4: cycle4}[c]()
