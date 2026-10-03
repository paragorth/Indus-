#!/usr/bin/env python3
"""S-DARK-60: A FUNCTIONAL PROFILE FOR EVERY HEAD (closer-slot sign).
For each head of the closer paradigm (S289/S291) plus the marked jars, the suffix-only endings W400 / W90-91 and
'no head': (1) object-type profile, use on sealings/tags and their carriers, seal <-> impression matching, moulded/incised
split; (2) companions: openers, numeral epithets, adjacent qualifier, middle length, fish words, quantity phrases, second
units, suffixes; (3) geography (sites, Mohenjo-daro areas, phases) and emblem / material / size; (4) pairwise functional
distinctness. Null: head labels permuted among texts within strata that exclude the tested dimension (1,000x):
  site x length-bin        for object type, carrier, mould/incised, seal<->impression match
  type x length-bin        for site shares and the foreign share
  site x type x length-bin for companions, areas, phases, emblem, material, size
All three merge levels (seq_raw / seq_strong / seq_all); IM77 replication through the bridge (M numbers).
Usage: python3 tools/dark_loop60.py <1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json, sys, csv, collections, math, random, re, os
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(60)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- sign classes (Wells) ----------------
HEAD_GROUP = {740: 'jar', 741: 'mjar', 742: 'mjar', 745: 'mjar', 520: 'arrow', 151: 'W151', 156: 'W156', 154: 'W154/158',
              158: 'W154/158', 527: 'box527', 526: 'box527', 226: 'W226', 617: 'W615/617', 615: 'W615/617', 236: 'W236',
              700: 'W700', 595: 'W595'}
SUF = {400: 'W400', 90: 'W90/91', 91: 'W90/91'}
HEADS_ORDER = ['jar', 'mjar', 'arrow', 'W151', 'W156', 'W154/158', 'box527', 'W226', 'W615/617', 'W236', 'W700', 'W595',
               'W400', 'W90/91', 'none']
OPENERS = {817, 861, 820, 920, 692}
CONN = {2, 60}
MARK = {1, 2, 31}
SHORT = set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30))
TALL = set(range(32, 40))
FIXED12 = {55, 56}
VAL = {**{i: i for i in range(3, 8)}, **{i: i - 10 for i in range(12, 21)}, **{i: i - 20 for i in range(25, 30)},
       **{i: i - 30 for i in range(32, 39)}, 39: 9}
NUMS = (SHORT | TALL) - MARK
FISH = {220, 240, 235, 233, 231, 226}
GOODS = {390, 405, 407, 900, 384, 388, 645, 904, 220}
FOREIGN_REGIONS = {'Persian Gulf', 'Mesopotamia', 'Iranian Plateau', 'Central Asia'}

def type_fine(t):
    t0 = t.split(':')[0]; sub = t.split(':')[1] if ':' in t else ''
    if t0 == 'SEAL':
        return {'S': 'seal_sq', 'R': 'seal_bar', 'C': 'seal_round', 'CY': 'seal_cyl', 'L': 'seal_round'}.get(sub, 'seal_oth')
    if t0 == 'TAB':
        return {'B': 'tab_mould', 'I': 'tab_inc', 'C': 'tab_cu'}.get(sub, 'tab_oth')
    if t0 == 'TAG': return 'sealing'
    if t0 in ('POT', 'POsT'): return 'pot'
    if t0 == 'BNGL': return 'bangle'
    if t0 == 'ROD': return 'rod'
    return 'misc'
def type_coarse(tf):
    if tf.startswith('seal'): return 'seal'
    if tf.startswith('tab'): return 'tablet'
    return tf
def lenbin(n): return 1 if n <= 1 else 2 if n == 2 else 3 if n == 3 else 4 if n == 4 else 5 if n <= 6 else 7
def emblem_group(sym):
    if not sym or sym in ('-', '?'): return 'unknown'
    if sym == 'None': return 'none'
    if sym.startswith('Bull1'): return 'unicorn'
    if sym in ('Bull', 'Bult', 'Buff', 'Zebu', 'Gaur', 'Gavi', 'CompBull'): return 'bovine_other'
    if sym == 'Elep': return 'elephant'
    if sym in ('Rhin', 'Tigr', 'Goat', 'Hare', 'Croc'): return 'other_animal'
    return 'other'
def material_group(m):
    m = (m or '').lower()
    if m in ('', '-'): return 'unknown'
    if 'steatite' in m: return 'steatite'
    if 'faience' in m or 'paste' in m: return 'faience'
    if 'clay' in m or 'terracotta' in m or 'ceramic' in m: return 'clay'
    if 'copper' in m or 'bronze' in m: return 'copper'
    if 'ivory' in m or 'bone' in m: return 'ivory/bone'
    return 'other'
def md_area(a):
    if a.startswith('DKG. (S)'): return 'DKG-S'
    if a.startswith('DKG. (N)') or a == 'DKG': return 'DKG-N/DKG'
    if a.startswith('DK'): return 'DK-other'
    if a.startswith('HR'): return 'HR'
    if a.startswith('VS'): return 'VS'
    if a.startswith('SD'): return 'SD'
    if a.startswith('L') or a.startswith('MN'): return 'L/MN'
    return 'unrecorded'
def phase_of(site, period, phase):
    if site == 'Mohenjo-daro':
        p = period.strip()
        return p if p in ('Early', 'Intermediate', 'Late') else None
    if site == 'Harappa':
        if period == '3' and phase in ('B', 'B/C', 'C'): return '3' + phase
        if phase.startswith('Stratum'):
            s = phase.split()[1]; return 'Str-' + ('I-II' if s in ('I', 'II') else 'III-IV' if s in ('III', 'IV') else 'V-VI')
        return None
    if site == 'Dholavira':
        return 'Dlv-' + period if period in ('4', '5', '6') else None
    if site == 'Kalibangan':
        return 'K-' + phase if phase in ('Early', 'Middle', 'Late') else None
    return None

# ---------------- head parse ----------------
def parse_head(seq, head_group, suf_map, openers, conn, nums, mark, fish, goods, fixed12, val, tall):
    """returns dict of per-text features; seq in reading order."""
    s = list(seq); suffix = []
    while len(s) > 1 and s[-1] in suf_map: suffix.append(s.pop())
    head = head_group.get(s[-1])
    if head is None:
        head = suf_map[suffix[-1]] if suffix else 'none'
        body = s if not suffix else s          # body = text without suffix
        head_sign = None
    else:
        head_sign = s[-1]; body = s[:-1]
    f = dict(head=head, head_sign=head_sign, suffix=(suf_map[suffix[-1]] if suffix else 'no'), n=len(seq))
    f['opener'] = 'yes' if (len(s) >= 1 and s[0] in openers) else 'no'
    f['conn'] = 'yes' if any(x in conn for x in s[:3]) else 'no'
    mid = list(body)
    if mid and mid[0] in openers:
        mid = mid[1:]
        if mid and mid[0] in conn: mid = mid[1:]
    f['midlen'] = len(mid)
    f['midbin'] = str(min(len(mid), 4)) + ('+' if len(mid) >= 4 else '')
    cnt = [x for x in body if x in nums]
    f['numeral'] = 'yes' if cnt else 'no'
    f['fixed12'] = 'yes' if any(x in fixed12 for x in body) else 'no'
    q = body[-1] if (head_sign is not None and body) else None
    f['qual'] = q
    if q in nums:
        f['epithet'] = ('T' if q in tall else 'S') + str(val.get(q, '?'))
    else:
        f['epithet'] = 'none'
    f['fish'] = 'yes' if any(x in fish for x in body) else 'no'
    f['quantity'] = 'yes' if any(body[i] in nums and body[i + 1] in goods for i in range(len(body) - 1)) else 'no'
    # second unit: a head sign occurring medially (not the final head)
    f['second_unit'] = 'yes' if any(x in head_group for x in body) else 'no'
    f['head_medial_tokens'] = [x for x in body if x in head_group]
    return f

# ---------------- load Wells ----------------
def complete_only_types_ok(ty):
    """end-intact fragments are admitted only for sealings (TAG), where whole texts are rare (S-DARK-16)"""
    return ty.startswith('TAG')
def load_wells(level, complete_only=False):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    g = lambda r, k: (r.get(k) or '')
    lothal = {}
    try:
        for r in json.load(open(ROOT + 'data/derived/lothal-sealings-frenez-tosi2005.json'))['rows']:
            lothal[r[0]] = r[2]
    except Exception:
        pass
    by_cisi = collections.defaultdict(list)
    for r in rows: by_cisi[r['cisi']].append(r)
    def parsed(r): return [x for x in (int(y) for y in re.findall(r'\d{3}', r['text'])) if x != 0]
    j = 0; T = []
    for c in C:
        while rows[j]['cisi'] != c['cisi']: j += 1
        r = rows[j]; j += 1
        s = c[level]
        if not s: continue
        raw = c['seq_raw']
        if list(reversed(parsed(r))) != raw:
            alt = [x for x in by_cisi[c['cisi']] if list(reversed(parsed(x))) == raw]
            r = alt[0] if alt else r
        text_ok = list(reversed(parsed(r))) == raw
        # seq is stored in reading order = reverse of the displayed text, so the reading END is the displayed LEFT end
        endok = (c['complete'] == 'Y') or (text_ok and r['text'].startswith('+'))
        if not endok: continue
        if c['complete'] != 'Y' and not complete_only_types_ok(c['type']): continue
        tf = type_fine(c['type'])
        foreign = g(r, 'region') in FOREIGN_REGIONS
        try: h = float(g(r, 'horizontal(mm)') or 0)
        except ValueError: h = 0.0
        t = dict(id=g(r, 'id'), cisi=c['cisi'], site=c['site'], complete=c['complete'] == 'Y', type=c['type'], tf=tf, tc=type_coarse(tf), foreign=foreign,
                 carrier=(c['type'].split(':')[1] if c['type'].startswith('TAG') and ':' in c['type'] else ('TAG' if tf == 'sealing' else None)),
                 back=lothal.get(c['cisi']) if tf == 'sealing' else None,
                 emblem=emblem_group(c['symbol']), material=material_group(c['material']), size=h if h > 0 else None,
                 area=md_area(c['area-section']) if c['site'] == 'Mohenjo-daro' else None,
                 phase=phase_of(c['site'], c['period'], c['phase']), seq=list(s), big=c['site'] in ('Mohenjo-daro', 'Harappa'))
        t.update(parse_head(t['seq'], HEAD_GROUP, SUF, OPENERS, CONN, NUMS, MARK, FISH, GOODS, FIXED12, VAL, TALL))
        if complete_only and not t['complete']: continue
        T.append(t)
    return T

# ---------------- load IM77 ----------------
M_HEAD = {342: 'jar', 343: 'mjar', 344: 'mjar', 345: 'mjar', 211: 'arrow', 12: 'W151', 15: 'W156+154/158', 254: 'box527',
          60: 'W226', 245: 'W615/617', 66: 'W236', 328: 'W700', 252: 'W595'}
M_SUF = {176: 'W400', 1: 'W90/91'}
M_OPEN = {267, 391, 293, 150}; M_CONN = {99, 100, 123}
M_TALL = set(range(87, 97)); M_SHORT = set(range(101, 121)); M_NUMS = M_TALL | M_SHORT
M_VAL = {87: 2, 88: 2, 89: 3, 90: 3, 91: 3, 92: 3, 93: 4, 94: 4, 95: 5, 96: 5, 102: 3, 103: 3, 104: 4, 105: 4, 106: 5, 107: 5,
         108: 6, 109: 6, 110: 7, 111: 7, 112: 7, 113: 8, 114: 8, 115: 9, 116: 9, 117: 10, 118: 10, 119: 8, 120: 9}
M_FISH = {59, 65, 67, 70, 72, 60}; M_GOODS = {161, 162, 167, 168, 169, 287, 59}
IM_TYPE = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tab_cu',
           'pottery graffito': 'pot', 'ivory/bone rod': 'rod', 'miscellaneous': 'misc', 'bronze implement': 'misc'}
def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        by[(r['text_no'], r['side'])].append(r)
    T = []
    for (tn, side), ls in by.items():
        ls.sort(key=lambda r: int(r['line']))
        seq = []
        for r in ls: seq += [int(x) for x in r['signs_clean'].split()]
        if 0 in seq or not seq: continue
        r = ls[0]
        t = dict(id=f'{tn}:{side}', site=r['site'], tf=IM_TYPE.get(r['object_type'], 'misc'), seq=seq, foreign=r['site'] == 'West Asian finds')
        t['tc'] = t['tf']
        t.update(parse_head(seq, M_HEAD, M_SUF, M_OPEN, M_CONN, M_NUMS, set(), M_FISH, M_GOODS, {121}, M_VAL, M_TALL))
        T.append(t)
    return T

# ---------------- permutation engine ----------------
def perm_profile(T, heads, dim_fn, strata_fn, nperm, subset=None, numeric=False):
    """dim_fn(t) -> category or None; strata_fn(t) -> stratum key. Returns dict head -> cat -> (obs, mean, sd, p_hi, p_lo), and n per head."""
    idx = [i for i, t in enumerate(T) if (subset is None or subset(t))]
    if not idx: return {}, {}
    hl = np.array([heads.index(T[i]['head']) if T[i]['head'] in heads else -1 for i in idx])
    cats_raw = [dim_fn(T[i]) for i in idx]
    if numeric:
        vals = np.array([v if v is not None else np.nan for v in cats_raw], float)
        H = len(heads)
        def stat(lab):
            out = np.full(H, np.nan)
            for h in range(H):
                m = (lab == h) & ~np.isnan(vals)
                if m.sum() >= 3: out[h] = vals[m].mean()
            return out
    else:
        cats = sorted({c for c in cats_raw if c is not None}, key=str)
        ci = {c: k for k, c in enumerate(cats)}
        cl = np.array([ci[c] if c is not None else -1 for c in cats_raw])
        H, K = len(heads), len(cats)
        def stat(lab):
            m = (lab >= 0) & (cl >= 0)
            return np.bincount(lab[m] * K + cl[m], minlength=H * K).reshape(H, K).astype(float)
    strata = collections.defaultdict(list)
    for k, i in enumerate(idx): strata[strata_fn(T[i])].append(k)
    groups = [np.array(v) for v in strata.values() if len(v) > 1]
    obs = stat(hl)
    null = np.empty((nperm,) + obs.shape)
    lab = hl.copy()
    for p in range(nperm):
        for gidx in groups:
            lab[gidx] = hl[gidx][rng.permutation(len(gidx))]
        null[p] = stat(lab)
    mean = np.nanmean(null, 0); sd = np.nanstd(null, 0)
    p_hi = (np.sum(null >= obs - 1e-9, 0) + 1) / (nperm + 1); p_lo = (np.sum(null <= obs + 1e-9, 0) + 1) / (nperm + 1)
    nh = {h: int((hl == k).sum()) for k, h in enumerate(heads)}
    res = {}
    if numeric:
        for k, h in enumerate(heads): res[h] = {'mean': (obs[k], mean[k], sd[k], p_hi[k], p_lo[k])}
    else:
        for k, h in enumerate(heads):
            res[h] = {c: (obs[k, ci[c]], mean[k, ci[c]], sd[k, ci[c]], p_hi[k, ci[c]], p_lo[k, ci[c]]) for c in cats}
    return res, nh

def star(p): return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
def report(title, res, nh, heads, min_n=10, show_all=False):
    P(f'\n### {title}')
    sig = []
    for h in heads:
        if h not in res or nh.get(h, 0) < min_n: continue
        r = res[h]; tot = sum(v[0] for v in r.values()) or 1
        parts = []
        for c, (o, m, s, ph, pl) in sorted(r.items(), key=lambda kv: -kv[1][0]):
            if o == 0 and m < 1: continue
            p = min(ph, pl); st = star(p)
            if st and o >= 3: sig.append((h, c, o, m, ph < pl))
            parts.append(f'{c} {int(o)} ({o / tot:.2f}; exp {m:.1f}{st})')
        P(f'  {h} [n={nh[h]}]: ' + '; '.join(parts[:9]))
    enr = [f'{h}:{c} {int(o)} vs {m:.1f}' for h, c, o, m, up in sig if up]
    dep = [f'{h}:{c} {int(o)} vs {m:.1f}' for h, c, o, m, up in sig if not up]
    P('  ENRICHED (p<0.05): ' + (', '.join(enr) if enr else 'none'))
    P('  DEPLETED (p<0.05): ' + (', '.join(dep) if dep else 'none'))
    return sig

# ---------------- cycles ----------------
def cycle1(T, heads, nperm):
    P(f'# S-DARK-60 cycle 1 ({LV}): object-type profile, carriers, impression match, mould/incised; nperm={nperm}')
    P(f'texts (complete, plus end-intact sealing fragments: {sum(1 for t in T if not t["complete"])}) {len(T)}; head counts: ' + ', '.join(f'{h} {sum(1 for t in T if t["head"] == h)}' for h in heads))
    SA = lambda t: (t['site'], lenbin(t['n']))
    res, nh = perm_profile(T, heads, lambda t: 'foreign' if t['foreign'] else t['tc'], lambda t: (t['foreign'], lenbin(t['n'])) if t['foreign'] else SA(t), nperm)
    report('Object type (coarse; foreign finds as one class; null: heads shuffled within site x length-bin)', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['tf'], SA, nperm)
    report('Object type (fine)', res, nh, heads)
    # sealings: carriers
    res, nh = perm_profile(T, heads, lambda t: t['carrier'], SA, nperm, subset=lambda t: t['tf'] == 'sealing')
    report('Sealings only: TAG subtype as recorded (TAG = plain; B = bale (S330); L, P, W, O, R, C, PC as coded)', res, nh, heads, min_n=3)
    res, nh = perm_profile(T, heads, lambda t: t['back'], SA, nperm, subset=lambda t: t['tf'] == 'sealing' and t['back'])
    report('Lothal sealings: back fastening (Frenez & Tosi 2005; S226)', res, nh, heads, min_n=2)
    # moulded vs incised
    res, nh = perm_profile(T, heads, lambda t: t['tf'], SA, nperm, subset=lambda t: t['tc'] == 'tablet')
    report('Tablets only: moulded (TAB:B) vs incised (TAB:I) vs copper (TAB:C)', res, nh, heads)
    # seal <-> impression matching: distinct seal texts with an exact sealing match anywhere
    seal_texts = collections.defaultdict(set); tag_texts = collections.defaultdict(set); tag_complete = collections.defaultdict(bool)
    for t in T:
        if t['tc'] == 'seal' and t['n'] >= 2: seal_texts[tuple(t['seq'])].add(t['site'])
        if t['tf'] == 'sealing' and t['n'] >= 2:
            tag_texts[tuple(t['seq'])].add(t['site']); tag_complete[tuple(t['seq'])] |= t['complete']
    # a sealing matches a seal text if it is complete and identical, or an end-intact fragment of >= 3 signs that is the seal text's end
    def tag_match(s): return any((u == s) if tag_complete[u] else (len(u) >= 3 and s[-len(u):] == u) for u in tag_texts)
    def seal_match(u): return (u in seal_texts) if tag_complete[u] else (len(u) >= 3 and any(v[-len(u):] == u for v in seal_texts))
    D = []
    for s, sites in seal_texts.items():
        h = parse_head(list(s), HEAD_GROUP, SUF, OPENERS, CONN, NUMS, MARK, FISH, GOODS, FIXED12, VAL, TALL)['head']
        D.append(dict(head=h, n=len(s), site=sorted(sites)[0], matched='yes' if tag_match(s) else 'no',
                      tab_match='yes' if any(tuple(t['seq']) == s for t in T if t['tc'] == 'tablet') else 'no'))
    res, nh = perm_profile(D, heads, lambda t: t['matched'], lambda t: lenbin(t['n']), nperm)
    report(f'Distinct complete seal texts (>=2 signs, {len(D)}): matched by a sealing anywhere (exact for complete sealings; an end-intact fragment of >= 3 signs must be the seal text\'s end; null within length-bin)', res, nh, heads)
    res, nh = perm_profile(D, heads, lambda t: t['tab_match'], lambda t: lenbin(t['n']), nperm)
    report('Distinct seal texts: exact match on a tablet anywhere', res, nh, heads)
    # sealings: share of sealing texts with a seal anywhere
    E = []
    for s, sites in tag_texts.items():
        h = parse_head(list(s), HEAD_GROUP, SUF, OPENERS, CONN, NUMS, MARK, FISH, GOODS, FIXED12, VAL, TALL)['head']
        E.append(dict(head=h, n=len(s), matched='yes' if seal_match(s) else 'no'))
    res, nh = perm_profile(E, heads, lambda t: t['matched'], lambda t: lenbin(t['n']), nperm)
    report(f'Distinct sealing texts incl. end-intact fragments (>=2 signs, {len(E)}): matched by a seal anywhere (same rule; ghost rate per head, cf. S-DARK-16)', res, nh, heads, min_n=3)
    # 1-sign texts share per head (flag for S-DARK-58 overlap)
    P('\n### Share of each head on 1-sign texts (reported only; S-DARK-58 covers minimal texts)')
    for h in heads:
        n = sum(1 for t in T if t['head'] == h); n1 = sum(1 for t in T if t['head'] == h and t['n'] == 1)
        if n: P(f'  {h}: {n1}/{n} = {n1 / n:.2f}')

def cycle2(T, heads, nperm, label='Wells'):
    P(f'# S-DARK-60 cycle 2 ({label} {LV}): companions; null = heads shuffled within site x type x length-bin; nperm={nperm}')
    SC = lambda t: (t['site'], t['tc'], lenbin(t['n']))
    SC2 = lambda t: (t['site'], t['tc'])
    sub2 = lambda t: t['n'] >= 2
    sigs = {}
    for name, fn, strat, sub in [('Opener present (texts >= 2 signs)', lambda t: t['opener'], SC, sub2),
                                 ('Connective W2/W60 in first three signs', lambda t: t['conn'], SC, sub2),
                                 ('Suffix after the head', lambda t: t['suffix'], SC, lambda t: t['n'] >= 2 and t['head'] not in ('W400', 'W90/91', 'none')),
                                 ('Numeral epithet directly before the head (series+value; S-DARK-44)', lambda t: t['epithet'], SC, lambda t: t['n'] >= 2 and t['head_sign'] is not None),
                                 ('Any count numeral in the body', lambda t: t['numeral'], SC, sub2),
                                 ('Fixed 12/24 (W55/W56) in the body', lambda t: t['fixed12'], SC, sub2),
                                 ('Fish word in the body', lambda t: t['fish'], SC, sub2),
                                 ('Quantity phrase (numeral + good) in the body', lambda t: t['quantity'], SC, sub2),
                                 ('Second unit: another head sign medial in the body', lambda t: t['second_unit'], SC, sub2),
                                 ('Middle length bins (opener unit and head removed; null within site x type, length free)', lambda t: t['midbin'], SC2, sub2)]:
        res, nh = perm_profile(T, heads, fn, strat, nperm, subset=sub)
        sigs[name] = report(name, res, nh, heads)
    # middle length mean
    res, nh = perm_profile(T, heads, lambda t: t['midlen'], SC2, nperm, subset=sub2, numeric=True)
    P('\n### Mean middle length (null within site x type)')
    for h in heads:
        if h in res and nh[h] >= 10:
            o, m, s, ph, pl = res[h]['mean']; P(f'  {h} [n={nh[h]}]: {o:.2f} vs null {m:.2f} (sd {s:.2f}) {star(min(ph, pl))}')
    # adjacent qualifier sets
    P('\n### Adjacent qualifier (sign before the head): distinct signs, entropy (bits), top 5 (texts >= 2 signs)')
    Q = {}
    for h in heads:
        c = collections.Counter(t['qual'] for t in T if t['head'] == h and t['qual'] is not None and t['n'] >= 2)
        if sum(c.values()) < 10: continue
        n = sum(c.values()); H = -sum(v / n * math.log2(v / n) for v in c.values())
        Q[h] = c
        P(f'  {h} [n={n}]: {len(c)} distinct, H={H:.2f}, top: ' + ', '.join(f'{k} x{v}' for k, v in c.most_common(5)))
    # pairwise qualifier overlap (histogram intersection) between heads
    hs = list(Q)
    P('  pairwise histogram-intersection of qualifier sets (S303 replication): ')
    for i in range(len(hs)):
        for k in range(i + 1, len(hs)):
            a, b = Q[hs[i]], Q[hs[k]]; na, nb = sum(a.values()), sum(b.values())
            ov = sum(min(a[x] / na, b.get(x, 0) / nb) for x in a)
            if ov >= 0.15: P(f'    {hs[i]} ~ {hs[k]}: {ov:.2f}')
    # head token position: final vs medial per head sign
    P('\n### Head sign tokens: final (head) vs medial (second-unit) per group')
    fin = collections.Counter(); med = collections.Counter()
    for t in T:
        if t['head'] in HEADS_ORDER and t['head_sign'] is not None: fin[t['head']] += 1
        for x in t['head_medial_tokens']:
            med[HEAD_GROUP.get(x, M_HEAD.get(x))] += 1
    for h in heads:
        if fin[h] + med[h] >= 10: P(f'  {h}: final {fin[h]}, medial {med[h]} ({med[h] / (fin[h] + med[h]):.2f} medial)')
    return sigs

def cycle3(T, heads, nperm):
    P(f'# S-DARK-60 cycle 3 ({LV}): geography, chronology, emblem/material/size, pairwise distinctness; nperm={nperm}')
    SB = lambda t: (t['tc'], lenbin(t['n']))
    SC = lambda t: (t['site'], t['tc'], lenbin(t['n']))
    def sitegrp(t):
        if t['foreign']: return 'foreign'
        return t['site'] if t['site'] in ('Mohenjo-daro', 'Harappa', 'Dholavira', 'Kalibangan', 'Lothal', 'Chanhu-daro') else 'other_home'
    res, nh = perm_profile(T, heads, sitegrp, SB, nperm)
    report('Site shares (null: heads shuffled within type x length-bin)', res, nh, heads)
    res, nh = perm_profile(T, heads, sitegrp, SB, nperm, subset=lambda t: t['tc'] == 'seal')
    report('Site shares, seals only', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['area'], SC, nperm, subset=lambda t: t['site'] == 'Mohenjo-daro' and t['area'] != 'unrecorded')
    report('Mohenjo-daro area-sections (null within type x length-bin)', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['phase'], SC, nperm, subset=lambda t: t['phase'] is not None)
    report('Phases (site-specific labels; null within site x type x length-bin)', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['emblem'], SC, nperm, subset=lambda t: t['tc'] == 'seal' and t['emblem'] != 'unknown')
    report('Emblem on seals (expected null per S-DARK-34)', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['material'], SC, nperm, subset=lambda t: t['material'] != 'unknown')
    report('Material (null within site x type x length-bin; S-DARK-35)', res, nh, heads)
    res, nh = perm_profile(T, heads, lambda t: t['size'], SC, nperm, subset=lambda t: t['tc'] == 'seal' and t['size'], numeric=True)
    P('\n### Mean seal width (mm), seals with a size (null within site x type x length-bin)')
    for h in heads:
        if h in res and nh[h] >= 10:
            o, m, s, ph, pl = res[h]['mean']; P(f'  {h} [n={nh[h]}]: {o:.1f} vs null {m:.1f} (sd {s:.1f}) {star(min(ph, pl))}')
    # ---- pairwise functional distinctness ----
    P('\n### Pairwise functional distinctness (profile over type_fine, site group, opener, suffix, epithet, fish, quantity, '
      'midbin, second unit; statistic = summed chi-square across dimensions; null = labels shuffled between the two heads '
      f'(a) freely, (b) within length-bin; {nperm}x; texts >= 2 signs)')
    dims = [lambda t: t['tf'], sitegrp, lambda t: t['opener'], lambda t: t['suffix'], lambda t: t['epithet'], lambda t: t['fish'],
            lambda t: t['quantity'], lambda t: t['midbin'], lambda t: t['second_unit']]
    dimnames = ['type', 'site', 'opener', 'suffix', 'epithet', 'fish', 'quantity', 'midlen', 'second']
    T2 = [t for t in T if t['n'] >= 2]
    byh = {h: [t for t in T2 if t['head'] == h] for h in heads}
    use = [h for h in heads if len(byh[h]) >= 15]
    def chi(A, B, fn):
        ca = collections.Counter(fn(t) for t in A); cb = collections.Counter(fn(t) for t in B)
        na, nb = len(A), len(B); tot = 0.0
        for c in set(ca) | set(cb):
            e_a = (ca[c] + cb[c]) * na / (na + nb); e_b = (ca[c] + cb[c]) * nb / (na + nb)
            if e_a > 0: tot += (ca[c] - e_a) ** 2 / e_a
            if e_b > 0: tot += (cb[c] - e_b) ** 2 / e_b
        return tot
    def allchi(A, B): return [chi(A, B, fn) for fn in dims]
    pairs = {}
    for i in range(len(use)):
        for k in range(i + 1, len(use)):
            A, B = byh[use[i]], byh[use[k]]; pool = A + B; na = len(A)
            obs = allchi(A, B); tot = sum(obs)
            nulls_free = []; nulls_len = []
            lb = [lenbin(t['n']) for t in pool]; groups = collections.defaultdict(list)
            for ix, l in enumerate(lb): groups[l].append(ix)
            lab = np.zeros(len(pool), int); lab[:na] = 1
            for p in range(min(nperm, 400)):
                perm = rng.permutation(len(pool)); A2 = [pool[x] for x in perm[:na]]; B2 = [pool[x] for x in perm[na:]]
                nulls_free.append(sum(allchi(A2, B2)))
                lab2 = lab.copy()
                for gidx in groups.values():
                    gi = np.array(gidx); lab2[gi] = lab[gi][rng.permutation(len(gi))]
                A3 = [pool[x] for x in range(len(pool)) if lab2[x] == 1]; B3 = [pool[x] for x in range(len(pool)) if lab2[x] == 0]
                nulls_len.append(sum(allchi(A3, B3)))
            pf = (sum(1 for v in nulls_free if v >= tot) + 1) / (len(nulls_free) + 1)
            pl = (sum(1 for v in nulls_len if v >= tot) + 1) / (len(nulls_len) + 1)
            top = sorted(zip(dimnames, obs), key=lambda kv: -kv[1])[:3]
            pairs[(use[i], use[k])] = (tot, pf, pl, top)
    npairs = len(pairs); bonf = 0.05 / max(npairs, 1)
    indist = []
    for (a, b), (tot, pf, pl, top) in sorted(pairs.items(), key=lambda kv: kv[1][1]):
        tag = 'DISTINCT' if pl < bonf else ('weak' if pl < 0.05 else 'INDISTINGUISHABLE')
        if tag != 'DISTINCT': indist.append((a, b, pl))
        P(f'  {a} vs {b} [n {len(byh[a])}/{len(byh[b])}]: chi {tot:.0f}, p_free {pf:.3f}, p_len {pl:.3f} -> {tag}; drivers ' +
          ', '.join(f'{d} {v:.0f}' for d, v in top))
    P(f'  pairs {npairs}, Bonferroni alpha {bonf:.4f}; distinct {npairs - len(indist)}, not distinct {len(indist)}')
    # components of the not-distinct graph -> number of functionally distinct heads
    parent = {h: h for h in use}
    def find(x):
        while parent[x] != x: x = parent[x]
        return x
    for a, b, pl in indist: parent[find(a)] = find(b)
    comps = collections.defaultdict(list)
    for h in use: comps[find(h)].append(h)
    P(f'  functionally distinct head groups (merging pairs not distinct after Bonferroni): {len(comps)} of {len(use)} heads tested')
    for c in comps.values(): P('    {' + ', '.join(c) + '}')
    return pairs, comps

def write_profiles(T, heads):
    """one row per head: the main shares, for loop60_head_profiles.csv (Wells, this level)."""
    rows = []
    for h in heads:
        X = [t for t in T if t['head'] == h]; n = len(X)
        if not n: continue
        X2 = [t for t in X if t['n'] >= 2]; n2 = len(X2) or 1
        def sh(fn, sub=X): return round(sum(1 for t in sub if fn(t)) / max(len(sub), 1), 3)
        seals = [t for t in X if t['tc'] == 'seal']; tabs = [t for t in X if t['tc'] == 'tablet']
        q = collections.Counter(t['qual'] for t in X2 if t['qual'] is not None)
        ep = collections.Counter(t['epithet'] for t in X2 if t['epithet'] != 'none')
        sz = [t['size'] for t in seals if t['size']]
        rows.append(dict(level=LV, head=h, n=n, n_ge2=len(X2),
                         seal=sh(lambda t: t['tc'] == 'seal'), tablet=sh(lambda t: t['tc'] == 'tablet'), sealing=sh(lambda t: t['tf'] == 'sealing'),
                         pot=sh(lambda t: t['tf'] == 'pot'), bangle=sh(lambda t: t['tf'] == 'bangle'), foreign=sh(lambda t: t['foreign']),
                         seal_square=sh(lambda t: t['tf'] == 'seal_sq', seals), seal_bar=sh(lambda t: t['tf'] == 'seal_bar', seals),
                         seal_round=sh(lambda t: t['tf'] in ('seal_round', 'seal_cyl'), seals),
                         tab_moulded=sh(lambda t: t['tf'] == 'tab_mould', tabs), tab_incised=sh(lambda t: t['tf'] == 'tab_inc', tabs), tab_copper=sh(lambda t: t['tf'] == 'tab_cu', tabs),
                         MD=sh(lambda t: t['site'] == 'Mohenjo-daro'), Harappa=sh(lambda t: t['site'] == 'Harappa'),
                         Dholavira=sh(lambda t: t['site'] == 'Dholavira'), Kalibangan=sh(lambda t: t['site'] == 'Kalibangan'), Lothal=sh(lambda t: t['site'] == 'Lothal'),
                         opener=sh(lambda t: t['opener'] == 'yes', X2), suffix_W400=sh(lambda t: t['suffix'] == 'W400', X2), suffix_W90=sh(lambda t: t['suffix'] == 'W90/91', X2),
                         numeral=sh(lambda t: t['numeral'] == 'yes', X2), fixed12=sh(lambda t: t['fixed12'] == 'yes', X2), fish=sh(lambda t: t['fish'] == 'yes', X2),
                         quantity=sh(lambda t: t['quantity'] == 'yes', X2), second_unit=sh(lambda t: t['second_unit'] == 'yes', X2),
                         midlen_mean=round(sum(t['midlen'] for t in X2) / n2, 2),
                         epithet_top=' '.join(f'{k}x{v}' for k, v in ep.most_common(3)), qual_distinct=len(q),
                         qual_top=' '.join(f'W{k}x{v}' for k, v in q.most_common(5)),
                         unicorn_share_seals=sh(lambda t: t['emblem'] == 'unicorn', [t for t in seals if t['emblem'] != 'unknown']),
                         steatite_share=sh(lambda t: t['material'] == 'steatite', [t for t in X if t['material'] != 'unknown']),
                         seal_width_mm=round(sum(sz) / len(sz), 1) if sz else ''))
    return rows

if __name__ == '__main__':
    heads = HEADS_ORDER
    if CY == 1:
        T = load_wells(LV); cycle1(T, heads, NP)
        open(OUT + f'loop60_c1_{LV}.txt', 'w').write('\n'.join(LOG))
    elif CY == 2:
        T = load_wells(LV, complete_only=True); cycle2(T, heads, NP, 'Wells')
        if LV == 'seq_raw':
            TI = load_im77(); hi = ['jar', 'mjar', 'arrow', 'W151', 'W156+154/158', 'box527', 'W226', 'W615/617', 'W236', 'W700', 'W595', 'W400', 'W90/91', 'none']
            P('\n\n########## IM77 replication (M numbers; W154/156/158 all = M15; W615/617 = M245 and W236 = M66 are proposed bridge entries) ##########')
            P(f'IM77 texts without breaks: {len(TI)}; heads: ' + ', '.join(f'{h} {sum(1 for t in TI if t["head"] == h)}' for h in hi))
            SA = lambda t: (t['site'], lenbin(t['n']))
            res, nh = perm_profile(TI, hi, lambda t: 'foreign' if t['foreign'] else t['tc'], SA, NP)
            report('IM77 object type (null within site x length-bin)', res, nh, hi)
            SB = lambda t: (t['tc'], lenbin(t['n']))
            res, nh = perm_profile(TI, hi, lambda t: t['site'], SB, NP)
            report('IM77 site shares (null within type x length-bin)', res, nh, hi)
            cycle2(TI, hi, NP, 'IM77')
        open(OUT + f'loop60_c2_{LV}.txt', 'w').write('\n'.join(LOG))
    elif CY == 3:
        T = load_wells(LV, complete_only=True); pairs, comps = cycle3(T, heads, NP)
        rows = write_profiles(T, heads)
        fn = OUT + 'loop60_head_profiles.csv'
        old = []
        if os.path.exists(fn):
            old = [r for r in csv.DictReader(open(fn)) if r['level'] != LV]
        with open(fn, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in old + rows: w.writerow(r)
        open(OUT + f'loop60_c3_{LV}.txt', 'w').write('\n'.join(LOG))
