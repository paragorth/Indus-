#!/usr/bin/env python3
"""S-DARK-62: A FUNCTIONAL PROFILE FOR EVERY OPENER, MARKER AND FISH WORD (mirror of S-DARK-60, which profiles heads).
Elements (Wells numbers; one text may carry several):
  openers  op817, op861, op820 (leaf / diamond / wheel), op920 (the 920-60(-741) phrase), op692 (692-60), and the
           candidate text-initial signs op503, op413, op501 (>= 20 tokens, >= 0.5 text-initial)
  markers  W2 in its two uses: W2_conn = directly after an opener sign; W2_free = anywhere else;
           W60_conn / W60_free likewise; W1 (short single stroke), W31 (tall single stroke), W32 (tall 2)
  fish     F220 plain, F240 whiskers, F235 hat, F233 bar, F231 stroke (W226 is a head, profiled in S-DARK-60)
Cycle 1  object type / carrier / mould-vs-incised / impressed-or-not; share of each type's texts carrying the element
Cycle 2  companions: element x head table, length, middle inventory, numerals, second units, sub-grammar (own order,
         own qualifier set)
Cycle 3  geography (site shares, Mohenjo-daro areas, phases, foreign), emblem / material / size, 'who writes it'
         (seal vs moulded-tablet batch; are opener-bearing seals impressed), pairwise distinctness -> how many
         functionally distinct openers / markers / fish words
Null: the element indicator is permuted among texts within strata that exclude the tested dimension (1,000x):
  site x length-bin        for object type, carrier, mould/incised, impression
  type x length-bin        for site shares and the foreign share
  site x type x length-bin for companions, areas, phases, emblem, material, size
All three merge levels (seq_raw / seq_strong / seq_all); IM77 replication through the bridge (M numbers).
Usage: python3 tools/dark_loop62.py <1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json, sys, csv, collections, math, re, os
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(62)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- sign classes (Wells) ----------------
HEAD_GROUP = {740: 'jar', 741: 'mjar', 742: 'mjar', 745: 'mjar', 520: 'arrow', 151: 'W151', 156: 'W156', 154: 'W154/158',
              158: 'W154/158', 527: 'box527', 526: 'box527', 226: 'W226', 617: 'W615/617', 615: 'W615/617', 236: 'W236',
              700: 'W700', 595: 'W595'}
SUF = {400: 'W400', 90: 'W90/91', 91: 'W90/91'}
OPENERS = {817: 'op817', 861: 'op861', 820: 'op820', 920: 'op920', 692: 'op692'}
CAND = {503: 'op503', 413: 'op413', 501: 'op501'}
CONN = {2: 'W2', 60: 'W60'}
MARKERS = {1: 'W1', 31: 'W31', 32: 'W32'}
FISH = {220: 'F220', 240: 'F240', 235: 'F235', 233: 'F233', 231: 'F231'}
SHORT = set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30))
TALL = set(range(32, 40))
NUMS = (SHORT | TALL) - {1, 2, 31}
GOODS = {390, 405, 407, 900, 384, 388, 645, 904, 220}
FOREIGN_REGIONS = {'Persian Gulf', 'Mesopotamia', 'Iranian Plateau', 'Central Asia'}
ELEMENTS = ['op817', 'op861', 'op820', 'op920', 'op692', 'op503', 'op413', 'op501',
            'W2_conn', 'W2_free', 'W60_conn', 'W60_free', 'W1', 'W31', 'W32',
            'F220', 'F240', 'F235', 'F233', 'F231']
OPEN_EL = ELEMENTS[:5]; CAND_EL = ELEMENTS[5:8]; MARK_EL = ELEMENTS[8:15]; FISH_EL = ELEMENTS[15:]

# IM77 (M numbers). M267 = W817 + W861 (one sign in Mahadevan); M99/100 = W2; M97/98 = W1; M86 = W31; M87 = W32
M_HEAD = {342: 'jar', 343: 'mjar', 344: 'mjar', 345: 'mjar', 211: 'arrow', 12: 'W151', 15: 'W156', 254: 'box527',
          60: 'W226', 245: 'W615/617', 66: 'W236', 328: 'W700', 252: 'W595'}
M_SUF = {176: 'W400', 1: 'W90/91'}
M_OPENERS = {267: 'op817+861', 391: 'op820', 293: 'op920', 150: 'op692'}
M_CAND = {}
M_CONN = {99: 'W2', 100: 'W2', 123: 'W60'}
M_MARKERS = {97: 'W1', 98: 'W1', 86: 'W31', 87: 'W32'}
M_FISH = {59: 'F220', 67: 'F240', 65: 'F235', 72: 'F233', 70: 'F231'}
M_TALL = set(range(87, 97)); M_SHORT = set(range(101, 121)); M_NUMS = (M_TALL | M_SHORT) - {86, 97, 98, 99, 100}
M_GOODS = {161, 162, 167, 168, 169, 287, 59}
M_ELEMENTS = ['op817+861', 'op820', 'op920', 'op692', 'W2_conn', 'W2_free', 'W60_conn', 'W60_free', 'W1', 'W31', 'W32',
              'F220', 'F240', 'F235', 'F233', 'F231']

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

# ---------------- element parse ----------------
def parse_elements(seq, openers, cand, conn, markers, fish, head_group, suf_map, nums, goods):
    s = list(seq); suffix = []
    while len(s) > 1 and s[-1] in suf_map: suffix.append(s.pop())
    head = head_group.get(s[-1])
    if head is None:
        head = suf_map[suffix[-1]] if suffix else 'none'; body = s; head_sign = None
    else:
        head_sign = s[-1]; body = s[:-1]
    f = dict(head=head, head_sign=head_sign, suffix=(suf_map[suffix[-1]] if suffix else 'no'), n=len(seq), els=set())
    op = openers.get(s[0]) if s else None
    if op is None and s and s[0] in cand: op = cand[s[0]]
    f['opener'] = op or 'none'
    if op: f['els'].add(op)
    opener_set = set(openers) | set(cand)
    for i, x in enumerate(s):
        if x in conn:
            use = '_conn' if (i >= 1 and s[i - 1] in opener_set) else '_free'
            f['els'].add(conn[x] + use)
        if x in markers: f['els'].add(markers[x])
        if x in fish and not (i == len(s) - 1 and head_sign == x): f['els'].add(fish[x])
    f['fish_any'] = 'yes' if any(x in fish for x in body) else 'no'
    f['nfish'] = sum(1 for x in body if x in fish)
    # middle = body minus opener unit (opener [+ connective])
    mid = list(body)
    if mid and mid[0] in opener_set:
        mid = mid[1:]
        if mid and mid[0] in conn: mid = mid[1:]
    f['mid'] = mid; f['midlen'] = len(mid); f['midbin'] = str(min(len(mid), 4)) + ('+' if len(mid) >= 4 else '')
    f['lenbin'] = str(lenbin(len(seq)))
    f['numeral'] = 'yes' if any(x in nums for x in body) else 'no'
    f['quantity'] = 'yes' if any(body[i] in nums and body[i + 1] in goods for i in range(len(body) - 1)) else 'no'
    f['second_unit'] = 'yes' if any(x in head_group for x in body) else 'no'
    f['mjar_any'] = 'yes' if any(head_group.get(x) == 'mjar' for x in s) else 'no'
    # 920 phrase form
    if op == 'op920' or op == 'op920':
        f['p920'] = '920-60-mjar' if (len(s) >= 3 and s[1] in conn and head_group.get(s[2]) == 'mjar') else ('920-60' if len(s) >= 2 and s[1] in conn else '920-other')
    else: f['p920'] = None
    f['nW2'] = sum(1 for x in s if conn.get(x) == 'W2')
    return f

def complete_only_types_ok(ty): return ty.startswith('TAG')
def load_wells(level, complete_only=False):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    g = lambda r, k: (r.get(k) or '')
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
        startok = (c['complete'] == 'Y') or (text_ok and r['text'].endswith('+'))   # reading START = displayed right end
        endok = (c['complete'] == 'Y') or (text_ok and r['text'].startswith('+'))
        # openers need an intact start, heads an intact end: keep complete texts, plus end-intact sealing fragments
        if not endok: continue
        if c['complete'] != 'Y' and not complete_only_types_ok(c['type']): continue
        tf = type_fine(c['type'])
        foreign = g(r, 'region') in FOREIGN_REGIONS
        try: h = float(g(r, 'horizontal(mm)') or 0)
        except ValueError: h = 0.0
        t = dict(id=g(r, 'id'), cisi=c['cisi'], site=c['site'], complete=c['complete'] == 'Y', startok=startok, type=c['type'], tf=tf,
                 tc=type_coarse(tf), foreign=foreign,
                 carrier=(c['type'].split(':')[1] if c['type'].startswith('TAG') and ':' in c['type'] else ('TAG' if tf == 'sealing' else None)),
                 emblem=emblem_group(c['symbol']), material=material_group(c['material']), size=h if h > 0 else None,
                 area=md_area(c['area-section']) if c['site'] == 'Mohenjo-daro' else None,
                 phase=phase_of(c['site'], c['period'], c['phase']), seq=list(s), big=c['site'] in ('Mohenjo-daro', 'Harappa'))
        t.update(parse_elements(t['seq'], OPENERS, CAND, CONN, MARKERS, FISH, HEAD_GROUP, SUF, NUMS, GOODS))
        if not t['startok']:
            t['els'] = {e for e in t['els'] if not e.startswith('op') and not e.endswith('_conn')}; t['opener'] = 'unk'
        if complete_only and not t['complete']: continue
        T.append(t)
    return T

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
        t = dict(id=f'{tn}:{side}', site=r['site'], tf=IM_TYPE.get(r['object_type'], 'misc'), seq=seq, foreign=r['site'] == 'West Asian finds',
                 complete=True, startok=True, carrier=None, emblem='unknown', material='unknown', size=None, area=None, phase=None)
        t['tc'] = t['tf'] if t['tf'] != 'tab_cu' else 'tablet'
        t.update(parse_elements(seq, M_OPENERS, M_CAND, M_CONN, M_MARKERS, M_FISH, M_HEAD, M_SUF, M_NUMS, M_GOODS))
        T.append(t)
    return T

# ---------------- permutation engine (binary element indicators) ----------------
def perm_elements(T, els, dim_fn, strata_fn, nperm, subset=None, numeric=False):
    """For each element: category counts among element-bearing texts, vs null where all element indicators are permuted
    jointly among texts within strata. Returns res[el][cat] = (obs, mean, sd, p_hi, p_lo), n[el]."""
    idx = [i for i, t in enumerate(T) if (subset is None or subset(t))]
    if not idx: return {}, {}
    E = len(els)
    I = np.zeros((len(idx), E), bool)
    for k, i in enumerate(idx):
        for e in T[i]['els']:
            if e in els: I[k, els.index(e)] = True
    cats_raw = [dim_fn(T[i]) for i in idx]
    if numeric:
        vals = np.array([v if v is not None else np.nan for v in cats_raw], float)
        ok = ~np.isnan(vals); v0 = np.where(ok, vals, 0.0)
        def stat(M):
            cnt = (M & ok[:, None]).sum(0); sm = v0 @ M
            return np.where(cnt >= 3, sm / np.maximum(cnt, 1), np.nan)
    else:
        cats = sorted({c for c in cats_raw if c is not None}, key=str)
        ci = {c: k for k, c in enumerate(cats)}
        K = len(cats)
        OH = np.zeros((len(idx), K), float)
        for k, c in enumerate(cats_raw):
            if c is not None: OH[k, ci[c]] = 1.0
        def stat(M): return M.T.astype(float) @ OH   # E x K
    strata = collections.defaultdict(list)
    for k, i in enumerate(idx): strata[strata_fn(T[i])].append(k)
    groups = [np.array(v) for v in strata.values() if len(v) > 1]
    obs = stat(I)
    null = np.empty((nperm,) + obs.shape)
    M = I.copy()
    for p in range(nperm):
        for gidx in groups: M[gidx] = I[gidx][rng.permutation(len(gidx))]
        null[p] = stat(M)
    mean = np.nanmean(null, 0); sd = np.nanstd(null, 0)
    p_hi = (np.sum(null >= obs - 1e-9, 0) + 1) / (nperm + 1); p_lo = (np.sum(null <= obs + 1e-9, 0) + 1) / (nperm + 1)
    n = {e: int(I[:, k].sum()) for k, e in enumerate(els)}
    res = {}
    if numeric:
        for k, e in enumerate(els): res[e] = {'mean': (obs[k], mean[k], sd[k], p_hi[k], p_lo[k])}
    else:
        for k, e in enumerate(els): res[e] = {c: (obs[k, ci[c]], mean[k, ci[c]], sd[k, ci[c]], p_hi[k, ci[c]], p_lo[k, ci[c]]) for c in cats}
    return res, n

def star(p): return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
def report(title, res, n, els, min_n=10, maxparts=9):
    P(f'\n### {title}')
    sig = []
    for e in els:
        if e not in res or n.get(e, 0) < min_n: continue
        r = res[e]; tot = sum(v[0] for v in r.values()) or 1
        parts = []
        for c, (o, m, s, ph, pl) in sorted(r.items(), key=lambda kv: -kv[1][0]):
            if o == 0 and m < 1: continue
            p = min(ph, pl); st = star(p)
            if st and (o >= 3 or m >= 3): sig.append((e, c, o, m, ph < pl))
            parts.append(f'{c} {int(o)} ({o / tot:.2f}; exp {m:.1f}{st})')
        P(f'  {e} [n={n[e]}]: ' + '; '.join(parts[:maxparts]))
    enr = [f'{e}:{c} {int(o)} vs {m:.1f}' for e, c, o, m, up in sig if up]
    dep = [f'{e}:{c} {int(o)} vs {m:.1f}' for e, c, o, m, up in sig if not up]
    P('  ENRICHED (p<0.05): ' + (', '.join(enr) if enr else 'none'))
    P('  DEPLETED (p<0.05): ' + (', '.join(dep) if dep else 'none'))
    return sig
def report_numeric(title, res, n, els, fmt='{:.2f}', min_n=10):
    P(f'\n### {title}')
    for e in els:
        if e in res and n.get(e, 0) >= min_n:
            o, m, s, ph, pl = res[e]['mean']
            if not np.isnan(o): P(f'  {e} [n={n[e]}]: ' + fmt.format(o) + ' vs null ' + fmt.format(m) + f' (sd {s:.2f}) {star(min(ph, pl))}')

def share_table(T, els, dim_fn, title, subset=None):
    """share of each category's texts that carry the element (the 'penetration' view)"""
    P(f'\n### {title}')
    X = [t for t in T if subset is None or subset(t)]
    cats = collections.Counter(dim_fn(t) for t in X if dim_fn(t) is not None)
    order = [c for c, _ in cats.most_common()]
    P('  ' + 'element'.ljust(10) + ''.join(f'{c[:11]:>13}' for c in order))
    P('  ' + 'n texts'.ljust(10) + ''.join(f'{cats[c]:>13}' for c in order))
    for e in els:
        row = []
        for c in order:
            k = sum(1 for t in X if dim_fn(t) == c and e in t['els'])
            row.append(f'{k}/{cats[c]}={k / cats[c]:.2f}' if cats[c] else '-')
        P('  ' + e.ljust(10) + ''.join(f'{x:>13}' for x in row))

# ---------------- cycle 1 ----------------
def cycle1(T, els, nperm, label='Wells'):
    P(f'# S-DARK-62 cycle 1 ({label} {LV}): object type, carrier, mould/incised, impression; nperm={nperm}')
    P(f'texts {len(T)} (complete {sum(1 for t in T if t["complete"])}, end-intact sealing fragments {sum(1 for t in T if not t["complete"])}); '
      'element counts: ' + ', '.join(f'{e} {sum(1 for t in T if e in t["els"])}' for e in els))
    SA = lambda t: (t['site'], lenbin(t['n']))
    SA2 = lambda t: (t['foreign'], lenbin(t['n'])) if t['foreign'] else SA(t)
    T2 = [t for t in T if t['n'] >= 2]
    res, n = perm_elements(T2, els, lambda t: 'foreign' if t['foreign'] else t['tc'], SA2, nperm)
    report('Object type (coarse; foreign finds one class; texts >= 2 signs; null: element shuffled within site x length-bin)', res, n, els)
    share_table(T2, els, lambda t: 'foreign' if t['foreign'] else t['tc'], 'Share of each type\'s texts (>= 2 signs) carrying the element')
    res, n = perm_elements(T2, els, lambda t: t['tf'], SA, nperm)
    report('Object type (fine)', res, n, els)
    share_table(T2, els, lambda t: t['tf'], 'Share of each fine type\'s texts (>= 2 signs) carrying the element',
                subset=lambda t: t['tf'] in ('seal_sq', 'seal_bar', 'seal_round', 'tab_mould', 'tab_inc', 'tab_cu', 'sealing', 'pot'))
    res, n = perm_elements(T2, els, lambda t: t['tf'], SA, nperm, subset=lambda t: t['tc'] == 'tablet')
    report('Tablets only: moulded (TAB:B) vs incised (TAB:I) vs copper (TAB:C)', res, n, els)
    if label == 'Wells':
        res, n = perm_elements(T2, els, lambda t: t['carrier'], SA, nperm, subset=lambda t: t['tf'] == 'sealing')
        report('Sealings only: TAG subtype (TAG plain; B bale; L, P, W, O, R, C, PC as coded)', res, n, els, min_n=3)
        # one-sign texts
        P('\n### One-sign texts: which elements stand alone (S-DARK-58 covers minimal texts; reported for the openers)')
        c = collections.Counter()
        for t in T:
            if t['n'] == 1:
                for e in t['els']: c[e] += 1
        P('  ' + ', '.join(f'{e} {c[e]}' for e in els if c[e]) + f' (of {sum(1 for t in T if t["n"] == 1)} one-sign texts)')
        # potters' marks: pots with an opener sign anywhere
        P('\n### Pots: opener-family signs on pots by period (S-DARK-23.2)')
        for t in T:
            if t['tf'] == 'pot' and any(x in OPENERS for x in t['seq']):
                P(f'  {t["cisi"]} {t["site"]} {t["type"]} period {t["phase"]} seq {t["seq"]}')
    # impression: distinct seal texts matched by a sealing / tablet
    seal_texts = collections.defaultdict(set); tag_texts = collections.defaultdict(set); tag_complete = collections.defaultdict(bool)
    tab_texts = collections.Counter()
    for t in T:
        if t['tc'] == 'seal' and t['n'] >= 2 and t['complete']: seal_texts[tuple(t['seq'])].add(t['site'])
        if t['tf'] == 'sealing' and t['n'] >= 2:
            tag_texts[tuple(t['seq'])].add(t['site']); tag_complete[tuple(t['seq'])] |= t['complete']
        if t['tc'] == 'tablet' and t['n'] >= 2: tab_texts[tuple(t['seq'])] += 1
    def tag_match(s): return any((u == s) if tag_complete[u] else (len(u) >= 3 and s[-len(u):] == u) for u in tag_texts)
    D = []
    for s, sites in seal_texts.items():
        f = parse_elements(list(s), *( (OPENERS, CAND, CONN, MARKERS, FISH, HEAD_GROUP, SUF, NUMS, GOODS) if label == 'Wells' else
                                      (M_OPENERS, M_CAND, M_CONN, M_MARKERS, M_FISH, M_HEAD, M_SUF, M_NUMS, M_GOODS)))
        D.append(dict(els=f['els'], n=len(s), site=sorted(sites)[0], matched='yes' if tag_match(s) else 'no',
                      tab_match='yes' if s in tab_texts else 'no'))
    res, n = perm_elements(D, els, lambda t: t['matched'], lambda t: lenbin(t['n']), nperm)
    report(f'Distinct complete seal texts (>= 2 signs, {len(D)}): matched by a sealing anywhere (exact, or an end-intact fragment >= 3 signs = the seal text\'s end; null within length-bin)', res, n, els)
    res, n = perm_elements(D, els, lambda t: t['tab_match'], lambda t: lenbin(t['n']), nperm)
    report('Distinct seal texts: exact match on a tablet anywhere', res, n, els)
    # tablets: batch size (copies per distinct text) per element
    P('\n### Tablets: copies per distinct text (batch factor) for texts carrying the element vs not, by site (texts >= 2 signs)')
    for site in ('Harappa', 'Mohenjo-daro'):
        cc = collections.Counter(tuple(t['seq']) for t in T if t['tc'] == 'tablet' and t['n'] >= 2 and t['site'] == site)
        if not cc: continue
        def els_of(s):
            f = parse_elements(list(s), *((OPENERS, CAND, CONN, MARKERS, FISH, HEAD_GROUP, SUF, NUMS, GOODS) if label == 'Wells' else
                                           (M_OPENERS, M_CAND, M_CONN, M_MARKERS, M_FISH, M_HEAD, M_SUF, M_NUMS, M_GOODS)))
            return f['els']
        E = {s: els_of(s) for s in cc}
        parts = []
        for e in els:
            w = [cc[s] for s in cc if e in E[s]]; wo = [cc[s] for s in cc if e not in E[s]]
            if len(w) >= 5: parts.append(f'{e}: {len(w)} texts, {sum(w)} tokens, {sum(w) / len(w):.1f} copies/text (others {sum(wo) / max(len(wo), 1):.1f})')
        P(f'  {site}: ' + '; '.join(parts))

# ---------------- cycle 2 ----------------
def cycle2(T, els, nperm, label='Wells', heads_order=None):
    P(f'# S-DARK-62 cycle 2 ({label} {LV}): companions; null = element shuffled within site x type x length-bin; nperm={nperm}')
    SC = lambda t: (t['site'], t['tc'], lenbin(t['n'])); SC2 = lambda t: (t['site'], t['tc'])
    T2 = [t for t in T if t['n'] >= 2]
    res, n = perm_elements(T2, els, lambda t: t['head'], SC, nperm)
    report('Element x HEAD table (closer-slot sign of the text; texts >= 2 signs)', res, n, els, maxparts=12)
    for name, fn, strat, sub in [('Suffix after the head', lambda t: t['suffix'], SC, lambda t: t['head'] not in ('W400', 'W90/91', 'none')),
                                 ('Any count numeral in the body (W1/W2/W31 excluded)', lambda t: t['numeral'], SC, None),
                                 ('Quantity phrase (numeral + good)', lambda t: t['quantity'], SC, None),
                                 ('Fish word anywhere in the body', lambda t: t['fish_any'], SC, None),
                                 ('Marked jar W741/742/745 anywhere', lambda t: t['mjar_any'], SC, None),
                                 ('Second unit (another head sign medial)', lambda t: t['second_unit'], SC, None),
                                 ('Text length bins (null within site x type, length free)', lambda t: t['lenbin'], SC2, None),
                                 ('Middle length bins (opener unit and head removed; null within site x type)', lambda t: t['midbin'], SC2, None)]:
        res, n = perm_elements(T2, els, fn, strat, nperm, subset=sub)
        report(name, res, n, els)
    res, n = perm_elements(T2, els, lambda t: t['n'], SC2, nperm, numeric=True)
    report_numeric('Mean text length (null within site x type)', res, n, els)
    # co-occurrence among elements
    P('\n### Element co-occurrence within one text (obs / expected under the site x type x length-bin shuffle)')
    res, n = perm_elements(T2, els, lambda t: tuple(sorted(e for e in t['els'] if e in els)), SC, min(nperm, 300))
    pair_obs = collections.Counter(); pair_exp = collections.Counter()
    for e in els:
        for c, (o, m, s, ph, pl) in res[e].items():
            for e2 in c:
                if e2 != e and e2 in els: pair_obs[(e, e2)] += o; pair_exp[(e, e2)] += m
    lines = []
    for (a, b), o in sorted(pair_obs.items()):
        if els.index(a) < els.index(b) and (o >= 5 or pair_exp[(a, b)] >= 5):
            lines.append(f'{a}+{b} {int(o)}/{pair_exp[(a, b)]:.1f}')
    P('  ' + '; '.join(lines))
    # the 920 phrase forms
    P('\n### op920: phrase form')
    c = collections.Counter(t['p920'] for t in T2 if t['p920'])
    P('  ' + ', '.join(f'{k} {v}' for k, v in c.most_common()))
    # W2 count per text
    c = collections.Counter(t['nW2'] for t in T2 if t['nW2'])
    P(f'\n### W2 tokens per text: ' + ', '.join(f'{k}x: {v}' for k, v in sorted(c.items())))
    # what follows W2 in each use / W60 in each use
    P('\n### Sign directly after the connective, by use (top 8)')
    conn_map = CONN if label == 'Wells' else M_CONN
    opener_set = set(OPENERS) | set(CAND) if label == 'Wells' else set(M_OPENERS)
    for cname in ('W2', 'W60'):
        for use in ('_conn', '_free'):
            cc = collections.Counter(); nn = 0
            for t in T2:
                s = t['seq']
                for i, x in enumerate(s):
                    if conn_map.get(x) == cname and ((i >= 1 and s[i - 1] in opener_set) == (use == '_conn')):
                        nn += 1
                        if i + 1 < len(s): cc[s[i + 1]] += 1
                        else: cc['END'] += 1
            P(f'  {cname}{use} [n={nn}]: ' + ', '.join(f'{k} x{v}' for k, v in cc.most_common(8)))
    # sign directly before free W2 / W1 / W31
    P('\n### Sign directly before the marker (top 8)')
    mk = MARKERS if label == 'Wells' else M_MARKERS
    for cname in ('W2_free', 'W1', 'W31', 'W32'):
        cc = collections.Counter(); nn = 0
        for t in T2:
            s = t['seq']
            for i, x in enumerate(s):
                hit = (conn_map.get(x) == 'W2' and not (i >= 1 and s[i - 1] in opener_set)) if cname == 'W2_free' else (mk.get(x) == cname)
                if hit:
                    nn += 1; cc[s[i - 1] if i >= 1 else 'START'] += 1
        P(f'  {cname} [n={nn}]: ' + ', '.join(f'{k} x{v}' for k, v in cc.most_common(8)))
    # ---- sub-grammar: own order and own middle inventory per opener ----
    P('\n### Sub-grammar test per opener: (a) order of sign pairs inside the middle vs the order in all other texts; '
      '(b) middle-inventory distinctness (JSD of middle-sign distribution vs the rest); null: opener label shuffled within site x type x length-bin')
    def pair_orders(X):
        c = collections.Counter()
        for t in X:
            m = t['mid']
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    if m[i] != m[j]: c[(m[i], m[j])] += 1
        return c
    def disagreements(A, B, minn=4):
        ca, cb = pair_orders(A), pair_orders(B); dis = 0; tested = 0
        for (a, b), k in ca.items():
            if a < b:
                ka, kb = ca[(a, b)], ca[(b, a)]; la, lb = cb[(a, b)], cb[(b, a)]
                if ka + kb >= minn and la + lb >= minn and ka != kb and la != lb:
                    tested += 1
                    if (ka > kb) != (la > lb): dis += 1
        return dis, tested
    def jsd(A, B):
        ca = collections.Counter(x for t in A for x in t['mid']); cb = collections.Counter(x for t in B for x in t['mid'])
        na, nb = sum(ca.values()), sum(cb.values())
        if not na or not nb: return float('nan')
        keys = set(ca) | set(cb); out = 0.0
        for k in keys:
            p, q = ca[k] / na, cb[k] / nb; m = (p + q) / 2
            if p: out += 0.5 * p * math.log2(p / m)
            if q: out += 0.5 * q * math.log2(q / m)
        return out
    strata = collections.defaultdict(list)
    for k, t in enumerate(T2): strata[SC(t)].append(k)
    groups = [np.array(v) for v in strata.values() if len(v) > 1]
    for e in [x for x in els if x.startswith('op')]:
        lab = np.array([e in t['els'] for t in T2])
        if lab.sum() < 15: continue
        A = [t for t, l in zip(T2, lab) if l]; B = [t for t, l in zip(T2, lab) if not l]
        d, tested = disagreements(A, B); j = jsd(A, B)
        nd = []; nj = []
        for p in range(min(nperm, 200)):
            l2 = lab.copy()
            for g in groups: l2[g] = lab[g][rng.permutation(len(g))]
            A2 = [t for t, l in zip(T2, l2) if l]; B2 = [t for t, l in zip(T2, l2) if not l]
            d2, t2 = disagreements(A2, B2); nd.append(d2 / max(t2, 1)); nj.append(jsd(A2, B2))
        rate = d / max(tested, 1)
        pd_ = (sum(1 for v in nd if v >= rate) + 1) / (len(nd) + 1); pj = (sum(1 for v in nj if v >= j) + 1) / (len(nj) + 1)
        ninv = len({x for t in A for x in t['mid']})
        P(f'  {e} [n={len(A)}; middle inventory {ninv} signs]: order disagreements {d}/{tested} = {rate:.2f} (null {np.mean(nd):.2f}, p {pd_:.2f}); '
          f'middle JSD {j:.3f} (null {np.mean(nj):.3f}, p {pj:.3f})')
    # middle inventory top signs per opener (qualifier sets)
    P('\n### First middle sign after the opener unit (top 6) and distinct count')
    for e in [x for x in els if x.startswith('op')]:
        c = collections.Counter(t['mid'][0] for t in T2 if e in t['els'] and t['mid'])
        if sum(c.values()) >= 10: P(f'  {e} [n={sum(c.values())}]: {len(c)} distinct; ' + ', '.join(f'{k} x{v}' for k, v in c.most_common(6)))
    # fish words: position in text, what precedes/follows
    P('\n### Fish words: relative position (0 = first sign, 1 = last), sign before and after (top 5)')
    fish_map = FISH if label == 'Wells' else M_FISH
    for e in [x for x in els if x.startswith('F')]:
        pos = []; before = collections.Counter(); after = collections.Counter()
        for t in T2:
            s = t['seq']
            for i, x in enumerate(s):
                if fish_map.get(x) == e:
                    pos.append(i / (len(s) - 1)); before[s[i - 1] if i else 'START'] += 1; after[s[i + 1] if i + 1 < len(s) else 'END'] += 1
        if pos:
            P(f'  {e} [tokens {len(pos)}]: mean rel. position {np.mean(pos):.2f}; before: ' + ', '.join(f'{k} x{v}' for k, v in before.most_common(5)) +
              '; after: ' + ', '.join(f'{k} x{v}' for k, v in after.most_common(5)))

# ---------------- cycle 3 ----------------
def cycle3(T, els, nperm, label='Wells'):
    P(f'# S-DARK-62 cycle 3 ({label} {LV}): geography, chronology, emblem/material/size, who writes it, pairwise distinctness; nperm={nperm}')
    SB = lambda t: (t['tc'], lenbin(t['n'])); SC = lambda t: (t['site'], t['tc'], lenbin(t['n']))
    T2 = [t for t in T if t['n'] >= 2]
    def sitegrp(t):
        if t['foreign']: return 'foreign'
        return t['site'] if t['site'] in ('Mohenjo-daro', 'Harappa', 'Dholavira', 'Kalibangan', 'Lothal', 'Chanhu-daro') else 'other_home'
    res, n = perm_elements(T2, els, sitegrp, SB, nperm)
    report('Site shares (texts >= 2 signs; null: element shuffled within type x length-bin)', res, n, els)
    share_table(T2, els, sitegrp, 'Share of each site\'s texts (>= 2 signs) carrying the element')
    res, n = perm_elements(T2, els, sitegrp, SB, nperm, subset=lambda t: t['tc'] == 'seal')
    report('Site shares, seals only', res, n, els)
    if label == 'Wells':
        res, n = perm_elements(T2, els, lambda t: t['area'], SC, nperm, subset=lambda t: t['site'] == 'Mohenjo-daro' and t['area'] != 'unrecorded')
        report('Mohenjo-daro area-sections (null within type x length-bin)', res, n, els)
        res, n = perm_elements(T2, els, lambda t: t['phase'], SC, nperm, subset=lambda t: t['phase'] is not None)
        report('Phases (site-specific labels; null within site x type x length-bin)', res, n, els)
        res, n = perm_elements(T2, els, lambda t: t['emblem'], SC, nperm, subset=lambda t: t['tc'] == 'seal' and t['emblem'] != 'unknown')
        report('Emblem on seals (expected null, S15/S-DARK-34)', res, n, els)
        res, n = perm_elements(T2, els, lambda t: t['material'], SC, nperm, subset=lambda t: t['material'] != 'unknown')
        report('Material (null within site x type x length-bin)', res, n, els)
        res, n = perm_elements(T2, els, lambda t: t['size'], SC, nperm, subset=lambda t: t['tc'] == 'seal' and t['size'], numeric=True)
        report_numeric('Mean seal width (mm), seals with a size (null within site x type x length-bin)', res, n, els, fmt='{:.1f}')
    # ---- who writes it: seal vs moulded tablet, per site ----
    P('\n### Who writes it: element share among Harappa moulded tablets, Harappa incised tablets, Harappa seals, Mohenjo-daro seals, '
      'Mohenjo-daro tablets, sealings (texts >= 2 signs; deduplicated = each distinct text once at that site)')
    cells = [('H tab_mould', lambda t: t['site'] == 'Harappa' and t['tf'] == 'tab_mould'), ('H tab_inc', lambda t: t['site'] == 'Harappa' and t['tf'] == 'tab_inc'),
             ('H seal', lambda t: t['site'] == 'Harappa' and t['tc'] == 'seal'), ('MD seal', lambda t: t['site'] == 'Mohenjo-daro' and t['tc'] == 'seal'),
             ('MD tablet', lambda t: t['site'] == 'Mohenjo-daro' and t['tc'] == 'tablet'), ('sealing', lambda t: t['tf'] == 'sealing'),
             ('other-site seal', lambda t: not t['big'] and not t['foreign'] and t['tc'] == 'seal')]
    if label != 'Wells':
        cells = [('H tablet', lambda t: t['site'] == 'Harappa' and t['tc'] == 'tablet'), ('H seal', lambda t: t['site'] == 'Harappa' and t['tc'] == 'seal'),
                 ('MD seal', lambda t: t['site'] == 'Mohenjodaro' and t['tc'] == 'seal'), ('MD tablet', lambda t: t['site'] == 'Mohenjodaro' and t['tc'] == 'tablet'),
                 ('sealing', lambda t: t['tf'] == 'sealing'), ('other seal', lambda t: t['site'] not in ('Mohenjodaro', 'Harappa') and not t['foreign'] and t['tc'] == 'seal')]
    P('  ' + 'element'.ljust(10) + ''.join(f'{c[0]:>22}' for c in cells))
    for e in els:
        row = []
        for name, fn in cells:
            X = [t for t in T2 if fn(t)]; D = {tuple(t['seq']) for t in X}
            k = sum(1 for t in X if e in t['els']); kd = sum(1 for s in D if any(e in t['els'] for t in X if tuple(t['seq']) == s))
            row.append(f'{k}/{len(X)}={k / len(X):.2f} d{kd}/{len(D)}' if X else '-')
        P('  ' + e.ljust(10) + ''.join(f'{x:>22}' for x in row))
    # ---- pairwise distinctness ----
    P('\n### Pairwise functional distinctness within class (openers; markers; fish). Profile over type_fine, site group, head, suffix, '
      'numeral, quantity, midbin, fish_any, second unit; statistic = summed chi-square; null = labels shuffled between the two '
      f'elements\' texts within length-bin, {min(nperm, 400)}x; texts carrying both are dropped; texts >= 2 signs')
    dims = [lambda t: t['tf'], sitegrp, lambda t: t['head'], lambda t: t['suffix'], lambda t: t['numeral'], lambda t: t['quantity'],
            lambda t: t['midbin'], lambda t: t['fish_any'], lambda t: t['second_unit']]
    dimnames = ['type', 'site', 'head', 'suffix', 'numeral', 'quantity', 'midlen', 'fish', 'second']
    def chi(A, B, fn):
        ca = collections.Counter(fn(t) for t in A); cb = collections.Counter(fn(t) for t in B)
        na, nb = len(A), len(B); tot = 0.0
        for c in set(ca) | set(cb):
            e_a = (ca[c] + cb[c]) * na / (na + nb); e_b = (ca[c] + cb[c]) * nb / (na + nb)
            if e_a > 0: tot += (ca[c] - e_a) ** 2 / e_a
            if e_b > 0: tot += (cb[c] - e_b) ** 2 / e_b
        return tot
    def allchi(A, B): return [chi(A, B, fn) for fn in dims]
    groups_out = {}
    classes = [('openers', [e for e in els if e.startswith('op')]), ('markers', [e for e in els if e.startswith('W')]), ('fish', [e for e in els if e.startswith('F')])]
    for cname, members in classes:
        use = [e for e in members if sum(1 for t in T2 if e in t['els']) >= 15]
        pairs = {}
        for i in range(len(use)):
            for k in range(i + 1, len(use)):
                A = [t for t in T2 if use[i] in t['els'] and use[k] not in t['els']]; B = [t for t in T2 if use[k] in t['els'] and use[i] not in t['els']]
                if len(A) < 10 or len(B) < 10: continue
                pool = A + B; na = len(A); obs = allchi(A, B); tot = sum(obs)
                lb = [lenbin(t['n']) for t in pool]; groups = collections.defaultdict(list)
                for ix, l in enumerate(lb): groups[l].append(ix)
                lab = np.zeros(len(pool), int); lab[:na] = 1; nulls = []
                for p in range(min(nperm, 400)):
                    lab2 = lab.copy()
                    for gidx in groups.values():
                        gi = np.array(gidx); lab2[gi] = lab[gi][rng.permutation(len(gi))]
                    A3 = [pool[x] for x in range(len(pool)) if lab2[x] == 1]; B3 = [pool[x] for x in range(len(pool)) if lab2[x] == 0]
                    nulls.append(sum(allchi(A3, B3)))
                pl = (sum(1 for v in nulls if v >= tot) + 1) / (len(nulls) + 1)
                top = sorted(zip(dimnames, obs), key=lambda kv: -kv[1])[:3]
                pairs[(use[i], use[k])] = (tot, pl, top, len(A), len(B))
        npairs = len(pairs); bonf = 0.05 / max(npairs, 1); indist = []
        P(f'  -- {cname}: {len(use)} elements, {npairs} pairs, Bonferroni alpha {bonf:.4f}')
        for (a, b), (tot, pl, top, na, nb) in sorted(pairs.items(), key=lambda kv: kv[1][1]):
            tag = 'DISTINCT' if pl < bonf else ('weak' if pl < 0.05 else 'INDISTINGUISHABLE')
            if tag != 'DISTINCT': indist.append((a, b, pl))
            P(f'    {a} vs {b} [n {na}/{nb}]: chi {tot:.0f}, p_len {pl:.3f} -> {tag}; drivers ' + ', '.join(f'{d} {v:.0f}' for d, v in top))
        parent = {h: h for h in use}
        def find(x):
            while parent[x] != x: x = parent[x]
            return x
        for a, b, pl in indist: parent[find(a)] = find(b)
        comps = collections.defaultdict(list)
        for h in use: comps[find(h)].append(h)
        P(f'    functionally distinct {cname} (merging pairs not distinct after Bonferroni): {len(comps)} of {len(use)} tested: ' +
          ' '.join('{' + ', '.join(c) + '}' for c in comps.values()))
        groups_out[cname] = comps
    return groups_out

def write_profiles(T, els, label='Wells'):
    rows = []
    T2 = [t for t in T if t['n'] >= 2]
    for e in els:
        X = [t for t in T2 if e in t['els']]; n = len(X)
        if not n: continue
        def sh(fn, sub=X): return round(sum(1 for t in sub if fn(t)) / max(len(sub), 1), 3)
        def pen(fn):  # share of texts of that type carrying e
            Y = [t for t in T2 if fn(t)]; return round(sum(1 for t in Y if e in t['els']) / max(len(Y), 1), 3)
        seals = [t for t in X if t['tc'] == 'seal']; tabs = [t for t in X if t['tc'] == 'tablet']
        hc = collections.Counter(t['head'] for t in X)
        sz = [t['size'] for t in seals if t['size']]
        rows.append(dict(source=label, level=LV, element=e, n_texts=n,
                         seal=sh(lambda t: t['tc'] == 'seal'), tablet=sh(lambda t: t['tc'] == 'tablet'), sealing=sh(lambda t: t['tf'] == 'sealing'),
                         pot=sh(lambda t: t['tf'] == 'pot'), foreign=sh(lambda t: t['foreign']),
                         pen_seal=pen(lambda t: t['tc'] == 'seal'), pen_tablet=pen(lambda t: t['tc'] == 'tablet'), pen_sealing=pen(lambda t: t['tf'] == 'sealing'),
                         pen_pot=pen(lambda t: t['tf'] == 'pot'), pen_H_tab_mould=pen(lambda t: t['site'] == 'Harappa' and t['tf'] == 'tab_mould'),
                         pen_H_tab_inc=pen(lambda t: t['site'] == 'Harappa' and t['tf'] == 'tab_inc'),
                         pen_H_seal=pen(lambda t: t['site'] == 'Harappa' and t['tc'] == 'seal'), pen_MD_seal=pen(lambda t: t['site'] in ('Mohenjo-daro', 'Mohenjodaro') and t['tc'] == 'seal'),
                         seal_square=sh(lambda t: t['tf'] == 'seal_sq', seals), seal_bar=sh(lambda t: t['tf'] == 'seal_bar', seals),
                         tab_moulded=sh(lambda t: t['tf'] == 'tab_mould', tabs), tab_incised=sh(lambda t: t['tf'] == 'tab_inc', tabs),
                         MD=sh(lambda t: t['site'] in ('Mohenjo-daro', 'Mohenjodaro')), Harappa=sh(lambda t: t['site'] == 'Harappa'),
                         head_jar=sh(lambda t: t['head'] == 'jar'), head_mjar=sh(lambda t: t['head'] == 'mjar'), head_arrow=sh(lambda t: t['head'] == 'arrow'),
                         head_none=sh(lambda t: t['head'] in ('none', 'W400', 'W90/91')), head_top=' '.join(f'{k}x{v}' for k, v in hc.most_common(4)),
                         suffix_W400=sh(lambda t: t['suffix'] == 'W400'), numeral=sh(lambda t: t['numeral'] == 'yes'), quantity=sh(lambda t: t['quantity'] == 'yes'),
                         fish=sh(lambda t: t['fish_any'] == 'yes'), mjar_any=sh(lambda t: t['mjar_any'] == 'yes'), second_unit=sh(lambda t: t['second_unit'] == 'yes'),
                         len_mean=round(sum(t['n'] for t in X) / n, 2), midlen_mean=round(sum(t['midlen'] for t in X) / n, 2),
                         unicorn_share_seals=sh(lambda t: t['emblem'] == 'unicorn', [t for t in seals if t['emblem'] != 'unknown']),
                         seal_width_mm=round(sum(sz) / len(sz), 1) if sz else ''))
    return rows

def save_profiles(rows, label):
    fn = OUT + 'loop62_opener_profiles.csv'
    old = []
    if os.path.exists(fn):
        old = [r for r in csv.DictReader(open(fn)) if not (r['level'] == LV and r['source'] == label)]
    with open(fn, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in old + rows: w.writerow(r)

if __name__ == '__main__':
    if CY == 1:
        T = load_wells(LV); cycle1(T, ELEMENTS, NP)
        if LV == 'seq_raw':
            TI = load_im77(); P('\n\n########## IM77 replication (M numbers) ##########'); cycle1(TI, M_ELEMENTS, NP, 'IM77')
        open(OUT + f'loop62_c1_{LV}.txt', 'w').write('\n'.join(LOG))
    elif CY == 2:
        T = load_wells(LV, complete_only=True); cycle2(T, ELEMENTS, NP)
        if LV == 'seq_raw':
            TI = load_im77(); P('\n\n########## IM77 replication (M numbers) ##########'); cycle2(TI, M_ELEMENTS, NP, 'IM77')
        open(OUT + f'loop62_c2_{LV}.txt', 'w').write('\n'.join(LOG))
    elif CY == 3:
        T = load_wells(LV, complete_only=True); cycle3(T, ELEMENTS, NP)
        save_profiles(write_profiles(T, ELEMENTS), 'Wells')
        if LV == 'seq_raw':
            TI = load_im77(); P('\n\n########## IM77 replication (M numbers) ##########'); cycle3(TI, M_ELEMENTS, NP, 'IM77')
            save_profiles(write_profiles(TI, M_ELEMENTS, 'IM77'), 'IM77')
        open(OUT + f'loop62_c3_{LV}.txt', 'w').write('\n'.join(LOG))
