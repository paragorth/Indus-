"""Loop 70 shared code: do scribes' line breaks fall at word boundaries?

Data
  Wells: data/raw/inscriptions.csv, 'text' in object order; reversed = reading order; '/' = a line
    break on the object (S-DARK-38). The ORDER of the segments is not recorded (S-DARK-8.2), so every
    Wells statistic is computed (a) order-agnostic (all segment orders, every candidate junction pair
    'last of one segment + first of another' counted) and (b) in the canonical order (after-'/' segment
    first = the merged-corpus reading order, which agreed with IM77 line 1 -> 2 in 8 of 9 matched texts,
    S-DARK-8.2b).
  IM77: data/im77/im77_corpus_lines.csv; sides with lines 1, 2 (,3), each line in reading order; line
    order KNOWN. Sign 0 = illegible.
  One copy per site x object type x text (the same break on duplicate copies is not new evidence).
  Merge levels (Wells only): seq_raw / seq_strong / seq_all from sign_allographs_levels.json.
  Glyph widths: data/derived/dark/loop38_glyph_widths.json (lipi font, w_rel); IM77 signs via the
    inverse bridge; unknown widths = 0.52 (median).

Units (all learned or listed WITHOUT looking at any line break)
  QH     qualifier + head: sign directly before a closer (S289 paradigm) that belongs to that closer's
         60%-mass left-partner set (S303 rule), learned on single-line complete texts.
  OPEN   opener phrase: 817/861/820 + W2, 820/920/692 + W60, 60 + 741 (920-60-741) (S331, S334).
  NUM    numeral + item: stroke numeral (not W1/W2 markers) directly before a non-numeral sign.
  FISH   fish words / arrow phrase: adjacent pair inside {fish 220/240/235/233/231, 705/706, 33, 520}
         with at least one fish or the 705/706-33-520 chain (S296-S300).
  FROZEN data-driven frozen pairs: adjacent pair seen in >= 5 single-line texts with P(y|x) >= 0.4 and
         P(x|y) >= 0.4 (both directions loyal), learned on single-line texts.
  UNIT   union of the five.
  MIDMID adjacency of two middle signs (neither sign a frame sign: opener, connective, closer,
         numeral, suffix W400/W90, fish/arrow-phrase signs) and not inside any unit.
  BOUND  every other adjacency (unit edge, frame-to-middle junction).
"""
import csv, json, re, itertools, collections, math
import numpy as np

ROOT = '/home/user/Indus-'
OUT = ROOT + '/data/derived/dark/'
LEVELS = ['seq_raw', 'seq_strong', 'seq_all']
HOME = {'Mohenjo-daro', 'Harappa'}

OPENERS = {817, 861, 820, 920, 692}
CONNECT = {2, 60}
CLOSERS = {740, 520, 156, 527, 617, 226, 390, 405, 154, 158, 15, 254, 12, 151, 236, 700}
NUMERALS = {3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
MARKERS = {1, 2}
SUFFIX = {400, 90}
FISH = {220, 240, 235, 233, 231}
ARROWPH = {705, 706, 33, 520}
OPEN_PAIRS = {(o, 2) for o in (817, 861, 820)} | {(o, 60) for o in (820, 920, 692)} | {(60, 741)}
CATS = ['QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT', 'MIDMID', 'BOUND']


# ------------------------------------------------------------------ merge maps / bridge / widths
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
            while v in mp and n < 10:
                v = mp[v]; n += 1
            mp[k] = v
        return mp
    return {'seq_raw': {}, 'seq_strong': close(strong), 'seq_all': close(alll)}


def bridge(proposals=True):
    b = {int(k): list(v) for k, v in json.load(open(ROOT + '/data/derived/bridge_extended.json')).items()}
    if proposals:
        for p in json.load(open(OUT + 'bridge_proposals.json'))['proposals']:
            b.setdefault(int(p['W']), [])
            if p['M'] not in b[int(p['W'])]:
                b[int(p['W'])].append(int(p['M']))
    return b


def widths():
    g = json.load(open(OUT + 'loop38_glyph_widths.json'))['signs']
    return {int(k): v['w_rel'] for k, v in g.items()}


# ------------------------------------------------------------------ corpora
def wells(level='seq_raw'):
    """Returns (one_line, multi) where one_line = list of dict(site, type, seq) and multi = list of
    dict(site, type, segs=[tuple,...] each in reading order, canonical order = after-'/' first)."""
    mp = merge_maps()[level]
    one, multi = [], []
    seen1, seenm = set(), set()
    for r in csv.DictReader(open(ROOT + '/data/raw/inscriptions.csv')):
        t = (r['text'] or '').strip()
        if not t or '[' in t or ']' in t:
            continue
        core = t.strip('+ ')
        try:
            segs = [tuple(int(x) for x in s.split('-') if x.strip()) for s in core.split('/')]
        except ValueError:
            continue
        if any(0 in s for s in segs) or any(len(s) == 0 for s in segs):
            continue
        segs = [tuple(mp.get(x, x) for x in reversed(s)) for s in segs]   # reading order within segment
        segs = list(reversed(segs))                                          # canonical: after-'/' first
        sub = r['type'].split(':')[0]
        if len(segs) == 1:
            if r['complete'] != 'Y' or len(segs[0]) < 2:
                continue
            key = (r['site'], sub, segs[0])
            if key in seen1:
                continue
            seen1.add(key); one.append(dict(site=r['site'], type=sub, seq=segs[0]))
        else:
            key = (r['site'], sub, tuple(segs))
            if key in seenm:
                continue
            seenm.add(key)
            multi.append(dict(site=r['site'], type=sub, segs=[tuple(s) for s in segs], cisi=r['cisi'],
                              symbol=r['symbol'], shape=r['shape']))
    return one, multi


def im77():
    rows = list(csv.DictReader(open(ROOT + '/data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['text_no'], r['side'])].append(r)
    one, multi = [], []
    seen1, seenm = set(), set()
    for (tno, side), v in by.items():
        if any(x['line'] == '9' for x in v):
            continue
        v = sorted(v, key=lambda x: x['line'])
        seqs = [tuple(int(s) for s in x['signs_clean'].split()) for x in v]
        if any(0 in s for s in seqs) or any(len(s) == 0 for s in seqs):
            continue
        site = v[0]['site']; typ = v[0]['object_type']
        if len(v) == 1 and v[0]['line'] == '0':
            if len(seqs[0]) < 2:
                continue
            key = (site, typ, seqs[0])
            if key in seen1:
                continue
            seen1.add(key); one.append(dict(site=site, type=typ, seq=seqs[0]))
        elif [x['line'] for x in v] in (['1', '2'], ['1', '2', '3']):
            key = (site, typ, tuple(seqs))
            if key in seenm:
                continue
            seenm.add(key)
            multi.append(dict(site=site, type=typ, segs=seqs, text_no=tno, side=side,
                              dirs=[x['direction'] for x in v]))
    return one, multi


def is_home(site):
    return site in HOME or site.startswith(('Mohenjo', 'Harappa'))


# ------------------------------------------------------------------ unit sets
def unit_sets(one, sign_map=None):
    """sign_map: function W -> set of signs in this corpus (identity for Wells, bridge for IM77)."""
    if sign_map is None:
        sign_map = lambda w: {w}
    def S(ws):
        out = set()
        for w in ws:
            out |= set(sign_map(w))
        return out
    closers, nums, fish, arrowph = S(CLOSERS), S(NUMERALS), S(FISH), S(ARROWPH)
    openers, connect, suffix, markers = S(OPENERS), S(CONNECT), S(SUFFIX), S(MARKERS)
    open_pairs = set()
    for a, b in OPEN_PAIRS:
        for x in sign_map(a):
            for y in sign_map(b):
                open_pairs.add((x, y))
    # QH: 60%-mass left partner set of each closer, closer text-final
    left = collections.defaultdict(collections.Counter)
    for t in one:
        s = t['seq']
        if len(s) >= 2 and s[-1] in closers:
            left[s[-1]][s[-2]] += 1
    qh = set()
    for h, c in left.items():
        tot = sum(c.values()); acc = 0
        for q, n in c.most_common():
            if acc >= 0.6 * tot:
                break
            qh.add((q, h)); acc += n
    # NUM
    num = set()
    big = collections.Counter()
    for t in one:
        s = t['seq']
        for a, b in zip(s, s[1:]):
            big[(a, b)] += 1
    for (a, b) in big:
        if a in nums and b not in nums and b not in (markers - nums):
            num.add((a, b))
    # FISH
    fishp = set()
    for (a, b) in big:
        if a in fish | arrowph and b in fish | arrowph and (a in fish or b in fish):
            fishp.add((a, b))
    for a in S({705, 706}):
        for b in S({33}):
            fishp.add((a, b))
    for a in S({33}):
        for b in S({520}):
            fishp.add((a, b))
    # FROZEN
    texts_with = collections.Counter(); lc = collections.Counter(); rc = collections.Counter()
    for t in one:
        s = t['seq']
        for p in set(zip(s, s[1:])):
            texts_with[p] += 1
        for a, b in zip(s, s[1:]):
            lc[a] += 1; rc[b] += 1
    frozen = set()
    for p, n in texts_with.items():
        a, b = p
        if n >= 5 and big[p] / lc[a] >= 0.4 and big[p] / rc[b] >= 0.4:
            frozen.add(p)
    frame = openers | connect | closers | nums | suffix | fish | arrowph | markers
    sets = dict(QH=qh, OPEN=open_pairs, NUM=num, FISH=fishp, FROZEN=frozen)
    sets['UNIT'] = qh | open_pairs | num | fishp | frozen
    return sets, frame


def pair_cat(p, sets, frame):
    """Categories an adjacency belongs to (list)."""
    out = [c for c in ('QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT') if p in sets[c]]
    if 'UNIT' not in out:
        if p[0] not in frame and p[1] not in frame:
            out.append('MIDMID')
        else:
            out.append('BOUND')
    return out


# ------------------------------------------------------------------ break enumeration
def cut_configs(T, k):
    """All ways to cut sequence T into k non-empty pieces (list of piece lists)."""
    n = len(T)
    for cuts in itertools.combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        yield [T[b[i]:b[i + 1]] for i in range(k)]


def junctions_known(pieces):
    return [(pieces[i][-1], pieces[i + 1][0]) for i in range(len(pieces) - 1)]


def junctions_agnostic(pieces):
    return [(pieces[i][-1], pieces[j][0]) for i in range(len(pieces)) for j in range(len(pieces)) if i != j]


def ink(seq, W):
    return sum(W.get(s, 0.52) for s in seq)


def imbalance(pieces, W):
    v = [ink(p, W) for p in pieces]
    return max(v) - min(v)


def null_configs(segs, mode, W):
    """For one multi-line text: the observed junction-pair list, and three null lists of (weight, junction
    list) under the uniform, ink-midpoint and ink-balance-matched nulls. mode = 'known' | 'canon' |
    'agnostic'. 'known'/'canon' use the given segment order; 'agnostic' averages over all orders."""
    k = len(segs)
    J = junctions_agnostic if mode == 'agnostic' else junctions_known
    obs = J(segs)
    obs_imb = imbalance(segs, W)
    orders = list(itertools.permutations(range(k))) if mode == 'agnostic' else [tuple(range(k))]
    uni, mid, bal = [], [], []
    for o in orders:
        T = sum((tuple(segs[i]) for i in o), ())
        confs = list(cut_configs(T, k))
        imbs = [imbalance(c, W) for c in confs]
        m = min(imbs)
        midc = [c for c, x in zip(confs, imbs) if x <= m + 1e-9]
        balc = [c for c, x in zip(confs, imbs) if x <= obs_imb + 1e-9]
        if not balc:
            balc = midc
        wo = 1.0 / len(orders)
        uni += [(wo / len(confs), J(c)) for c in confs]
        mid += [(wo / len(midc), J(c)) for c in midc]
        bal += [(wo / len(balc), J(c)) for c in balc]
    return obs, dict(uniform=uni, midpoint=mid, balance=bal)


def poisbin_p_low(ps, obs):
    """P(X <= obs) for X = sum of independent Bernoulli(ps)."""
    dist = np.zeros(len(ps) + 1); dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[:int(obs) + 1].sum()), float(dist[int(obs):].sum())


def category_test(texts, mode, sets, frame, W, cats=CATS):
    """O = texts with >= 1 junction pair in the category; E under each null; P_low / P_high."""
    res = {}
    pre = [null_configs(t['segs'], mode, W) for t in texts]
    for c in cats:
        inside = lambda js: any(c in pair_cat(p, sets, frame) for p in js)
        O = sum(1 for obs, _ in pre if inside(obs))
        row = dict(O=O, n=len(texts))
        for nm in ('uniform', 'midpoint', 'balance'):
            ps = [sum(w for w, js in nl[nm] if inside(js)) for _, nl in pre]
            E = sum(ps)
            pl, ph = poisbin_p_low(ps, O)
            row[nm] = dict(E=E, OE=(O / E if E > 0 else float('nan')), p_low=pl, p_high=ph)
        res[c] = row
    return res


def fmt_row(label, r):
    s = f"{label}: O {r['O']}/{r['n']}"
    for nm in ('uniform', 'midpoint', 'balance'):
        x = r[nm]
        s += f" | {nm[:3]} E {x['E']:.1f} O/E {x['OE']:.2f} pL {x['p_low']:.3f} pH {x['p_high']:.3f}"
    return s
