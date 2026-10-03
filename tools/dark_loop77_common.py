"""Loop 77 common: THE PROTO-ELAMITE TABLET SIGNATURE IN INDUS.
S-DARK-73.2 found PE middles on one tablet share elements 3x beyond chance (Linear B names on a tablet 1.0x).
Here the identical statistic (pair_stats copied verbatim from tools/dark_loop73_c2.py) is run on Indus analogues of
'entries on one tablet':
  (a) faces of one multi-sided object (inscriptions.csv id n.k; moulded identical face-sets collapsed, S-DARK-37),
  (b) Harappa (and Mohenjo-daro) tablets found together in one find-group (area x block x room/grid),
  (c) Lothal: dies impressed on one sealing; dies of the warehouse deposit (loop71 catalogue).
Element sets per Indus text (S310/S331 parser of tools/dark_loop56.py, QUAL = S303 qualifiers learnt per level):
  FULL  all signs of the text (formula included)
  MID   NAME + COUNT tokens (the loop73 Indus 'middle')
  DES   designation only: NAME tokens minus every frame-inventory sign (openers, markers, marked jars, suffixes, closers,
        fish, numerals, every S303 qualifier of any closer) and minus the sign directly before the closer.
Nulls: (1) designations permuted across groups within stratum (site x object type x designation length; PE: numeral
system as in 73.2, and length), N x; (2) Markov-2 generator trained on the pooled designations of the analysis
(same lengths, same group structure), N x. Element frequency classes rare < 5 / mid 5-19 / freq >= 20 over the pool.
"""
import sys, os, json, csv, re, collections, math, random
ROOT = '/home/user/Indus-/'
sys.path.insert(0, ROOT + 'tools')
os.chdir(ROOT)
DARK = ROOT + 'data/derived/dark/'
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ---------------- statistic: verbatim from tools/dark_loop73_c2.py ----------------
def pair_stats(groups, fcls):
    """groups: list of lists of (pos, mid). returns counts."""
    tot = sh = 0; rare = mid = freq = 0; ident = 0; adj_t = adj_s = 0; nadj_t = nadj_s = 0
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = g[i][1]; b = g[j][1]
                if a == b: ident += 1; continue
                tot += 1; S = set(a) & set(b)
                isadj = abs(g[i][0] - g[j][0]) == 1
                if isadj: adj_t += 1
                else: nadj_t += 1
                if S:
                    sh += 1
                    if isadj: adj_s += 1
                    else: nadj_s += 1
                    cl = {fcls[x] for x in S}
                    if 'rare' in cl: rare += 1
                    if 'mid' in cl: mid += 1
                    if 'freq' in cl: freq += 1
    return dict(pairs=tot, share=sh / tot if tot else float('nan'), rare=rare / tot if tot else 0, mid=mid / tot if tot else 0,
                freq=freq / tot if tot else 0, ident=ident, adj=adj_s / adj_t if adj_t else float('nan'),
                nadj=nadj_s / nadj_t if nadj_t else float('nan'), adj_n=adj_t)

KEYS = ['share', 'rare', 'mid', 'freq', 'adj', 'nadj']

def markov2(items, r):
    """items: list of designations (tuples). returns generated designations, same lengths, order-2 Markov chain
    (back-off to order 1, then unigram) trained on items."""
    t2 = collections.defaultdict(list); t1 = collections.defaultdict(list); st = []; uni = []
    for m in items:
        if not m: continue
        st.append(m[0]); uni += list(m)
        for k in range(1, len(m)):
            t1[m[k - 1]].append(m[k])
            if k >= 2: t2[(m[k - 2], m[k - 1])].append(m[k])
    out = []
    for m in items:
        if not m: out.append(m); continue
        s = [r.choice(st)]
        while len(s) < len(m):
            c = t2.get(tuple(s[-2:])) if len(s) >= 2 else None
            if not c: c = t1.get(s[-1])
            s.append(r.choice(c) if c else r.choice(uni))
        out.append(tuple(s))
    return out

def within(label, items, nperm=1000, seed=77, markov=True, quiet=False):
    """items: list of (group, pos, des, stratum). Same statistic as S-DARK-73.2 'within'; nulls: stratified permutation
    across groups, and Markov-2 regeneration. Returns dict."""
    items = [x for x in items if len(x[2]) >= 1]
    cnt = collections.Counter(x for _, _, m, _ in items for x in set(m))
    fcls = collections.defaultdict(lambda: 'rare')
    fcls.update({x: 'rare' if c < 5 else 'mid' if c < 20 else 'freq' for x, c in cnt.items()})
    def groups(its):
        g = collections.defaultdict(list)
        for d, p, m, s in its: g[d].append((p, m))
        return [v for v in g.values() if len(v) >= 2]
    G = groups(items)
    obs = pair_stats(G, fcls)
    if obs['pairs'] == 0 and obs['ident'] == 0:
        if not quiet: P(f'  [{label}] no groups with >= 2 designations')
        return None
    r = random.Random(seed); byS = collections.defaultdict(list)
    for i, (d, p, m, s) in enumerate(items): byS[s].append(i)
    nulls = []; mk = []
    mids0 = [m for _, _, m, _ in items]
    for _ in range(nperm):
        mids = list(mids0)
        for s, idx in byS.items():
            vals = [mids[i] for i in idx]; r.shuffle(vals)
            for i, v in zip(idx, vals): mids[i] = v
        nulls.append(pair_stats(groups([(d, p, mm, s) for (d, p, _, s), mm in zip(items, mids)]), fcls))
    if markov:
        for _ in range(nperm):
            g = markov2(mids0, r)
            mk.append(pair_stats(groups([(d, p, mm, s) for (d, p, _, s), mm in zip(items, g)]), fcls))
    out = dict(obs=obs, n_groups=len(G), n_items=sum(len(v) for v in G))
    line = f'  [{label}] groups {len(G)}, designations {out["n_items"]}, non-identical pairs {obs["pairs"]}, identical {obs["ident"]} (perm {sum(x["ident"] for x in nulls)/nperm:.1f}'
    if markov: line += f', M2 {sum(x["ident"] for x in mk)/nperm:.1f}'
    line += '); '
    for k in KEYS:
        rec = {}
        for nm, NL in (('perm', nulls), ('m2', mk)):
            if not NL: continue
            v = [x[k] for x in NL if not (isinstance(x[k], float) and math.isnan(x[k]))]
            if not v or (isinstance(obs[k], float) and math.isnan(obs[k])): rec[nm] = None; continue
            mu = sum(v) / len(v); p = (1 + sum(1 for x in v if x >= obs[k])) / (1 + len(v))
            rec[nm] = dict(null=mu, ratio=obs[k] / mu if mu else float('nan'), p=p)
        out[k] = dict(obs=obs[k], **rec)
        if k in ('share', 'rare', 'adj') and rec.get('perm'):
            s = f'{k} {obs[k]:.3f} vs perm {rec["perm"]["null"]:.3f} (x{rec["perm"]["ratio"]:.2f}, P {rec["perm"]["p"]:.3f})'
            if rec.get('m2'): s += f' / M2 {rec["m2"]["null"]:.3f} (x{rec["m2"]["ratio"]:.2f}, P {rec["m2"]["p"]:.3f})'
            line += s + '; '
    vi = [x['ident'] for x in nulls]
    out['ident'] = dict(obs=obs['ident'], perm=sum(vi) / len(vi), p=(1 + sum(1 for x in vi if x >= obs['ident'])) / (1 + len(vi)))
    if not quiet: P(line + f'rare-pair n {round(obs["rare"]*obs["pairs"])}, adjacent pairs {obs["adj_n"]}')
    out['fcls'] = None
    return out

def shared_elements(items):
    """which elements are shared by non-identical pairs inside a group; also how often each sits in each position class."""
    g = collections.defaultdict(list)
    for d, p, m, s in items: g[d].append(m)
    c = collections.Counter()
    for L in g.values():
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if L[i] != L[j]:
                    for x in set(L[i]) & set(L[j]): c[x] += 1
    return c

# ---------------- PE (loop73 entries) ----------------
def pe_items():
    import dark_loop73_common as L73
    E = L73.pe_entries()
    a = []; b = []
    for i, e in enumerate(E):
        a.append((e['tablet'], i, e['mid'], e['system'] or 'none'))
        b.append((e['tablet'], i, e['mid'], min(len(e['mid']), 5)))
    return a, b

# ---------------- Indus: parser ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
def build_qual(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q
def parse(s, QUAL):
    """identical to tools/dark_loop56.py parse"""
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab
def frame_inventory(QUAL):
    F = set(OPEN) | MARK | MJAR | SUF | set(CL) | FISH | NUM
    for q in QUAL.values(): F |= q
    return F
def element_sets(seq, QUAL, FR):
    if not seq: return dict(FULL=(), MID=(), DES=())
    lab = parse(seq, QUAL)
    mid = tuple(a for a, l in zip(seq, lab) if l in ('NAME', 'COUNT'))
    ci = next((k for k, l in enumerate(lab) if l == 'CLOSER'), None)
    des = tuple(a for k, (a, l) in enumerate(zip(seq, lab)) if l == 'NAME' and a not in FR and not (ci is not None and k == ci - 1))
    return dict(FULL=tuple(seq), MID=mid, DES=des, lab=lab)

# ---------------- Indus: faces from inscriptions.csv (loop37 loader) ----------------
RAW = ROOT + 'data/raw/inscriptions.csv'
def merge_maps():
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    ms = collections.defaultdict(collections.Counter); ma = collections.defaultdict(collections.Counter)
    for r in C:
        if len(r['seq_raw']) == len(r['seq_strong']) == len(r['seq_all']):
            for a, b, c in zip(r['seq_raw'], r['seq_strong'], r['seq_all']): ms[a][b] += 1; ma[a][c] += 1
    return {'seq_raw': {}, 'seq_strong': {a: c.most_common(1)[0][0] for a, c in ms.items()},
            'seq_all': {a: c.most_common(1)[0][0] for a, c in ma.items()}}
def bad(v): return v is None or v.strip() in ('-', '--', '- -', '')
def otype(t):
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing', 'BNGL': 'bangle'}.get(t.split(':')[0], 'other')
def load_objects(level):
    """oid -> object with faces (reading order, 000/999 dropped), find-spot fields, and element sets per face."""
    mp = merge_maps()[level]
    objs = collections.OrderedDict()
    for r in csv.DictReader(open(RAW)):
        oid, _, k = r['id'].partition('.')
        toks = [int(t) for t in re.findall(r'\d{3}', r['text'])][::-1]
        seq = [mp.get(t, t) for t in toks if t not in (0, 999)]
        area = None if bad(r['area-section']) else r['area-section'].strip()
        o = objs.setdefault(oid, dict(oid=oid, cisi=r['cisi'], site=r['site'], type=r['type'], ot=otype(r['type']),
                                      area=area, block=None if bad(r['block-house']) else r['block-house'].strip(),
                                      room=None if bad(r['room-grid']) else r['room-grid'].strip(),
                                      exc=r['excavation-idno'].strip(), period=r['period'].strip(), faces=[]))
        o['faces'].append(dict(k=int(k or 1), seq=seq, complete=r['complete'] == 'Y', has0=0 in toks))
    for o in objs.values(): o['faces'].sort(key=lambda f: f['k'])
    QUAL = build_qual([f['seq'] for o in objs.values() for f in o['faces'] if f['seq']])
    FR = frame_inventory(QUAL)
    for o in objs.values():
        for f in o['faces']: f.update(element_sets(f['seq'], QUAL, FR))
    return objs, QUAL, FR
def collapse(objs):
    """S-DARK-13 / 37: one die per identical set of face texts for moulded / impressed types, within site."""
    seen = set(); out = []
    for o in objs:
        if o['type'].startswith('TAG') or o['type'] in ('TAB:B', 'TAB:C', 'MDLN'):
            key = (o['site'], o['type'], tuple(sorted(tuple(f['seq']) for f in o['faces'])))
            if key in seen: continue
            seen.add(key)
        out.append(o)
    return out

# ---------------- IM77 (bridged to Wells space as in tools/dark_loop56.py im77_objects) ----------------
def im77_sides():
    BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    PROP = json.load(open(ROOT + 'data/derived/dark/bridge_proposals.json'))
    M2W = {}; prop = set()
    for w, ms in BR.items():
        for m in ms: M2W.setdefault(m, []).append(int(w))
    for p in PROP['proposals']:
        if p['M'] not in M2W: prop.add(p['M'])
        M2W.setdefault(p['M'], []).append(p['W'])
    for m in M2W: M2W[m] = min(M2W[m])
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[(r['text_no'], r['side'])].append(r)
    T = collections.defaultdict(list); nprop = 0; tot = 0
    for (tn, sd), rs in by.items():
        rs = sorted(rs, key=lambda r: int(r['line']))
        ms = [int(x) for r in rs for x in r['signs_clean'].split() if x.strip() and x != '0']
        seq = []
        for m in ms:
            tot += 1
            if m in prop: nprop += 1
            seq.append(M2W.get(m, 10000 + m))
        site = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa'}.get(rs[0]['site'], rs[0]['site'])
        T[tn].append(dict(side=int(sd), seq=seq, site=site, ot=rs[0]['object_type'], locus=rs[0]['locus'].strip(),
                          level=rs[0]['level'].strip()))
    QUAL = build_qual([s['seq'] for L in T.values() for s in L if s['seq']])
    FR = frame_inventory(QUAL)
    for L in T.values():
        for s in L: s.update(element_sets(s['seq'], QUAL, FR))
    return T, nprop / tot

def save(name, obj):
    json.dump(obj, open(DARK + name + '.json', 'w'), indent=1, default=str)
    open(DARK + name + '_log.txt', 'w').write('\n'.join(LOG) + '\n')
