"""S-DARK-76 common: HOW OFTEN DOES ONE DESIGNATION RECUR, AND WHERE?  Data layer.

A DOCUMENT is one physical object (one act). Its designations = the set of distinct S-DARK-26 middles
(NAME + COUNT residue of the S310/S331 frame parser, tools/dark_loop56.py) of its faces, so repeated impressions of one
seal on one sealing, and the two faces of one object carrying the same text, count once (S-DARK-13 die collapse:
identical faces, 000 wildcard, fragment inside a fuller face of the same object).
Object kinds: seal (SEAL), sealing (TAG), tablet_m (moulded TAB:B / TAB:C), tablet_i (incised TAB:I), pot (POT), other.
Two document levels:
  'act'  every object is a document (an impression on a sealing = one act; a moulded tablet = one copy)
  'die'  sealings and moulded tablets with the identical full text set at one site are ONE document (one seal / one
         mould); seals, incised tablets and pots are never collapsed (each is its own instrument / act).
Count faces (numerals + W700 only, the Harappa voucher count side) carry no designation and are skipped.
Face filters: 'strict' = complete faces only (Wells complete == 'Y'); 'loose' = every face, lost signs dropped.
Sources: data/raw/inscriptions.csv (5,680 rows, the current build; merges from the canonical file's maps, tools/dark_loop37),
IM77 (data/im77, M numbers bridged to Wells via bridge_extended + S-DARK-27 proposals, unbridged kept opaque).
"""
import sys, os, json, csv, re, collections, math, random
ROOT = '/home/user/Indus-/'
sys.path.insert(0, ROOT + 'tools'); os.chdir(ROOT)
from dark_loop37 import load_faces, merge_maps, RAW
_a = sys.argv; sys.argv = ['x', '0']
import dark_loop56 as D56
sys.argv = _a
from dark_loop65 import collapse_dies, head_of
DARK = ROOT + 'data/derived/dark/'
OPEN = D56.OPEN; MARK = D56.MARK
COUNTSET = set(D56.NUM) | {700}   # numeral(s) + W700 = a count face (voucher), not a designation

def kind_of(t):
    t0 = t.split(':')[0]
    if t0 == 'SEAL': return 'seal'
    if t0 == 'TAG': return 'sealing'
    if t0 == 'TAB': return 'tablet_m' if t in ('TAB:B', 'TAB:C') else 'tablet_i'
    if t0 == 'POT': return 'pot'
    return 'other'

def _parse_all(seqs):
    Q = D56.build_qual([dict(seq=list(s)) for s in seqs if s])
    return lambda s: D56.parse(list(s), Q) if s else []

def face_info(seq, parse):
    lab = parse(seq)
    mid = tuple(a for a, l in zip(seq, lab) if l in ('NAME', 'COUNT'))
    pre = tuple(a for a, l in zip(seq, lab) if l in ('OPENER', 'MARKER'))
    head = tuple(a for a, l in zip(seq, lab) if l in ('TITLE', 'CLOSER', 'SUFFIX'))
    return dict(seq=tuple(seq), mid=mid, pre=pre, head=head, hgroup=head_of(seq), opener=bool(seq) and seq[0] in OPEN,
                count_face=bool(seq) and all(a in COUNTSET for a in seq))

def wells_docs(level='seq_raw', filt='strict'):
    objs = load_faces(level)
    mp = merge_maps()[level]
    raw = {}
    for r in csv.DictReader(open(RAW)):
        toks = [int(t) for t in re.findall(r'\d{3}', r['text'])][::-1]
        raw[r['id']] = ([mp.get(t, t) if t not in (0, 999) else 0 for t in toks], r)
    allseq = [f['seq'] for o in objs.values() for f in o['faces'] if f['seq']]
    parse = _parse_all(allseq)
    docs = []
    for oid, o in objs.items():
        for f in o['faces']:
            f['toks'], rr = raw.get(f'{oid}.{f["k"]}', raw.get(oid, ([], None)))
        r0 = raw.get(f'{oid}.1', raw.get(oid, ([], None)))[1] or {}
        dies = collapse_dies(o['faces'])
        faces = []
        for d in dies:
            leg = [f for f in d if f['seq'] and (f['complete'] or filt == 'loose')]
            if not leg: continue
            best = max(leg, key=lambda f: (f['complete'], len(f['seq'])))
            fi = face_info(best['seq'], parse); fi['n_imp'] = len(d); fi['complete'] = best['complete']
            faces.append(fi)
        if not faces: continue
        docs.append(dict(oid=oid, cisi=o['cisi'], site=o['site'], type=o['type'], kind=kind_of(o['type']), faces=faces,
                         area=r0.get('area-section', '-'), block=r0.get('block-house', '-'), period=r0.get('period', '-'),
                         big=o['site'] in ('Mohenjo-daro', 'Harappa')))
    return docs

def im77_docs(filt='strict'):
    BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    PROP = json.load(open(DARK + 'bridge_proposals.json'))
    M2W = {}
    for w, ms in BR.items():
        for m in ms: M2W.setdefault(m, []).append(int(w))
    for p in PROP['proposals']: M2W.setdefault(p['M'], []).append(p['W'])
    M2W = {m: min(v) for m, v in M2W.items()}
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[r['text_no']].append(r)
    KIND = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet_m', 'copper tablet': 'tablet_c', 'pottery graffito': 'pot'}
    sides_all = []; tmp = []
    for tn, rs in by.items():
        sides = collections.OrderedDict()
        for r in sorted(rs, key=lambda r: (int(r['side']), int(r['line']))):
            ms = [int(x) for x in r['signs_clean'].split() if x.strip()]
            if r['line'] == '9': continue
            sides.setdefault(r['side'], []).extend(ms)
        fl = []
        for sd, ms in sides.items():
            lost = 0 in ms
            if lost and filt == 'strict': continue
            seq = [M2W.get(m, 10000 + m) for m in ms if m != 0]
            if seq: fl.append(seq); sides_all.append(seq)
        if fl:
            site = {'Mohenjodaro': 'Mohenjo-daro'}.get(rs[0]['site'], rs[0]['site'])
            tmp.append((tn, site, KIND.get(rs[0]['object_type'], 'other'), fl, rs[0]))
    parse = _parse_all(sides_all)
    docs = []
    for tn, site, kind, fl, r0 in tmp:
        uniq = []
        for s in fl:
            if tuple(s) not in [u for u in uniq]: uniq.append(tuple(s))
        faces = [face_info(list(s), parse) for s in uniq]
        for f in faces: f['n_imp'] = 1; f['complete'] = True
        docs.append(dict(oid='IM' + tn, cisi='IM' + tn, site=site, type=r0['object_type'], kind=kind, faces=faces,
                         area=r0.get('locus', '-'), block='-', period=r0.get('level', '-'), big=site in ('Mohenjo-daro', 'Harappa')))
    return docs

def collapse_level(docs, lev):
    """'act' = docs as they are; 'die' = sealings and moulded tablets with an identical full-text set at one site merged."""
    if lev == 'act': return docs
    out = []; seen = {}
    for d in docs:
        if d['kind'] in ('sealing', 'tablet_m'):
            key = (d['site'], d['kind'], tuple(sorted(f['seq'] for f in d['faces'])))
            if key in seen: seen[key]['ncopies'] += 1; continue
            d = dict(d, ncopies=1); seen[key] = d
        out.append(d)
    return out

def tokens(docs, minel=2, kinds=None):
    """(doc index, designation, face) tokens; one per distinct designation per document."""
    out = []
    for i, d in enumerate(docs):
        if kinds and d['kind'] not in kinds: continue
        seen = set()
        for f in d['faces']:
            m = f['mid']
            if len(m) >= minel and m not in seen and not f['count_face']:
                seen.add(m); out.append((i, m, f))
    return out

# ---------------- comparators ----------------
C76 = DARK + 'loop76_corpora/'
UR3_TITLE = {'dumu', 'dub-sar', 'lugal', 'arad2', 'arad2-zu', 'lu2', 'ensi2', 'ugula', 'nu-banda3', 'dam-gar3', 'sanga', 'sukkal',
             'szabra', 'agrig', 'sipa', 'kiszib3', 'dam', 'szesz', 'nin', 'ama', 'mu', 'nita', 'kal-ga', 'lugal-kal-ga', 'nita-kal-ga'}
def _okur(w): return w and w not in UR3_TITLE and not w.startswith('{d}szu-{d}suen') and 'x' not in w.split('-') and not w.startswith('_') and '$' not in w
def ur3_sealings():
    """(doc = tablet pid, designation = seal owner name, frame = rest of the legend, site); one per tablet x legend."""
    out = []; seen = set()
    for l in open(C76 + 'ur3_seal_impr.jsonl'):
        r = json.loads(l)
        if not _okur(r['owner']): continue
        k = (r['pid'], tuple(r['legend']))
        if k in seen: continue
        seen.add(k); out.append(dict(doc=r['pid'], des=r['owner'], frame=tuple(r['legend'][1:]), site=r['site'], text=tuple(r['legend'])))
    return out
def ur3_seals():
    """one document per DISTINCT legend (= one seal) per site; designation = owner name."""
    seen = set(); out = []
    for x in ur3_sealings():
        k = (x['site'], x['text'])
        if k in seen: continue
        seen.add(k); out.append(dict(x, doc='S' + str(len(out))))
    return out
def ur3_admin():
    out = []; seen = set()
    for l in open(C76 + 'ur3_admin_names.jsonl'):
        r = json.loads(l)
        if not _okur(r['name']): continue
        k = (r['pid'], r['name'])
        if k in seen: continue
        seen.add(k); out.append(dict(doc=r['pid'], des=r['name'], frame=(r['slot'],), site=r['site'], text=(r['slot'], r['name'])))
    return out
def linb_personnel():
    """DAMOS personnel-series names with tablet (tools/dark_loop73_common.linb_names_tab rules); frame = series."""
    import dark_loop73_common as L73
    out = []; seen = set()
    for x in L73.linb_names_tab():
        k = (x['tablet'], x['name'])
        if k in seen: continue
        seen.add(k); out.append(dict(doc=x['tablet'], des=x['name'], frame=(x['series'],), site=x['site'], text=(x['series'],) + x['name']))
    return out

# ---------------- statistics ----------------
def recur_share(toks):
    """toks: list of (doc, des). share of tokens whose designation occurs in another document."""
    docs = collections.defaultdict(set)
    for d, m in toks: docs[m].add(d)
    return sum(1 for d, m in toks if len(docs[m]) >= 2) / len(toks) if toks else float('nan')
def profile(toks):
    """per-designation document counts: types, tokens, mean docs per type, share of types on >= 2 docs, max."""
    docs = collections.defaultdict(set)
    for d, m in toks: docs[m].add(d)
    c = [len(v) for v in docs.values()]
    return dict(tokens=len(toks), types=len(c), mean=sum(c) / len(c) if c else float('nan'),
                share2=sum(1 for v in c if v >= 2) / len(c) if c else float('nan'), max=max(c) if c else 0,
                recur=recur_share(toks))
def yule_rho(counts):
    """MLE of the Yule-Simon rho on frequency counts (>= 1); small rho = heavy tail (strongly recurring items)."""
    from scipy.special import betaln
    from scipy.optimize import minimize_scalar
    x = [c for c in counts if c >= 1]
    if len(x) < 5: return float('nan')
    def nll(r): return -sum(math.log(r) + betaln(k, r + 1) for k in x)
    return minimize_scalar(nll, bounds=(0.05, 50), method='bounded').x
def zipf_slope(counts): return D56.zipf(counts)
def q(v, p):
    v = sorted(x for x in v if not (isinstance(x, float) and math.isnan(x)))
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')
def sub(toks, n, r):
    return r.sample(toks, n) if len(toks) > n else list(toks)

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def save(name):
    open(DARK + name + '.txt', 'w').write('\n'.join(LOG) + '\n')
