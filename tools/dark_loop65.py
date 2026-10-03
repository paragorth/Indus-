"""S-DARK-65: sealings and tags that carry TWO OR MORE DIFFERENT seal impressions = two parties on one act?
Shared loader for the loop-65 cycles. Builds on the loop-37 face loader (id n.k = recorded face/impression of one
object in data/raw/inscriptions.csv, signs reversed to reading order, 000 dropped; seq_raw / seq_strong / seq_all
from the S268 merge maps) and the loop-60 head parser (closer paradigm S289).
An IMPRESSION here is one recorded face of a TAG object. Faces are collapsed to one DIE when their raw tokens agree
position by position with 000 as a wildcard, or when one (a fragment) is a substring of the other (S-DARK-13).
Faces with no legible sign (all 000) are counted as 'illegible impressions' but take part in no test.
Usage: python3 tools/dark_loop65_c<N>.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, csv, json, re, collections, random, itertools, math
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import (ROOT, RAW, CANON, BRIDGE, IM77, merge_maps, otype, load_faces, learn_qual, make_parser,
                         OPEN, MARK, SUF, CL, FISH, NUM, pval, Mname, load_im77)

LOTHAL = ROOT + '/data/derived/lothal-sealings-frenez-tosi2005.json'
OUT = ROOT + '/data/derived/dark/'

# ---- heads (loop 60, S289 paradigm) ----
HEAD_GROUP = {740: 'jar', 741: 'mjar', 742: 'mjar', 745: 'mjar', 520: 'arrow', 151: 'W151', 156: 'W156', 154: 'W154/158',
              158: 'W154/158', 527: 'box527', 526: 'box527', 226: 'W226', 617: 'W615/617', 615: 'W615/617', 236: 'W236',
              700: 'W700', 595: 'W595'}
SUF_MAP = {400: 'W400', 90: 'W90/91', 91: 'W90/91'}
M_HEAD = {342: 'jar', 343: 'mjar', 344: 'mjar', 345: 'mjar', 211: 'arrow', 12: 'W151', 15: 'W156/154', 254: 'box527',
          60: 'W226', 245: 'W615/617', 66: 'W236', 328: 'W700', 252: 'W595'}
M_SUF = {176: 'W400', 1: 'W90/91'}
M_OPEN = {267, 391, 293, 150}; M_MARK = {99, 100, 123}
M_NUM = set(range(87, 97)) | set(range(101, 121))
M_CL = list(M_HEAD)

def head_of(seq, HG=HEAD_GROUP, SM=SUF_MAP, OP=OPEN):
    """closer-slot head of a text in reading order; 'open-final' when an opener sign stands last (S286)."""
    s = list(seq)
    while len(s) > 1 and s[-1] in SM: s.pop()
    if not s: return 'none'
    if s[-1] in HG: return HG[s[-1]]
    if s[-1] in OP: return 'open-final'
    return 'none'

def lothal_backs():
    d = json.load(open(LOTHAL))
    return {r[0]: dict(nimp=r[1], back=r[2], ctx=r[3]) for r in d['rows']}

# ---- die collapse ----
def same_die(fa, fb):
    """fa, fb: face dicts with 'toks' (raw tokens incl. 0, reading order) and 'complete'."""
    a, b = fa['toks'], fb['toks']
    if not a or not b: return False
    if len(a) == len(b) and all(x == y or x == 0 or y == 0 for x, y in zip(a, b)): return True
    sa = [x for x in a if x]; sb = [x for x in b if x]
    if not sa or not sb: return False
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    fs = fa if len(sa) <= len(sb) else fb
    if len(short) >= 2 and not fs['complete']:
        for i in range(len(long_) - len(short) + 1):
            if long_[i:i + len(short)] == short: return True
    return False

def collapse_dies(faces):
    """group faces of one object into dies; returns list of dies (each a list of faces); legible first."""
    dies = []
    for f in faces:
        for d in dies:
            if same_die(f, d[0]): d.append(f); break
        else: dies.append([f])
    return dies

def load_sealings(level='seq_raw'):
    """every TAG object -> dict(oid,cisi,site,type,carrier,back,ctx,nimp_ft,faces,dies,imps,illegible)
    imps = list of distinct legible impressions (one face per die, the most complete / longest)."""
    objs = load_faces(level)
    mp = merge_maps()[level]
    backs = lothal_backs()
    # raw tokens incl. zeros, in reading order, with merges applied
    raw = {}
    for r in csv.DictReader(open(RAW)):
        toks = [int(t) for t in re.findall(r'\d{3}', r['text'])][::-1]
        raw[r['id']] = [mp.get(t, t) if t not in (0, 999) else 0 for t in toks]
    S = collections.OrderedDict()
    for oid, o in objs.items():
        if o['ot'] != 'sealing': continue
        for f in o['faces']:
            f['toks'] = raw.get(f'{oid}.{f["k"]}', raw.get(oid, []))
        dies = collapse_dies(o['faces'])
        imps = []
        illeg = 0
        for d in dies:
            leg = [f for f in d if f['seq']]
            if not leg: illeg += 1; continue
            best = max(leg, key=lambda f: (f['complete'], len(f['seq'])))
            best = dict(best, ndup=len(d))
            imps.append(best)
        lb = backs.get(o['cisi'], {})
        S[oid] = dict(oid=oid, cisi=o['cisi'], site=o['site'], type=o['type'],
                      carrier=(o['type'].split(':')[1] if ':' in o['type'] else '-'),
                      back=lb.get('back'), ctx=lb.get('ctx'), nimp_ft=lb.get('nimp'),
                      faces=o['faces'], dies=dies, imps=imps, illegible=illeg, sides=o['sides'], big=o['big'])
    return objs, S

def seal_texts(objs, min_len=1):
    """set of seal face texts (reading order tuples) and a site map"""
    T = collections.defaultdict(set)
    for o in objs.values():
        if o['ot'] == 'seal':
            for f in o['faces']:
                if len(f['seq']) >= min_len: T[tuple(f['seq'])].add(o['site'])
    return T

_MCACHE = {}
def matches_seal(face, seals):
    """exact match for complete faces; for fragments (>= 2 signs) a contiguous substring of some seal text"""
    s = tuple(face['seq'])
    if not s: return None
    key = (id(seals), s, bool(face['complete']))
    if key in _MCACHE: return _MCACHE[key]
    r = None
    if s in seals: r = 'exact'
    elif not face['complete'] and len(s) >= 2:
        sub = seals.get('_substr')
        if sub is None:
            sub = set()
            for t in list(seals):
                if t == '_substr': continue
                for L in range(2, len(t)):
                    for i in range(len(t) - L + 1): sub.add(t[i:i + L])
            seals['_substr'] = sub
        if s in sub: r = 'fragment'
    _MCACHE[key] = r
    return r

def fmt(seq): return '-'.join(map(str, seq)) if seq else '(none)'

def middle_fn(parse, kinds=('NAME', 'COUNT', 'TITLE')):
    def middle(s):
        lab = parse(list(s)); return tuple(w for w, l in zip(s, lab) if l in kinds)
    return middle

def is_count_face_M(s):
    return bool(s) and all(w in M_NUM or w == 328 for w in s) and any(w in M_NUM for w in s)

def pair_stats(a, b, middle, name_only, head, seals=None, fa=None, fb=None):
    ma, mb = middle(a), middle(b); na, nb = name_only(a), name_only(b)
    ha, hb = head(a), head(b)
    st = dict(shared_any=bool(set(a) & set(b)), shared_mid=bool(set(ma) & set(mb)), shared_name=bool(set(na) & set(nb)),
              same_mid=(sorted(ma) == sorted(mb) and len(ma) >= 1), ident=(list(a) == list(b)),
              jacc=len(set(a) & set(b)) / len(set(a) | set(b)) if (set(a) | set(b)) else 0.0,
              same_head=(ha == hb and ha != 'none'), both_jar=(ha == hb == 'jar'), heads=tuple(sorted((ha, hb))),
              lendiff=abs(len(a) - len(b)), maxlen=max(len(a), len(b)), minlen=min(len(a), len(b)),
              frame_only_one=((len(ma) == 0) != (len(mb) == 0)), frame_only_both=(len(ma) == 0 and len(mb) == 0),
              short_long=(min(len(a), len(b)) <= 2 and max(len(a), len(b)) >= 4),
              opener_one=((a[0] in OPEN) != (b[0] in OPEN)) if a and b else False,
              opener_both=(a[0] in OPEN and b[0] in OPEN) if a and b else False)
    if seals is not None and fa is not None:
        xa = matches_seal(fa, seals) is not None; xb = matches_seal(fb, seals) is not None
        st['match_one'] = xa != xb; st['match_both'] = xa and xb; st['match_none'] = not (xa or xb)
    return st
