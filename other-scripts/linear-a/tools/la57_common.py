#!/usr/bin/env python3
"""LA-57 EVERY BUREAUCRACY ON EARTH VOTES: shared code.

Administrative corpora from unrelated record-keeping systems are reduced to one abstract form:
  doc = {'id', 'sys', 'toks': [('T', opaque_type) | ('N', value_or_None, has_frac) | ('L',)]}
('L' = line / cord-group break).  No sign values are used anywhere; type strings are opaque ids.
Role truth (controls only, never used for Linear A) comes from conventional meanings of the
known systems:  COM commodity, TOT total, PER person, PLA place, TRA transaction, HDR office/header,
UNI unit.  Khipus: TOT = top cords (a physical attachment class, not an arithmetic one).
Systems: PC proto-cuneiform (Uruk IV-III), UR3 Ur III admin, OB Old Babylonian admin, EB Ebla admin (Syria), OA Old Assyrian
admin (CDLI), LB Linear B KN+PY (DAMOS), KH Inca khipus (Open Khipu Repository), PLANT (made-up, test only).
"""
import os, re, sys, json, math, random, hashlib, collections, csv, sqlite3, unicodedata, pickle
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la57_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
PE = os.path.join(HERE, '..', '..', 'proto-elamite')
KHDB = os.path.join(SCRATCH, 'codelib', 'open-khipu-repository-2.1.0', 'data', 'khipu.db')
ROLES = ['COM', 'TOT', 'PER', 'PLA', 'TRA', 'HDR', 'UNI']
KNOWN = ['PC', 'UR3', 'OB', 'EB', 'OA', 'LB', 'KH']
CIV2 = {'PC': 'CUN', 'UR3': 'CUN', 'OB': 'CUN', 'EB': 'CUN', 'OA': 'CUN', 'LB': 'AEG', 'KH': 'AND'}
CIV = {'PC': 'MESO', 'UR3': 'MESO', 'OB': 'MESO', 'EB': 'SYRIA', 'OA': 'ASSUR', 'LB': 'AEGEAN', 'KH': 'ANDES'}


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _cache(name, fn):
    p = os.path.join(CK, name + '.json')
    if os.path.exists(p):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(p))]
    out = fn()
    json.dump(out, open(p, 'w'))
    return out


# =============================================================== Linear A
def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        toks = []
        for t in ins['tokens']:
            if t['t'] == 'word':
                toks.append(('T', '-'.join(t['s'])))
            elif t['t'] == 'logo':
                toks.append(('T', 'L:' + t['v']))
            elif t['t'] == 'num':
                toks.append(('N', float(t['v']), bool(t['frac'])))
            elif t['t'] == 'nl' and toks and toks[-1][0] != 'L':
                toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if any(x[0] == 'T' for x in toks):
            out.append({'id': ins['id'], 'sys': 'LA', 'site': ins['site'], 'toks': toks})
    return out


# =============================================================== Linear B (DAMOS)
def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def _lb():
    from la6_common import LB_COM, LIQ_COM, DRYU, LIQU
    WORD = re.compile(r'^[a-z0-9*]+(-[a-z0-9*]+)*$')
    MEAS = set('TVZSMNPQ')
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m or m.group(1) not in ('KN', 'PY'):
            continue
        toks = []
        for ln in (d.get('content') or '').split('\n'):
            meas, logo, any_ = None, None, False
            for t in _strip(ln).split():
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'v.', 'r.'):
                    continue
                b = s.split('+')[0]
                if b in LB_COM or re.fullmatch(r'\*1\d\d', b) or (b.isupper() and len(b) >= 3 and b not in MEAS):
                    toks.append(('T', 'L:' + b)); logo = b; any_ = True
                elif s in MEAS:
                    meas = s
                elif s.isdigit():
                    v = float(int(s))                           # written number, as with every other system
                    if toks and toks[-1][0] == 'N' and meas:       # S 2 V 3: minor units of the same amount
                        toks[-1] = ('N', toks[-1][1], True)
                    else:
                        toks.append(('N', v, bool(meas))); any_ = True
                    meas = None
                elif WORD.match(s) and '-' in s:
                    toks.append(('T', s)); any_ = True
            if any_:
                toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if any(x[0] == 'T' for x in toks):
            out.append({'id': h, 'sys': 'LB', 'site': m.group(1), 'series': m.group(2), 'toks': toks})
    return out


def lb_docs():
    return _cache('lb_docs', _lb)


LB_TRUTH_W = {
    'TOT': 'to-so to-sa to-so-de to-sa-de to-so-pa',
    'TRA': 'a-pu-do-si de-ka-sa-to do-so-mo e-ko-si e-ke e-ko-te a-ke-re a-ke-re-se di-do-si o-u-di-do-si '
           'di-do-ke a-pe-do-ke a-pe-e-si a-pe-o-te pa-ro o-da-a2 e-e-si me-ta-pe-mo o-pe-ro o-pe-ro-si',
    'PLA': 'pa-i-to ku-do-ni-ja a-mi-ni-so ko-no-so tu-ri-so ru-ki-to da-wo e-ra su-ri-mo pu-ro pa-ki-ja-ne '
           'ro-u-so ka-ra-do-ro ri-jo ti-mi-to-a-ko a-pu2-we a-ke-re-wa e-ra-to pe-to-no me-ta-pa za-ma-e-wi-ja '
           'ri-jo-no ku-ta-to qa-mo si-ja-du-we ra-to e-ko-me-no ka-ru-no u-ta-no a-ka-wo-ne ti-ri-to '
           'ko-tu-we e-ki-no-jo pa-ra-ja da-*22-to do-ti-ja ra-su-to tu-ni-ja a-pa-ta-wa e-ra-jo '
           'ka-to-ro u-pa-ta-ro qa-ra pi-ja-se-ra a-ra-ka-te-ja ku-ko-no-jo',
}
LB_TRUTH = {w: r for r, ws in LB_TRUTH_W.items() for w in ws.split()}


def lb_truth(docs):
    out, first, tot = {}, collections.Counter(), collections.Counter()
    for d in docs:
        ws = [x[1] for x in d['toks'] if x[0] == 'T']
        for i, w in enumerate(ws):
            tot[w] += 1
            if d.get('series', '').startswith('D') and d['site'] == 'KN' and i == 0:
                first[w] += 1
    for w, c in tot.items():
        if w.startswith('L:'):
            out[w] = 'COM'
        elif w in LB_TRUTH:
            out[w] = LB_TRUTH[w]
        elif first[w] >= max(1, 0.5 * c):
            out[w] = 'PER'
    return out


# =============================================================== CDLI (UR3, OB, EB, OA)
UR_INT = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000,
          'diš': 1, 'aš': 1}
UR_CAPU = {'barig': 60, 'ban2': 10}
NUMRE = re.compile(r"^(\d+(?:/\d+)?)\(([a-z']+\d?)(?:@[a-z0-9]+)*\)$")


def _cdli(period_pred, nmax, tag):
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if period_pred(row['period']) and 'dministrative' in row.get('genre', ''):
            keep['P%06d' % int(row['id_text'])] = row['provenience'].split(' (')[0]
    docs, cur, toks, surf = [], None, [], None

    def flush():
        if cur and sum(1 for x in toks if x[0] == 'N') >= 2 and sum(1 for x in toks if x[0] == 'T') >= 2:
            t = list(toks)
            while t and t[-1][0] == 'L':
                t.pop()
            docs.append({'id': cur, 'sys': tag, 'site': keep[cur], 'toks': t})
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            cur = raw[1:8] if raw[1:8] in keep else None
            toks, surf = [], 'obverse'
            continue
        if not cur:
            continue
        if raw.startswith('@'):
            w = raw[1:].split()[0] if raw[1:].split() else ''
            surf = 'seal' if w in ('seal', 'envelope') else surf
            continue
        if surf == 'seal' or not raw[:1].isdigit():
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
        if not m:
            continue
        body = re.sub(r'[\[\]#?!<>*]|\(\$.*?\$\)', '', m.group(1))
        line, nums = [], []
        for x in body.replace('_', ' ').split():
            mm = NUMRE.match(x)
            if mm:
                nums.append((mm.group(1), mm.group(2)))
                continue
            if nums:
                line.append(_numtok(nums, line)); nums = []
            if x in ('...', 'x') or x.startswith('x-') or x == 'n' or x.startswith('n('):
                continue
            line.append(('T', x.lower()))
        if nums:
            line.append(_numtok(nums, line))
        if line:
            toks.extend(line); toks.append(('L',))
    flush()
    rng = random.Random(seed('la57-' + tag))
    rng.shuffle(docs)
    return docs[:nmax]


def _numtok(nums, line):
    try:
        capish = any(u in UR_CAPU for _, u in nums)
        v = Fr(0)
        for n, u in nums:
            f = Fr(n)
            v += f * UR_CAPU[u] if u in UR_CAPU else f * UR_INT[u]
        v = float(v)
        return ('N', v, abs(v - round(v)) > 1e-9 or capish)
    except (KeyError, ValueError, ZeroDivisionError):
        return ('N', None, False)


def ur3_docs():
    return _cache('ur3_docs', lambda: _cdli(lambda p: p.startswith('Ur III'), 5000, 'UR3'))


def ob_docs():
    return _cache('ob_docs', lambda: _cdli(lambda p: p.startswith('Old Babylonian') or
                                           p.startswith('Early Old Babylonian'), 5000, 'OB'))


def eb_docs():
    return _cache('eb_docs', lambda: _cdli(lambda p: p.startswith('Ebla'), 5000, 'EB'))


def oa_docs():
    return _cache('oa_docs', lambda: _cdli(lambda p: p.startswith('Old Assyrian'), 5000, 'OA'))


MESO_W = {
    'COM': 'udu u8 masz2 ud5 sila4 kir11 gu4 ab2 amar ansze masz gukkal udu-nita2 dusu2 szah2 sila4-ga amar-ga '
           'gu4-niga u8-sig5 udu-niga sze ziz2 kasz ninda i3 zu2-lum tug2 siki ku3-babbar urudu gesz ga numun '
           'i3-gesz i3-nun sze-ba gig kasz-saga dabin esza zi3 munu4 sum mun masz2-gal kusz sze-bar zu2-lum-ma '
           'gesztin sze-gesz-i3 ku3-gi an-na zabar kaskal sa-gal sza3-gal ku6 muszen',
    'UNI': 'sila3 gur ban2 barig gin2 ma-na dug gu2 sar iku sa pi gu2-un gi-na',
    'TRA': 'ba-ti ba-zi zi-ga i3-dab5 ba-ug7 ba-an-ti mu-du sa2-du11 szu-ba-ti ba-ab-dab5 mu-kux(du) ba-an-zi '
           'i3-gal2 ba-na-zi ba-usz2 mu-kux(DU) szu-ba-an-ti ma-hi-ir ma-hir na-din id-di-in i-din it-ta-din '
           'e-ti-ir nadnu ba-an-dab5 i-na-an-di-nu',
    'TOT': 'szu-nigin2 szunigin szu-nigin pap',
    'HDR': 'ugula nu-banda3 szabra sanga ensi2 kiszib3 giri3 ki iti mu u4',
}
MESO = {w: r for r, ws in MESO_W.items() for w in ws.split()}


def meso_truth(docs):
    tr = dict(MESO)
    pn, tot = collections.Counter(), collections.Counter()
    for d in docs:
        ts = [x for x in d['toks'] if x[0] != 'L']
        for i, x in enumerate(ts):
            if x[0] != 'T':
                continue
            w = x[1]; tot[w] += 1
            if i > 0 and ts[i - 1][0] == 'T':
                p = ts[i - 1][1]
                if p in ('giri3', 'kiszib3', 'szu', 'igi') or (p == 'ki' and w.endswith('-ta')):
                    pn[w] += 1
    out = {}
    for w, c in tot.items():
        if w in tr:
            out[w] = tr[w]
        elif '{ki}' in w or w.startswith('{uru}') or w.startswith('{kur}'):
            out[w] = 'PLA'
        elif w.startswith('{disz}') or w.startswith('{m}') or w.startswith('{f}') or w.startswith('{1}'):
            out[w] = 'PER'
        elif pn[w] >= max(1, 0.5 * c) and not w.startswith('{d}'):
            out[w] = 'PER'
    return out


# =============================================================== proto-cuneiform
PC_W = {'COM': 'UDU U8 UDUNITA KIR11 SILA4 MASZ MASZ2 UD5 GU4 AB2 AMAR ANSZE SZAH2 KU6 MUSZEN SZE NINDA KASZ GAR '
               'I3 TUG2 SIG2 GA KU3 URUDU GADA NAGA BAPPIR GISZ MUN ZIZ2 SZIM DUH',
        'UNI': 'DUG', 'HDR': 'EN SANGA E2', 'TRA': 'BA DU GI'}
PC_T = {w: r for r, ws in PC_W.items() for w in ws.split()}


def _pc():
    sys.path.insert(0, os.path.join(PE, 'tools'))
    from common import base, system_of
    import pe38_common as P
    T = json.load(open(os.path.join(PE, 'data', 'pe2_pc_corpus.json')))
    out = []
    for t in T:
        toks = []
        for l in t['lines']:
            if l['surface'] in ('seal',):
                continue
            nums = [[n, c] for n, c in l['numerals'] if isinstance(n, int)]
            sysn = system_of(l['numerals']) if l['numerals'] else None
            line = []
            if sysn:
                val = P.num_value(nums, sysn) if len(nums) == len(l['numerals']) else None
                line.append(('N', float(val) if val is not None else None, sysn not in ('SDB',)))
            for s in l['signs']:
                if s == 'x' or s.startswith('x'):
                    continue
                line.append(('T', base(s)))
            if line:
                toks.extend(line); toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if sum(1 for x in toks if x[0] == 'T') >= 1 and sum(1 for x in toks if x[0] == 'N') >= 1:
            out.append({'id': t['id'], 'sys': 'PC', 'site': t['provenience'].split(' (')[0], 'toks': toks})
    return out


def pc_docs():
    return _cache('pc_docs', _pc)


def pc_truth(docs):
    out = {}
    for d in docs:
        for x in d['toks']:
            if x[0] == 'T':
                b = re.sub(r'~.*$', '', x[1]).strip('|')
                if x[1] in PC_T:
                    out[x[1]] = PC_T[x[1]]
                elif b in PC_T and not any(c in b for c in '+.x&'):
                    out[x[1]] = PC_T[b]
    return out


# =============================================================== khipus
def _kh():
    c = sqlite3.connect(KHDB)
    col = {}
    for kid, cid, c1, o1, c2 in c.execute('select KHIPU_ID,CORD_ID,COLOR_CD_1,OPERATOR_1,COLOR_CD_2 '
                                         'from ascher_cord_color order by color_id'):
        col.setdefault(cid, (c1 or '?') + ((o1 or '') + (c2 or '') if c2 else ''))
    # cord value: knot clusters ordered from the far end; unit cluster = long / figure-eight knots
    kc = collections.defaultdict(list)
    kn = collections.defaultdict(lambda: collections.Counter())
    for cid, clid, sp in c.execute('select CORD_ID,CLUSTER_ID,START_POS from knot_cluster'):
        kc[cid].append((float(sp or 0), clid))
    for cid, tc, clid, nt in c.execute('select CORD_ID,TYPE_CODE,CLUSTER_ID,NUM_TURNS from knot'):
        t = (tc or '').strip("'")
        if t in ('S', 'SP'):
            kn[clid]['v'] += 1
        elif t in ('L', 'LL'):
            kn[clid]['v'] += int(nt or 0); kn[clid]['u'] = 1
        elif t in ('E', 'EE'):
            kn[clid]['v'] += 1; kn[clid]['u'] = 1
    val = {}
    for cid, L in kc.items():
        L.sort(reverse=True)            # farthest from the primary cord first
        v, place = 0, 0
        for k, (sp, clid) in enumerate(L):
            if k == 0 and not kn[clid]['u']:
                place = 1               # no unit cluster: units are zero
            v += kn[clid]['v'] * 10 ** place
            place += 1
        val[cid] = v
    cords = collections.defaultdict(list)
    for kid, cid, pf, att, cl, cord_ord, apos, lev, clas in c.execute(
            'select KHIPU_ID,CORD_ID,PENDANT_FROM,ATTACHED_TO,CLUSTER_ID,CORD_ORDINAL,ATTACH_POS,CORD_LEVEL,'
            'CORD_CLASSIFICATION from cord'):
        if lev not in (1, None) or clas in ('M', 'K', 'X', 'A', 'B', 'C', 'D', 'E', 'F'):
            continue
        cords[kid].append((float(apos or 0), cord_ord or 0, cl, cid, clas))
    out = []
    for kid, L in cords.items():
        L.sort()
        toks, occ, lastcl = [], [], None
        for apos, o, cl, cid, clas in L:
            if lastcl is not None and cl != lastcl and toks and toks[-1][0] != 'L':
                toks.append(('L',))
            lastcl = cl
            toks.append(('T', 'K:' + col.get(cid, '?')))
            occ.append('TOT' if clas in ('T', 'TPA') else 'NON')
            if cid in val and val[cid] > 0:
                toks.append(('N', float(val[cid]), False))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if len(occ) >= 3:
            out.append({'id': 'KH%d' % kid, 'sys': 'KH', 'site': 'K', 'toks': toks, 'occ': occ})
    return out


def kh_docs():
    return _cache('kh_docs', _kh)


# =============================================================== planted (test only)
def plant_docs(n_tok, rng):
    from la45_common import planted_docs
    docs, tr = planted_docs(n_tok, rng)
    m = {'PERSON': 'PER', 'PLACE': 'PLA', 'COMMODITY': 'COM', 'VERB': 'TRA', 'TOTAL': 'TOT'}
    out = []
    for d in docs:
        t = []
        for x in d['toks']:
            t.append(tuple(x) if x[0] == 'T' else ('N', x[1], x[2]))
            if x[0] == 'N':
                t.append(('L',))
        while t and t[-1][0] == 'L':
            t.pop()
        out.append({'id': d['id'], 'sys': 'PLANT', 'site': 'P', 'toks': t})
    return out, {w: m[r] for w, r in tr.items() if r in m}


LOADERS = {'PC': (pc_docs, pc_truth), 'UR3': (ur3_docs, meso_truth), 'OB': (ob_docs, meso_truth),
           'EB': (eb_docs, meso_truth), 'OA': (oa_docs, meso_truth), 'LB': (lb_docs, lb_truth), 'KH': (kh_docs, None)}


def ntok(docs):
    return sum(1 for d in docs for x in d['toks'] if x[0] != 'L')


def draw(docs, n, rng):
    idx = list(range(len(docs))); rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= n:
            break
        out.append(docs[i]); c += sum(1 for x in docs[i]['toks'] if x[0] != 'L')
    return out


# =============================================================== shuffles
def shuffle_types(docs, rng):
    ws = [x for d in docs for x in d['toks'] if x[0] == 'T']
    rng.shuffle(ws); it = iter(ws)
    return [dict(d, toks=[next(it) if x[0] == 'T' else x for x in d['toks']]) for d in docs]


def shuffle_order(docs, rng):
    out = []
    for d in docs:
        t = [x for x in d['toks'] if x[0] != 'L']; rng.shuffle(t)
        out.append(dict(d, toks=t))
    return out


# =============================================================== features
FEAT = ['pos', 'first', 'last', 'linit', 'lfin', 'nextN', 'prevN', 'adjN', 'nextW', 'prevW', 'mag', 'frac',
        'sum', 'max', 'nrank', 'doclen', 'numshare', 'single', 'linelen', 'dist', 'lineidx', 'lastline',
        'wordsline', 'nonum_doc',
        't_freq', 't_disp', 't_rep', 't_lent', 't_rent', 't_adjN', 't_first', 't_sum', 't_lfin', 't_single',
        't_hapax', 't_nbrf', 't_linit', 't_lastline', 't_mag', 't_max', 't_dist', 't_pos']
NF = len(FEAT)
FI = {f: i for i, f in enumerate(FEAT)}


def _close(a, b):
    return a is not None and b is not None and abs(a - b) <= 0.005 * max(1.0, abs(b))


def features(docs, truth=None):
    """Return X (n_occ x NF), types (list), labels (list of role or None), doc index per occurrence."""
    rows, types, labs, didx = [], [], [], []
    for di, d in enumerate(docs):
        seq = d['toks']
        lines, cur = [], []
        for x in seq:
            if x[0] == 'L':
                if cur: lines.append(cur)
                cur = []
            else:
                cur.append(x)
        if cur: lines.append(cur)
        flat = [(li, x) for li, l in enumerate(lines) for x in l]
        nums = [x[1] for _, x in flat if x[0] == 'N']
        kv = [v for v in nums if v is not None]
        nw = sum(1 for _, x in flat if x[0] == 'T')
        ntot = len(flat)
        numpos = [k for k, (_, x) in enumerate(flat) if x[0] == 'N']
        # sum flags per number position (value equals sum of >=2 other numbers in doc or of the run before/after it)
        sumflag, rankf, maxf = {}, {}, {}
        tot_all = sum(kv)
        for j, k in enumerate(numpos):
            v = flat[k][1][1]
            s = False
            if v is not None and len(kv) >= 3:
                if _close(tot_all - v, v):
                    s = True
                else:
                    acc, cnt = 0.0, 0
                    for kk in reversed(numpos[:j]):
                        vv = flat[kk][1][1]
                        if vv is None: break
                        acc += vv; cnt += 1
                        if cnt >= 2 and _close(acc, v): s = True; break
                    if not s:
                        acc, cnt = 0.0, 0
                        for kk in numpos[j + 1:]:
                            vv = flat[kk][1][1]
                            if vv is None: break
                            acc += vv; cnt += 1
                            if cnt >= 2 and _close(acc, v): s = True; break
            sumflag[k] = s
            rankf[k] = (sum(1 for u in kv if v is not None and u < v) / max(1, len(kv) - 1)) if v is not None else 0.5
            maxf[k] = v is not None and len(kv) >= 2 and v >= max(kv)
        wi = 0
        occ = d.get('occ')
        for k, (li, x) in enumerate(flat):
            if x[0] != 'T':
                continue
            line = lines[li]
            pos_in_line = line.index(x) if x in line else 0
            lw = [y for y in line if y[0] == 'T']
            nxt = flat[k + 1] if k + 1 < ntot and flat[k + 1][0] == li else None
            prv = flat[k - 1] if k > 0 and flat[k - 1][0] == li else None
            nextN = nxt is not None and nxt[1][0] == 'N'
            prevN = prv is not None and prv[1][0] == 'N'
            ak = k + 1 if nextN else (k - 1 if prevN else None)
            av = flat[ak][1] if ak is not None else None
            dist = min([abs(k - p) for p in numpos], default=99)
            r = [wi / max(1, nw - 1), wi == 0, wi == nw - 1,
                 bool(lw) and lw[0] is x, bool(lw) and lw[-1] is x,
                 nextN, prevN, nextN or prevN,
                 nxt is not None and nxt[1][0] == 'T', prv is not None and prv[1][0] == 'T',
                 math.log10(1 + av[1]) / 4 if av and av[1] is not None else 0.0,
                 bool(av and av[2]), bool(ak is not None and sumflag.get(ak)),
                 bool(ak is not None and maxf.get(ak)), rankf.get(ak, 0.0) if ak is not None else 0.0,
                 min(1.0, math.log(1 + ntot) / math.log(200)), len(nums) / max(1, ntot), nw == 1,
                 min(1.0, len(line) / 10), min(1.0, dist / 5), li / max(1, len(lines) - 1), li == len(lines) - 1,
                 min(1.0, len(lw) / 6), len(nums) == 0]
            rows.append(r)
            types.append(x[1]); didx.append(di)
            if occ is not None:
                labs.append(occ[wi] if occ[wi] != 'NON' else 'NON')
            else:
                labs.append(truth.get(x[1]) if truth else None)
            wi += 1
    X = np.zeros((len(rows), NF), np.float32)
    if not rows:
        return X, types, labs, np.array(didx)
    X[:, :24] = np.array(rows, np.float32)
    # type-level aggregates (rank-normalised among types within this corpus draw)
    T = collections.defaultdict(list)
    for i, t in enumerate(types):
        T[t].append(i)
    tl = list(T)
    ti = {t: k for k, t in enumerate(tl)}
    cnt = np.array([len(T[t]) for t in tl], float)
    left, right = collections.defaultdict(set), collections.defaultdict(set)
    for d in docs:
        ws = [x for x in d['toks'] if x[0] != 'L']
        for a, b in zip(ws, ws[1:]):
            ka = a[1] if a[0] == 'T' else '#N'
            kb = b[1] if b[0] == 'T' else '#N'
            if a[0] == 'T': right[a[1]].add(kb)
            if b[0] == 'T': left[b[1]].add(ka)
    logf = {t: math.log(len(T[t])) for t in tl}
    agg = np.zeros((len(tl), 18))
    occ_t = np.array([ti[t] for t in types])
    for t in tl:
        ii = T[t]; k = ti[t]
        ds = collections.Counter(didx[i] for i in ii)
        nb = [logf.get(y, 0.0) for y in (left[t] | right[t]) if y != '#N']
        agg[k] = [math.log(len(ii)), len(ds) / len(ii), sum(c for c in ds.values() if c > 1) / len(ii),
                  len(left[t]) / len(ii), len(right[t]) / len(ii), X[ii, FI['adjN']].mean(),
                  X[ii, FI['first']].mean(), X[ii, FI['sum']].mean(), X[ii, FI['lfin']].mean(),
                  X[ii, FI['single']].mean(), float(len(ii) == 1), np.mean(nb) if nb else 0.0,
                  X[ii, FI['linit']].mean(), X[ii, FI['lastline']].mean(), X[ii, FI['mag']].mean(),
                  X[ii, FI['max']].mean(), X[ii, FI['dist']].mean(), X[ii, FI['pos']].mean()]
    # rank-normalise each aggregate across types (ties averaged)
    for j in range(agg.shape[1]):
        v = agg[:, j]
        order = np.argsort(v, kind='mergesort')
        rk = np.empty(len(v)); rk[order] = np.arange(len(v))
        # average ties
        uv, inv = np.unique(v, return_inverse=True)
        sums = np.bincount(inv, rk); cnts = np.bincount(inv)
        rk = (sums / cnts)[inv]
        agg[:, j] = rk / max(1, len(v) - 1)
    X[:, 24:] = agg[occ_t]
    return X, types, labs, np.array(didx)


# =============================================================== AUC
def auc(score, y):
    """y boolean. Rank AUC (ties averaged)."""
    y = np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    s = score[order]
    rk = np.empty(len(s)); rk[order] = np.arange(1, len(s) + 1)
    uv, inv = np.unique(s, return_inverse=True)
    sums = np.bincount(inv, rk[order]); cnts = np.bincount(inv)
    r = np.empty(len(s)); r[order] = (sums / cnts)[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_matrix(S, y):
    """S: n x m scores; returns m AUCs (ties broken by rank average approximated with tiny jitter)."""
    y = np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[1], np.nan)
    from scipy.stats import rankdata
    R = rankdata(S, axis=0)
    return (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip() + '\n')


# =============================================================== pooled draws and label sets
def load():
    F = pickle.load(open(os.path.join(CK, 'feats.pkl'), 'rb'))
    sysd = {}
    for k in KNOWN:
        Xs, labs, types = [], [], []
        for j in range(4):
            f = F[(k, j)]
            Xs.append(f['X']); labs += f['labs']; types += [(j, t) for t in f['types']]
        sysd[k] = dict(X=np.vstack(Xs), labs=labs, types=types)
    return F, sysd


def label_sets(s, role, nnull, rng):
    """rows used for role (labelled rows), real y and nnull permuted y (type-level within corpus)."""
    labs = s['labs']
    rows = np.array([i for i, l in enumerate(labs) if l is not None])
    y = np.array([labs[i] == role for i in rows])
    Ys = [y]
    if s.get('kh'):
        for _ in range(nnull):
            Ys.append(rng.permutation(y))
    else:
        tl = sorted({s['types'][i][1] for i in rows})
        tr = {}
        for i in rows:
            tr[s['types'][i][1]] = labs[i]
        for _ in range(nnull):
            perm = rng.permutation(len(tl))
            m = {tl[a]: tr[tl[b]] for a, b in zip(range(len(tl)), perm)}
            Ys.append(np.array([m[s['types'][i][1]] == role for i in rows]))
    return rows, np.array(Ys)


