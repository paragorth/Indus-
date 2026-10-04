"""pe38 TYPE INFERENCE FOR PROTO-ELAMITE SIGNS: shared code.

Every sign type (base sign; compounds kept whole) gets one role from
  NAM person/group name, PRO profession/class of person, COM commodity, ANI animal category,
  MEA container/measure, HDR header/office, VRB transaction verb, TOT total marker, QUA qualifier.
A role assignment is scored by (i) a collapsed Dirichlet-multinomial over structural occurrence features,
(ii) type-level accounting rules (cost table), (iii) a line type-checker (pe38_core.c: entries need a
countable head and no header/total sign; totals need a total marker or the head role of the entry they
close; headers need an office/name/verb sign; no total marker off total lines).
Corpora in one abstract format: docs = [{id, site, lines: [{toks, kind, sys, val, surf}]}], toks = type
strings or None (wildcard: 'x', numerals-as-signs excluded).
Controls: proto-cuneiform (CDLI, data/pe2_pc_corpus.json) and Ur III (CDLI dump in the scratchpad), both
with role labels from conventional sign / word meanings, used ONLY to score.
"""
import collections, ctypes, hashlib, json, math, os, random, re, sys
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe38_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa: E402

SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
ROLES = ['NAM', 'PRO', 'COM', 'ANI', 'MEA', 'HDR', 'VRB', 'TOT', 'QUA']
R = len(ROLES)
RI = {r: i for i, r in enumerate(ROLES)}
CARD = [4, 5, 5, 6, 5, 4, 4, 5]
NF = len(CARD)

_so = os.path.join(CK, 'pe38_core.so')
if not os.path.exists(_so):
    os.system('gcc -O2 -shared -fPIC -o %s %s -lm' % (_so, os.path.join(HERE, 'pe38_core.c')))
_lib = ctypes.CDLL(_so)
_I = np.ctypeslib.ndpointer(np.int32, flags='C')
_D = np.ctypeslib.ndpointer(np.float64, flags='C')
_lib.run_chain.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _I, ctypes.c_int, _I, _I, _D, ctypes.c_int,
                           _I, _I, _I, _I, _I, _I, _I, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                           ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_double, ctypes.c_ulonglong,
                           _I, _I, ctypes.c_int, _D]
_lib.run_chain.restype = ctypes.c_int
_lib.count_viol.argtypes = [ctypes.c_int, _I, _I, _I, _I, _I]
_lib.count_viol.restype = ctypes.c_int


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ numerals
SEX = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000}
CAPV = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720}
BV = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 120, 'N48': 1200, 'N51': 1, 'N54': 10, 'N46': 120}
SYSCODE = {None: 0, 'SDB': 1, 'C': 2, 'C*': 2, 'B': 3, 'N23': 3, 'mod*': 3, 'S-frac': 4}


def num_value(nums, sysn):
    try:
        if sysn == 'SDB':
            return Fr(sum(n * SEX[c] for n, c in nums))
        if sysn == 'C':
            return Fr(sum(n * CAPV[c] for n, c in nums))
        if sysn == 'B':
            return Fr(sum(n * BV[c] for n, c in nums))
    except (KeyError, TypeError):
        return None
    return None


def mark_kinds(lines):
    """lines: list of dicts with toks, sys (0-4), val, surf; sets kind and prev (index of closed entry)."""
    numbered = [i for i, l in enumerate(lines) if l['sys']]
    off = [i for i in numbered if lines[i]['surf'] not in ('obverse', 'o')]
    obv_entries = [i for i in numbered if lines[i]['surf'] in ('obverse', 'o')]
    run = collections.defaultdict(list)   # system -> values since last closure
    for i, l in enumerate(lines):
        l['prev'] = -1
        if not l['sys']:
            l['kind'] = 0 if i == 0 else 1
            continue
        closed = False
        vals = run[l['sys']]
        if l['val'] is not None and len(vals) >= 2 and all(v is not None for _, v in vals) and \
                sum(v for _, v in vals) == l['val']:
            closed = True
        if not closed and len(off) == 1 and i == off[0] and len(obv_entries) >= 2:
            closed = True       # pe21: the reverse is reserved for the total
        if closed:
            l['kind'] = 3
            l['prev'] = vals[-1][0] if vals else (obv_entries[-1] if obv_entries else -1)
            run[l['sys']] = []
        else:
            l['kind'] = 2 if l['surf'] in ('obverse', 'o') else 4
            run[l['sys']].append((i, l['val']))
    return lines


# ------------------------------------------------------------------ corpora
def _sign_tok(s):
    if s == 'x' or s.startswith('x'):
        return None
    if re.match(r'^N\d', s):        # PC: numeral-shaped signs written as signs
        return 'SG:' + s
    return base(s)


def pe_docs():
    out = []
    for t in load():
        lines = []
        for l in t['lines']:
            toks = [_sign_tok(s) for s in l['signs'] if is_sign(s) or s.startswith('x')]
            sysn = system_of(l['numerals']) if l['numerals'] else None
            clean = all(isinstance(n, int) for n, _ in l['numerals']) and not l.get('lacuna')
            val = num_value(l['numerals'], sysn) if (sysn and clean) else None
            if not toks and not sysn:
                continue
            lines.append({'toks': toks, 'sys': SYSCODE.get(sysn, 3) if sysn else 0, 'val': val,
                          'surf': l['surface'], 'cnt': val if sysn == 'SDB' else None})
        if lines:
            out.append({'id': t['id'], 'site': t['provenience'].split(' (')[0], 'lines': mark_kinds(lines)})
    return out


def pc_docs(period=None):
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in T:
        if period and t['period'] != period:
            continue
        lines = []
        for l in t['lines']:
            nums = [[n, c] for n, c in l['numerals'] if isinstance(n, int)]
            toks = [_sign_tok(s) for s in l['signs']]
            sysn = system_of(l['numerals']) if l['numerals'] else None
            clean = len(nums) == len(l['numerals']) and not l.get('lacuna')
            val = num_value(nums, sysn) if (sysn and clean) else None
            if not toks and not sysn:
                continue
            lines.append({'toks': toks, 'sys': SYSCODE.get(sysn, 3) if sysn else 0, 'val': val,
                          'surf': l['surface'], 'cnt': val if sysn == 'SDB' else None})
        if lines:
            out.append({'id': t['id'], 'site': t['provenience'].split(' (')[0], 'period': t['period'],
                        'lines': mark_kinds(lines)})
    return out


UR_INT = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000}
UR_CAPU = {'barig': 60, 'ban2': 10}
NUMRE = re.compile(r"^(\d+(?:/\d+)?)\(([a-z']+\d?)\)$")


def ur3_docs_all():
    fn = os.path.join(CK, 'ur3_docs.json')
    if os.path.exists(fn):
        D = json.load(open(fn))
        for d in D:
            for l in d['lines']:
                l['val'] = Fr(l['val']) if l['val'] is not None else None
                l['cnt'] = l['val'] if l['sys'] == 1 else None
        return D
    import csv
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and 'dministrative' in row.get('genre', ''):
            keep['P%06d' % int(row['id_text'])] = row['provenience'].split(' (')[0]
    docs, cur, surf, lines = [], None, None, []

    def flush():
        if cur and len(lines) >= 2 and sum(1 for l in lines if l['sys']) >= 2:
            docs.append({'id': cur, 'site': keep[cur], 'lines': mark_kinds(lines)})
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            cur = raw[1:8] if raw[1:8] in keep else None
            surf, lines = 'obverse', []
            continue
        if not cur:
            continue
        if raw.startswith('@'):
            w = raw[1:].split()[0] if raw[1:].split() else ''
            if w in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge'):
                surf = w
            elif w in ('seal', 'envelope'):
                surf = 'seal'
            continue
        if surf == 'seal' or not raw[:1].isdigit():
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
        if not m:
            continue
        body = re.sub(r'[\[\]#?!<>*]|\(\$.*?\$\)', '', m.group(1))
        tk = body.split()
        toks, nums = [], []
        for x in tk:
            mm = NUMRE.match(x)
            if mm:
                nums.append((mm.group(1), mm.group(2)))
            elif x in ('...', 'x') or x.startswith('x-') or x == 'n' or x.startswith('n('):
                toks.append(None)
            else:
                toks.append(x.lower())
        sysn, val = 0, None
        if nums:
            capish = any(u in UR_CAPU for _, u in nums) or any(t in ('sila3', 'gur') for t in toks if t)
            try:
                if capish:
                    gur = 'gur' in toks
                    v = Fr(0)
                    for n, u in nums:
                        f = Fr(n)
                        v += f * UR_CAPU[u] if u in UR_CAPU else f * UR_INT[u] * (300 if gur else 1)
                    sysn, val = 2, v
                else:
                    sysn, val = 1, sum(Fr(n) * UR_INT[u] for n, u in nums)
            except KeyError:
                sysn, val = 3, None
        if not toks and not sysn:
            continue
        lines.append({'toks': toks, 'sys': sysn, 'val': val, 'surf': surf})
    flush()
    json.dump(docs, open(fn, 'w'), default=lambda x: str(x))
    for d in docs:
        for l in d['lines']:
            l['cnt'] = l['val'] if l['sys'] == 1 else None
    return docs


def ntok(docs):
    return sum(1 for d in docs for l in d['lines'] for t in l['toks'] if t)


def sample_size(docs, n, rng):
    """Random whole documents until the token count reaches n (PE size)."""
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= n:
            break
        out.append(docs[i]); c += sum(1 for l in docs[i]['lines'] for t in l['toks'] if t)
    return out


# ------------------------------------------------------------------ labels (controls only, never used to fit in cycle 1)
PC_LAB = {}
for _r, _ws in {'ANI': 'UDU U8 UDUNITA KIR11 SILA4 MASZ MASZ2 UD5 GU4 AB2 AMAR ANSZE SZAH2 KU6 MUSZEN',
                'COM': 'SZE NINDA KASZ GAR I3 TUG2 SIG2 GA KU3 URUDU GADA NAGA BAPPIR GISZ MUN ZIZ2 SZIM DUH',
                'MEA': 'DUG',
                'PRO': 'SAL KUR ERIM NAR SUKKAL NAMESZDA',
                'HDR': 'EN SANGA E2',
                'VRB': 'BA DU GI',
                'QUA': 'GAL TUR'}.items():
    for _w in _ws.split():
        PC_LAB[_w] = _r
UR_LAB = {}
for _r, _ws in {'ANI': 'udu u8 masz2 ud5 sila4 kir11 gu4 ab2 amar ansze masz gukkal udu-nita2 dusu2 szah2 sila4-ga '
                       'amar-ga gu4-niga u8-sig5 udu-niga',
                'COM': 'sze ziz2 kasz ninda i3 zu2-lum tug2 siki ku3-babbar urudu gesz ga numun i3-gesz i3-nun '
                       'sze-ba gig kasz-saga dabin esza zi3 munu4 sum mun',
                'MEA': 'sila3 gur ban2 barig gin2 ma-na dug gu2 sar iku ninda-du8 sa',
                'PRO': 'gurusz geme2 dumu lu2 erin2 szu-gi4 nar muhaldim dam dumu-munus dub-sar sukkal aga3-us2 '
                       'ug3-il2 lu2-huN-ga2 kinkin2 kurusda unu3 sipa',
                'HDR': 'ugula nu-banda3 szabra sanga ensi2 kiszib3 nam-ra-ak',
                'VRB': 'ba-ti ba-zi zi-ga i3-dab5 ba-ug7 ba-an-ti mu-du sa2-du11 la2-ia3 szu-ba-ti ba-ab-dab5 '
                       'mu-kux(du) ba-an-zi i3-gal2 ba-na-zi',
                'TOT': 'szu-nigin2 szunigin szu-nigin',
                'QUA': 'niga u2 nita2 munus gal tur saga sig5 us2 babbar ge6 mu-2 mu-1 hi-a'}.items():
    for _w in _ws.split():
        UR_LAB[_w] = _r


# ------------------------------------------------------------------ building the sampler input
def _sizebin(v):
    if v is None:
        return 0
    v = float(v)
    return 1 if v <= 1 else 2 if v <= 3 else 3 if v <= 9 else 4 if v <= 59 else 5


def _dbin(n):
    return 0 if n == 1 else 1 if n <= 3 else 2 if n <= 9 else 3 if n <= 29 else 4


def build(docs, minocc=2):
    tcount = collections.Counter(t for d in docs for l in d['lines'] for t in l['toks'] if t)
    types = sorted(w for w, c in tcount.items() if c >= minocc)
    ti = {w: i for i, w in enumerate(types)}
    tdocs = collections.defaultdict(set)
    for di, d in enumerate(docs):
        for l in d['lines']:
            for t in l['toks']:
                if t in ti:
                    tdocs[ti[t]].add(di)
    otype, ofeat, ls, lt, lk, lp = [], [], [0], [], [], []
    st = [collections.Counter() for _ in types]
    lineof = collections.defaultdict(list)   # type -> lines (global index)
    L0 = 0
    for di, d in enumerate(docs):
        L = d['lines']
        nl = len(L)
        for i, l in enumerate(L):
            toks = l['toks']
            n = len(toks)
            for j, t in enumerate(toks):
                lt.append(ti[t] if t in ti else -1)
                if t not in ti:
                    continue
                k = ti[t]
                slot = 0 if n == 1 else 1 if j == 0 else 3 if j == n - 1 else 2
                pos = 4 if nl == 1 else 0 if i == 0 else 1 if i == 1 else 3 if i == nl - 1 else 2

                def rel(m):
                    if m < 0 or m >= nl:
                        return 0
                    return 1 if not L[m]['sys'] else (2 if L[m]['sys'] == l['sys'] else 3)
                f = [slot, l['kind'], l['sys'], _sizebin(l['val']) if l['sys'] else 0, pos, rel(i + 1), rel(i - 1),
                     _dbin(len(tdocs[k]))]
                otype.append(k); ofeat.extend(f)
                s = st[k]
                s['n'] += 1; s['k%d' % l['kind']] += 1
                if l['sys']:
                    s['num'] += 1
                    s['cnt'] += l['sys'] == 1; s['cap'] += l['sys'] == 2
                    if l['sys'] == 1 and l['val'] is not None:
                        s['cntv'] += 1; s['big'] += l['val'] >= 10
                s['multi'] += n >= 2; s['alone'] += n == 1; s['fin'] += slot == 3; s['init'] += slot == 1
                lineof[k].append(L0 + i)
            lt_k = l['kind']
            lk.append(lt_k)
            lp.append(L0 + l['prev'] if l['prev'] >= 0 else -1)
            ls.append(len(lt))
        L0 += nl
    nL = len(lk)
    # dependent lines: the lines a type sits on, plus total lines that close a line it sits on
    closers = collections.defaultdict(list)
    for li in range(nL):
        if lp[li] >= 0:
            closers[lp[li]].append(li)
    dstart, dlist = [0], []
    for k in range(len(types)):
        dep = set(lineof[k])
        for li in list(dep):
            dep.update(closers.get(li, []))
        dlist.extend(sorted(dep)); dstart.append(len(dlist))
    A = lambda x: np.ascontiguousarray(np.array(x, np.int32))
    stats = []
    for k in range(len(types)):
        s = st[k]; n = max(s['n'], 1); num = max(s['num'], 1)
        stats.append(dict(n=s['n'], docs=len(tdocs[k]), p_num=s['num'] / n, p_hdr=s['k0'] / n, p_un=s['k1'] / n,
                          p_tot=s['k3'] / n, p_cnt=s['cnt'] / num if s['num'] else 0, p_cap=s['cap'] / num if s['num'] else 0,
                          p_big=s['big'] / s['cntv'] if s['cntv'] else 0, p_multi=s['multi'] / n, p_alone=s['alone'] / n,
                          p_fin=s['fin'] / n, p_init=s['init'] / n))
    return dict(types=types, ti=ti, otype=A(otype), ofeat=A(ofeat), ls=A(ls), lt=A(lt), lk=A(lk), lp=A(lp),
                dstart=A(dstart), dlist=A(dlist), nL=nL, stats=stats,
                nocc=np.array([s['n'] for s in stats]), ndocs=np.array([s['docs'] for s in stats]))


def rule_cost(B, w=2.0, scale='sqrt'):
    """Pre-registered type-level accounting rules (cycle 1). cost[t, r] = w * f(n) per broken rule."""
    C = np.zeros((len(B['types']), R))
    for t, s in enumerate(B['stats']):
        f = math.sqrt(s['n']) if scale == 'sqrt' else 1.0
        head = s['p_fin'] + s['p_alone']
        br = {
            'NAM': (s['p_num'] < 0.5) + (s['p_multi'] < 0.5) + (s['p_fin'] > 0.5),
            'PRO': (s['p_num'] < 0.5) + (s['p_cnt'] < 0.6) + (head < 0.5) + (s['p_big'] > 0.3),
            'ANI': (s['p_num'] < 0.5) + (s['p_cnt'] < 0.6) + (head < 0.5) + (s['p_big'] < 0.2),
            'COM': (s['p_num'] < 0.5) + (head < 0.5) + (s['p_cap'] < 0.3),
            'MEA': (s['p_num'] < 0.5) + (s['p_cap'] < 0.5) + (s['p_fin'] < 0.3),
            'HDR': (s['p_hdr'] < 0.4) * 1,
            'VRB': (s['p_un'] + s['p_hdr'] + s['p_tot'] < 0.5) + (s['p_alone'] > 0.5) + (s['docs'] < 3),
            'TOT': (s['p_tot'] < 0.5) + (s['docs'] < 3),
            'QUA': (s['p_multi'] < 0.8) + (s['p_alone'] > 0.1),
        }
        for r, b in br.items():
            C[t, RI[r]] = w * f * b
    return C


def run_chains(B, cost, nchains=16, steps=400_000, burn=None, thin=2000, seed0=1, alpha=0.5, beta=1.0,
               lam_g=2.0, T0=8.0, fixed=None):
    nT = len(B['types'])
    burn = burn if burn is not None else steps // 2
    maxs = (steps - burn) // thin + 1
    card = np.array(CARD, np.int32)
    cost = np.ascontiguousarray(cost.ravel(), np.float64)
    fx = np.full(nT, -1, np.int32) if fixed is None else np.ascontiguousarray(fixed, np.int32)
    allS, viol = [], []
    for c in range(nchains):
        rng = np.random.default_rng(seed0 * 1000 + c)
        assign = rng.integers(0, R, nT).astype(np.int32)
        assign[fx >= 0] = fx[fx >= 0]
        S = np.zeros((maxs, nT), np.int32)
        tr = np.zeros(1)
        ns = _lib.run_chain(nT, R, NF, card, len(B['otype']), B['otype'], B['ofeat'], cost, B['nL'], B['ls'], B['lt'],
                            B['lk'], B['lp'], B['dstart'], B['dlist'], fx, alpha, beta, lam_g, steps, burn, thin, T0,
                            seed0 * 7919 + c + 1, assign, S, maxs, tr)
        allS.append(S[:ns]); viol.append(tr[0])
    return np.stack(allS), viol


def count_viol(B, assign):
    return _lib.count_viol(B['nL'], B['ls'], B['lt'], B['lk'], B['lp'], np.ascontiguousarray(assign, np.int32))


def summarize(S):
    C, N, T = S.shape
    cnt = np.zeros((T, R))
    for r in range(R):
        cnt[:, r] = (S == r).sum(axis=(0, 1))
    mode = cnt.argmax(1); freq = cnt.max(1) / (C * N)
    cm = np.zeros((C, T), int)
    for c in range(C):
        cm[c] = np.stack([(S[c] == r).sum(0) for r in range(R)], 1).argmax(1)
    agree = (cm == mode[None, :]).mean(0)
    return mode, freq, agree, cnt / (C * N)


def forced(B, S, thr=0.9, minocc=3):
    mode, freq, agree, P = summarize(S)
    out = {}
    for t, w in enumerate(B['types']):
        if B['nocc'][t] >= minocc and freq[t] >= thr and agree[t] >= thr:
            out[w] = (ROLES[mode[t]], float(freq[t]), float(agree[t]))
    return out, mode, freq, agree, P


def score_labels(B, frc, mode, lab, minocc=3):
    """Strict accuracy of forced labelled types; accuracy of modal role on all labelled types; per-role recall;
    best-mapping accuracy (each inferred role mapped to its most common true label)."""
    idx = [t for t, w in enumerate(B['types']) if w in lab and B['nocc'][t] >= minocc]
    fl = [(w, frc[w][0], lab[w]) for w in frc if w in lab]
    strict = sum(a == b for _, a, b in fl) / len(fl) if fl else float('nan')
    allacc = sum(ROLES[mode[t]] == lab[B['types'][t]] for t in idx) / len(idx) if idx else float('nan')
    rec = {}
    for r in set(lab.values()):
        ii = [t for t in idx if lab[B['types'][t]] == r]
        if ii:
            rec[r] = (sum(ROLES[mode[t]] == r for t in ii), len(ii))
    mp = collections.defaultdict(collections.Counter)
    for t in idx:
        mp[ROLES[mode[t]]][lab[B['types'][t]]] += 1
    best = sum(c.most_common(1)[0][1] for c in mp.values()) / len(idx) if idx else float('nan')
    maj = collections.Counter(lab[B['types'][t]] for t in idx).most_common(1)[0][1] / len(idx) if idx else float('nan')
    return dict(n_lab=len(idx), n_forced_lab=len(fl), strict=strict, acc_all=allacc, recall=rec, bestmap=best,
                majority=maj, forced_lab=fl)


def shuffle_entries(docs, rng, scope='entry'):
    """Null: sign tokens permuted across entry-line slots (kinds 2, 4) of the corpus ('entry'), or across every
    slot of every line ('all'). Line kinds, numerals and string lengths are kept."""
    kinds = (2, 4) if scope == 'entry' else (0, 1, 2, 3, 4)
    pos = [(di, li, j) for di, d in enumerate(docs) for li, l in enumerate(d['lines']) if l['kind'] in kinds
           for j in range(len(l['toks']))]
    toks = [docs[di]['lines'][li]['toks'][j] for di, li, j in pos]
    rng.shuffle(toks)
    new = [dict(d, lines=[dict(l, toks=list(l['toks'])) for l in d['lines']]) for d in docs]
    for (di, li, j), t in zip(pos, toks):
        new[di]['lines'][li]['toks'][j] = t
    return new


# ------------------------------------------------------------------ "good solutions" (chains do not mix: each
# tempered chain freezes in its own mode, so forced = same role in >= thr of the best-scoring chains)
def good_forced(out, q=0.34, thr=0.9, minocc=3):
    v = np.array(out['viol'], float)
    cut = np.quantile(v, q)
    good = [c for c in range(len(v)) if v[c] <= cut]
    cm = np.array(out['chain_modes'])[good]
    res = {}
    for t, w in enumerate(out['types']):
        if out['nocc'][w] < minocc:
            continue
        bc = np.bincount(cm[:, t], minlength=R)
        if bc.max() / len(good) >= thr:
            res[w] = ROLES[int(bc.argmax())]
    return res, len(good)


def score_forced(fr, lab):
    fl = [(w, r, lab[w]) for w, r in fr.items() if w in lab]
    strict = sum(a == b for _, a, b in fl) / len(fl) if fl else float('nan')
    return strict, fl
