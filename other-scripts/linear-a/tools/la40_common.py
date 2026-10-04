"""LA-40 'type inference as a compiler does it': shared code.

Every word type gets one role from ROLES; every tablet must 'type-check' under fixed
structural rules (the same for Linear A and every control). No sound values, no Linear B
word meanings are used for Linear A. Linear B word classes are used only to score the
positive control.
"""
import collections, ctypes, json, math, os, random, re, subprocess, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la23_common as L23

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la40_ckpt')
os.makedirs(CK, exist_ok=True)
ROLES = ['P', 'L', 'C', 'H', 'T', 'M']   # person/group, place, commodity/qualifier, heading/transaction, total, number-word/measure
R = len(ROLES)
CARD = [4, 4, 7, 3, 3, 4]               # slot, next, numbin, closure, linepos, prev
TCARD = 4                                # type-level: number of documents bin

_so = os.path.join(HERE, 'la40_core.so')
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(HERE, 'la40_core.c')):
    subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', _so, os.path.join(HERE, 'la40_core.c'), '-lm'])
_lib = ctypes.CDLL(_so)
_P = np.ctypeslib.ndpointer
_lib.run_chain.restype = ctypes.c_int
_lib.run_chain.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _P(np.int32), ctypes.c_int, _P(np.int32), _P(np.int32),
                           _P(np.int32), ctypes.c_int, _P(np.float64), _P(np.int32), ctypes.c_int, _P(np.int32),
                           ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                           ctypes.c_double, ctypes.c_ulonglong, _P(np.int32), _P(np.int32), ctypes.c_int]


def seed(name):
    return zlib.crc32(name.encode()) % 2 ** 31


# ------------------------------------------------------------------ corpora
def la_docs():
    D = L23.la_docs(site=None, support='Tablet')
    return [dict(id=d['id'], site=d['site'], lines=[[t for t in ln if t[0] != 'X'] for ln in d['lines']]) for d in D]


def lb_docs_all():
    D = L23.lb_docs(sites=('KN', 'PY'))
    out = []
    for d in D:
        lines = [[t for t in ln if t[0] != 'X'] for ln in d['lines']]
        lines = [ln for ln in lines if ln]
        if any(t[0] in ('W', 'W1') for ln in lines for t in ln):
            out.append(dict(id=d['id'], site=d['site'], series=d['series'], lines=lines))
    return out


def lb_sample(D, ntok, rng):
    """Random KN+PY documents until the word-token count reaches ntok (Linear A size)."""
    idx = list(range(len(D))); rng.shuffle(idx)
    out, n = [], 0
    for i in idx:
        out.append(D[i]); n += sum(1 for ln in D[i]['lines'] for t in ln if t[0] in ('W', 'W1'))
        if n >= ntok:
            break
    return out


def is_word(t):
    return t[0] in ('W', 'W1')


# ------------------------------------------------------------------ structural features
def _numbin(v):
    if v is None: return 0
    if v < 1: return 1
    if v == 1: return 2
    if v <= 4: return 3
    if v <= 19: return 4
    if v <= 99: return 5
    return 6


def occurrences(docs):
    """One record per word token with structural features only."""
    occ = []
    for di, d in enumerate(docs):
        lines = d['lines']
        nl = len(lines)
        run = []          # entry numbers since the last closure
        first_word_seen = False
        for li, ln in enumerate(lines):
            line_closed = False
            for j, t in enumerate(ln):
                if not is_word(t):
                    continue
                # previous token kind in line (dividers skipped)
                prev = 0
                for q in range(j - 1, -1, -1):
                    if ln[q][0] == 'D': continue
                    prev = {'W': 1, 'W1': 1, 'N': 2, 'L': 3}.get(ln[q][0], 0); break
                if not first_word_seen and prev == 0:
                    slot = 0
                elif prev == 0:
                    slot = 1
                elif prev == 1:
                    slot = 2
                else:
                    slot = 3
                first_word_seen = True
                nxt = 3
                for q in range(j + 1, len(ln)):
                    if ln[q][0] == 'D': continue
                    nxt = {'N': 0, 'L': 1, 'W': 2, 'W1': 2}.get(ln[q][0], 3); break
                num = None
                for q in range(j + 1, len(ln)):   # first number later in the same line (words may intervene)
                    if ln[q][0] == 'N': num = ln[q][1]; break
                clo = 0
                if num is not None and len(run) >= 2 and num >= 3:
                    s = 0
                    for k in range(len(run) - 1, -1, -1):
                        s += run[k]
                        if len(run) - k >= 2 and (s == num or (s >= 20 and abs(s - num) <= 1)):
                            clo = 1; break
                    if clo == 0:
                        S = sum(run)
                        if num >= max(run) and 0.7 * S <= num <= 1.3 * S:
                            clo = 2
                linepos = 0 if li == 0 else (2 if li == nl - 1 else 1)
                occ.append(dict(doc=di, line=li, w=t[1], slot=slot, nxt=nxt, numbin=_numbin(num), num=num,
                                clo=clo, linepos=linepos, prev=prev, nlines=nl))
                if clo == 1:
                    line_closed = True
            # update entry-number run with this line's entry numbers (N not preceded by N)
            for j, t in enumerate(ln):
                if t[0] == 'N':
                    pk = None
                    for q in range(j - 1, -1, -1):
                        if ln[q][0] == 'D': continue
                        pk = ln[q][0]; break
                    if pk != 'N':
                        run.append(t[1])
            if line_closed:
                run = []
    return occ


def occ_viol(o):
    """Violation count per role for one occurrence (fixed type rules, identical for all corpora)."""
    v = np.zeros(R)
    lone = (o['numbin'] == 0 and o['nxt'] == 3 and o['slot'] != 0)
    v[0] = lone                                                     # P: an entry name carries a count or heads a phrase
    v[1] = lone                                                     # L: same rule as P (P/L split only post hoc)
    v[2] = not (o['nxt'] == 1 or o['prev'] == 3 or o['slot'] == 2)  # C: touches a logogram or qualifies a word
    v[3] = (o['slot'] == 1 and o['nxt'] == 0) or (o['linepos'] == 2 and o['nlines'] > 1 and o['slot'] != 0)  # H: not a counted list line; precedes the list
    v[4] = (o['numbin'] == 0) or (o['clo'] == 0 and o['linepos'] != 2)  # T: has a number that closes, or is the last line
    v[5] = o['slot'] in (0, 1) or not (o['prev'] == 2 or o['nxt'] == 0)  # M: inside an entry, next to a number
    return v


def _tbin(n):
    return 0 if n == 1 else 1 if n == 2 else 2 if n <= 5 else 3


def build(docs, lam=3.0):
    occ = occurrences(docs)
    types = sorted(set(o['w'] for o in occ))
    ti = {w: i for i, w in enumerate(types)}
    nT, nO = len(types), len(occ)
    otype = np.array([ti[o['w']] for o in occ], np.int32)
    ofeat = np.array([[o['slot'], o['nxt'], o['numbin'], o['clo'], o['linepos'], o['prev']] for o in occ], np.int32).ravel()
    ohead = np.array([o['doc'] if (o['slot'] in (0, 1) and o['numbin'] > 0) else -1 for o in occ], np.int32)
    viol = np.zeros((nT, R))
    nd = collections.defaultdict(set)
    for o, t in zip(occ, otype):
        viol[t] += occ_viol(o)
        nd[t].add(o['doc'])
    viol *= lam
    tbin = np.array([_tbin(len(nd[t])) for t in range(nT)], np.int32)
    ndocs = np.array([len(nd[t]) for t in range(nT)])
    nocc = np.bincount(otype, minlength=nT)
    return dict(occ=occ, types=types, ti=ti, otype=otype, ofeat=ofeat, ohead=ohead, viol=viol.ravel().astype(np.float64),
                tbin=tbin, ndocs=ndocs, nocc=nocc, nD=len(docs))


def run_chains(B, nchains=16, steps=1_000_000, burn=None, thin=2000, seed0=1, alpha=0.5, beta=1.0, lam_h=3.0, T0=8.0,
               hetero=(1, 1, 1, 0, 0, 0), viol=None):
    nT = len(B['types'])
    burn = burn if burn is not None else steps // 2
    maxs = (steps - burn) // thin + 1
    card = np.array(CARD, np.int32)
    hm = np.array(hetero, np.int32)
    V = B['viol'] if viol is None else viol
    allS = []
    for c in range(nchains):
        rng = np.random.default_rng(seed0 * 1000 + c)
        assign = rng.integers(0, R, nT).astype(np.int32)
        S = np.zeros((maxs, nT), np.int32)
        ns = _lib.run_chain(nT, R, len(CARD), card, len(B['otype']), B['otype'], B['ofeat'], B['ohead'], B['nD'], V,
                            B['tbin'], TCARD, hm, alpha, beta, lam_h, steps, burn, thin, T0,
                            seed0 * 7919 + c + 1, assign, S, maxs)
        allS.append(relabel(S[:ns], B))
    return np.stack(allS)  # chains x samples x types


def relabel(S, B):
    """Convention: of the two entry roles, L is the one whose types recur on more documents (P/L rules are identical)."""
    S = S.copy()
    lg = np.log(B['ndocs'])
    for i in range(len(S)):
        a = S[i] == 0; b = S[i] == 1
        if a.sum() and b.sum() and lg[a].mean() > lg[b].mean():
            S[i][a] = 1; S[i][b] = 0
    return S


def summarize(S):
    """Per type: pooled modal role, its frequency, and the share of chains whose own mode agrees."""
    C, N, T = S.shape
    cnt = np.zeros((T, R))
    for r in range(R):
        cnt[:, r] = (S == r).sum(axis=(0, 1))
    mode = cnt.argmax(1); freq = cnt.max(1) / (C * N)
    cm = np.zeros((C, T), int)
    for c in range(C):
        cc = np.stack([(S[c] == r).sum(0) for r in range(R)], 1)
        cm[c] = cc.argmax(1)
    agree = (cm == mode[None, :]).mean(0)
    return mode, freq, agree, cnt / (C * N)


def forced_table(B, S, thr=0.9, minocc=2):
    mode, freq, agree, P = summarize(S)
    rows = []
    for t, w in enumerate(B['types']):
        if B['nocc'][t] >= minocc and freq[t] >= thr and agree[t] >= thr:
            rows.append((w, ROLES[mode[t]], float(freq[t]), float(agree[t]), int(B['nocc'][t]), int(B['ndocs'][t])))
    return rows, mode, freq, agree


def shuffle_words(docs, rng, by_site=True):
    """Null: word strings permuted across all word slots (within site), structure kept."""
    groups = collections.defaultdict(list)
    for di, d in enumerate(docs):
        for li, ln in enumerate(d['lines']):
            for j, t in enumerate(ln):
                if is_word(t):
                    groups[d['site'] if by_site else 0].append((di, li, j))
    new = [dict(d, lines=[list(ln) for ln in d['lines']]) for d in docs]
    for g, pos in groups.items():
        words = [docs[di]['lines'][li][j] for di, li, j in pos]
        rng.shuffle(words)
        for (di, li, j), w in zip(pos, words):
            new[di]['lines'][li][j] = w
    return new


# ------------------------------------------------------------------ Linear B labels (control scoring only)
LB_PLACES = set(x.upper() for x in '''pu-ro pi-*82 me-ta-pa pe-to-no pa-ki-ja-ne a-pu2 a-ke-re-wa ro-u-so ka-ra-do-ro ri-jo
ti-mi-to-a-ke-e ra-u-ra-ta sa-ma-ra a-si-ja-ti-ja e-ra-te-re-wa za-ma-e-wi-ja e-sa-re-wi-ja re-u-ko-to-ro pa-ki-ja-ni-ja
e-ke-ra-ne ne-do-wo-ta ku-pa-ri-so ko-ri-to a-te-re-wa-ja me-te-to i-te-re-wa za-e-to-ro sa-ri-no-te e-na-po-ro a-pu2-we
ro-o-wa a-ka-wo-ne o-wi-to-no ti-no e-ra-po-ri-ja u-pa-ra-ki-ja a-ra-tu-a re-ka-ta-ne de-we-ro e-re-e pi-*82-de pu-ro-de
ko-no-so a-mi-ni-so pa-i-to ku-do-ni-ja tu-ri-so da-wo e-ko-so se-to-i-ja ra-to su-ri-mo u-ta-no ka-ta-ra-i ri-jo-no ru-ki-to
ku-ta-to qa-ra ti-ri-to pu-na-so da-*22-to e-ra a-pa-ta-wa tu-ni-ja si-ra-ri-ja do-ti-ja ra-su-to pu-so wi-na-to
ka-u-ma-ti-ja ku-ta-i-to qa-mo ra-ja di-ka-ta a-mi-ni-so-de ko-no-so-de pa-i-to-de da-*83-ja da-*22-ti-ja e-ra-de
ku-ni-su pe-ri-ta ra-ti-ja tu-ti-ja e-ki-no-jo-de e-ki-no-jo ka-ra-mo ma-ro-pi o-ru-ma-to pa-ra-ke-we te-ta-ra-ne
po-ra a-ke-re-wa-de ro-u-so-de ka-ra-do-ro-de me-ta-pa-de pe-to-no-de pa-ki-ja-si a-po-ne-we u-ru-pi-ja-jo
ti-mi-to a-ke-e za-ma-e-wi-ja ru-ke-wo-do-so a-ke-ne-u'''.split())
LB_HEAD = set(x.upper() for x in '''a-pu-do-si o-pe-ro o-pe-ra o-pe-ro-sa o-pe-ro-ta de-de-ku-me-na do-so-mo di-do-si
o-u-di-do-si e-ke e-ko-si e-ke-qe o-na-to ko-to-na ki-ti-me-na ke-ke-me-na pa-ro a-ke-re a-ko-ra ta-ra-si-ja a-pe-o-te
a-pe-e-si a-pe-a-sa a-pe-o-ta e-ne-ka wo-ze wo-zo-te o-u-wo-ze wo-zo-me-no pe-re a-pe-do-ke e-qe-o a-pu-ke-ka ke-ra-me-ja
o-ne-ra pa-ra-ke-ta-wo do-ke qe-te-a qe-te-a2 a-ke-ra2-te pe-ri-te ta-ra-si-ja i-je-to a-ke-ro de-ka-sa-to qe-te-jo'''.split())
LB_TOTAL = set(x.upper() for x in 'to-so to-sa to-so-de to-sa-de to-so-pa to-sa-pa to-so-jo to-o'.split())
LB_COMM = set(x.upper() for x in '''ko-ri-ja-do-no ko-ri-a2-da-na ku-mi-no sa-sa-ma ma-ra-tu-wo se-ri-no ku-pa-ro mi-ta ka-na-ko po-ni-ki-jo
pe-ma pe-mo e-ra-wo me-ri wo-no ri-no ka-po e-ra3-wo ne-wa ne-wo pa-ra-ja pa-ra-jo ke-ra-e-wo e-ru-ta-ra ku-ru-so a-ka-ra-no
sa-pi-ti-ne-wo ko-no a-ro-mo ko-ro-to ki-ta-no ka-ka-re-a re-u-ka ki-ri-ta po-ka po-ku-ta pa-we-a pa-we-a2 tu-na-no ko-to-no
ki-to e-ne-wo-pe-za e-ru-to-ro ku-te-so e-rja pe-ru-si-nu-wo pe-ru-si-nwa ko-wa ko-wo e-pi-ko-wo ka-ko ku-ru-so-jo'''.split()) - {'KO-WA', 'KO-WO'}


def lb_labels():
    pers = L23.lb_name_label()
    lab = {}
    for w in pers: lab[w] = 'P'
    for w in LB_PLACES: lab[w] = 'L'
    for w in LB_COMM: lab[w] = 'C'
    for w in LB_HEAD: lab[w] = 'H'
    for w in LB_TOTAL: lab[w] = 'T'
    return lab
