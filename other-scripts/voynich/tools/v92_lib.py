"""v92: WHAT DO THE PAGE-RECURRING WORDS DO?  Shared helpers.

v90 found that whole words and word frames (word minus its middle) recur on their own page beyond spelling habit.
v92 asks what those recurring units do, by tests that separate the name of a thing from habit.
Corpora are v72-format page lists: [{'id','sec','lang','hand','lines':[{'w':[str], 'ps':bool}]}], one character per
glyph unit. Plants = real medieval texts (v89 texts.json) written through a lossy merge code and the planted v72 surface.
"""
import os, sys, json, math, random, re, hashlib, pickle
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v92_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


# ------------------------------------------------------------------ units
def frame(w):
    """word minus its middle (v90 scheme 0: first 3 and last 3 glyph units kept, the rest replaced by '.')."""
    n = len(w)
    if n <= 6: return w
    return w[:3] + '.' + w[-3:]


def split_half(pid):
    """train (0) / held-out (1) folios: leaf parity (both sides of a leaf together)."""
    return V.leaf_half(pid)


# ------------------------------------------------------------------ corpora
TEXTS = None


def _texts():
    global TEXTS
    if TEXTS is None: TEXTS = json.load(open(os.path.join(DATA, 'v89_ckpt', 'texts.json')))
    return TEXTS


def entries(tid, cap=36000):
    T = _texts()[tid]; out = []; n = 0
    for u in T['units']:
        ws = [re.sub(r'[^a-z]', '', w.lower()) for w in u['tok']]
        ws = [w for w in ws if w]
        if len(ws) >= 3: out.append(ws); n += len(ws)
        if n >= cap: break
    k = len(out)
    return [('b%d' % min(5, i * 6 // k), ws) for i, ws in enumerate(out)]


def surfaced(ents, seed, prefix):
    words = [w for s, ws in ents for w in ws]
    code = V.payload_code(words, seed=seed, mode='merge')
    pages = V._pages_from_entries(ents, None, line_w=9, page_tok=160, cap=36000, prefix=prefix)
    S = V.surface(V.encode_payload(pages, code), seed=seed + 1)
    for p in S:
        for l in p['lines']: l.pop('orig', None)
    return S


PLANTS = {'P_KONRAD': 'konrad_plants', 'P_CIRCA': 'circa_instans_fr', 'P_APIC': 'apicius_eng',
          'P_HYGIN': 'hyginus_astr', 'P_CULP': 'culpeper'}


def within_page_shuffle(pages, seed):
    """all words of a page dealt at random to its word slots (line lengths kept): keeps every page-recurrence,
    destroys syntax and line structure."""
    rng = random.Random(seed); out = []
    for p in pages:
        ws = [w for l in p['lines'] for w in l['w']]; rng.shuffle(ws); it = iter(ws)
        out.append(dict(p, lines=[dict(l, w=[next(it) for _ in l['w']]) for l in p['lines']]))
    return out


def within_line_shuffle(pages, seed):
    rng = random.Random(seed); out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = l['w'][:]; rng.shuffle(ws); nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


GEN_P = {
    # lexical page mood + junction draw + onset harmony + line moods (v82/v84 family)
    'G_LX': dict(base='junc', p_vert=0.0, p_cite=0.0, p_mod=0.4, k=2, M=4, mood_src='fit', mood_beta=1.2, mood_scope='line',
                 mood_rate=0.3, agr='harm', agr_src=1, agr_lam=1.0, H=4, hseed=921, bias_copies=True, mech='lx',
                 cite_win=60, cite_scope='page', gm_beta=0.0, gm_feat='uni', gm_rho=0.0, gm_line=0.0, lx_beta=1.8,
                 lx_rho=0.0, lx_line=0.0, lx_K=0, sd_n=0, sd_p=0.0, sd_mod=0.0, sd_keep=0.0, sd_alpha=0.5),
    # per-page seed vocabulary, copy-and-vary, plus junction draw and harmony
    'G_SEED': dict(base='junc', p_vert=0.0, p_cite=0.05, p_mod=0.5, k=2, M=4, mood_src='fit', mood_beta=1.0, mood_scope='line',
                   mood_rate=0.3, agr='harm', agr_src=1, agr_lam=1.0, H=4, hseed=922, bias_copies=True, mech='seed',
                   cite_win=60, cite_scope='page', gm_beta=0.0, gm_feat='uni', gm_rho=0.0, gm_line=0.0, lx_beta=0.0,
                   lx_rho=0.0, lx_line=0.0, lx_K=0, sd_n=8, sd_p=0.12, sd_mod=0.4, sd_keep=0.5, sd_alpha=0.5),
    # glyph page mood (spelling mood over word bodies) + stack base with vertical copies
    'G_GM': dict(base='stack', p_vert=0.1, p_cite=0.1, p_mod=0.5, k=2, M=4, mood_src='fit', mood_beta=1.0, mood_scope='line',
                 mood_rate=0.3, agr='harm', agr_src=1, agr_lam=1.0, H=4, hseed=923, bias_copies=True, mech='gm',
                 cite_win=60, cite_scope='page', gm_beta=2.0, gm_feat='bi', gm_rho=0.5, gm_line=0.1, lx_beta=0.0,
                 lx_rho=0.0, lx_line=0.0, lx_K=0, sd_n=0, sd_p=0.0, sd_mod=0.0, sd_keep=0.0, sd_alpha=0.5),
}


def corpus(name):
    """named corpus, cached."""
    c = pload('corp_%s.pkl' % name)
    if c is not None: return c
    if name in ('ZL3b', 'IT2a', 'GC2a'):
        c = V.voynich(name)
    elif name in PLANTS:
        i = sorted(PLANTS).index(name)
        c = surfaced(entries(PLANTS[name]), 9200 + 11 * i, prefix=name[2:4].lower())
    elif name.endswith('_WPS'):
        c = within_page_shuffle(corpus(name[:-4]), 9290)
    elif name.endswith('_WLS'):
        c = within_line_shuffle(corpus(name[:-4]), 9291)
    elif name == 'G_SELF':
        c = V.gen_selfcit(corpus('ZL3b'), seed=9293, p_copy=0.3)
    elif name in GEN_P:
        import v84_lib as L84
        c = L84.gen_page(corpus('ZL3b'), dict(GEN_P[name]), seed=9294 + sorted(GEN_P).index(name))
    else:
        raise KeyError(name)
    psave('corp_%s.pkl' % name, c)
    return c


def recurrence_profile(pages, unit=frame):
    """share of tokens whose unit (count 2-20 in the corpus) recurs on another line of its page."""
    cnt = Counter(unit(w) for p in pages for l in p['lines'] for w in l['w'])
    ev = hit = 0
    for p in pages:
        lines = [set(unit(w) for w in l['w']) for l in p['lines']]
        for li, l in enumerate(p['lines']):
            for w in l['w']:
                u = unit(w)
                if 2 <= cnt[u] <= 20:
                    ev += 1; hit += any(u in s for j, s in enumerate(lines) if j != li)
    return hit / max(ev, 1), ev


def sha_obj(o): return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()
