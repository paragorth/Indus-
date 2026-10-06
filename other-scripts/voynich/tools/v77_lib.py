"""v77 THE KILL SWEEP: shared helpers.

Every grade C Voynich guess in FINDINGS.md is put to its own stated kill test (or the strongest test possible on
existing data), on ZL3b and IT2a, discovery/held-out leaves, and on fitted generator text, so that a 'survive'
means more than generator statistics.

Corpus format (as v72_lib): list of pages; page = dict(id, sec, lang, hand, quire, lines=[dict(w=[words], ps=bool)]).
Words are strings of glyph units (C=ch S=sh T=cth K=ckh P=cph F=cfh, lowercase EVA otherwise).

Generators (all fitted per group = section|Currier language unless stated):
  SELFCIT  copy a word from the last 60 words of the page, one-glyph change with p 0.5, else draw (v72)
  SC10     the same with copy rate 0.10
  MK2      glyph-trigram resynthesis with a separate model for line-initial words (v72)
  JUNC     junction resynthesis (next word from words that follow the same last glyph), line-initial pool (v72)
  STACK    new here: the strongest line-aware copy-and-vary generator we can build without meaning:
           positional pools (paragraph-first line / body; first / interior / last word), junction draws for
           interior words, vertical copy-and-vary from the line above at the same position (p 0.25) and
           within-page self-citation (p 0.25), each copy changed in one glyph with p 0.4.
"""
import os, sys, json, math, random, re, collections, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v77_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V72

GALL = set('ktpfTKPF')
FRAME = set('oainylr')


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


_CACHE = {}


def voy(name='ZL3b'):
    if name not in _CACHE:
        _CACHE[name] = V72.voynich(name)
    return _CACHE[name]


leaf_half = V72.leaf_half


def _cum(c):
    ks = list(c); v = np.cumsum([c[k] for k in ks]).astype(float); return ks, v / v[-1]


def _draw(kv, rng):
    ks, cp = kv
    return ks[min(int(np.searchsorted(cp, rng.random())), len(ks) - 1)]


def _grouped(pages, key):
    """temporarily set sec to the grouping key so v72 generators fit per group."""
    out = []
    for p in pages:
        q = dict(p); q['_sec'] = p['sec']
        q['sec'] = key(p); out.append(q)
    return out


def _ungroup(pages):
    out = []
    for p in pages:
        q = dict(p); q['sec'] = q.pop('_sec'); out.append(q)
    return out


def gkey(p):
    return '%s|%s' % (p['sec'], p.get('lang', '-'))


def mutate(w, gl, rng):
    j = rng.randrange(len(w))
    return w[:j] + _draw(gl, rng) + w[j + 1:]


def gen_stack(pages, seed=1, p_vert=0.25, p_cite=0.25, p_mod=0.4, window=60):
    rng = random.Random(seed)
    POOL = defaultdict(list); FOL = defaultdict(list)
    GL = defaultdict(Counter)
    for p in pages:
        g = p['sec']
        for l in p['lines']:
            n = len(l['w'])
            for i, w in enumerate(l['w']):
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                POOL[(g, l['ps'], pc)].append(w)
                GL[g].update(w)
                if i > 0: FOL[(g, l['ps'], pc, l['w'][i - 1][-1])].append(w)
    GLc = {g: _cum(c) for g, c in GL.items()}
    out = []
    for p in pages:
        g = p['sec']; hist = []; nl = []; prev = None
        for l in p['lines']:
            n = len(l['w']); ws = []
            for i in range(n):
                pc = 'F' if i == 0 else ('L' if i == n - 1 else 'M')
                u = rng.random()
                if prev is not None and u < p_vert and len(prev) > 0:
                    j = min(i, len(prev) - 1) if pc != 'L' else len(prev) - 1
                    w = prev[j]
                    if rng.random() < p_mod: w = mutate(w, GLc[g], rng)
                elif len(hist) >= 3 and u < p_vert + p_cite:
                    w = rng.choice(hist[-window:])
                    if rng.random() < p_mod: w = mutate(w, GLc[g], rng)
                else:
                    c = None
                    if i > 0: c = FOL.get((g, l['ps'], pc, ws[-1][-1]))
                    if not c: c = POOL.get((g, l['ps'], pc)) or POOL[(g, False, 'M')] or POOL[(g, True, 'M')]
                    w = rng.choice(c)
                ws.append(w); hist.append(w)
            nl.append(dict(l, w=ws)); prev = ws
        out.append(dict(p, lines=nl))
    return out


GENS = dict(SELFCIT=V72.gen_selfcit, SC10=lambda pg, seed=1: V72.gen_selfcit(pg, seed, p_copy=0.10),
            MK2=V72.gen_mk2, JUNC=V72.gen_junction, STACK=gen_stack)


def generate(name, gen, seed, key=gkey):
    P = voy(name)
    return _ungroup(GENS[gen](_grouped(P, key), seed=seed))


def texts(gens=('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'), seeds=(771,), names=('ZL3b', 'IT2a'), gen_on=('ZL3b',)):
    """yield (label, pages): the real texts, then generators fitted to gen_on with fresh seeds."""
    for n in names:
        yield n, voy(n)
    for n in gen_on:
        for g in gens:
            for s in seeds:
                yield '%s:%s:%d' % (n[:2], g, s), generate(n, g, s)


def lines_of(pages, half=None, drop_ps=False):
    for p in pages:
        if half is not None and leaf_half(p['id']) != half: continue
        for li, l in enumerate(p['lines']):
            if drop_ps and l['ps']: continue
            yield p, li, l


def to_v61(pages):
    out = []
    for p in pages:
        L = p['lines']
        for i, l in enumerate(L):
            out.append({'words': list(l['w']), 'base': None, 'marked': None, 'page': p['id'], 'sec': p['sec'],
                        'lang': p.get('lang', '?'), 'hand': p.get('hand', '?'), 'para_start': l['ps'],
                        'para_end': (i + 1 == len(L)) or L[i + 1]['ps']})
    return out


def zgen(v, gens):
    """position of value v against a list of generator values."""
    g = np.array(gens, float)
    return float((v - g.mean()) / (g.std() + 1e-9)) if len(g) > 1 else float('nan')


def fmt(x, d=3):
    return ('%.' + str(d) + 'f') % x if isinstance(x, (float, np.floating)) else str(x)
