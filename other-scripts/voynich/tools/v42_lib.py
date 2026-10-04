"""v42 THE VOYNICH AMONG DELIBERATELY MEANINGLESS BOOKS: shared library.

New reference object: the Codex Seraphinianus (Serafini 1976-78, ~360 pages written by one hand, declared asemic
by its author). Data: M. Ponzi's neural-network transliteration (github.com/marcoponzi/codex_seraphinianus_ocr,
CS_OCR_TRANSLITERATION.txt, 33 stroke-level units, validation character error ~18%), kept in the scratchpad.
It is used as raw symbol data only.

Because the CS text is machine-read with ~18% character errors, every comparison is also made against
NOISE-MATCHED versions of the other corpora: the same OCR-like channel (substitution with the unit's nearest
context twin, unit duplication, unit deletion, word split / merge) applied to languages, generators, gibberish
and the Voynich.

Corpus = list of documents; document = list of lines; line = list of words (strings of one-character units).
"""
import os, sys, json, re, random, math
import numpy as np
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v42_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v42'
CSF = os.path.join(SCR, 'cs', 'CS_OCR_TRANSLITERATION.txt')
V2_FIRST = 179          # first PDF page of the second volume (Ponzi)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ Codex Seraphinianus
def cs_pages(drop_num=True):
    """-> list of pages {'id','page','sec','paras':[[line,...],...]}; a '#' text area = a paragraph.
    '.' inside a box = word break (Ponzi); tokens with '?' dropped; tokens with Z (numeral glyph) dropped."""
    pages = {}; order = []; cur = None; pg = None
    for ln in open(CSF, encoding='utf-8'):
        ln = ln.rstrip('\n')
        m = re.match(r'###PAGE\s+(\d+)', ln)
        if m:
            n = int(m.group(1)); pid = f'p{n:03d}'
            pg = pages.setdefault(pid, {'id': pid, 'page': n, 'sec': 'V1' if n < V2_FIRST else 'V2', 'paras': []})
            if pid not in order: order.append(pid)
            cur = None; continue
        if ln.startswith('#'):
            cur = None; continue
        if pg is None or not ln.strip(): continue
        ws = [w for w in re.split(r'[ .]+', ln.strip()) if w]
        ws = [w.replace('id', 'n') if False else w for w in ws]
        ws = [w for w in ws if '?' not in w and not (drop_num and 'Z' in w) and re.fullmatch(r'[A-Za-z]+', w)]
        if not ws: continue
        if cur is None: cur = []; pg['paras'].append(cur)
        cur.append(ws)
    return [pages[p] for p in order if pages[p]['paras']]


def pages_docs(P, key=None):
    """pages -> documents (one per section or per page)."""
    docs = defaultdict(list)
    for p in P:
        k = p['sec'] if key == 'sec' else ('*' if key is None else p[key])
        docs[k].extend(l for pa in p['paras'] for l in pa)
    return list(docs.values())


def collapse(word):
    """Ponzi's 'single' scheme: runs of a repeated unit -> one unit (removes minim-count noise)."""
    return re.sub(r'(.)\1+', r'\1', word)


def map_docs(docs, f):
    return [[[f(w) for w in l if f(w)] for l in d] for d in docs]


# ------------------------------------------------------------------ OCR-like noise channel
def twins(docs):
    """each unit's nearest unit by left/right context profile (visual twins are context twins in a script)."""
    ctx = defaultdict(Counter)
    for d in docs:
        for l in d:
            for w in l:
                x = '^' + w + '$'
                for i in range(1, len(x) - 1):
                    ctx[x[i]]['L' + x[i - 1]] += 1; ctx[x[i]]['R' + x[i + 1]] += 1
    units = [u for u in ctx if sum(ctx[u].values()) >= 5]
    keys = sorted({k for u in units for k in ctx[u]})
    M = np.array([[ctx[u][k] for k in keys] for u in units], float)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    S = M @ M.T; np.fill_diagonal(S, -1)
    tw = {u: units[int(S[i].argmax())] for i, u in enumerate(units)}
    return tw


def noisy_word(w, rng, tw, rate):
    out = []
    ps, pd, pi = rate * 0.5, rate * 0.25, rate * 0.25
    for c in w:
        r = rng.random()
        if r < ps: out.append(tw.get(c, c))
        elif r < ps + pd: continue
        elif r < ps + pd + pi: out.append(c); out.append(c)
        else: out.append(c)
    return ''.join(out)


def noise_docs(docs, rate, seed=0, split=0.01, merge=0.01):
    """OCR channel at character error `rate` (half substitutions with the context twin, a quarter deletions,
    a quarter duplications), plus word splits / merges at 1% each."""
    if rate <= 0: return docs
    rng = random.Random(seed); tw = twins(docs)
    out = []
    for d in docs:
        nd = []
        for l in d:
            nl = []
            for w in l:
                v = noisy_word(w, rng, tw, rate)
                if not v: continue
                if len(v) >= 4 and rng.random() < split:
                    k = rng.randint(1, len(v) - 1); nl += [v[:k], v[k:]]
                elif nl and rng.random() < merge:
                    nl[-1] = nl[-1] + v
                else: nl.append(v)
            if nl: nd.append(nl)
        out.append(nd)
    return out


# ------------------------------------------------------------------ corpora
def voynich_docs(name, part=None, minw=10):
    import v21_lib as V
    P = V.voynich_pages(name, minw=minw)
    if part: P = [p for p in P if p['sec'].endswith(part)]
    docs = {}
    for p in P: docs.setdefault(p['sec'], []).extend(l for pa in p['paras'] for l in pa)
    return list(docs.values())


def all_corpora():
    """v31 training corpora + generators + Voynich + Codex Seraphinianus. -> {name: (cls, docs)}"""
    import v31_lib as L31, v31_gen as G
    C = json.load(open(L31.CORPORA))
    out = {k: (v['cls'], v['docs']) for k, v in C.items() if v['cls'] != 'TEST'}
    for k, d in G.build(C).items(): out[k] = ('GEN', [d])
    out['V_ZL'] = ('TEST', voynich_docs('ZL3b'))
    out['V_IT'] = ('TEST', voynich_docs('IT2a'))
    P = cs_pages()
    out['S_CS'] = ('TEST', pages_docs(P, 'sec'))
    out['S_CS1'] = ('TEST', pages_docs([p for p in P if p['sec'] == 'V1']))
    out['S_CS2'] = ('TEST', pages_docs([p for p in P if p['sec'] == 'V2']))
    return out
