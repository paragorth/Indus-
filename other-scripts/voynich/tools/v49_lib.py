"""v49 THE WORDS THAT BREAK THE RULES ARE THE CONTENT: shared code.

The v45 rule model (v26 minimal genome + backoff, 2-fold cross-fit by pages) marks every word that lies outside
every choice the rules allow at its slot ('off-table', PIT u < 0). Those words are pulled out as a separate lexicon,
split by line role (paragraph-first line vs body), and tested for behaving like subject terms / borrowed names.

Corpora (all through the identical v45 crossfit):
  V, VI          Voynich ZL3b / IT2a (v45 checkpoints)
  GEN0, GEN1, PL v45 generators (no message)
  GENN           GEN0 + planted rare-word noise (novel spliced words; 20% of paragraph-first-line tokens, 6% of body)
  GENT           GEN0 + planted subject terms (positive control for the term tests): each paragraph gets a novel
                 term in its first line (early), repeated 1-2 times later in the paragraph; 40% of terms are shared
                 with another page of the same section
  BRe            REAL vernacular herbal with Latin names: Brumati, Flora medico-economica (1844, Italian, Latin
                 binomials in every entry), each entry = one paragraph, wrapped 9 words/line, pushed through a planted
                 rule-bound glyph encoding (letter -> Voynich glyph chunk, neighbour-conditioned k/t twin, positional
                 ch->sh twin in paragraph-first lines and line-initial words). Ground truth: Latin binomial tokens.
"""
import os, sys, re, json, math, random, html, unicodedata, glob
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v45_lib as L45
from v45_lib import crossfit, meta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(ROOT, 'data', 'v49_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ normalisation (v48 E3 ladder, GLY units)
E3 = str.maketrans({'k': 't', 'f': 'p', 'K': 't', 'T': 't', 'F': 'p', 'P': 'p', 'S': 'C'})


def norm(w):
    return w.translate(E3)


# E3 + all four gallows merged (p, f -> t; benched cph, cfh -> cth): removes the paragraph-gallows choice
GM = str.maketrans({'k': 't', 'f': 't', 'p': 't', 'K': 'T', 'F': 'T', 'P': 'T', 'S': 'C'})


def gnorm(w):
    return w.translate(GM)


# ------------------------------------------------------------------ Brumati herbal
def _ascii(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn')


def brumati_entries():
    """Entries of Brumati's Flora (all classes): list of dict(cls, paras=[text...]) where an entry begins at a
    numbered species header ('155. M. verde.') or a genus header ('CIX. Menta.')."""
    files = sorted(glob.glob(os.path.join(SCR, 'v21', 'flora', '*.html')))
    roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
             'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV']
    out = []
    for f in sorted(files, key=lambda x: roman.index(os.path.basename(x)[:-5]) if os.path.basename(x)[:-5] in roman else 99):
        cls = os.path.basename(f)[:-5]
        t = open(f, encoding='utf-8', errors='replace').read()
        i = t.find('Informazioni sulla fonte'); t = t[i:]
        t = re.sub(r'<style.*?</style>', '', t, flags=re.S)
        t = re.sub(r'<span[^>]*data-mw[^>]*>.*?</span>', '', t, flags=re.S)
        t = re.sub(r'<(p|br|div|dd|dt|li|h\d)[^>]*>', '\n', t)
        t = re.sub(r'<[^>]+>', '', t); t = html.unescape(t)
        t = re.sub(r'\[p\. \d+[^\]]*\]', ' ', t)
        t = re.sub(r'[ \t]+', ' ', t)
        j = t.find('Classe '); t = t[j:]
        k = t.find('Note');
        paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', t)]
        paras = [p for p in paras if p and not p.startswith(('{', '.mw', 'Questo testo', 'Classe ', 'Or.'))]
        cur = None
        for p in paras:
            if re.match(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', p):
                if cur: out.append(cur)
                cur = dict(cls=cls, paras=[p])
            elif cur is not None:
                cur['paras'].append(p)
        if cur: out.append(cur)
    return [e for e in out if sum(len(p.split()) for p in e['paras']) >= 15]


def latin_name(entry):
    """Ground truth: the Latin binomial = the text before ' It.' / ' Ver.' / ' Off.' in the first sub-paragraph that
    has one (Brumati writes 'Mentha viridis. It. Menta comune...'). Returns a set of lower-case tokens."""
    for p in entry['paras'][1:4]:
        m = re.match(r'^([A-Z][a-z]+(?: [a-zA-Z]+){0,3})\.\s+(It|Ver|Off|Ted)\b', _ascii(p))
        if m: return set(w.lower() for w in m.group(1).split())
        m = re.match(r'^([A-Z][a-z]+ [a-z]+)\.', _ascii(p))
        if m and m.group(1).split()[0].endswith(('a', 'us', 'um', 'is', 'on', 'es', 'ia', 'e')): return set(w.lower() for w in m.group(1).split())
    return set()


def brumati_words(text):
    t = _ascii(text).lower().replace("'", ' ').replace('’', ' ')
    return [w for w in re.findall(r'[a-z]+', t) if len(w) >= 1]


# planted rule-bound encoding: letter -> glyph chunk
ENC = dict(a='a', e='o', i='e', o='y', u='ee', n='iin', r='r', l='l', t='t', s='d', c='C', d='ar', m='m', p='p',
           v='f', g='g', b='T', f='P', h='s', q='q', z='K', x='F', j='ei', k='Ca', w='Ce', y='oy')


def encode_word(w, pf, first, rng):
    out = ''.join(ENC[c] for c in w)
    out = re.sub(r'(?<=o)t', 'k', out)                 # neighbour-conditioned twin: t after o is written k
    if (pf or first) and rng.random() < 0.6:            # positional twin, as v40 ch/sh
        out = out.replace('C', 'S')
    if out[0] in 'eio' and len(out) > 1 and rng.random() < 0.5: out = 'q' + out   # optional word-initial q- mark
    return out


def brumati_corpus(seed=49, cap_tokens=40000, line_w=9, front=False):
    rng = random.Random(seed)
    ents = brumati_entries()
    pages = []; cur = None; ntok = 0
    groups = {}
    for e in ents:
        r = e['cls']
        groups.setdefault(r, len(groups))
    for e in ents:
        if ntok >= cap_tokens: break
        lat = latin_name(e)
        ws, lab = [], []
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            pw = brumati_words(p)
            nlat = 0
            if lat and pi in (1, 2, 3) and not any(lab):
                while nlat < len(pw) and pw[nlat] in lat: nlat += 1
            ws += pw; lab += [i < nlat for i in range(len(pw))]
        if len(ws) < 15: continue
        if front and any(lab):
            i0 = lab.index(True); i1 = i0
            while i1 < len(lab) and lab[i1]: i1 += 1
            ws = ws[i0:i1] + ws[:i0] + ws[i1:]; lab = lab[i0:i1] + lab[:i0] + lab[i1:]
        lines = [ws[i:i + line_w] for i in range(0, len(ws), line_w)]
        if len(lines[-1]) < 2 and len(lines) > 1: lines[-2] += lines.pop()
        labs = [lab[i:i + line_w] for i in range(0, len(lab), line_w)]
        if len(labs[-1]) < 2 and len(labs) > 1: labs[-2] += labs.pop()
        para_enc = []; para_plain = []
        for li, ln in enumerate(lines):
            enc = [encode_word(w, li == 0, k == 0, rng) for k, w in enumerate(ln)]
            para_enc.append(enc); para_plain.append([(w, b) for w, b in zip(ln, labs[li])])
        sec = 'G%d' % min(5, groups[e['cls']] // 4)
        if cur is None or cur['_n'] >= 180 or cur['sec'] != sec:
            cur = dict(id='br%03d' % len(pages), sec=sec, cls=e['cls'], paras=[], plain=[], _n=0); pages.append(cur)
        cur['paras'].append(para_enc); cur['plain'].append(para_plain); cur['_n'] += len(ws); ntok += len(ws)
    for p in pages: p.pop('_n')
    return pages


# ------------------------------------------------------------------ generator controls
def novel_word(vocab_list, vocab, rng):
    for _ in range(100):
        a = vocab_list[rng.randrange(len(vocab_list))]; b = vocab_list[rng.randrange(len(vocab_list))]
        if len(a) < 2 or len(b) < 2: continue
        w = a[:rng.randint(1, len(a) - 1)] + b[rng.randint(1, len(b) - 1):]
        if w not in vocab and len(w) >= 3: return w
    return 'qoTeCdy' + str(rng.randrange(9))


def gen_noise(C, seed=491, r_pf=0.20, r_body=0.06):
    rng = random.Random(seed)
    voc = [w for p in C for pa in p['paras'] for l in pa for w in l]
    vs = set(voc)
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for li, l in enumerate(pa):
                r = r_pf if li == 0 else r_body
                npa.append([novel_word(voc, vs, rng) if rng.random() < r else w for w in l])
            q['paras'].append(npa)
        out.append(q)
    return out


def gen_terms(C, seed=492):
    rng = random.Random(seed)
    voc = [w for p in C for pa in p['paras'] for l in pa for w in l]
    vs = set(voc)
    bysec = defaultdict(list)
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            if bysec[p['sec']] and rng.random() < 0.4:
                term = bysec[p['sec']][rng.randrange(len(bysec[p['sec']]))]
            else:
                term = novel_word(voc, vs, rng); vs.add(term); bysec[p['sec']].append(term)
            npa = [list(l) for l in pa]
            if npa and npa[0]:
                k = min(len(npa[0]) - 1, rng.choice([0, 1, 1, 2]))
                npa[0][k] = term
                for _ in range(rng.randint(1, 2)):
                    if len(npa) > 1:
                        li = rng.randrange(1, len(npa));
                        if npa[li]: npa[li][rng.randrange(len(npa[li]))] = term
            q['paras'].append(npa)
        out.append(q)
    return out


# ------------------------------------------------------------------ corpora + residuals
def get_corpus(name):
    p = os.path.join(CK, f'corpus_{name}.json')
    if os.path.exists(p): return json.load(open(p))
    if name in ('V', 'VI', 'GEN0', 'GEN1', 'PL'): return L45.get_corpus(name)
    if name == 'GENN': C = gen_noise(L45.get_corpus('GEN0'))
    elif name == 'GENT': C = gen_terms(L45.get_corpus('GEN0'))
    elif name.endswith('_G'):
        C = []
        for pg in get_corpus(name[:-2]):
            q = dict(pg); q['paras'] = [[[gnorm(w) for w in l] for l in pa] for pa in pg['paras']]; C.append(q)
    elif name == 'BRe': C = brumati_corpus()
    elif name == 'BRf': C = brumati_corpus(front=True)
    else: raise ValueError(name)
    json.dump(C, open(p, 'w'))
    return C


def residual(name):
    if name in ('V', 'VI', 'GEN0', 'GEN1', 'PL'): return L45.residual(name)
    p = os.path.join(CK, f'resid_{name}.json')
    if os.path.exists(p): return json.load(open(p))
    C = get_corpus(name)
    R = crossfit(C)
    out = {pid: dict(recs=r) for pid, (r, e) in R.items()}
    json.dump(out, open(p, 'w'))
    return out


def illus_of(name, C):
    """'Illustration type' label per page: Voynich illus letter (with Currier language as stratum); controls: sec."""
    if name.split('_')[0] in ('V', 'VI'):
        m = meta()
        return {p['id']: (m.get(p['id'], {}).get('illus') or '?', m.get(p['id'], {}).get('lang') or '?') for p in C}
    return {p['id']: (p['sec'], '-') for p in C}


def tokens_table(name):
    """One record per token: page, para, line, k, n, word, off(bool), role ('pf'/'body'), latin (BRe only)."""
    C = get_corpus(name); R = residual(name)
    out = []
    for p in C:
        if p['id'] not in R: continue
        recs = R[p['id']]['recs']
        plain = p.get('plain')
        for r in recs:
            w, qi, li, k, n = r[0], r[1], r[2], r[3], r[4]
            lat = bool(plain[qi][li][k][1]) if plain else None
            pw = plain[qi][li][k][0] if plain else None
            out.append(dict(page=p['id'], para=qi, line=li, k=k, n=n, w=w, off=r[8] < 0, role='pf' if li == 0 else 'body',
                            lat=lat, pw=pw))
    return out
