#!/usr/bin/env python3
"""LA-8 shared: build typed token streams for the grammar-evolution experiment.

Corpora (each a list of documents; a document is a list of (token, kind, key)):
  LA   Linear A administrative documents (corpus.json; tablets, nodules, roundels, lames, bars,
       sealings, labels) with >= 2 tokens. Words are sign strings used only as identifiers.
  LB   Linear B (DAMOS) documents, parsed to the same token kinds. No sound value is used:
       transliterations are identifiers only.
  PE   Proto-Elamite (CDLI-derived pe_corpus.json), one token per sign, one NUM per numeral group.
  LAS  Linear A with tokens shuffled inside each document (order control).
  PFLAT, PREC  planted synthetic corpora (a small flat form; a small recursive language).
Kinds: W1 one-sign word, W2 multi-sign word, L logogram, NUM integer, FRAC fraction/measure, S PE sign.
Key (used for novel tokens): kind + last sign for W2, kind otherwise (L: base logogram family is too
specific for novel items, so L key is just L).
"""
import json, os, random, re, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
OUT = os.path.join(D, 'la8')
PE_PATH = os.path.join(HERE, '..', '..', 'proto-elamite', 'data', 'pe_corpus.json')
ADMIN = {'Tablet', 'Nodule', 'Roundel', 'Lames (short thin tablet)', 'Sealing', 'Label', '3-sided bar', '4-sided bar'}


def tok(token, kind, last=None):
    key = kind + (':' + last if (kind == 'W2' and last) else '')
    return (token, kind, key)


def build_la():
    c = json.load(open(os.path.join(D, 'corpus.json')))
    docs = []
    for ins in c:
        if ins['support'] not in ADMIN: continue
        d = []
        for t in ins['tokens']:
            if t['t'] == 'word':
                s = t['s']
                if len(s) == 1: d.append(tok('W:' + s[0], 'W1'))
                else: d.append(tok('W:' + '-'.join(s), 'W2', s[-1]))
            elif t['t'] == 'logo':
                d.append(tok('L:' + t['v'], 'L'))
            elif t['t'] == 'num':
                if t['v'] > 0 or not t['frac']: d.append(tok('NUM', 'NUM'))
                if t['frac']: d.append(tok('FRAC', 'FRAC'))
            elif t['t'] == 'frac':
                d.append(tok('FRAC', 'FRAC'))
        if len(d) >= 2: docs.append({'id': ins['id'], 'site': ins['site'], 'toks': d})
    return docs


LB_UNITS = set('TVZSMNPQL')
LB_DROP = {',', '/', '//', ':', 'vac.', 'sup.', 'mut.', 'inf.', 'vest.', 'v.', 'v.↓', 'lat.', 'X', 'deest', 'mut', 'sup', 'inf'}


def clean_lb(t):
    t = unicodedata.normalize('NFD', t)
    t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    t = re.sub(r"[\[\]⟦⟧⌞⌟⸢⸣'\"?!<>{}]", '', t)
    return t.strip('-')


def build_lb():
    docs = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        x = json.loads(line)
        if not x.get('content'): continue
        d = []
        pending_unit = False
        for raw in x['content'].split():
            if re.match(r'^\.[0-9A-Za-z]+$', raw) or raw.startswith('.') and len(raw) <= 4: continue
            t = clean_lb(raw)
            if not t or t in LB_DROP or t.endswith('.'): continue
            if re.fullmatch(r'\d+', t):
                if pending_unit: pending_unit = False; continue
                if d and d[-1][1] == 'NUM': continue
                d.append(tok('NUM', 'NUM')); continue
            pending_unit = False
            if t in LB_UNITS:
                d.append(tok('FRAC', 'FRAC')); pending_unit = True; continue
            if any('\u0370' <= ch <= '\u03ff' for ch in t): continue
            if t.lower() in ('supra', 'infra', 'sigillum', 'graffito', 'angustum', 'dextra', 'sinistra', 'prior', 'verso', 'recto', 'margo', 'textus', 'deleted', 'erasa', 'erasure'): continue
            if t.lower() in ('vacat', 'vac', 'lat', 'sup', 'inf', 'mut', 'vest', 'deest', 'fr', 'sigillum'): continue
            if re.fullmatch(r"[A-Z*0-9;+±]+(:[mf])?", t):
                d.append(tok('L:' + t, 'L')); continue
            if any(ch.islower() for ch in t):
                s = [p for p in t.split('-') if p]
                if not s or not re.fullmatch(r'[a-z0-9*\-]+', t): continue
                if len(s) == 1 and (len(s[0]) > 3 or s[0] in ('qs', 'vs', 'fr', 'cf', 'sq', 'nm')): continue
                if len(s) == 1: d.append(tok('W:' + s[0], 'W1'))
                else: d.append(tok('W:' + '-'.join(s), 'W2', s[-1]))
                continue
            if re.fullmatch(r"[A-Z*0-9;+±:]+[A-Z0-9]*", t) and not re.fullmatch(r'\d+', t):
                d.append(tok('L:' + t, 'L')); continue
        if len(d) >= 2: docs.append({'id': x['heading'], 'site': x['heading'][:2], 'toks': d})
    return docs


def build_pe():
    pe = json.load(open(PE_PATH))
    docs = []
    for p in pe:
        if p.get('genre') not in ('Administrative', ''): continue
        d = []
        for l in p['lines']:
            for s in l['signs']:
                if s in ('x', 'X', '...'): continue
                d.append(tok('S:' + s, 'S'))
            if l['numerals']: d.append(tok('NUM', 'NUM'))
        if len(d) >= 2: docs.append({'id': p['id'], 'site': p.get('provenience', ''), 'toks': d})
    return docs


def ntok(docs): return sum(len(d['toks']) for d in docs)


def size_match(docs, target, seed):
    rng = random.Random(seed)
    idx = list(range(len(docs))); rng.shuffle(idx)
    out, n = [], 0
    for i in idx:
        if n >= target: break
        out.append(docs[i]); n += len(docs[i]['toks'])
    return out


def shuffle_within(docs, seed):
    rng = random.Random(seed)
    out = []
    for d in docs:
        t = list(d['toks']); rng.shuffle(t)
        out.append({'id': d['id'], 'site': d['site'], 'toks': t})
    return out


# ---------------- planted corpora ----------------
SYL = ['A', 'E', 'I', 'O', 'U'] + [c + v for c in 'DJKMNPQRSTWZ' for v in 'AEIOU']


def zipf_lexicon(rng, n, lens=(2, 4), suffixes=None):
    words = set()
    while len(words) < n:
        k = rng.randint(*lens)
        w = [rng.choice(SYL) for _ in range(k)]
        if suffixes: w[-1] = rng.choice(suffixes)
        words.add('-'.join(w))
    words = sorted(words); rng.shuffle(words)
    wts = [1.0 / (i + 1) ** 1.05 for i in range(n)]
    return words, wts


def wtok(w):
    s = w.split('-')
    return tok('W:' + w, 'W1' if len(s) == 1 else 'W2', s[-1] if len(s) > 1 else None)


def planted_flat(target, seed=7):
    """A form: [HEAD] [LOGO] (NAME [LOGO] NUM [FRAC])+ [TOTAL NUM]. True classes recorded."""
    rng = random.Random(seed)
    head = ['KA-RU', 'DE-ME', 'PO-NI', 'SA-TO', 'TU-KE']
    tot = ['ZO-RA', 'QE-QE']
    logo = ['L:G1', 'L:G2', 'L:G3', 'L:G4', 'L:G5', 'L:G6']
    names, nw = zipf_lexicon(rng, 3000, (2, 4))
    truth = {}
    for w in head: truth['W:' + w] = 'HEAD'
    for w in tot: truth['W:' + w] = 'TOTAL'
    for w in logo: truth[w] = 'LOGO'
    docs, n = [], 0
    while n < target:
        d = []
        if rng.random() < 0.6: d.append(wtok(rng.choice(head)))
        lg = rng.choice(logo)
        if rng.random() < 0.4: d.append(tok(lg, 'L'))
        for _ in range(rng.randint(1, 8)):
            w = rng.choices(names, nw)[0]; truth.setdefault('W:' + w, 'NAME')
            d.append(wtok(w))
            if rng.random() < 0.25: d.append(tok(rng.choice(logo), 'L'))
            d.append(tok('NUM', 'NUM'))
            if rng.random() < 0.2: d.append(tok('FRAC', 'FRAC'))
        if rng.random() < 0.4:
            d.append(wtok(rng.choice(tot))); d.append(tok('NUM', 'NUM'))
        docs.append({'id': 'F%d' % len(docs), 'site': 'P', 'toks': d}); n += len(d)
    truth['NUM'] = 'NUM'; truth['FRAC'] = 'FRAC'
    return docs, truth


def planted_rec(target, seed=11):
    """A small language with centre-embedding relative clauses.
    S -> NP VP ; NP -> DET? N [NUM] | DET? N REL NP V (object relative: centre-embedding, p=.25)
    VP -> V | V NP | V NP P NP. Open N (suffix -A/-O), open V (suffix -TE/-SI); closed DET, REL, P."""
    rng = random.Random(seed)
    det = ['TA', 'TO', 'TE']; rel = ['QI-SE', 'QO']; prep = ['PE-DA', 'KU-MI', 'NE', 'SO-WE']
    nouns, nwt = zipf_lexicon(rng, 2000, (2, 4), ['A', 'O'])
    verbs, vwt = zipf_lexicon(rng, 1200, (2, 3), ['TE', 'SI'])
    truth = {}
    for w in det: truth['W:' + w] = 'DET'
    for w in rel: truth['W:' + w] = 'REL'
    for w in prep: truth['W:' + w] = 'P'

    def N(d):
        w = rng.choices(nouns, nwt)[0]; truth.setdefault('W:' + w, 'N'); d.append(wtok(w))

    def V(d):
        w = rng.choices(verbs, vwt)[0]; truth.setdefault('W:' + w, 'V'); d.append(wtok(w))

    def NP(d, depth):
        if rng.random() < 0.5: d.append(wtok(rng.choice(det)))
        N(d)
        if rng.random() < 0.15: d.append(tok('NUM', 'NUM'))
        if depth < 4 and rng.random() < 0.25:
            d.append(wtok(rng.choice(rel))); NP(d, depth + 1); V(d)

    def VP(d, depth):
        V(d); r = rng.random()
        if r < 0.45: NP(d, depth)
        elif r < 0.7: NP(d, depth); d.append(wtok(rng.choice(prep))); NP(d, depth)

    docs, n = [], 0
    while n < target:
        d = []
        for _ in range(rng.randint(1, 2)):
            NP(d, 0); VP(d, 0)
        docs.append({'id': 'R%d' % len(docs), 'site': 'P', 'toks': d}); n += len(d)
    truth['NUM'] = 'NUM'
    return docs, truth


def build_all(seed=0):
    os.makedirs(OUT, exist_ok=True)
    la = build_la(); T = ntok(la)
    lb_full = build_lb(); pe_full = build_pe()
    corp = {'LA': la, 'LB': size_match(lb_full, T, seed + 1), 'PE': size_match(pe_full, T, seed + 2),
            'LAS': shuffle_within(la, seed + 3)}
    pf, tf = planted_flat(T); pr, tr = planted_rec(T)
    corp['PFLAT'] = pf; corp['PREC'] = pr
    info = {k: {'docs': len(v), 'tokens': ntok(v), 'types': len({t[0] for d in v for t in d['toks']})} for k, v in corp.items()}
    info['LB_full'] = {'docs': len(lb_full), 'tokens': ntok(lb_full)}; info['PE_full'] = {'docs': len(pe_full), 'tokens': ntok(pe_full)}
    for k, v in corp.items():
        json.dump(v, open(os.path.join(OUT, 'corpus_%s.json' % k), 'w'))
    json.dump({'PFLAT': tf, 'PREC': tr}, open(os.path.join(OUT, 'planted_truth.json'), 'w'))
    json.dump(info, open(os.path.join(OUT, 'corpus_info.json'), 'w'), indent=1)
    return info


if __name__ == '__main__':
    info = build_all()
    for k, v in info.items(): print(k, v)
    for k in ('LA', 'LB', 'PE'):
        docs = json.load(open(os.path.join(OUT, 'corpus_%s.json' % k)))
        print(k, Counter(t[1] for d in docs for t in d['toks']).most_common(), [' '.join(t[0] for t in d['toks'][:12]) for d in docs[:3]])
