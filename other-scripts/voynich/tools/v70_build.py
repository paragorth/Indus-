"""v70 build: cut ink units from every corpus and embed them (no transcription alphabet used).

Corpora
  V    real Voynich, 25 Beinecke pages (Q20 + f58), v18 word boxes
  L    real Latin, 28 CREMMA-Medieval-Lat pages (6 MSS, abbreviations), v18 word boxes
  SV   Voynich ZL text (Q20 + f58 paragraphs) rendered in the synthetic Voynich-like hand (truth = EVA units)
  SG_* generator texts trained on that EVA text (self-citation, table-grille, char trigram Markov), same hand
  SL_* six natural-language texts in the same hand by a fixed letter->shape key, with two planted
       ligature units (top-2 letter bigrams drawn joined) and one planted split letter (drawn as two
       separate strokes 'h' 'h')
Output (scratch, not repo): per corpus per theta: embeddings (PCA 30, page-centred), unit widths,
word/line/page ids, truth labels for synthetic corpora.
Usage: python3 v70_build.py [corpus ...]
"""
import sys, os, json, re, random, unicodedata, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import *
from v70_synth import render
import gen

ARR = os.path.join(SCR, 'v70', 'arr'); os.makedirs(ARR, exist_ok=True)
THETAS = [0, 2.5, 4]
NWORDS = 9000
Q20 = ['f58r', 'f58v'] + [f'f{n}{s}' for n in list(range(103, 109)) + list(range(111, 117)) for s in 'rv']
MULTI = {'cth': 'T', 'ckh': 'K', 'cph': 'P', 'cfh': 'F', 'ch': 'C', 'sh': 'S'}
INV = {v: k for k, v in MULTI.items()}


def eva_lines():
    L = json.load(open(os.path.join(DER, 'ZL3b_lines.json')))
    out = []
    for r in L:
        if r['folio'] in Q20 and r['ltype'] == 'P':
            ws = [re.sub(r'[^a-z]', '', w) for w in r['words']]
            ws = [w for w in ws if w]
            if ws:
                out.append({'words': [''.join(MULTI.get(g, g) for g in vglyphs(w)) for w in ws]})
    return out


def text_words(name):
    D = os.path.join(HERE, '..', 'data')
    fn = {'la': 'plain/la.txt', 'lac': 'pg218.txt', 'it': 'pg45334.txt', 'es': 'pg2000.txt',
          'de': 'pg22367.txt', 'cs': 'plain/cs.txt'}[name]
    t = open(os.path.join(D, fn), encoding='utf8', errors='ignore').read()
    if 'START OF' in t:
        t = t.split('START OF', 1)[1]
    t = unicodedata.normalize('NFKD', t.lower())
    t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    ws = re.findall(r'[a-z]+', t)
    ws = ws[2000:2000 + NWORDS]
    return ws


def lines_of(words, rng):
    out, i = [], 0
    while i < len(words):
        k = rng.randint(8, 11); out.append(words[i:i + k]); i += k
    return out


def lang_units(ws, seed):
    """letter -> shape key with planted ligatures and a planted split letter."""
    rng = random.Random(seed)
    lc = collections.Counter(c for w in ws for c in w)
    bc = collections.Counter(w[i:i + 2] for w in ws for i in range(len(w) - 1))
    letters = [c for c, _ in lc.most_common()]
    shapes = [s for s in 'oaecinrlsydktpfqmgxvzbujw']
    rng.shuffle(shapes)
    split = letters[4] if len(letters) > 4 else letters[-1]
    key = {}
    j = 0
    for c in letters:
        if c == split:
            continue
        key[c] = shapes[j % len(shapes)]; j += 1
    ligs = [b for b, _ in bc.most_common(10) if split not in b][:2]
    units = []
    for w in ws:
        u, i = [], 0
        while i < len(w):
            if w[i:i + 2] in ligs:
                u.append(('LIG:' + w[i:i + 2], key[w[i]] + key[w[i + 1]])); i += 2
            elif w[i] == split:
                u.append(('SPL1:' + split, 'h')); u.append(('SPL2:' + split, 'h')); i += 1
            else:
                u.append((w[i], key[w[i]])); i += 1
        units.append(u)
    return units, {'ligs': ligs, 'split': split}


def synth_corpus(name, seed=0):
    rng = random.Random(seed)
    if name == 'SV' or name.startswith('SG'):
        el = eva_lines()
        if name == 'SG_self':
            el = gen.self_citation(el, seed=seed + 1)
        elif name == 'SG_grille':
            el = gen.table_grille(el, seed=seed + 1)
        elif name == 'SG_mk3':
            el = gen.char_markov(el, 2, seed + 1)
        lines = []
        for r in el:
            lines.append([[(INV.get(g, g), INV.get(g, g)) for g in w] for w in r['words']])
        meta = {}
    else:
        ws = text_words(name[3:])
        units, meta = lang_units(ws, seed)
        lines = lines_of(units, rng)
    pages = render([[[s for _, s in w] for w in ln] for ln in lines], seed=seed + 7)
    # attach truth labels
    flat = [w for ln in lines for w in ln]
    i = 0
    for p in pages:
        for w in p['words']:
            w['truth'] = [t for t, _ in flat[i]]
            i += 1
    return pages, meta


def build(name):
    fn = os.path.join(ARR, name + '.json')
    if os.path.exists(fn):
        return
    recs = []   # per word: page, line, k, word string
    P = {th: [] for th in THETAS}; meta_u = {th: [] for th in THETAS}
    meta = {}
    if name in ('V', 'L'):
        pages = json.load(open(os.path.join(DER, 'v18_words.json' if name == 'V' else 'v18_latin_words.json')))
        for pi, pg in enumerate(pages):
            if name == 'V':
                con = load_con(os.path.join(SCR, 'v18', 'img', pg['folio'] + '.jpg'))
            else:
                fo, f = pg['folio'].split(':', 1)
                con = load_con(os.path.join(SCR, 'v18', 'lat', 'data', fo, f + '.jpg'), 2000)
            ws = [w for w in pg['words']]
            res = extract_words(con, ws, pg['pitch'], THETAS)
            base = len(recs)
            for w in ws:
                recs.append({'pg': pi, 'folio': pg['folio'], 'li': w['li'], 'k': w['k'], 'word': w['word']})
            for th in THETAS:
                r = res[th]
                P[th].append(np.array(r['P'], np.uint8).reshape(-1, PH, PW))
                for wid, wd, pos in zip(r['wid'], r['w'], r['pos']):
                    meta_u[th].append((base + wid, wd, pos, ''))
            print(name, pi, len(recs), flush=True)
    else:
        pages, meta = synth_corpus(name)
        for pi, pg in enumerate(pages):
            ws = pg['words']
            res = extract_words(pg['con'], ws, pg['pitch'], THETAS)
            base = len(recs)
            for w in ws:
                recs.append({'pg': pi, 'folio': f'{name}_{pi}', 'li': w['li'], 'k': w['k'], 'word': w['word'],
                             'truth': w['truth']})
            for th in THETAS:
                r = res[th]
                P[th].append(np.array(r['P'], np.uint8).reshape(-1, PH, PW))
                # truth per unit: truth units whose centre falls inside the unit's x-span
                for wid, wd, pos, ab in zip(r['wid'], r['w'], r['pos'], r['ab']):
                    w = ws[wid]; x0 = max(0, w['x0'] - 2)
                    lab = [t for t, (u, xa, xb) in zip(w['truth'], w['units']) if x0 + ab[0] <= (xa + xb) / 2 < x0 + ab[1]]
                    meta_u[th].append((base + wid, wd, pos, '+'.join(lab)))
    for th in THETAS:
        X = np.concatenate(P[th]) if P[th] else np.zeros((0, PH, PW), np.uint8)
        np.save(os.path.join(ARR, f'{name}_t{th}_P.npy'), X)
    json.dump({'recs': recs, 'units': {str(th): meta_u[th] for th in THETAS}, 'meta': meta}, open(fn, 'w'))
    print('done', name, len(recs), {th: len(meta_u[th]) for th in THETAS}, meta, flush=True)


CORPORA = ['V', 'L', 'SV', 'SG_self', 'SG_grille', 'SG_mk3', 'SL_la', 'SL_lac', 'SL_it', 'SL_es', 'SL_de', 'SL_cs']

if __name__ == '__main__':
    names = sys.argv[1:] or CORPORA
    with ProcessPoolExecutor(2) as ex:
        list(ex.map(build, names))
