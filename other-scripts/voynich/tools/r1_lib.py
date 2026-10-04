"""R1 random-hypothesis machine: shared data layer for Proto-Elamite, Linear A, Voynich.

A corpus is a dict:
  name, docs: list of {id, split('A'|'B'|'C'), lines: list[list[str]], label, stratum, site}
Fixed tokens (never partitioned): '_' word space, '#' numeral, '?' damaged/rare, '|' line break.
Splits are by document, seeded: A 40% (train), B 30% (stage-1 score), C 30% (held-out re-test).
"""
import json, os, random, re, hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OS = os.path.dirname(os.path.dirname(HERE))           # other-scripts
VOY = os.path.join(OS, 'voynich', 'data')
LA = os.path.join(OS, 'linear-a', 'data')
PE = os.path.join(OS, 'proto-elamite', 'data')
RES = os.path.join(VOY, 'results', 'r1')
os.makedirs(RES, exist_ok=True)
FIX = {'_', '#', '?', '|'}

GLYPH_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]


def vglyphs(w):
    for a, b in GLYPH_MULTI:
        w = w.replace(a, b)
    return [c if c.isalpha() else '?' for c in w]


def split_of(docid, seed=0):
    h = int(hashlib.md5(f'{seed}:{docid}'.encode()).hexdigest(), 16) % 100
    return 'A' if h < 40 else ('B' if h < 70 else 'C')


def pe_base(sign):
    if sign.startswith('|'):
        parts = re.split(r'([+.x&])', sign.strip('|'))
        return '|' + ''.join(re.sub(r'~[A-Za-z0-9]+', '', p) for p in parts) + '|'
    return re.sub(r'~[A-Za-z0-9]+', '', sign)


C_CODES = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N29B', 'N39N', 'N28'}
B_CODES = {'N51', 'N51G', 'N54', 'N54G', 'N46'}
FRAC_CODES = {'N02', 'N08', 'N08A', 'N8B', 'N8A'}


def pe_system(numerals):
    codes = {c.replace('N1@', 'N01@') if c != 'N1' else 'N01' for _, c in numerals}
    if not codes:
        return None
    if any('@' in c for c in codes):
        return 'C*' if any(c.split('@')[0] in C_CODES for c in codes) else 'mod*'
    if codes & C_CODES:
        return 'C'
    if codes & B_CODES:
        return 'B'
    if 'N23' in codes:
        return 'N23'
    if codes & FRAC_CODES:
        return 'S-frac'
    return 'SDB'


def load_voynich(trans='ZL3b', seed=0):
    recs = json.load(open(os.path.join(VOY, 'derived', trans + '_lines.json')))
    docs = {}
    for r in recs:
        if r['ltype'] != 'P' or not r['words']:
            continue
        d = docs.setdefault(r['folio'], {'id': r['folio'], 'lines': [], 'label': r['illus'],
                                         'stratum': r['lang'], 'site': r['hand'], 'words': [],
                                         'para_start': []})
        toks = []
        for i, w in enumerate(r['words']):
            if i:
                toks.append('_')
            toks.extend(vglyphs(w))
        d['lines'].append(toks)
        d['words'].append([tuple(vglyphs(w)) for w in r['words']])
        d['para_start'].append(bool(r['para_start']))
    out = list(docs.values())
    for d in out:
        d['split'] = split_of(d['id'], seed)
    return {'name': 'voynich', 'docs': out}


def load_linear_a(seed=0):
    C = json.load(open(os.path.join(LA, 'corpus.json')))
    out = []
    for x in C:
        lines, cur, words, wcur = [], [], [], []
        nums_after = []   # per line: list of (word tuple, next token type/value)
        toks = x['tokens']
        for i, t in enumerate(toks):
            if t['t'] == 'nl':
                if cur:
                    lines.append(cur); words.append(wcur)
                cur, wcur = [], []
                continue
            if t['t'] == 'word':
                if cur:
                    cur.append('_')
                cur.extend(t['s'])
                nxt = toks[i + 1] if i + 1 < len(toks) else {'t': 'end'}
                wcur.append((tuple(t['s']), nxt['t'], nxt.get('v')))
            elif t['t'] == 'num':
                if cur:
                    cur.append('_')
                cur.append('#')
            elif t['t'] == 'logo':
                if cur:
                    cur.append('_')
                cur.append('L:' + logo_group(t['v']))
            elif t['t'] == 'unk':
                if cur:
                    cur.append('_')
                cur.append('?')
        if cur:
            lines.append(cur); words.append(wcur)
        if not lines:
            continue
        out.append({'id': x['id'], 'lines': lines, 'words_ctx': words, 'label': x['support'],
                    'stratum': None, 'site': x['site'], 'split': split_of(x['id'], seed)})
    return {'name': 'linear_a', 'docs': out}


def logo_group(v):
    g = v.split('+')[0].strip('*').strip("'")
    return g


def load_proto_elamite(seed=0):
    T = json.load(open(os.path.join(PE, 'pe_corpus.json')))
    out = []
    for t in T:
        lines, ents = [], []
        for li, l in enumerate(t['lines']):
            sg = [('?' if s == 'x' else pe_base(s)) for s in l['signs']
                  if s == 'x' or s.startswith('M') or s.startswith('|')]
            toks = list(sg)
            if l['numerals']:
                toks.append('#')
            if toks:
                lines.append(toks)
            if sg and l['numerals'] and not (li == 0):
                ents.append({'signs': sg, 'system': pe_system(l['numerals']),
                             'line': len(lines) - 1})
        if not lines:
            continue
        out.append({'id': t['id'], 'lines': lines, 'entries': ents, 'label': t.get('genre'),
                    'stratum': None, 'site': t['provenience'], 'split': split_of(t['id'], seed)})
    return {'name': 'proto_elamite', 'docs': out}


LOADERS = {'voynich': load_voynich, 'linear_a': load_linear_a, 'proto_elamite': load_proto_elamite}


def load(name, seed=0):
    return LOADERS[name](seed=seed)


def sign_counts(corpus):
    c = Counter()
    for d in corpus['docs']:
        for l in d['lines']:
            c.update(t for t in l if t not in FIX)
    return c


def global_shuffle(corpus, rng):
    """Permute all non-fixed tokens across the whole corpus (keeps every line length,
    every fixed-token position and the unigram counts; destroys order and co-occurrence)."""
    pool = [t for d in corpus['docs'] for l in d['lines'] for t in l if t not in FIX]
    rng.shuffle(pool)
    it = iter(pool)
    docs = []
    for d in corpus['docs']:
        nd = dict(d)
        nd['lines'] = [[t if t in FIX else next(it) for t in l] for l in d['lines']]
        docs.append(nd)
    return {'name': corpus['name'], 'docs': docs}


def save(name, obj):
    p = os.path.join(RES, name)
    tmp = p + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, p)


def done(name):
    return os.path.exists(os.path.join(RES, name))
