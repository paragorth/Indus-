"""v53: evolve the machine that wrote it.

Genetic search over small meaning-preserving writing procedures applied to
real medieval plaintexts.  Each program = segmentation + optional abbreviation
+ optional in-word reordering + wording mode + substitution table kind +
a short ordered list of decoration ops (neighbour-dependent padding, line
markers, filler words) + a line width.  Tables are built by frequency-rank
matching of plaintext units to Voynich glyph pieces (from a BPE of the target
half), so the search is over procedure shape, not over a cipher family.

Every program has an exact decoder that uses only the table and the program
(DP parse over its own codewords; reserved padding stripped by rule).  The
decodability constraint is measured: decode(encode(x)) vs x.
"""
import json, math, os, random, re, unicodedata, hashlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v53_ckpt')
os.makedirs(CK, exist_ok=True)

GLYPH_MULTI = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]


def vglyphs(w):
    for a, b in GLYPH_MULTI:
        w = w.replace(a, b)
    return w


# ------------------------------------------------------------------ Voynich
def load_voynich(name='ZL3b'):
    recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    lines = []
    for r in recs:
        if r['ltype'] != 'P':
            continue
        ws = [vglyphs(w) for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
        ws = [w for w in ws if w]
        if ws:
            lines.append({'folio': r['folio'], 'words': ws, 'lang': r.get('lang'),
                          'para_start': r.get('para_start')})
    return lines


def split_halves(lines):
    fol = sorted(set(l['folio'] for l in lines), key=lambda f: [l['folio'] for l in lines].index(f))
    A = set(fol[0::2])
    return [l['words'] for l in lines if l['folio'] in A], [l['words'] for l in lines if l['folio'] not in A]


# ------------------------------------------------------------------ plaintexts
SPECIAL = {'ſ': 's', 'ß': 'ss', 'ꝑ': 'p', 'ꝓ': 'p', 'ꝗ': 'q', 'ꝰ': 'us', '⁊': 'et', 'ꝯ': 'con', 'æ': 'ae',
           'œ': 'oe', 'ł': 'l', 'đ': 'd', 'ȝ': 'z', 'þ': 'th', 'ð': 'd', 'ø': 'o', 'ı': 'i', 'ȷ': 'j'}
HEB = set('אבגדהוזחטיכךלמםנןסעפףצץקרשת')
FINAL_HEB = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}


def norm_word(w):
    out = []
    for ch in w.lower():
        ch = SPECIAL.get(ch, ch)
        for c in unicodedata.normalize('NFD', ch):
            if c in HEB:
                out.append(FINAL_HEB.get(c, c))
            elif 'a' <= c <= 'z':
                out.append(c)
    return ''.join(out)


def _lines_from_wordlists(wls, width=9):
    out = []
    for wl in wls:
        ws = [norm_word(w) for w in wl]
        ws = [w for w in ws if w]
        if ws:
            out.append(ws)
    return out


def load_corpora():
    """name -> (lang, genre, list of lines of normalised words)."""
    f = os.path.join(CK, 'corpora.json')
    if os.path.exists(f):
        return json.load(open(f))
    C = {}
    v30 = json.load(open(os.path.join(DATA, 'v30_ckpt', 'corpora.json')))['corpora']

    def pages(keys):
        ls = []
        for k in keys:
            for pg in v30[k]:
                ls.extend(_rewrap(pg, 9))
        return _lines_from_wordlists(ls)
    C['DE_herb'] = ('de', 'herbal', pages(['G_Bav2']))
    C['DE_rel'] = ('de', 'religious', pages(['G_Alem']))
    C['IT_med'] = ('it', 'medical', pages(['I_Ita']))
    C['IT_verse'] = ('it', 'verse', pages(['I_Com1', 'I_Com2']))
    C['LA_verse'] = ('la', 'verse', pages(['I_Lat']))
    C['CS_chron'] = ('cs', 'chronicle', pages(['C_Old']))
    h = json.load(open(os.path.join(DATA, 'derived', 'v21_herbals.json')))
    C['LA_ency'] = ('la', 'encyclopedic', _lines_from_wordlists([l for e in h['LA'] for p in e['paras'] for l in p]))
    C['IT_herb'] = ('it', 'herbal', _lines_from_wordlists([l for e in h['IT'] for p in e['paras'] for l in p]))
    txt = open(os.path.join(DATA, 'v23_ckpt', 'he_mishneh_torah.txt'), encoding='utf-8').read()
    ws = [norm_word(w) for w in txt.split()]
    ws = [w for w in ws if w]
    C['HE_law'] = ('he', 'legal', [ws[i:i + 9] for i in range(0, len(ws), 9)])
    json.dump(C, open(f, 'w'))
    return C


def _rewrap(page, n):
    ws = [w for l in page for w in (l if isinstance(l, list) else [l])]
    return [ws[i:i + n] for i in range(0, len(ws), n)]


def words_of(lines):
    return [w for l in lines for w in l]


def vowels_for(lang):
    return set('אהוי') if lang == 'he' else set('aeiouy')


# ------------------------------------------------------------------ BPE on Voynich (target half)
def learn_bpe(words, nmerges):
    cnt = Counter(words)
    vocab = {w: list(w) for w in cnt}
    merges = []
    for _ in range(nmerges):
        pc = Counter()
        for w, c in cnt.items():
            s = vocab[w]
            for a, b in zip(s, s[1:]):
                pc[(a, b)] += c
        if not pc:
            break
        (a, b), n = pc.most_common(1)[0]
        if n < 5:
            break
        merges.append((a, b))
        for w in vocab:
            s = vocab[w]
            i = 0; t = []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    t.append(a + b); i += 2
                else:
                    t.append(s[i]); i += 1
            vocab[w] = t
    return merges, vocab


def piece_classes(words, vocab):
    """ranked piece lists per position class S/I/M/F and pooled."""
    cl = {k: Counter() for k in 'SIMF'}
    pooled = Counter()
    for w in words:
        s = vocab[w]
        pooled.update(s)
        if len(s) == 1:
            cl['S'][s[0]] += 1
        else:
            cl['I'][s[0]] += 1; cl['F'][s[-1]] += 1
            for p in s[1:-1]:
                cl['M'][p] += 1
    out = {k: [p for p, _ in v.most_common()] for k, v in cl.items()}
    out['P'] = [p for p, _ in pooled.most_common()]
    return out


def line_enriched(lines, first=True, k=6):
    """glyph pieces enriched at line start (first glyph of first word) / line end."""
    pos, allc = Counter(), Counter()
    for l in lines:
        for i, w in enumerate(l):
            g = w[0] if first else w[-1]
            allc[g] += 1
            if (i == 0 and first) or (i == len(l) - 1 and not first):
                pos[g] += 1
    tp, ta = sum(pos.values()), sum(allc.values())
    sc = {g: (pos[g] / tp) / (allc[g] / ta) for g in pos if pos[g] >= 10}
    return [g for g, _ in sorted(sc.items(), key=lambda x: -x[1])][:k]


# ------------------------------------------------------------------ statistics panel
def H(c):
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def MI(pairs):
    j = Counter(pairs); a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs)
    return H(a) + H(b) - H(j)


def tvd(c1, c2):
    n1, n2 = sum(c1.values()) or 1, sum(c2.values()) or 1
    ks = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in ks)


def panel(lines, ntok=2000):
    """scalar stats + distributions for a text given as lines of glyph-string words (Voynich glyph units)."""
    toks = []
    L = []
    for l in lines:
        if not l:
            continue
        L.append(l)
        toks.extend(l)
        if len(toks) >= ntok:
            break
    if len(toks) < 500:
        return None
    wl = Counter(min(len(w), 12) for w in toks)
    g1 = Counter(c for w in toks for c in w)
    fg = Counter(w[0] for w in toks); lg = Counter(w[-1] for w in toks)
    big = Counter()
    for w in toks:
        s = '^' + w + '$'
        big.update(zip(s, s[1:]))
    ctx = Counter()
    for (a, b), c in big.items():
        ctx[a] += c
    hjoint = H(big); h2 = hjoint - H(ctx)
    irr = 0.0; nb = sum(big.values())
    for (a, b), c in big.items():
        p = c / nb; q = (big.get((b, a), 0) + 0.5) / (nb + 0.5 * len(big))
        irr += p * math.log2(p / q)
    lfirst = [(i == 0, w[0]) for l in L for i, w in enumerate(l)]
    llast = [(i == len(l) - 1, w[-1]) for l in L for i, w in enumerate(l)]
    junc = [(a[-1], b[0]) for l in L for a, b in zip(l, l[1:])]
    adjrep = sum(a == b for l in L for a, b in zip(l, l[1:])) / max(1, sum(len(l) - 1 for l in L))
    linerep = sum(len(l) - len(set(l)) for l in L) / len(toks)
    wc = Counter(toks)
    ttr = len(wc) / len(toks)
    hap = sum(1 for v in wc.values() if v == 1) / len(wc)
    return {'_wl': wl, '_g1': g1, '_fg': fg, '_lg': lg,
            'h2': h2, 'irr': irr, 'mi_lf': MI(lfirst), 'mi_ll': MI(llast), 'mi_j': MI(junc),
            'adjrep': adjrep, 'linerep': linerep, 'ttr': ttr, 'hapax': hap,
            'wpl': len(toks) / len(L), 'mwl': sum(len(w) for w in toks) / len(toks)}


SCAL = ['h2', 'irr', 'mi_lf', 'mi_ll', 'mi_j', 'adjrep', 'linerep', 'ttr', 'hapax', 'wpl', 'mwl']
DIST = ['_wl', '_g1', '_fg', '_lg']


def chunks(lines, ntok=2000):
    out, cur, n = [], [], 0
    for l in lines:
        cur.append(l); n += len(l)
        if n >= ntok:
            out.append(cur); cur, n = [], 0
    return out


def target_profile(lines, ntok=2000):
    """mean stats and sd across chunks; distribution stats against pooled."""
    ch = chunks(lines, ntok)
    ps = [panel(c, ntok) for c in ch]
    ps = [p for p in ps if p]
    pooled = panel(lines, 10 ** 9)
    T = {'dist': {k: dict(pooled[k]) for k in DIST}, 'mean': {}, 'sd': {}}
    for k in SCAL:
        v = [p[k] for p in ps]
        m = sum(v) / len(v)
        T['mean'][k] = m
        T['sd'][k] = max((sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5, 0.05 * abs(m), 1e-3)
    for k in DIST:
        v = [tvd(p[k], pooled[k]) for p in ps]
        T['mean'][k] = 0.0
        T['sd'][k] = max(sum(v) / len(v), 0.01)
    return T


def distance(p, T, detail=False):
    if p is None:
        return 99.0, {}
    z = {}
    for k in SCAL:
        z[k] = (p[k] - T['mean'][k]) / T['sd'][k]
    for k in DIST:
        z[k] = tvd(p[k], Counter(T['dist'][k])) / T['sd'][k]
    d = sum(math.log1p(abs(v)) for v in z.values()) / len(z)
    return (d, z) if detail else (d, None)


# ------------------------------------------------------------------ program space
SEGS = ['letter', 'syll', 'bigram', 'vc']
ABBR = ['none', 'none', 'skel', 'trunc4', 'trunc6', 'susp']
REORD = ['none', 'none', 'rev', 'rot', 'oddeven']
WORDING = ['keep', 'keep', 'unit', 'group2', 'group3', 'split4']
TABLES = ['single', 'pos', 'pos', 'homo2', 'homo3', 'rot2', 'rot3', 'pair', 'linepos']
BPE_M = [0, 25, 60, 150]
OPS = ['PFX', 'SFX', 'FILL', 'LINEM', 'LINEF']
RULES = ['prev', 'linepos', 'self', 'const']


def random_program(rng, corpus=None, corpora=None):
    ops = []
    for _ in range(rng.choice([0, 0, 1, 1, 2, 3, 4])):
        ops.append([rng.choice(OPS), rng.choice([1, 2, 3]), rng.choice(RULES), rng.randrange(8), rng.choice([2, 3, 4, 6])])
    return {'corpus': corpus if corpus else rng.choice(sorted(corpora)),
            'seg': rng.choice(SEGS), 'abbr': rng.choice(ABBR), 'reord': rng.choice(REORD),
            'wording': rng.choice(WORDING), 'table': rng.choice(TABLES), 'bpe': rng.choice(BPE_M),
            'jit': rng.randrange(3), 'width': rng.randrange(24, 60), 'ops': ops, 'seed': rng.randrange(10 ** 6)}


def mutate(p, rng, corpora=None, p_corpus=0.0):
    q = json.loads(json.dumps(p))
    for _ in range(rng.choice([1, 1, 2, 3])):
        g = rng.choice(['seg', 'abbr', 'reord', 'wording', 'table', 'bpe', 'jit', 'width', 'ops', 'ops', 'ops', 'seed'])
        if g == 'seg': q['seg'] = rng.choice(SEGS)
        elif g == 'abbr': q['abbr'] = rng.choice(ABBR)
        elif g == 'reord': q['reord'] = rng.choice(REORD)
        elif g == 'wording': q['wording'] = rng.choice(WORDING)
        elif g == 'table': q['table'] = rng.choice(TABLES)
        elif g == 'bpe': q['bpe'] = rng.choice(BPE_M)
        elif g == 'jit': q['jit'] = rng.randrange(3)
        elif g == 'width': q['width'] = max(18, min(70, q['width'] + rng.randint(-8, 8)))
        elif g == 'seed': q['seed'] = rng.randrange(10 ** 6)
        else:
            r = rng.random()
            if r < 0.3 and len(q['ops']) < 5:
                q['ops'].insert(rng.randrange(len(q['ops']) + 1), [rng.choice(OPS), rng.choice([1, 2, 3]), rng.choice(RULES), rng.randrange(8), rng.choice([2, 3, 4, 6])])
            elif r < 0.5 and q['ops']:
                q['ops'].pop(rng.randrange(len(q['ops'])))
            elif r < 0.6 and len(q['ops']) > 1:
                i, j = rng.sample(range(len(q['ops'])), 2); q['ops'][i], q['ops'][j] = q['ops'][j], q['ops'][i]
            elif q['ops']:
                o = q['ops'][rng.randrange(len(q['ops']))]
                k = rng.randrange(5)
                o[k] = [rng.choice(OPS), rng.choice([1, 2, 3]), rng.choice(RULES), rng.randrange(8), rng.choice([2, 3, 4, 6])][k]
    if corpora and rng.random() < p_corpus:
        q['corpus'] = rng.choice(sorted(corpora))
    return q


def crossover(a, b, rng):
    c = {}
    for k in a:
        c[k] = json.loads(json.dumps(rng.choice([a, b])[k]))
    if rng.random() < 0.5:
        c['ops'] = json.loads(json.dumps(a['ops'][:len(a['ops']) // 2] + b['ops'][len(b['ops']) // 2:]))[:5]
    return c


def motif_key(p):
    return (p['seg'], p['abbr'], p['reord'], p['wording'], p['table'], p['bpe'],
            tuple(sorted(set(o[0] + ':' + o[2] for o in p['ops']))))


# ------------------------------------------------------------------ unit segmentation (cached)
def seg_word(w, seg, V):
    if seg == 'letter':
        return list(w)
    if seg == 'bigram':
        return [w[i:i + 2] for i in range(0, len(w), 2)]
    out, cur = [], ''
    if seg == 'syll':      # C*V+ chunks, trailing consonants attached to the last chunk
        i = 0
        while i < len(w):
            cur += w[i]
            if w[i] in V and (i + 1 == len(w) or w[i + 1] not in V):
                out.append(cur); cur = ''
            i += 1
        if cur:
            if out: out[-1] += cur
            else: out.append(cur)
        return out
    if seg == 'vc':        # V*C+ chunks
        i = 0
        while i < len(w):
            cur += w[i]
            if w[i] not in V and (i + 1 == len(w) or w[i + 1] in V):
                out.append(cur); cur = ''
            i += 1
        if cur:
            if out: out[-1] += cur
            else: out.append(cur)
        return out
    raise ValueError(seg)


def abbr_word(w, ab, V):
    if ab == 'none' or len(w) <= 2:
        return w
    if ab == 'skel':
        return w[0] + ''.join(c for c in w[1:-1] if c not in V) + w[-1]
    if ab == 'trunc4':
        return w[:4]
    if ab == 'trunc6':
        return w[:6]
    if ab == 'susp':
        return w[:3] + w[-1] if len(w) > 4 else w
    raise ValueError(ab)


def reorder(u, r):
    if r == 'none' or len(u) < 2:
        return list(u)
    if r == 'rev':
        return u[::-1]
    if r == 'rot':
        return u[1:] + u[:1]
    if r == 'oddeven':
        return u[0::2] + u[1::2]


def unreorder(u, r):
    if r == 'none' or len(u) < 2:
        return list(u)
    if r == 'rev':
        return u[::-1]
    if r == 'rot':
        return u[-1:] + u[:-1]
    if r == 'oddeven':
        n = len(u); k = (n + 1) // 2
        out = [None] * n
        out[0::2] = u[:k]; out[1::2] = u[k:]
        return out


def info_kept(words, ab, V):
    """1 - H(orig word | abbreviated) / H(orig word), word unigram level."""
    if ab == 'none':
        return 1.0
    c = Counter(words)
    by = defaultdict(Counter)
    for w, n in c.items():
        by[abbr_word(w, ab, V)][w] += n
    N = sum(c.values())
    Hc = sum(sum(g.values()) / N * H(g) for g in by.values())
    return 1 - Hc / H(c)


# ------------------------------------------------------------------ the machine
class Env:
    """precomputed target pieces and plaintext samples."""

    def __init__(self, target_lines, corpora, n_plain=2400, offset=0, rng_seed=0):
        self.target_lines = target_lines
        tw = words_of(target_lines)
        self.bpe = {}
        self.vocabset = {}
        for M in BPE_M:
            merges, vocab = learn_bpe(tw, M)
            self.bpe[M] = piece_classes(tw, vocab)
            self.vocabset[M] = set(x for s in vocab.values() for x in s)
        self._pcache = {}
        self.linit = line_enriched(target_lines, True)
        self.lfin = line_enriched(target_lines, False)
        self.corpora = corpora
        self.plain = {}
        for name, (lang, genre, lines) in corpora.items():
            ws = words_of(lines)
            st = offset % max(1, len(ws) - n_plain)
            self.plain[name] = (lang, ws[st:st + n_plain], ws)
        self._ucache = {}
        self._icache = {}

    def units(self, p):
        key = (p['corpus'], p['seg'], p['abbr'], p['reord'])
        if key not in self._ucache:
            lang, ws, allw = self.plain[p['corpus']]
            V = vowels_for(lang)
            uw = []
            for w in ws:
                a = abbr_word(w, p['abbr'], V)
                uw.append(reorder(seg_word(a, p['seg'], V), p['reord']))
            ik = (p['corpus'], p['abbr'])
            if ik not in self._icache:
                self._icache[ik] = info_kept(allw, p['abbr'], V)
            self._ucache[key] = (uw, self._icache[ik])
        return self._ucache[key]


def _greedy(w, vs, maxlen):
    out, i = [], 0
    while i < len(w):
        for j in range(min(len(w), i + maxlen), i, -1):
            if w[i:j] in vs or j == i + 1:
                out.append(w[i:j]); i = j; break
    return out


def env_pieces(env, M, ops):
    """piece classes of the target AFTER the program's padding is stripped (the program's own reading)."""
    sig = (M, tuple((o[0], tuple(o[5])) for o in ops if o[5]))
    if sig in env._pcache:
        return env._pcache[sig]
    if len(sig[1]) == 0:
        env._pcache[sig] = env.bpe[M]
        return env.bpe[M]
    vs = env.vocabset[M]; ml = max(map(len, vs))
    fill = set(x for k, sel in sig[1] if k == 'FILL' for x in sel)
    cl = {k: Counter() for k in 'SIMF'}; pooled = Counter()
    for l in env.target_lines:
        for i, w in enumerate(l):
            if w in fill:
                continue
            for k, sel in sig[1][::-1]:
                if k == 'SFX' or (k == 'LINEF' and i == len(l) - 1):
                    for r in sorted(sel, key=len, reverse=True):
                        if w.endswith(r) and len(w) > len(r):
                            w = w[:-len(r)]; break
                elif k == 'PFX' or (k == 'LINEM' and i == 0):
                    for r in sorted(sel, key=len, reverse=True):
                        if w.startswith(r) and len(w) > len(r):
                            w = w[len(r):]; break
            sg = _greedy(w, vs, ml)
            pooled.update(sg)
            if len(sg) == 1:
                cl['S'][sg[0]] += 1
            else:
                cl['I'][sg[0]] += 1; cl['F'][sg[-1]] += 1
                for x in sg[1:-1]:
                    cl['M'][x] += 1
    out = {k: [p for p, _ in v.most_common()] for k, v in cl.items()}
    out['P'] = [p for p, _ in pooled.most_common()]
    env._pcache[sig] = out
    return out


def _out_words(uw, wording):
    """plaintext unit-words -> list of output unit groups, plus boundary flags (True = plaintext word ends here)."""
    out = []
    if wording == 'keep':
        return [(u, True) for u in uw if u]
    if wording == 'unit':
        for u in uw:
            for i, x in enumerate(u):
                out.append(([x], i == len(u) - 1))
        return out
    if wording in ('group2', 'group3'):
        n = 2 if wording == 'group2' else 3
        stream = [(x, i == len(u) - 1) for u in uw for i, x in enumerate(u)]
        for i in range(0, len(stream), n):
            g = stream[i:i + n]
            out.append(([x for x, _ in g], g[-1][1]))
        return out
    if wording == 'split4':
        for u in uw:
            for i in range(0, len(u), 4):
                out.append((u[i:i + 4], i + 4 >= len(u)))
        return out


def _cls(i, n):
    if n == 1: return 'S'
    if i == 0: return 'I'
    if i == n - 1: return 'F'
    return 'M'


def build(p, env):
    """returns machine dict: tables, reserved sets, decoder index."""
    p = clean(p)
    rng = random.Random(p['seed'])
    pcs = env.bpe[p['bpe']]
    uw, ik = env.units(p)
    groups = _out_words(uw, p['wording'])
    # reserved pieces for decoration ops
    reserved = {'PFX': [], 'SFX': [], 'FILL': [], 'LINEM': [], 'LINEF': []}
    used = set()
    for o in p['ops']:
        kind, k, rule, off, every = o
        src = {'PFX': pcs['I'], 'SFX': pcs['F'], 'FILL': pcs['S'], 'LINEM': env.linit, 'LINEF': env.lfin}[kind]
        sel = [x for x in src[off:] if x not in used][:k] or [x for x in src if x not in used][:k]
        o.append(sel)  # resolved reserved items appended as o[5]
        used.update(sel)
    pcs = env_pieces(env, p['bpe'], p['ops'])
    tkind = p['table']
    # unit counts by class
    if tkind in ('pos', 'linepos'):
        ccount = defaultdict(Counter)
        for g, _ in groups:
            for i, x in enumerate(g):
                ccount[_cls(i, len(g))][x] += 1
    else:
        ccount = {'P': Counter(x for g, _ in groups for x in g)}
    tables = {}

    def ranked(cnt, plist, h=1, shift=0):
        units = [u for u, _ in cnt.most_common()]
        plist = [x for x in plist if x not in used]
        if shift:
            plist = plist[shift:] + plist[:shift]
        if p['jit']:
            plist = list(plist)
            for i in range(0, len(plist) - 1):
                if rng.random() < 0.15 * p['jit']:
                    plist[i], plist[i + 1] = plist[i + 1], plist[i]
        t = {}
        j = 0
        taken = set(plist)
        P = max(1, min(len(plist), 40))
        extra = ((plist[a] + plist[b]) for a in range(P) for b in range(P))
        for u in units:
            codes = []
            for _ in range(h):
                if j < len(plist):
                    codes.append(plist[j]); j += 1
                else:   # overflow: two-piece composite codeword
                    for c in extra:
                        if c not in taken:
                            taken.add(c); codes.append(c); break
            if codes:
                t[u] = codes
        return t
    if tkind == 'single':
        tables['P'] = ranked(ccount['P'], pcs['P'])
    elif tkind in ('homo2', 'homo3'):
        tables['P'] = ranked(ccount['P'], pcs['P'], h=int(tkind[-1]))
    elif tkind in ('rot2', 'rot3'):
        g = int(tkind[-1])
        for t in range(g):
            tables['R%d' % t] = ranked(ccount['P'], pcs['P'], shift=7 * t)
    elif tkind == 'pair':
        units = [u for u, _ in ccount['P'].most_common()]
        I = [x for x in pcs['I'] if x not in used]; F = [x for x in pcs['F'] if x not in used]
        q = max(2, int(math.ceil(math.sqrt(len(units)))))
        q = min(q, len(F))
        tables['PAIR'] = {u: [I[r // q] + '|' + F[r % q]] for r, u in enumerate(units) if r // q < len(I)}
    else:
        for c in 'SIMF':
            tables[c] = ranked(ccount.get(c, Counter()), pcs[c] if pcs[c] else pcs['P'])
        if tkind == 'linepos':
            for c in 'SIMF':
                tables['L' + c] = ranked(ccount.get(c, Counter()), pcs[c] if pcs[c] else pcs['P'], shift=5)
    inv = {name: {code: u for u, codes in t.items() for code in codes} for name, t in tables.items()}
    return {'p': p, 'groups': groups, 'tables': tables, 'inv': inv, 'reserved_ops': p['ops'], 'info': ik}


def _hash(s, k):
    if k <= 1:
        return 0
    return (sum(ord(c) for c in s) * 2654435761) % k


def encode(m, max_words=None, max_tokens=2100):
    """-> lines of output words (strings), payload glyph count, total glyph count, unit-group list per output word."""
    p = m['p']; tk = p['table']; W = p['width']
    lines, cur, curlen = [], [], 0
    lineno = 0
    prev_last = '^'
    pay = tot = 0
    gid = []
    curgid = []
    ops = m['reserved_ops']
    groups = m['groups'][:max_words] if max_words else m['groups']
    ntoks = 0
    for gi, (g, bnd) in enumerate(groups):
        def make(line_pos, lineno, prev_last):
            core = []
            n = len(g)
            for i, x in enumerate(g):
                if tk == 'single':
                    cs = m['tables']['P'].get(x)
                elif tk in ('homo2', 'homo3'):
                    cs = m['tables']['P'].get(x)
                    if cs:
                        cs = [cs[_hash(core[-1][-1] if core else prev_last, len(cs))]]
                elif tk in ('rot2', 'rot3'):
                    cs = m['tables']['R%d' % (lineno % int(tk[-1]))].get(x)
                elif tk == 'pair':
                    cs = m['tables']['PAIR'].get(x)
                elif tk == 'linepos':
                    cs = m['tables'][('L' if line_pos == 0 else '') + _cls(i, n)].get(x)
                else:
                    cs = m['tables'][_cls(i, n)].get(x)
                core.append(cs[0].replace('|', '') if cs else '?')
            word = ''.join(core)
            payload = len(word)
            extra_words = []
            for o in ops:
                kind, k, rule, off, every, sel = o
                if not sel:
                    continue
                key = {'prev': prev_last, 'linepos': str(min(line_pos // 3, 3)), 'self': word[:1], 'const': ''}[rule]
                r = sel[_hash(key, len(sel))]
                if kind == 'PFX':
                    word = r + word
                elif kind == 'SFX':
                    word = word + r
                elif kind == 'LINEM' and line_pos == 0:
                    word = r + word
                elif kind == 'FILL' and (gi % every) == every - 1:
                    extra_words.append(r)
            return word, payload, extra_words
        word, payload, extra = make(len(cur), lineno, prev_last)
        wl = len(word) + sum(len(e) for e in extra)
        if cur and curlen + wl > W:
            # close line (LINEF)
            for o in ops:
                if o[0] == 'LINEF' and o[5]:
                    cur[-1] = cur[-1] + o[5][_hash(cur[-1][:1], len(o[5]))]
                    tot += len(o[5][0])
            lines.append(cur); gid.append(curgid); lineno += 1
            cur, curlen, curgid = [], 0, []
            word, payload, extra = make(0, lineno, prev_last)
        cur.append(word); curgid.append(gi); pay += payload; tot += len(word)
        for e in extra:
            cur.append(e); curgid.append(-1); tot += len(e)
        curlen += wl
        prev_last = cur[-1][-1]
        ntoks += 1 + len(extra)
        if max_tokens and ntoks >= max_tokens:
            break
    if cur:
        lines.append(cur); gid.append(curgid)
    return lines, pay, tot, gid


def decode_lines(m, lines):
    """inverse: lines of output (or Voynich) words -> list of (unit list, boundary_known) per payload word; coverage."""
    p = m['p']; tk = p['table']; ops = m['reserved_ops']
    fill = set(x for o in ops if o[0] == 'FILL' for x in o[5])
    pfx = [o[5] for o in ops if o[0] == 'PFX' and o[5]]
    sfx = [o[5] for o in ops if o[0] == 'SFX' and o[5]]
    lm = [o[5] for o in ops if o[0] == 'LINEM' and o[5]]
    lf = [o[5] for o in ops if o[0] == 'LINEF' and o[5]]
    out = []
    known = total = 0
    for lineno, l in enumerate(lines):
        l = list(l)
        if lf and l:
            w = l[-1]
            for sel in lf[::-1]:
                for r in sorted(sel, key=len, reverse=True):
                    if w.endswith(r) and len(w) > len(r):
                        w = w[:-len(r)]; break
            l[-1] = w
        pos = 0
        for wi, w in enumerate(l):
            if w in fill:
                continue
            # strip decorations in reverse order of application
            for o in ops[::-1]:
                kind, sel = o[0], o[5]
                if not sel:
                    continue
                if kind == 'SFX':
                    for r in sorted(sel, key=len, reverse=True):
                        if w.endswith(r) and len(w) > len(r):
                            w = w[:-len(r)]; break
                elif kind == 'PFX' or (kind == 'LINEM' and pos == 0):
                    for r in sorted(sel, key=len, reverse=True):
                        if w.startswith(r) and len(w) > len(r):
                            w = w[len(r):]; break
            units = parse(m, w, pos, lineno)
            total += 1
            if units is not None:
                known += 1
            out.append(units)
            pos += 1
    return out, known / max(1, total)


def parse(m, w, line_pos, lineno):
    """DP parse of a word into the program's codewords -> unit list or None."""
    tk = m['p']['table']; inv = m['inv']
    n = len(w)
    if tk in ('single', 'homo2', 'homo3'):
        tabs = lambda i, j: inv['P']
    elif tk in ('rot2', 'rot3'):
        t = inv['R%d' % (lineno % int(tk[-1]))]
        tabs = lambda i, j: t
    elif tk == 'pair':
        t = {k.replace('|', ''): v for k, v in inv['PAIR'].items()}
        tabs = lambda i, j: t
    else:
        pre = 'L' if (tk == 'linepos' and line_pos == 0) else ''

        def tabs(i, j):
            if i == 0 and j == n: return inv[pre + 'S']
            if i == 0: return inv[pre + 'I']
            if j == n: return inv[pre + 'F']
            return inv[pre + 'M']
    # dp over positions: fewest pieces
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, min(n, i + 12) + 1):
            u = tabs(i, j).get(w[i:j])
            if u is not None and (best[j] is None or len(best[j]) > len(best[i]) + 1):
                best[j] = best[i] + [u]
    return best[n]


def evaluate(p, env, T, n_check=300):
    m = build(p, env)
    lines, pay, tot, gid = encode(m)
    pn = panel(lines)
    d, _ = distance(pn, T)
    # decodability: decode the first n_check output words and compare with the encoded groups
    sub, cnt = [], 0
    for l in lines:
        sub.append(l); cnt += len(l)
        if cnt >= n_check:
            break
    dec, cov = decode_lines(m, sub)
    tg = [g for g, _ in m['groups'][:sum(1 for l in gid[:len(sub)] for g in l if g >= 0)]]
    if len(tg) == len(dec) and tg:
        acc = sum(sum(1 for a, b in zip(g, u or []) if a == b) / len(g) if g else 1 for g, u in zip(tg, dec)) / len(tg)
    else:
        truth = [x for g in tg for x in g]
        got = [x for u in dec if u for x in u]
        acc = SequenceMatcher(None, truth[:600], got[:600], autojunk=False).ratio() if truth else 0.0
    unk = sum(1 for g, _ in m['groups'] for x in g if x not in _all_units(m))
    nunits = sum(len(g) for g, _ in m['groups'])
    lost = unk / max(1, nunits)
    bnd = 1.0 if (p['wording'] == 'keep' or p['table'] in ('pos', 'linepos')) else 0.7
    retention = acc * (1 - lost) * m['info'] * bnd
    fit = d + 3.0 * max(0.0, 0.85 - acc * (1 - lost)) + 1.0 * (1 - m['info']) + 0.5 * (1 - bnd)
    return {'fit': fit, 'd': d, 'acc': acc, 'lost': lost, 'info': m['info'], 'ret': retention,
            'payload_share': pay / max(1, tot)}


def _all_units(m):
    if '_au' not in m:
        m['_au'] = set(u for t in m['tables'].values() for u in t)
    return m['_au']


# ------------------------------------------------------------------ the GA
def run_ga(env, T, corpora_names, pop_size, gens, seed, log=None, fixed_corpus=None, elite_frac=0.1):
    rng = random.Random(seed)
    pop = [random_program(rng, corpus=fixed_corpus, corpora=corpora_names) for _ in range(pop_size)]
    allrec = []
    for gen in range(gens):
        scored = []
        for p in pop:
            try:
                r = evaluate(p, env, T)
            except Exception as e:
                r = {'fit': 99.0, 'd': 99.0, 'acc': 0, 'lost': 1, 'info': 0, 'ret': 0, 'payload_share': 0, 'err': str(e)[:80]}
            scored.append((r['fit'], p, r))
        scored.sort(key=lambda x: x[0])
        allrec.extend((s[0], s[1], s[2], gen) for s in scored)
        if log:
            b = scored[0]
            cc = Counter(s[1]['corpus'] for s in scored[:pop_size // 10])
            log(f"gen {gen} best {b[0]:.3f} d {b[2]['d']:.3f} acc {b[2]['acc']:.2f} {b[1]['corpus']} {motif_key(b[1])} top10% corpora {dict(cc.most_common(5))}")
        ne = max(2, int(elite_frac * pop_size))
        elite = [s[1] for s in scored[:ne]]
        new = [json.loads(json.dumps(e)) for e in elite]
        while len(new) < pop_size:
            def tour():
                c = rng.sample(scored[:pop_size // 2], 3)
                return min(c, key=lambda x: x[0])[1]
            if rng.random() < 0.3:
                ch = crossover(tour(), tour(), rng)
            else:
                ch = tour()
            ch = mutate(ch, rng, corpora_names, p_corpus=0.0 if fixed_corpus else 0.05)
            if rng.random() < 0.05:
                ch = random_program(rng, corpus=fixed_corpus, corpora=corpora_names)
            new.append(ch)
        pop = new
    return allrec


def clean(p):
    q = json.loads(json.dumps(p))
    for o in q['ops']:
        del o[5:]
    return q
