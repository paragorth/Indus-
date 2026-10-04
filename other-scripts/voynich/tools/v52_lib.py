"""v52 THE BOOK IS A DATABASE: shared library.

Inverted problem: every word is a record key built from fields (slots), not a word of speech.
A random FIELD SCHEMA cuts each word (list of glyph units) into K = 2..5 slots by K-1 random cut rules
applied left to right from a pointer p:
  ('fix', k)    cut at p+k
  ('end', k)    cut at max(p, L-k)
  ('set', S)    cut before the first unit at or after p that is in the random unit set S
  ('run', S)    cut after the maximal run of S-units starting at p
Each slot keeps its top-N values (random N) plus OTHER: a random slot alphabet.

For every slot s and external variable v the score is the shuffle-corrected mutual information
E[s,v] = MI(s;v) - mean MI(s; v_null), v_null = page variables permuted across pages, token variables
permuted among the tokens of the same page; normalised by min(H(s), H(v)).
Variables fall in groups (PAGE / POS / NBR for the Voynich). The FIELD SCORE
  FS = sum_g max_s max(0, e[s,g] - sum_{g' != g} e[s,g'])
is large only when different slots are each PURE trackers of different groups (a field code).
DEP = conditional inter-slot dependence: sum over adjacent slot pairs of normalised
I(s_i; s_j | cell) minus its within-cell permutation null (cell = joint of the group targets); a real
field code has slots that are independent given what they encode.
"""
import os, sys, json, math, random
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v52_ckpt'); os.makedirs(CK, exist_ok=True)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


# ---------------------------------------------------------------- corpora
def _enc(vals):
    m = {}
    return np.array([m.setdefault(v, len(m)) for v in vals], dtype=np.int64)


def make_corpus(words, page, varvals, pagevars, groups, name):
    """words: list of unit tuples per token; page: list page ids; varvals: {var: list values}."""
    tmap = {}
    tt = np.array([tmap.setdefault(tuple(w), len(tmap)) for w in words], dtype=np.int64)
    types = [None] * len(tmap)
    for w, i in tmap.items():
        types[i] = w
    uc = Counter(u for w in words for u in w)
    return dict(name=name, types=types, tok_type=tt, page=_enc(page), page_raw=list(page),
                vars={k: _enc(v) for k, v in varvals.items()}, pagevars=set(pagevars), groups=dict(groups),
                top_units=[u for u, _ in uc.most_common(24)], n=len(words))


def voynich_corpus(name='ZL3b', words_override=None):
    recs = json.load(open(os.path.join(vlib.DATA, 'derived', name + '_lines.json')))
    # page variables = per-folio majority
    fol = defaultdict(lambda: defaultdict(Counter))
    for r in recs:
        for k in ('illus', 'quire', 'hand', 'lang'):
            fol[r['folio']][k][str(r[k])] += len(r['words'])
    fv = {f: {k: c.most_common(1)[0][0] for k, c in d.items()} for f, d in fol.items()}
    W, P, V = [], [], defaultdict(list)
    for r in recs:
        ws = r['words']
        keep = [i for i, w in enumerate(ws) if '?' not in w and w]
        for j, i in enumerate(keep):
            w = ws[i]
            W.append(tuple(vlib.glyphs(w))); P.append(r['folio'])
            for k, kk in (('SEC', 'illus'), ('QUIRE', 'quire'), ('HAND', 'hand'), ('LANG', 'lang')):
                V[k].append(fv[r['folio']][kk])
            if r['ltype'] != 'P':
                pos = 'L'
            elif r['para_start']:
                pos = 'F'
            elif j == 0:
                pos = 'I'
            elif j == len(keep) - 1:
                pos = 'E'
            else:
                pos = 'M'
            V['POS'].append(pos)
            V['NBR'].append(vlib.glyphs(ws[keep[j - 1]])[-1] if j > 0 else '^')
            V['NXT'].append(vlib.glyphs(ws[keep[j + 1]])[0] if j + 1 < len(keep) else '$')
    if words_override is not None:
        W = words_override
    groups = dict(SEC='PAGE', QUIRE='PAGE', HAND='PAGE', LANG='PAGE', POS='POS', NBR='NBR', NXT='NBR')
    return make_corpus(W, P, V, {'SEC', 'QUIRE', 'HAND', 'LANG'}, groups, name)


def replace_words(C, words, name):
    """Same external variables, new token strings."""
    D = dict(C)
    tmap = {}
    D['tok_type'] = np.array([tmap.setdefault(tuple(w), len(tmap)) for w in words], dtype=np.int64)
    types = [None] * len(tmap)
    for w, i in tmap.items():
        types[i] = w
    D['types'] = types
    uc = Counter(u for w in words for u in w)
    D['top_units'] = [u for u, _ in uc.most_common(24)]
    D['name'] = name
    return D


def raw_values(C):
    """Recover the categorical value labels by token (codes are fine for the generators)."""
    return {k: v for k, v in C['vars'].items()}


# ---------------------------------------------------------------- nulls
def null_vars(C, rng, R=3):
    """R null realisations of every variable."""
    page = C['page']; npg = page.max() + 1
    # tokens of each page
    order = np.argsort(page, kind='stable')
    bounds = np.searchsorted(page[order], np.arange(npg + 1))
    first_tok = order[bounds[:-1]]
    outs = []
    for _ in range(R):
        perm = rng.permutation(npg)
        d = {}
        for k, v in C['vars'].items():
            if k in C['pagevars']:
                pv = v[first_tok]          # page value
                d[k] = pv[perm][page]
            else:
                nv = v.copy()
                for p in range(npg):
                    ix = order[bounds[p]:bounds[p + 1]]
                    if len(ix) > 1:
                        nv[ix] = v[rng.permutation(ix)]
                d[k] = nv
        outs.append(d)
    return outs


def shuffled_corpus(C, rng):
    """META-NULL: the external variables replaced by one null realisation."""
    D = dict(C)
    D['vars'] = null_vars(C, rng, 1)[0]
    D['name'] = C['name'] + '-VARSHUF'
    return D


def markov_cell_words(C, rng, cellvars=('SEC', 'POS'), order=2):
    """Markov resynthesis conditioned on a cell: unit n-gram per cell (backoff to global)."""
    cell = np.zeros(C['n'], dtype=np.int64)
    for k in cellvars:
        cell = cell * (C['vars'][k].max() + 1) + C['vars'][k]
    toks = [C['types'][t] for t in C['tok_type']]
    cnt = defaultdict(Counter); gl = defaultdict(Counter)
    for w, c in zip(toks, cell):
        s = ('^',) * order + tuple(w) + ('$',)
        for i in range(order, len(s)):
            h = s[i - order:i]
            cnt[(c, h)][s[i]] += 1; gl[h][s[i]] += 1
    cache = {}
    def draw(c, h):
        key = (c, h)
        if key not in cache:
            src = cnt.get(key)
            if not src or sum(src.values()) < 3:
                src = gl[h]
            it = list(src.items())
            cache[key] = ([a for a, _ in it], np.cumsum([b for _, b in it], dtype=float))
        a, cs = cache[key]
        return a[int(np.searchsorted(cs, rng.random() * cs[-1], side='right'))]
    out = []
    for c in cell:
        h = ('^',) * order; w = []
        while len(w) < 15:
            u = draw(c, h)
            if u == '$':
                break
            w.append(u); h = h[1:] + (u,)
        out.append(tuple(w) if w else ('o',))
    return out


# ---------------------------------------------------------------- schemas
def random_schema(rng, units):
    K = rng.randint(2, 5)
    rules = []
    for _ in range(K - 1):
        t = rng.choice(['fix', 'end', 'set', 'set', 'run'])
        if t in ('fix', 'end'):
            rules.append((t, rng.randint(1, 4)))
        else:
            pool = units[:20]
            S = tuple(sorted(rng.sample(pool, rng.randint(1, 6))))
            rules.append((t, S))
    N = rng.choice([6, 12, 24, 48])
    return dict(rules=rules, N=N)


def cut(w, rules):
    L = len(w); p = 0; out = []
    for t, a in rules:
        if t == 'fix':
            q = min(L, p + a)
        elif t == 'end':
            q = max(p, L - a)
        elif t == 'set':
            q = L
            for i in range(p, L):
                if w[i] in a:
                    q = i; break
        else:  # run
            q = p
            while q < L and w[q] in a:
                q += 1
        out.append(w[p:q]); p = q
    out.append(w[p:])
    return out


def slot_codes(C, sch):
    """K x n int arrays (top-N values + OTHER)."""
    K = len(sch['rules']) + 1
    tcuts = [cut(w, sch['rules']) for w in C['types']]
    tt = C['tok_type']
    tf = np.bincount(tt, minlength=len(C['types']))
    out = []
    for s in range(K):
        vals = {}
        tv = np.empty(len(tcuts), dtype=np.int64)
        for i, cs in enumerate(tcuts):
            tv[i] = vals.setdefault(''.join(cs[s]) if cs[s] else '-', len(vals))
        freq = np.bincount(tv, weights=tf, minlength=len(vals))
        keep = np.argsort(-freq)[:sch['N']]
        m = np.full(len(vals), len(keep), dtype=np.int64)
        m[keep] = np.arange(len(keep))
        out.append(m[tv][tt])
    return out, tcuts


# ---------------------------------------------------------------- information
def H(x):
    c = np.bincount(x); c = c[c > 0]; p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def MI(x, y):
    ny = int(y.max()) + 1
    j = np.bincount(x * ny + y); j = j[j > 0]; p = j / j.sum()
    return H(x) + H(y) - float(-(p * np.log2(p)).sum())


def CMI(x, y, z):
    """I(x;y|z) = H(x,z)+H(y,z)-H(x,y,z)-H(z)."""
    nx = int(x.max()) + 1; ny = int(y.max()) + 1
    xz = z * nx + x; yz = z * ny + y; xyz = (z * nx + x) * ny + y
    _, xyz = np.unique(xyz, return_inverse=True)
    return H(xz) + H(yz) - H(xyz) - H(z)


def score(C, sch, nulls, mask=None, varlist=None, dep=True, rng=None):
    codes, _ = slot_codes(C, sch)
    if mask is not None:
        codes = [c[mask] for c in codes]
    varlist = varlist or list(C['vars'])
    groups = sorted(set(C['groups'][v] for v in varlist))
    K = len(codes)
    E = np.zeros((K, len(varlist)))
    Hs = [H(c) for c in codes]
    for j, v in enumerate(varlist):
        y = C['vars'][v] if mask is None else C['vars'][v][mask]
        hy = H(y)
        nys = [(nd[v] if mask is None else nd[v][mask]) for nd in nulls]
        for s in range(K):
            if Hs[s] < 1e-9:
                continue
            real = MI(codes[s], y)
            nm = np.mean([MI(codes[s], yn) for yn in nys])
            E[s, j] = (real - nm) / max(1e-9, min(Hs[s], hy))
    G = np.zeros((K, len(groups)))
    for gi, g in enumerate(groups):
        idx = [j for j, v in enumerate(varlist) if C['groups'][v] == g]
        G[:, gi] = np.clip(E[:, idx].max(1), 0, None)
    FS = 0.0; tgt = {}
    for gi, g in enumerate(groups):
        pure = G[:, gi] - (G.sum(1) - G[:, gi])
        s = int(np.argmax(pure))
        if pure[s] > 0:
            FS += pure[s]; tgt[g] = s
    res = dict(FS=FS, TOT=float(G.sum()), E=E.round(4).tolist(), G=G.round(4).tolist(), groups=groups,
               varlist=varlist, H=[round(h, 3) for h in Hs], tgt=tgt, K=K)
    if dep:
        # conditional dependence of adjacent slots given the joint of the group-best variables
        z = np.zeros(len(codes[0]), dtype=np.int64)
        for gi, g in enumerate(groups):
            idx = [j for j, v in enumerate(varlist) if C['groups'][v] == g]
            v = varlist[idx[int(np.argmax(E[:, idx].max(0)))]]
            y = C['vars'][v] if mask is None else C['vars'][v][mask]
            z = z * (int(y.max()) + 1) + y
        _, z = np.unique(z, return_inverse=True)
        rng = rng or np.random.default_rng(0)
        order = np.argsort(z, kind='stable'); b = np.searchsorted(z[order], np.arange(z.max() + 2))
        dsum = 0.0; draw = 0.0
        for s in range(K - 1):
            a, c = codes[s], codes[s + 1]
            if Hs[s] < 1e-9 or Hs[s + 1] < 1e-9:
                continue
            real = CMI(a, c, z)
            cp = c.copy()
            for q in range(len(b) - 1):
                ix = order[b[q]:b[q + 1]]
                if len(ix) > 1:
                    cp[ix] = c[rng.permutation(ix)]
            nul = CMI(a, cp, z)
            dsum += (real - nul) / min(Hs[s], Hs[s + 1]); draw += real - nul
        res['DEP'] = dsum / max(1, K - 1)
    return res


def page_split(C, seed=0):
    rng = np.random.default_rng(seed)
    npg = C['page'].max() + 1
    A = rng.random(npg) < 0.5
    return A[C['page']]


# ---------------------------------------------------------------- controls
PL_S1 = [('q', 'o'), ('o',), ('C',), ('S',), ('d',), ('y',), ('s',), ('o', 'k')]
PL_S2 = [('k',), ('t',), ('k', 'e'), ('t', 'e'), ('p',), ('f',), ('k', 'C'), ('t', 'C')]
PL_S4 = [('y',), ('d', 'y'), ('e', 'y'), ('a', 'i', 'i', 'n'), ('o', 'l'), ('a', 'm')]


def planted_words(C, seed=0, p=0.6):
    """Synthetic catalogue of field codes in a Voynich-like alphabet on the real token labels.
    slot1 tracks SEC, slot2 tracks POS, slot3 serial (free), slot4 tracks NBR."""
    rng = np.random.default_rng(seed)
    al = ['a', 'i', 'n', 'r', 'l', 'e', 'o', 'd']
    serial = list({tuple(rng.choice(al, rng.integers(1, 4))) for _ in range(80)})[:40]
    sec, pos, nbr = C['vars']['SEC'], C['vars']['POS'], C['vars']['NBR']
    pref1 = rng.permutation(len(PL_S1)); pref2 = rng.permutation(len(PL_S2)); pref4 = rng.integers(0, len(PL_S4), nbr.max() + 1)
    W, T = [], []
    for i in range(C['n']):
        a = pref1[sec[i] % len(PL_S1)] if rng.random() < p else rng.integers(len(PL_S1))
        b = pref2[pos[i] % len(PL_S2)] if rng.random() < p else rng.integers(len(PL_S2))
        c = rng.integers(len(serial))
        d = pref4[nbr[i]] if rng.random() < p * 0.8 else rng.integers(len(PL_S4))
        W.append(PL_S1[a] + PL_S2[b] + serial[c] + PL_S4[d]); T.append((a, b, c, d))
    return W, np.array(T)


def opaque(strings, seed):
    chars = sorted(set(''.join(strings)))
    rng = random.Random(seed); sy = ['u%02d' % i for i in range(len(chars))]; rng.shuffle(sy)
    m = dict(zip(chars, sy))
    return [tuple(m[c] for c in s) for s in strings]


def gorila_corpus(seed=0):
    """Real catalogue designations: Linear A document ids (GORILA style: site + series + serial)."""
    import re
    d = json.load(open(os.path.join(vlib.ROOT, '..', 'linear-a', 'data', 'corpus.json')))
    ids, V = [], defaultdict(list)
    for x in d:
        s = re.sub(r'[^A-Za-z0-9]', '', x['id'])
        if not s:
            continue
        ids.append(s)
        V['SITE'].append(x['site']); V['SUPPORT'].append(x['support']); V['PERIOD'].append(x['context'] or '?')
    W = opaque(ids, seed)
    m = re.compile(r'^([A-Z]+?)((?:W[a-c]|Z[a-g]|[A-Z][a-z])?)(\d.*)?$')
    true = []
    for s in ids:
        g = m.match(s)
        true.append((g.group(1), g.group(2) or '-', g.group(3) or '-') if g else (s, '-', '-'))
    C = make_corpus(W, list(range(len(W))), V, {'SITE', 'SUPPORT', 'PERIOD'},
                    dict(SITE='SITE', SUPPORT='SUPPORT', PERIOD='PERIOD'), 'GORILA')
    return C, true


def unicode_corpus(seed=0, nmax=4000):
    """Real designation system: Unicode letter names abbreviated to the first 2 letters of each name word
    (e.g. LATIN SMALL LETTER A WITH ACUTE -> LASMLEAWIAC); external: block, case, decomposed."""
    import unicodedata
    blocks = [(0x0100, 0x024F, 'LatinExt'), (0x0370, 0x03FF, 'Greek'), (0x0400, 0x04FF, 'Cyrillic'),
              (0x0530, 0x058F, 'Armenian'), (0x10A0, 0x10FF, 'Georgian'), (0x1E00, 0x1EFF, 'LatinAdd'),
              (0x1F00, 0x1FFF, 'GreekExt'), (0x0600, 0x06FF, 'Arabic'), (0x0590, 0x05FF, 'Hebrew')]
    codes, V, true = [], defaultdict(list), []
    for a, b, nm in blocks:
        for cp in range(a, b + 1):
            ch = chr(cp); n = unicodedata.name(ch, '')
            if not n or not unicodedata.category(ch).startswith('L'):
                continue
            ws = n.split()
            codes.append(''.join(w[:2] for w in ws))
            V['BLOCK'].append(nm); V['CASE'].append(unicodedata.category(ch))
            V['DECOMP'].append('D' if unicodedata.decomposition(ch) else '-')
            true.append(ws)
    rng = random.Random(seed)
    idx = list(range(len(codes))); rng.shuffle(idx); idx = idx[:nmax]
    codes = [codes[i] for i in idx]; V = {k: [v[i] for i in idx] for k, v in V.items()}; true = [true[i] for i in idx]
    W = opaque(codes, seed)
    C = make_corpus(W, list(range(len(W))), V, set(V), dict(BLOCK='BLOCK', CASE='CASE', DECOMP='DECOMP'), 'UNICODE')
    return C, true


def nmi(a, b):
    _, a = np.unique(a, return_inverse=True); _, b = np.unique(b, return_inverse=True)
    return MI(a, b) / max(1e-9, min(H(a), H(b)))
