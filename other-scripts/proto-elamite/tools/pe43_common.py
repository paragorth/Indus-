"""pe43: THE SIGNS EVOLVE LIKE A FAMILY TREE, AND THE TREE IS THE DICTIONARY.

Shared loaders and the form 'phylogeny' (editorial derivation graph):
  compound |A+B|  -> parents = its component forms (A, B)
  variant  X~b    -> parent X   (X~a1 -> X~a -> X)
  modified X@g    -> parent X
A form's BASES are the root signs reached through these edges.

Corpora (same token format):
  PE : data/pe_corpus.json (1,585 tablets)
  PC : data/pe2_pc_corpus.json (proto-cuneiform administrative, Uruk IV / III)
  LEX: proto-cuneiform lexical lists read straight from the CDLI ATF dump (catalogue
       subgenre = list name), used ONLY as the answer key for the PC control.
  PLANT: synthetic corpus with a known variant-coining process (make_plant).
"""
import csv, json, os, re, random, sys
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe43_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('PE43_SCR', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad')

NUMTOK = re.compile(r'^\d')


def is_form(s):
    if not s or s in ('x', 'X', '...'):
        return False
    if NUMTOK.match(s) or s.startswith('N') and re.match(r'^N\d', s):
        return False
    return True


def comp_parts(form):
    """Component simple forms of a compound (numerals and X dropped)."""
    s = form.strip('|')
    s = re.sub(r'[()]', ' ', s)
    out = []
    for p in re.split(r'[x.+&%\s]+', s):
        if not p or p in ('X', 'x') or NUMTOK.match(p) or p[0] in '~@' or re.match(r'^N\d', p):
            continue
        out.append(p)
    return out


def strip_one(f):
    """Remove the last modifier of a simple form: ~var or @mod. None if bare."""
    m = re.match(r'^(.*?)(~[A-Za-z0-9]+|@[a-z0-9]+)$', f)
    if not m or not m.group(1):
        return None
    g = m.group(1)
    # X~a1 -> X~a (numeric suffix of a letter variant)
    m2 = re.match(r'^(.*~[A-Za-z]+)\d+$', f)
    if m2:
        return m2.group(1)
    return g


def parents(form):
    if form.startswith('|'):
        return comp_parts(form)
    p = strip_one(form)
    return [p] if p else []


def root(f):
    return re.sub(r'(~[A-Za-z0-9]+|@[a-z0-9]+)', '', f)


def bases(form):
    if form.startswith('|'):
        return sorted({root(p) for p in comp_parts(form)})
    return [root(form)]


def is_derived(form):
    return form.startswith('|') or '~' in form or '@' in form


def depth(form, _m={}):
    if form in _m:
        return _m[form]
    ps = parents(form)
    d = 0 if not ps else 1 + max(depth(p) for p in ps)
    _m[form] = d
    return d


# ---------------------------------------------------------------- loaders
def _mk_lines(lines, getforms, getcodes, getsurf):
    out = []
    for i, l in enumerate(lines):
        fs = [f for f in getforms(l) if is_form(f)]
        cs = getcodes(l)
        if not fs and not cs:
            continue
        out.append({'forms': fs, 'codes': cs, 'surf': getsurf(l), 'li': i})
    if out and out[0]['forms'] and not out[0]['codes']:
        out[0]['header'] = 1
    return out


def load_pe():
    T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    R = []
    for t in T:
        L = _mk_lines(t['lines'], lambda l: [s for s in l['signs'] if s.startswith('M') or s.startswith('|')],
                      lambda l: [c for _, c in l['numerals']], lambda l: 0 if l['surface'] == 'obverse' else 1)
        if L:
            R.append({'id': t['id'], 'site': t['provenience'], 'lines': L})
    return R


def load_pc():
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    R = []
    for t in T:
        L = _mk_lines(t['lines'], lambda l: l['signs'], lambda l: [c for _, c in l['numerals']],
                      lambda l: 0 if l['surface'] == 'obverse' else 1)
        if L:
            R.append({'id': t['id'], 'site': t['provenience'], 'period': t['period'], 'lines': L})
    return R


LEXMAP = {'Lu2 A': 'PERSONS', 'Officials': 'PERSONS', 'Animals': 'ANIMALS', 'Fish': 'FISH',
          'Birds': 'BIRDS', 'Plant': 'PLANTS', 'Wood': 'WOOD', 'Vessels': 'VESSELS',
          'Metal': 'METAL', 'Grain': 'GRAIN', 'Tribute': 'TRIBUTE', 'Cities': 'PLACES',
          'Geography': 'PLACES', 'Pigs': 'ANIMALS'}


def load_lex():
    """{pid: (family, [forms...])} for archaic lexical witnesses (forms keep variants)."""
    fn = os.path.join(CK, 'lex.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    csv.field_size_limit(10 ** 9)
    fam = {}
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), errors='replace')):
        if not (r['period'].startswith('Uruk') and 'exical' in r['genre']):
            continue
        sg = r['subgenre']
        for k, v in LEXMAP.items():
            if k in sg and ';' not in sg:
                if r['id_text'].isdigit():
                    fam['P%06d' % int(r['id_text'])] = v
                break
    out, cur, buf = {}, None, []
    with open(os.path.join(SCR, 'cdli.atf'), errors='replace') as f:
        for line in f:
            if line.startswith('&P'):
                if cur:
                    out[cur] = (fam[cur], buf)
                cur = line[1:8] if line[1:8] in fam else None
                buf = []
                continue
            if cur is None:
                continue
            m = re.match(r"^([0-9]+[a-z0-9.']*)\.\s+(.*)$", line.rstrip('\n'))
            if not m:
                continue
            raw = re.sub(r'\[[^\]]*\]', ' ', m.group(2))
            parts = raw.split(',', 1)
            body = parts[1] if len(parts) == 2 else parts[0]
            for w in body.split():
                w = re.sub(r'[#!?*<>\[\]]', '', w)
                if is_form(w) and not re.match(r'^\d+\(', w) and not w.startswith('$'):
                    buf.append(w)
    if cur:
        out[cur] = (fam[cur], buf)
    json.dump(out, open(fn, 'w'))
    return out


def lex_families(min_n=3, share=0.6):
    """root sign -> lexical family, only when concentrated in one list."""
    L = load_lex()
    c = defaultdict(Counter)
    for pid, (fm, forms) in L.items():
        for f in forms:
            for b in bases(f):
                c[b][fm] += 1
    out = {}
    for b, cc in c.items():
        n = sum(cc.values())
        f, k = cc.most_common(1)[0]
        if n >= min_n and k / n >= share:
            out[b] = f
    return out


# ---------------------------------------------------------------- contexts
def code_rarity(T):
    c = Counter(cd for t in T for l in t['lines'] for cd in l['codes'])
    return c


def token_contexts(T):
    """yield (tablet index, form, SYS, POS) for every form token."""
    cr = code_rarity(T)
    for ti, t in enumerate(T):
        for l in t['lines']:
            sysk = min(l['codes'], key=lambda c: (cr[c], c)) if l['codes'] else 'none'
            n = len(l['forms'])
            for j, f in enumerate(l['forms']):
                if l.get('header'):
                    pos = 'H'
                elif n == 1:
                    pos = 'S'
                elif j == 0:
                    pos = 'F'
                elif j == n - 1:
                    pos = 'L'
                else:
                    pos = 'M'
                yield ti, f, sysk, pos + str(l['surf'])


# ---------------------------------------------------------------- planted corpus
def make_plant(seed=0, n_tab=1500, K=12, per_fam=14, n_codes=10, coin_rate=0.9, comp_in=0.8,
               vigor=True):
    """Synthetic archive with a known variant-coining process.

    K meaning families of root signs; each family has its own numeral-code profile and
    position habits (meaning -> context). Tablets have a time t ~ U(0,1) and a topic family.
    Coining: each root sign spawns variants (~a, ~b, ...) at random birth times; families have a
    heritable coining rate (vigor). A variant is used (instead of its parent) with probability
    rising after its birth. Compounds |A+B| are born from two roots, same family with prob comp_in.
    Returns tablets (same format as load_pe), truth dict."""
    rng = np.random.default_rng(seed)
    roots = [f'S{k:02d}{i:02d}' for k in range(K) for i in range(per_fam)]
    fam = {r: int(r[1:3]) for r in roots}
    fam_rate = rng.gamma(2.0, 0.5, K) if vigor else np.ones(K)
    fam_codes = [rng.dirichlet(np.ones(n_codes) * 0.3) for _ in range(K)]
    fam_pos = [rng.dirichlet(np.ones(3)) for _ in range(K)]
    freq = {r: rng.pareto(1.2) + 0.2 for r in roots}
    births = {}   # form -> (parent(s), birth time)
    kids = defaultdict(list)
    for r in roots:
        nv = rng.poisson(coin_rate * fam_rate[fam[r]] * min(3, freq[r]))
        for v in range(nv):
            b = rng.uniform(0, 1)
            f = r + '~' + 'abcdefghij'[v % 10]
            births[f] = ([r], b)
            kids[r].append(f)
    n_comp = int(0.15 * len(roots))
    for _ in range(n_comp):
        a = rng.choice(roots)
        if rng.random() < comp_in:
            b = rng.choice([x for x in roots if fam[x] == fam[a] and x != a])
        else:
            b = rng.choice([x for x in roots if fam[x] != fam[a]])
        f = f'|{a}+{b}|'
        if f in births:
            continue
        births[f] = ([a, b], rng.uniform(0, 1))
        kids[a].append(f)
    T = []
    by_fam = defaultdict(list)
    for r in roots:
        by_fam[fam[r]].append(r)
    for ti in range(n_tab):
        t = rng.uniform(0, 1)
        topic = rng.integers(K)
        L = []
        nl = 1 + rng.poisson(4)
        for li in range(nl):
            k = topic if rng.random() < 0.75 else rng.integers(K)
            pool = by_fam[k]
            w = np.array([freq[r] for r in pool]); w /= w.sum()
            r = pool[rng.choice(len(pool), p=w)]
            # choose the form: any alive descendant, used with prob rising after birth
            cands = [r]
            for f in kids[r]:
                if t > births[f][1]:
                    cands.append(f)
            f = cands[rng.integers(len(cands))] if len(cands) > 1 and rng.random() < 0.6 else r
            # a second sign: same family often
            forms = [f]
            if rng.random() < 0.4:
                pool2 = by_fam[k if rng.random() < 0.7 else rng.integers(K)]
                forms.append(pool2[rng.integers(len(pool2))])
            pos = rng.choice(3, p=fam_pos[k])
            if pos == 1:
                forms = forms[::-1]
            code = f'N{rng.choice(n_codes, p=fam_codes[k]):02d}'
            L.append({'forms': forms, 'codes': [code], 'surf': int(li > 4), 'li': li})
        T.append({'id': f'PL{seed}_{ti}', 'site': 'plant', 't': float(t), 'lines': L})
    truth = {'fam': {r: f'F{fam[r]}' for r in roots}, 'births': {f: v[1] for f, v in births.items()},
             'fam_rate': fam_rate.tolist()}
    return T, truth


# ---------------------------------------------------------------- co-occurrence graph
def tablet_bases(T, vmap=None):
    """list of sets of root signs per tablet; vmap optionally remaps forms -> other root."""
    out = []
    for t in T:
        s = set()
        for l in t['lines']:
            for f in l['forms']:
                if vmap and f in vmap:
                    s.add(vmap[f])
                else:
                    s.update(bases(f))
        out.append(s)
    return out


def cooc(sets, idx):
    n = len(idx)
    A = np.zeros((n, n))
    for s in sets:
        ii = [idx[b] for b in s if b in idx]
        if len(ii) < 2:
            continue
        w = 1.0 / (len(ii) - 1)
        ii = np.array(ii)
        A[np.ix_(ii, ii)] += w
    np.fill_diagonal(A, 0)
    return A


def modularity(A, lab):
    W2 = A.sum()
    if W2 == 0:
        return 0.0
    lab = np.asarray(lab)
    inn = (A * (lab[:, None] == lab[None, :])).sum()
    dc = np.bincount(lab, weights=A.sum(1))
    return inn / W2 - ((dc / W2) ** 2).sum()


def relabel(lab):
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def forest_labels(par):
    n = len(par)
    lab = np.arange(n)
    # follow parents to root (par[i] == -1 root); assumes acyclic
    for i in range(n):
        j, seen = i, 0
        while par[j] >= 0 and seen <= n:
            j = par[j]; seen += 1
        lab[i] = j
    return relabel(lab)


def pair_precision(lab, truth_lab):
    """share of same-family pairs (under lab) whose members share the truth family (both labelled)."""
    same = tot = 0
    by = defaultdict(list)
    for i, l in enumerate(lab):
        if truth_lab[i] is not None:
            by[l].append(truth_lab[i])
    for l, v in by.items():
        c = Counter(v)
        n = len(v)
        tot += n * (n - 1) / 2
        same += sum(x * (x - 1) / 2 for x in c.values())
    return same / tot if tot else float('nan'), tot


def ari(a, b):
    from math import comb
    ct = Counter(zip(a, b))
    sa = Counter(a); sb = Counter(b); n = len(a)
    s_ij = sum(comb(v, 2) for v in ct.values())
    s_a = sum(comb(v, 2) for v in sa.values()); s_b = sum(comb(v, 2) for v in sb.values())
    e = s_a * s_b / comb(n, 2)
    mx = (s_a + s_b) / 2
    return (s_ij - e) / (mx - e) if mx != e else 0.0
