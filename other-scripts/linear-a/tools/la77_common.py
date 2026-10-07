"""LA-77 'sign-groups as organisms': mutant families, random phylogenies, phylogeography.

Sign-groups (2+ syllabic signs, cleanly read, no edge/continuation/unsure flags, damage-aware
corpus_ra.json) that differ by one substitution or one insertion/deletion are joined into
families. Trees over families are built under random mutation models; their edges are read as
transmissions between find-sites. Site coordinates come from the la31 gazetteer; deposit phases
(context field) are used ONLY in cycle 3, after the inferred directions were frozen by sha256.
No sound values are used.
"""
import os, sys, json, re, collections, math, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'la77'); CKPT = os.path.join(DATA, 'la77_ckpt')
for p in (OUT, CKPT):
    os.makedirs(p, exist_ok=True)
from la31_common import SITES, hav  # gazetteer only

NAME2CODE = {v[0]: k for k, v in SITES.items()}
BAD_FL = {'edgeR', 'edgeL', 'cont', 'part', 'unid', 'unsure', 'nodraw', 'erased'}
POS = ['ini', 'med', 'fin']          # position classes
OPS = ['sub', 'ins']                  # ins = one form has an extra sign (undirected: indel)
KCLS = [(o, p) for o in OPS for p in POS]


def _logo(s):
    return bool(re.fullmatch(r'\*[4-9]\d\d.*', s)) or s in ('VS', 'VAS') or bool(re.search(r'[a-z]', s))


def load_docs():
    from la15_common import load_la
    keep = {d['id'] for d in load_la()}
    C = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    docs = []
    for d in C:
        if d['site'] not in NAME2CODE or d['id'] not in keep:
            continue
        ws = []
        for t in d['tokens']:
            if t['t'] != 'word' or t.get('st') not in ('read', 'damaged'):
                continue
            if set(t.get('fl', [])) & BAD_FL:
                continue
            s = tuple(t['s'])
            if len(s) < 2 or any(_logo(x) for x in s):
                continue
            ws.append(s)
        docs.append(dict(id=d['id'], site=NAME2CODE[d['site']], support=d.get('support', ''),
                         context=d.get('context', ''), words=ws))
    return docs


def edit1(a, b):
    """Return (op, posclass, index) if a,b differ by one sub or one indel, else None."""
    la, lb = len(a), len(b)
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) != 1:
            return None
        i = diff[0]
        return ('sub', 'ini' if i == 0 else ('fin' if i == la - 1 else 'med'), i)
    if abs(la - lb) != 1:
        return None
    if la < lb:
        a, b = b, a; la, lb = lb, la      # a longer
    for i in range(la):
        if a[:i] + a[i + 1:] == b:
            return ('ins', 'ini' if i == 0 else ('fin' if i == la - 1 else 'med'), i)
    return None


def families(types):
    """types: list of tuples. Returns edges [(i,j,op,pos)] and components (lists of idx, size>=2)."""
    idx = {t: i for i, t in enumerate(types)}
    edges = []
    # neighbour generation via deletion keys (fast)
    bylen = collections.defaultdict(list)
    for i, t in enumerate(types):
        bylen[len(t)].append(i)
    # substitution: wildcard keys
    wk = collections.defaultdict(list)
    for i, t in enumerate(types):
        for k in range(len(t)):
            wk[(t[:k], '*', t[k + 1:])].append(i)
    seen = set()
    for g in wk.values():
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                i, j = g[x], g[y]
                e = edit1(types[i], types[j])
                if e and len(types[i]) >= 3 and (i, j) not in seen:   # share >= 2 signs
                    seen.add((i, j)); edges.append((i, j, e[0], e[1]))
    for i, t in enumerate(types):
        for k in range(len(t)):
            sh = t[:k] + t[k + 1:]
            if len(sh) >= 2 and sh in idx:
                j = idx[sh]
                key = (min(i, j), max(i, j))
                if key not in seen:
                    e = edit1(t, sh)
                    seen.add(key); edges.append((key[0], key[1], e[0], e[1]))
    # components
    par = list(range(len(types)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j, _, _ in edges:
        par[f(i)] = f(j)
    comp = collections.defaultdict(list)
    for i in range(len(types)):
        comp[f(i)].append(i)
    fams = [sorted(c) for c in comp.values() if len(c) >= 2]
    return edges, fams


def type_table(docs):
    """types, per-type site token counts (dict code->n)."""
    cnt = collections.defaultdict(collections.Counter)
    for d in docs:
        for w in d['words']:
            cnt[w][d['site']] += 1
    types = sorted(cnt)
    return types, [cnt[t] for t in types]


def opportunities(t):
    """Number of mutation opportunities per class for a word of length L (alphabet-free)."""
    L = len(t)
    o = {('sub', 'ini'): 1, ('sub', 'fin'): 1, ('sub', 'med'): max(L - 2, 0),
         ('ins', 'ini'): 2, ('ins', 'fin'): 2, ('ins', 'med'): max(L - 2, 0) + max(L - 1, 0)}
    return np.array([o[k] for k in KCLS], float)


def shuffle_within_site(docs, rng):
    """Control: sign tokens of clean words permuted within site, word lengths kept."""
    out = []
    pool = collections.defaultdict(list)
    for d in docs:
        for w in d['words']:
            pool[d['site']].extend(w)
    for s in pool:
        rng.shuffle(pool[s])
    ptr = collections.Counter()
    for d in docs:
        ws = []
        for w in d['words']:
            k = ptr[d['site']]; ws.append(tuple(pool[d['site']][k:k + len(w)])); ptr[d['site']] += len(w)
        out.append(dict(d, words=ws))
    return out


def permute_sites(docs, rng):
    """Null: site labels permuted over documents within support class."""
    by = collections.defaultdict(list)
    for k, d in enumerate(docs):
        by[d['support']].append(k)
    lab = [d['site'] for d in docs]
    new = list(lab)
    for ks in by.values():
        p = rng.permutation(len(ks))
        for a, b in zip(ks, p):
            new[a] = lab[ks[b]]
    return [dict(d, site=new[k]) for k, d in enumerate(docs)]


def km_matrix(codes):
    return np.array([[hav(SITES[a][1:], SITES[b][1:]) for b in codes] for a in codes])


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
