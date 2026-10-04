"""v36 IS THE A->B SHIFT A FEATURAL SOUND CHANGE?  Shared library.

A rewrite rule (lhs, rhs, lc, rc) from the v30 search is applied ALONE to the source word types;
each changed word is cut into glyph units and aligned (difflib) with its output.  From the
alignment we collect, per rule:
  targets  glyph types of the source that are deleted or replaced (mass 1 per rule, spread by token count)
  contexts glyph types immediately left / right of the changed span (word edge '#' counted separately)
  subs     1:1 replaced pairs a -> b (equal-length replace spans)
Statistics, each against a frequency-stratified permutation of glyph labels on the feature-similarity
matrix (= random glyph classes of the same size, mass profile and frequency):
  COH  mass-weighted mean pairwise feature similarity of the targeted glyph types (natural class?)
  CTX  same for the context glyph types
  SUB  mean feature similarity of a and b in 1:1 substitutions (minimal feature change?)
Feature descriptions: Voynich = v25 hand stroke primitives (weighted Jaccard) and v25 font-image
similarity; Hangul = v25 jamo stroke primitives; Latin-script corpora = phonetic value of the letter
(binary features, Jaccard), combining marks with a MARK feature.
"""
import os, sys, json, random, difflib, collections, unicodedata
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v18_lib import glyphs as vglyphs
import v25_shapes as S
from v30_lib import rules_by_first, apply_word

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v36_ckpt'); os.makedirs(CK, exist_ok=True)
CK30 = os.path.join(ROOT, 'data', 'v30_ckpt')
LOOPS = os.path.join(ROOT, 'loops')

# ------------------------------------------------------------------ phonetic letter features
_V = {'V'}
PHON = {
    'a': {'V', 'low'}, 'e': {'V', 'front', 'mid'}, 'i': {'V', 'front', 'high'}, 'y': {'V', 'front', 'high', 'round'},
    'o': {'V', 'back', 'mid', 'round'}, 'u': {'V', 'back', 'high', 'round'},
    'v': {'V', 'back', 'high', 'round', 'C', 'labial', 'fric', 'voiced'},  # medieval u/v
    'w': {'C', 'glide', 'labial', 'voiced', 'round'}, 'j': {'C', 'glide', 'palatal', 'voiced', 'high', 'front'},
    'b': {'C', 'stop', 'labial', 'voiced'}, 'p': {'C', 'stop', 'labial'}, 'm': {'C', 'nasal', 'labial', 'voiced'},
    'f': {'C', 'fric', 'labial'}, 'd': {'C', 'stop', 'coronal', 'voiced'}, 't': {'C', 'stop', 'coronal'},
    'n': {'C', 'nasal', 'coronal', 'voiced'}, 's': {'C', 'fric', 'coronal', 'sibilant'}, 'ſ': {'C', 'fric', 'coronal', 'sibilant'},
    'z': {'C', 'fric', 'coronal', 'sibilant', 'voiced'}, 'ß': {'C', 'fric', 'coronal', 'sibilant'},
    'l': {'C', 'liquid', 'coronal', 'voiced'}, 'r': {'C', 'liquid', 'coronal', 'voiced', 'trill'},
    'c': {'C', 'stop', 'dorsal'}, 'k': {'C', 'stop', 'dorsal'}, 'q': {'C', 'stop', 'dorsal', 'round'},
    'g': {'C', 'stop', 'dorsal', 'voiced'}, 'h': {'C', 'fric', 'dorsal', 'glottal'}, 'x': {'C', 'stop', 'dorsal', 'fric', 'sibilant'},
}
MARKS = {
    '́': {'MARK', 'long'}, '̂': {'MARK', 'long'}, '̌': {'MARK', 'palatal'}, '̊': {'MARK', 'long', 'round'},
    '̈': {'MARK', 'front'}, '̃': {'MARK', 'ABBR', 'nasal'}, '̄': {'MARK', 'ABBR', 'nasal'},
    '̅': {'MARK', 'ABBR'}, '̾': {'MARK', 'ABBR'}, '̇': {'MARK', 'ABBR'},
    'ͦ': {'MARK', 'V', 'back', 'mid', 'round'}, 'ͤ': {'MARK', 'V', 'front', 'mid'}, 'ͥ': {'MARK', 'V', 'front', 'high'},
    'ͧ': {'MARK', 'V', 'back', 'high', 'round'}, 'ͣ': {'MARK', 'V', 'low'}, 'ᵃ': {'MARK', 'V', 'low'},
    'ˢ': {'MARK', 'C', 'fric', 'coronal', 'sibilant'}, '᷑': {'MARK', 'ABBR', 'liquid', 'coronal'},
    '⁊': {'ABBR'}, '&': {'ABBR'}, 'ꝙ': {'ABBR', 'C', 'stop', 'dorsal'}, 'ꝑ': {'ABBR', 'C', 'stop', 'labial'},
    'ꝓ': {'ABBR', 'C', 'stop', 'labial'}, 'ꝗ': {'ABBR', 'C', 'stop', 'dorsal'}, 'ꝯ': {'ABBR'}, 'ꝰ': {'ABBR'},
}


def phon_feats(c):
    if c in PHON:
        return PHON[c]
    if c in MARKS:
        return MARKS[c]
    if unicodedata.category(c).startswith('M'):
        return {'MARK', 'X:' + c}
    b = unicodedata.normalize('NFD', c)[0]
    if b in PHON and b != c:
        return PHON[b] | {'MARK'}
    return {'X:' + c}


# ------------------------------------------------------------------ similarity matrices
def wjacc(a, b):
    ks = set(a) | set(b)
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in ks); den = sum(max(a.get(k, 0), b.get(k, 0)) for k in ks)
    return num / den if den else 0.0


def jacc(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


class Feat:
    """Glyph inventory + similarity matrix + binary feature sets for enrichment."""
    def __init__(self, kind, alph, freq):
        self.kind = kind
        self.alph = list(alph); self.ix = {g: i for i, g in enumerate(self.alph)}
        self.freq = np.array([freq.get(g, 0) for g in self.alph], float)
        n = len(self.alph)
        if kind == 'vhand':
            F = [S.VOYNICH[g] for g in self.alph]
            self.M = np.array([[wjacc(F[i], F[j]) for j in range(n)] for i in range(n)])
            self.sets = [set(f) for f in F]
        elif kind == 'vimg':
            import v25_image as I
            M = I.image_sims('voynich', self.alph)['combo']
            self.M = (M - M.min()) / (M.max() - M.min())
            self.sets = [set(S.VOYNICH[g]) for g in self.alph]
        elif kind == 'hangul':
            F = [S.HANGUL.get(g, {'X:' + g: 1}) for g in self.alph]
            self.M = np.array([[wjacc(F[i], F[j]) for j in range(n)] for i in range(n)])
            self.sets = [set(f) for f in F]
        elif kind == 'phon':
            F = [phon_feats(g) for g in self.alph]
            self.M = np.array([[jacc(F[i], F[j]) for j in range(n)] for i in range(n)])
            self.sets = F
        elif kind == 'vrandom':  # random decomposition with the same primitive budget (not used as null)
            raise ValueError
        self.strata = self._strata()

    def _strata(self, k=4):
        o = np.argsort(-self.freq); st = np.zeros(len(o), int)
        for r, i in enumerate(o):
            st[i] = min(k - 1, r * k // len(o))
        return st

    def perm(self, rng):
        p = np.arange(len(self.alph))
        for s in np.unique(self.strata):
            idx = np.where(self.strata == s)[0]; q = idx.copy(); rng.shuffle(q); p[idx] = q
        return p


# ------------------------------------------------------------------ tokenisers
def tok_for(script):
    if script == 'voynich':
        return vglyphs
    return list


# ------------------------------------------------------------------ rule effects
def rule_effect(rule, types, tok):
    """types: Counter word->count.  Returns dict(targets=Counter, ctx=Counter, subs=Counter, edge=float, mass=int)."""
    rb = rules_by_first([tuple(rule)])
    T = collections.Counter(); X = collections.Counter(); SB = collections.Counter(); OUT = collections.Counter()
    edge = 0.0; tot = 0.0; mass = 0
    for w, c in types.items():
        o = apply_word(w, rb)
        if o == w:
            continue
        a = tok(w); b = tok(o)
        mass += c
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'equal':
                continue
            for g in a[i1:i2]:
                T[g] += c
            for g in b[j1:j2]:
                OUT[g] += c
            if op == 'replace' and i2 - i1 == j2 - j1:
                for x, y in zip(a[i1:i2], b[j1:j2]):
                    if x != y:
                        SB[(x, y)] += c
            left = a[i1 - 1] if i1 > 0 else '#'; right = a[i2] if i2 < len(a) else '#'
            for g in (left, right):
                tot += c
                if g == '#':
                    edge += c
                else:
                    X[g] += c
    return dict(targets=T, ctx=X, subs=SB, out=OUT, edge=edge / tot if tot else 0.0, mass=mass)


def pooled(effects, key, alph_ix):
    """Each rule contributes mass 1, spread over its glyph types by token count; unknown glyphs dropped."""
    v = np.zeros(len(alph_ix))
    for e in effects:
        C = e[key]; s = sum(c for g, c in C.items() if g in alph_ix)
        if s <= 0:
            continue
        for g, c in C.items():
            if g in alph_ix:
                v[alph_ix[g]] += c / s
    return v


def coh(v, M):
    W = np.outer(v, v); np.fill_diagonal(W, 0)
    d = W.sum()
    return (W * M).sum() / d if d > 0 else np.nan


def sub_sim(effects, F, Mp=None):
    M = F.M if Mp is None else Mp
    num = den = 0.0
    for e in effects:
        s = sum(c for (a, b), c in e['subs'].items() if a in F.ix and b in F.ix)
        if s <= 0:
            continue
        for (a, b), c in e['subs'].items():
            if a in F.ix and b in F.ix:
                num += (c / s) * M[F.ix[a], F.ix[b]]; den += c / s
    return num / den if den else np.nan


def stats(effects, F, nperm=2000, seed=0):
    """Observed COH / CTX / SUB with frequency-stratified label-permutation null (z, percentile)."""
    rng = np.random.default_rng(seed)
    vt = pooled(effects, 'targets', F.ix); vc = pooled(effects, 'ctx', F.ix)
    obs = dict(COH=coh(vt, F.M), CTX=coh(vc, F.M), SUB=sub_sim(effects, F))
    null = {k: [] for k in obs}
    for _ in range(nperm):
        p = F.perm(rng); Mp = F.M[np.ix_(p, p)]
        null['COH'].append(coh(vt, Mp)); null['CTX'].append(coh(vc, Mp)); null['SUB'].append(sub_sim(effects, F, Mp))
    out = {}
    for k in obs:
        a = np.array(null[k], float); a = a[~np.isnan(a)]
        if np.isnan(obs[k]) or len(a) == 0:
            out[k] = dict(obs=np.nan, z=np.nan, pct=np.nan); continue
        out[k] = dict(obs=float(obs[k]), z=float((obs[k] - a.mean()) / (a.std() + 1e-12)), pct=float((a < obs[k]).mean()),
                      p=float(((a >= obs[k]).sum() + 1) / (len(a) + 1)), null_mean=float(a.mean()))
    out['edge'] = float(np.mean([e['edge'] for e in effects])) if effects else np.nan
    out['n_targets'] = int((vt > 0).sum()); out['n_subs'] = int(sum(1 for e in effects if e['subs']))
    return out


def enrichment(effects, F, nperm=2000, seed=0):
    """Feature-level: share of targeted mass on glyphs carrying feature f vs share under permuted labels."""
    rng = np.random.default_rng(seed)
    vt = pooled(effects, 'targets', F.ix); vt = vt / max(vt.sum(), 1e-12)
    feats = sorted(set().union(*F.sets))
    H = np.array([[1.0 if f in s else 0.0 for s in F.sets] for f in feats])  # feat x glyph
    obs = H @ vt
    nulls = np.array([H[:, F.perm(rng)] @ vt for _ in range(nperm)])
    res = []
    for i, f in enumerate(feats):
        a = nulls[:, i]
        res.append(dict(f=f, obs=float(obs[i]), null=float(a.mean()), p_hi=float(((a >= obs[i]).sum() + 1) / (nperm + 1)),
                        p_lo=float(((a <= obs[i]).sum() + 1) / (nperm + 1))))
    return res


# ------------------------------------------------------------------ v30 corpora / rule sets
_C = None


def corpora30():
    global _C
    if _C is None:
        _C = json.load(open(os.path.join(CK30, 'corpora.json')))['corpora']
    return _C


def load_rules(path, k=None):
    d = json.load(open(path))
    rules = [tuple(r) for r in d['path'][-1]['rules']]
    return (rules[:k] if k else rules), d


def counts_of(pages):
    return collections.Counter(w for p in pages for w in p)


def glyph_freq(types, tok):
    f = collections.Counter()
    for w, c in types.items():
        for g in tok(w):
            f[g] += c
    return f


def write_rows(path, rows, header=None):
    with open(path, 'a', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(x) for x in r) + ' |\n')
