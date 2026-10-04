"""v29 FEATURES SPREAD.  Does a stroke feature of the Voynich glyphs spread across a word (vowel harmony,
palatal agreement) or across word junctions?

A feature is (tier, value): tier = the set of glyphs that carry the feature at all (others are transparent),
value = a binary split of the tier.  Agreement = phi between the values of two tier members.
Statistics (per feature):
  tier1..3 : tier member and the d-th next tier member in the same word (transparent glyphs skipped)
  raw1..3  : two tier glyphs exactly d glyphs apart in the same word
  junc     : last tier member of a word and first tier member of the next word on the same line
  dir      : words with >= 3 tier members: agree(last two) - agree(first two)  (> 0: suffix agrees = left-to-right)
Nulls: Markov-2 (glyph trigram) within-word resynthesis for the within-word statistics; within-line word
shuffle for junctions; frequency-matched relabelling of glyph identities for the 'is it the stroke feature'
question; max-statistic over the whole search (leave-one-out on the resyntheses) for the search correction.
Controls: Turkish, Hungarian, Finnish (vowel harmony), Old Czech (palatal agreement), Latin, Hangul jamo
with v25 stroke shapes, and harmony planted into Voynich text.
"""
import os, sys, json, re, random
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v18_lib import glyphs as vglyphs
import v25_shapes as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v29_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('V29_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad')
TOK = 150000
STATS = ['tier1', 'tier2', 'tier3', 'raw1', 'raw2', 'raw3', 'junc', 'dir']

# ------------------------------------------------------------------ featural transcriptions (phonology)
_V = dict
PH = {
    'a': 'V LOW BACK', 'á': 'V LOW BACK LONG', 'à': 'V LOW BACK', 'â': 'V LOW BACK',
    'e': 'V MID FRONT', 'é': 'V MID FRONT LONG', 'ě': 'V MID FRONT PAL', 'ë': 'V MID FRONT',
    'i': 'V HIGH FRONT', 'í': 'V HIGH FRONT LONG', 'î': 'V HIGH FRONT',
    'ı': 'V HIGH BACK',
    'o': 'V MID BACK ROUND', 'ó': 'V MID BACK ROUND LONG', 'ô': 'V MID BACK ROUND',
    'ö': 'V MID FRONT ROUND', 'ő': 'V MID FRONT ROUND LONG',
    'u': 'V HIGH BACK ROUND', 'ú': 'V HIGH BACK ROUND LONG', 'ů': 'V HIGH BACK ROUND LONG', 'û': 'V HIGH BACK ROUND',
    'ü': 'V HIGH FRONT ROUND', 'ű': 'V HIGH FRONT ROUND LONG',
    'ä': 'V LOW FRONT',
    'b': 'C STOP LAB VOI', 'c': 'C STOP COR', 'ç': 'C STOP PAL', 'č': 'C STOP PAL', 'd': 'C STOP COR VOI',
    'ď': 'C STOP PAL VOI', 'f': 'C FRIC LAB', 'g': 'C STOP DOR VOI', 'ğ': 'C FRIC DOR VOI', 'h': 'C FRIC DOR',
    'j': 'C LIQ PAL VOI', 'k': 'C STOP DOR', 'l': 'C LIQ COR VOI', 'ł': 'C LIQ DOR VOI', 'm': 'C NAS LAB VOI',
    'n': 'C NAS COR VOI', 'ň': 'C NAS PAL VOI', 'p': 'C STOP LAB', 'q': 'C STOP DOR', 'r': 'C LIQ COR VOI',
    'ř': 'C LIQ PAL VOI', 's': 'C FRIC COR', 'ś': 'C FRIC PAL', 'š': 'C FRIC PAL', 'ş': 'C FRIC PAL',
    't': 'C STOP COR', 'ť': 'C STOP PAL', 'v': 'C FRIC LAB VOI', 'w': 'C LIQ LAB VOI', 'x': 'C FRIC DOR',
    'z': 'C FRIC COR VOI', 'ž': 'C FRIC PAL VOI', 'ź': 'C FRIC PAL VOI', 'ż': 'C FRIC PAL VOI',
}
LANG_Y = {'fi': 'V HIGH FRONT ROUND', 'cs': 'V HIGH FRONT', 'hu': 'C LIQ PAL VOI', 'tr': 'C LIQ PAL VOI',
          'la': 'V HIGH FRONT ROUND'}
LANG_S = {'hu': 'C FRIC PAL'}


def phon_shapes(lang, letters):
    out = {}
    for c in letters:
        if c == 'y':
            f = LANG_Y[lang]
        elif c in LANG_S and lang in ('hu',) and c == 's':
            f = LANG_S['hu']
        elif c in PH:
            f = PH[c]
        else:
            continue
        out[c] = {k: 1 for k in f.split()}
    return out


# ------------------------------------------------------------------ corpora: list of lines, line = list of word tuples
def voynich_lines(src='ZL3b', lang=None, fold=None):
    d = json.load(open(os.path.join(DATA, 'derived', f'{src}_lines.json')))
    keep = set(S.VOYNICH)
    out = []
    for L in d:
        if L['ltype'] != 'P' or (lang and L['lang'] != lang):
            continue
        if fold is not None and int((re.search(r'(\d+)', L['folio']) or re.search(r'(\d)', '1')).group(1)) % 2 != fold:
            continue
        cur = []
        for w, u in zip(L['words'], L['uncertain']):
            g = tuple(vglyphs(w)) if not (u or '?' in w) else None
            if g and all(x in keep for x in g):
                cur.append(g)
            else:
                if cur: out.append(cur)
                cur = []          # an unreadable word breaks the junction chain
        if cur:
            out.append(cur)
    return out


def _chunk(words, n=8):
    return [words[i:i + n] for i in range(0, len(words), n)]


def _cap(lines, tok):
    out, n = [], 0
    for L in lines:
        out.append(L); n += sum(len(w) for w in L)
        if n >= tok:
            break
    return out


def leipzig(code, lang, tok=TOK):
    f = os.path.join(SCR, 'harm', f'{code}_news_2020_10K', f'{code}_news_2020_10K-sentences.txt')
    lines = []
    for s in open(f, encoding='utf-8'):
        s = s.split('\t', 1)[-1]
        if lang == 'tr':
            s = s.replace('I', 'ı').replace('İ', 'i')
        s = s.lower()
        ws = re.findall(r"[^\W\d_]+", s)
        cur = []
        for w in ws:
            if all(c in PH or c == 'y' for c in w):
                cur.append(tuple(w))
            else:
                if cur: lines.extend(_chunk(cur))
                cur = []
        if cur:
            lines.extend(_chunk(cur))
    random.Random(1).shuffle(lines)  # Leipzig sentences are sorted alphabetically
    return _cap(lines, tok)


def plain_lines(lang, tok=TOK):
    t = open(os.path.join(DATA, 'plain', f'{lang}.txt'), encoding='utf-8').read().lower()
    if lang == 'la':
        t = t.replace('j', 'i').replace('v', 'u')
    lines = []
    for raw in t.split('\n'):
        cur = []
        for w in re.findall(r"[^\W\d_]+", raw):
            if all(c in PH or c == 'y' for c in w):
                cur.append(tuple(w))
            else:
                if cur: lines.extend(_chunk(cur))
                cur = []
        if cur:
            lines.extend(_chunk(cur))
    return _cap(lines, tok)


_CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
_JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
_JONG = ['', 'ㄱ', 'ㄲ', 'ㄱㅅ', 'ㄴ', 'ㄴㅈ', 'ㄴㅎ', 'ㄷ', 'ㄹ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ', 'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ', 'ㅁ', 'ㅂ', 'ㅂㅅ',
         'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def _hd(syl):
    o = ord(syl) - 0xAC00
    c, r = divmod(o, 588); j, f = divmod(r, 28)
    return [_CHO[c], _JUNG[j]] + list(_JONG[f])


def hangul_lines(tok=TOK):
    lines = []
    for line in open(os.path.join(SCR, 'ko_nsmc.txt'), encoding='utf-8').read().split('\n')[1:]:
        p = line.split('\t')
        if len(p) < 2:
            continue
        cur = []
        for w in p[1].split():
            if w and all(0xAC00 <= ord(c) <= 0xD7A3 for c in w):
                cur.append(tuple(j for c in w for j in _hd(c)))
            else:
                if cur: lines.append(cur)
                cur = []
        if cur:
            lines.append(cur)
    return _cap(lines, tok)


def plant(lines, tierset, a_set, p=0.7, seed=0, junction=0.0):
    """Planted harmony: in each word, every tier glyph is switched (with prob p) to the value class of the
    word's first tier glyph, using a fixed one-to-one pairing between A and B glyphs."""
    A = sorted(a_set); B = sorted(set(tierset) - set(a_set))
    toA = dict(zip(B, A)); toB = dict(zip(A, B))
    rng = random.Random(seed)
    out = []
    for L in lines:
        nl = []
        prev = None
        for w in L:
            w = list(w)
            tv = [g in a_set for g in w if g in tierset]
            if tv:
                v0 = tv[0]
                if prev is not None and rng.random() < junction:
                    v0 = prev
                for i, g in enumerate(w):
                    if g in tierset and rng.random() < p:
                        if v0 and g in toA: w[i] = toA[g]
                        elif (not v0) and g in toB: w[i] = toB[g]
                tv2 = [g in a_set for g in w if g in tierset]
                prev = tv2[-1]
            nl.append(tuple(w))
        out.append(nl)
    return out


# ------------------------------------------------------------------ encoding
class Enc:
    def __init__(self, lines, alph):
        self.alph = alph
        idx = {g: i for i, g in enumerate(alph)}
        ids, wid, ln = [], [], []
        w = 0
        self.nwords = 0
        for L in lines:
            for word in L:
                if all(g in idx for g in word):
                    ids.extend(idx[g] for g in word); wid.extend([w] * len(word)); w += 1
                    self.nwords += 1
                else:
                    w += 1
            w += 1  # line break: no junction
        self.ids = np.array(ids, dtype=np.int16); self.wid = np.array(wid, dtype=np.int32)


def alphabet(lines, shapes, minc=20):
    c = Counter(g for L in lines for w in L for g in w)
    return [g for g in shapes if c[g] >= minc], c


def _phi(x, y):
    n = len(x)
    if n < 30:
        return np.nan
    mx = x.mean(); my = y.mean()
    den = mx * (1 - mx) * my * (1 - my)
    if den <= 1e-9:
        return np.nan
    return ((x * y).mean() - mx * my) / np.sqrt(den)


def _pairmat(x, y, n):
    return np.bincount(x.astype(np.int64) * n + y, minlength=n * n).reshape(n, n).astype(np.float64)


def raw_mats(E):
    if getattr(E, '_raw', None) is None:
        n = len(E.alph); ids, wid = E.ids, E.wid
        E._raw = [_pairmat(ids[:-g][wid[:-g] == wid[g:]], ids[g:][wid[:-g] == wid[g:]], n) for g in (1, 2, 3)]
        E._tier = {}
    return E._raw


def tier_mats(E, m):
    raw_mats(E)
    key = m.tobytes()
    if key in E._tier:
        return E._tier[key]
    n = len(E.alph); ids, wid = E.ids, E.wid
    T = np.flatnonzero(m[ids]); gg = ids[T]; ww = wid[T]
    M = []
    for d in (1, 2, 3):
        s = ww[d:] == ww[:-d]
        M.append(_pairmat(gg[:-d][s], gg[d:][s], n))
    st = np.r_[True, ww[1:] != ww[:-1]] if len(T) else np.zeros(0, bool)
    en = np.r_[ww[1:] != ww[:-1], True] if len(T) else np.zeros(0, bool)
    si = np.flatnonzero(st); ei = np.flatnonzero(en); fw = ww[si]
    s = fw[1:] == fw[:-1] + 1
    M.append(_pairmat(gg[ei[:-1][s]], gg[si[1:][s]], n))
    k = (ei - si + 1) >= 3
    M.append(_pairmat(gg[si[k]], gg[si[k] + 1], n)); M.append(_pairmat(gg[ei[k] - 1], gg[ei[k]], n))
    if len(E._tier) > 5000:
        E._tier.clear()
    E._tier[key] = M
    return M


def _phim(M, v):
    n = M.sum()
    if n < 30:
        return np.nan
    mx = M.sum(1) @ v / n; my = M.sum(0) @ v / n; pxy = v @ M @ v / n
    den = mx * (1 - mx) * my * (1 - my)
    if den <= 1e-9:
        return np.nan
    return (pxy - mx * my) / np.sqrt(den)


def _agree(M, v):
    n = M.sum()
    return (v @ M @ v + (1 - v) @ M @ (1 - v)) / n


def stats(E, m, v):
    """m, v: bool arrays over the alphabet (tier mask, value).  Returns array len(STATS)."""
    R = raw_mats(E); Mt = tier_mats(E, m)
    vf = (v & m).astype(np.float64); mm = m.astype(np.float64)
    out = np.full(len(STATS), np.nan)
    for d in range(3):
        out[d] = _phim(Mt[d], vf)
        out[3 + d] = _phim(R[d] * np.outer(mm, mm), vf)
    out[6] = _phim(Mt[3], vf)
    if Mt[4].sum() >= 30:
        out[7] = _agree(Mt[5], vf) - _agree(Mt[4], vf)
    return out


# ------------------------------------------------------------------ nulls
def markov2(lines, seed):
    rng = random.Random(seed)
    T = defaultdict(Counter)
    for L in lines:
        for w in L:
            ww = ('<', '<') + tuple(w) + ('>',)
            for i in range(2, len(ww)):
                T[(ww[i - 2], ww[i - 1])][ww[i]] += 1
    tab = {k: (list(c.keys()), np.cumsum(list(c.values())) / sum(c.values())) for k, c in T.items()}
    out = []
    for L in lines:
        nl = []
        for _ in L:
            a, b = '<', '<'; w = []
            while True:
                ks, cp = tab[(a, b)]
                x = ks[int(np.searchsorted(cp, rng.random()))]
                if x == '>' or len(w) > 25:
                    break
                w.append(x); a, b = b, x
            if w:
                nl.append(tuple(w))
        out.append(nl)
    return out


def wshuffle(lines, seed):
    rng = random.Random(seed)
    out = []
    for L in lines:
        L = list(L); rng.shuffle(L); out.append(L)
    return out


def freq_bins(alph, cnt, size=4):
    order = sorted(range(len(alph)), key=lambda i: -cnt[alph[i]])
    return [order[i:i + size] for i in range(0, len(order), size)]


def relabel(bins, rng, n):
    p = np.arange(n)
    for b in bins:
        q = list(b); rng.shuffle(q)
        p[b] = q
    return p


# ------------------------------------------------------------------ feature generation
def features(alph, shapes, cnt, n_random=2000, n_class=600, seed=0, min_share=0.05, min_tier=1500):
    prims = sorted({p for g in alph for p in shapes[g]})
    P = {p: np.array([shapes[g].get(p, 0) > 0 for g in alph]) for p in prims}
    fr = np.array([cnt[g] for g in alph], dtype=float)
    feats, seen = [], set()

    def add(m, v, name, kind):
        a = m & v; b = m & ~v
        if not a.any() or not b.any():
            return
        tm = fr[m].sum()
        if tm < min_tier or min(fr[a].sum(), fr[b].sum()) < min_share * tm:
            return
        key = (tuple(np.flatnonzero(m)), tuple(np.flatnonzero(a)))
        key2 = (key[0], tuple(np.flatnonzero(b)))
        if key in seen or key2 in seen:
            return
        seen.add(key)
        feats.append(dict(m=m.copy(), v=v.copy(), name=name, kind=kind))

    allm = np.ones(len(alph), bool)
    for p in prims:                      # principled: one stroke, tier = all or = carriers of another stroke
        add(allm, P[p], f'{p}|all', 'stroke')
        for q in prims:
            if q != p:
                add(P[q], P[p], f'{p}|{q}', 'stroke')
                add(~P[q], P[p], f'{p}|not {q}', 'stroke')
    rng = random.Random(seed)
    tries = 0
    while sum(f['kind'] == 'rstroke' for f in feats) < n_random and tries < n_random * 50:
        tries += 1
        k = rng.choice([0, 1, 1, 2, 2, 3])
        m = allm.copy(); nm = []
        for _ in range(k):
            q = rng.choice(prims); neg = rng.random() < 0.3
            m = m & (~P[q] if neg else P[q]); nm.append(('!' if neg else '') + q)
        if rng.random() < 0.5:
            p1 = rng.choice(prims); v = P[p1].copy(); vn = p1
        else:
            p1, p2 = rng.sample(prims, 2); op = rng.choice(['or', 'and', 'andnot'])
            v = P[p1] | P[p2] if op == 'or' else (P[p1] & P[p2] if op == 'and' else P[p1] & ~P[p2])
            vn = f'{p1} {op} {p2}'
        add(m, v, f'{vn}|{"&".join(nm) or "all"}', 'rstroke')
    tries = 0
    n = len(alph)
    while sum(f['kind'] == 'class' for f in feats) < n_class and tries < n_class * 50:
        tries += 1
        k = rng.randint(3, max(3, min(n, 12)))
        tier = rng.sample(range(n), k); j = rng.randint(1, k - 1)
        m = np.zeros(n, bool); m[tier] = True; v = np.zeros(n, bool); v[tier[:j]] = True
        add(m, v, 'class:' + ''.join(alph[i] if len(alph[i]) == 1 else f'[{alph[i]}]' for i in tier[:j]) + '/' +
            ''.join(alph[i] if len(alph[i]) == 1 else f'[{alph[i]}]' for i in tier[j:]), 'class')
    return feats


def fname(f, alph):
    A = [alph[i] for i in np.flatnonzero(f['m'] & f['v'])]; B = [alph[i] for i in np.flatnonzero(f['m'] & ~f['v'])]
    return f"{f['name']} [{' '.join(A)} / {' '.join(B)}]"


# ------------------------------------------------------------------ the full run for one corpus
def run(lines, shapes, R=12, n_random=2000, n_class=600, seed=0, feats=None, log=print):
    alph, cnt = alphabet(lines, shapes)
    if feats is None:
        feats = features(alph, shapes, cnt, n_random=n_random, n_class=n_class, seed=seed)
    E = Enc(lines, alph)
    obs = np.array([stats(E, f['m'], f['v']) for f in feats])
    nullM = np.zeros((R,) + obs.shape); nullJ = np.zeros((R,) + obs.shape)
    for r in range(R):
        Em = Enc(markov2(lines, seed * 1000 + r), alph)
        nullM[r] = [stats(Em, f['m'], f['v']) for f in feats]
        Ej = Enc(wshuffle(lines, seed * 1000 + r), alph)
        nullJ[r] = [stats(Ej, f['m'], f['v']) for f in feats]
    null = nullM.copy(); null[:, :, 6] = nullJ[:, :, 6]
    return dict(alph=alph, cnt=cnt, feats=feats, obs=obs, null=null, nwords=E.nwords, ntok=len(E.ids))


def zscores(res):
    mu = np.nanmean(res['null'], 0); sd = np.nanstd(res['null'], 0, ddof=1)
    sd = np.where(sd < 0.02, 0.02, sd)   # floor: avoid huge z from degenerate nulls (0.02 phi)
    return (res['obs'] - mu) / sd, res['obs'] - mu


def maxnull(res):
    """Leave-one-out max-|z| over the search on each null replicate (search correction)."""
    N = res['null']; R = N.shape[0]
    out = []
    for r in range(R):
        rest = np.delete(N, r, 0)
        mu = np.nanmean(rest, 0); sd = np.nanstd(rest, 0, ddof=1); sd = np.where(sd < 0.02, 0.02, sd)
        z = (N[r] - mu) / sd
        out.append(np.nanmax(z, 0))
    return np.array(out)  # R x nstats
