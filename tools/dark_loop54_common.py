"""Loop 54: re-grade every dictionary entry and grammar statement against Markov nulls.
Shared code: corpus loading (three merge levels, die regime), Markov chains fitted within site x object type
(order 1 and 2 with lengths kept, as in loop 41; order 1 with an END state and free lengths as an extra), the S366
slot-grammar generator (M* = textreuse + closerdep + type + open + site), the IM77-only text set in Mahadevan numbers,
and a battery of per-claim statistics. Numbers only; no interpretation."""
import json, csv, random, collections, bisect, itertools, math, sys, os, statistics as st
ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
sys.path.insert(0, ROOT + 'tools')
BIG = ('Mohenjo-daro', 'Harappa')

OP = {817, 861, 820}; OP5 = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
SHORT = {1, 3, 4, 5, 16, 17, 18}; TALL = {31, 32, 33, 34}; NUM = SHORT | TALL | {2, 55, 56}
COUNTED = {390, 405, 406, 407, 900, 220, 740, 700}
FISHQ = [235, 240, 233, 231]; FISH = set(FISHQ) | {220}

def otype(t): return t.split(':')[0]

def load_corpus(level='seq_all', regime='die', sites=None, exclude_unknown=True):
    """-> list of (meta, tuple(seq)); meta has site, type. Die regime as in loop 41."""
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    out = []; seen = set()
    for r in C:
        s = tuple(r[level])
        if not s: continue
        if exclude_unknown and r['site'] == 'Unknown': continue
        if sites is not None and r['site'] not in sites: continue
        if regime == 'die':
            if otype(r['type']) in ('TAB', 'TAG') or r['type'].startswith('POT:T:s'): k = (r['site'], otype(r['type']), s)
            else: k = (r['cisi'] if r['cisi'] not in ('-', '') else id(r), s)
        elif regime == 'rows': k = ('row', id(r))
        else: raise ValueError(regime)
        if k in seen: continue
        seen.add(k); out.append(({'site': r['site'], 'type': r['type'], 'ot': otype(r['type']), 'cisi': r['cisi']}, s))
    return out

# ---------------------------------------------------------------- IM77-only texts (Mahadevan numbers)
def im77_new_texts():
    """The 324 'certainly new' IM77 texts of S-DARK-27.3 (loop27_sets.json 'new'), one text per (text_no, side), in M numbers."""
    new = {tuple(x) for x in json.load(open(DARK + 'loop27_sets.json'))['new']}
    lines = collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')):
        if r['line'] == '9': continue
        lines[(r['text_no'], r['side'])].append(r)
    out = []
    for k, ls in lines.items():
        if k not in new: continue
        ls = sorted(ls, key=lambda r: int(r['line'])); seq = []
        for l in ls: seq += [int(t) for t in l['signs_clean'].split()]
        seq = tuple(x for x in seq if x != 0)
        if not seq: continue
        ot = ls[0]['object_type']
        ot = 'SEAL' if ot == 'seal' else 'TAG' if ot == 'sealing' else 'TAB' if 'tablet' in ot else 'POT' if 'pottery' in ot else 'MISC'
        out.append(({'site': ls[0]['site'], 'type': ot, 'ot': ot, 'cisi': k[0]}, seq))
    return out

def bridge_w2m(contextual=True):
    b = json.load(open(DARK + 'loop27_sets.json'))['bridge']
    out = {int(k): set(v) for k, v in b.items()}
    if contextual: out.setdefault(34, {95})   # S-DARK-27.3 contextual mapping (Harappa tablet opener before W700 = M328)
    return out

def to_m(signs, br):
    """translate a set of Wells signs to the union of their Mahadevan numbers (None if any is unbridged)."""
    out = set()
    for w in signs:
        if w not in br: return None
        out |= br[w]
    return out

# ---------------------------------------------------------------- chains
class Chain:
    """order-k chain fitted per group (site x object type) with START; 'E' transitions kept only if end=True."""
    def __init__(self, T, order=1, end=False, group=lambda m: (m['site'], m['ot'])):
        self.k = order; self.end = end; self.group = group
        self.m = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
        self.uni = collections.defaultdict(collections.Counter)
        for meta, s in T:
            g = group(meta); h = ('S',) * order
            for x in list(s) + (['E'] if end else []):
                self.m[g][h][x] += 1; h = (h + (x,))[1:]
            self.uni[g].update(s)
        self.cum = {}
        for g, tab in self.m.items():
            for h, cnt in tab.items():
                ks = list(cnt); ws = list(itertools.accumulate(cnt[k] for k in ks)); self.cum[(g, h)] = (ks, ws)
            ks = list(self.uni[g]); ws = list(itertools.accumulate(self.uni[g][k] for k in ks)); self.cum[(g, ())] = (ks, ws)
    def draw(self, g, h, rnd, allow_end):
        c = self.cum.get((g, h))
        if c is None: c = self.cum[(g, ())]
        ks, ws = c
        for _ in range(20):
            x = ks[bisect.bisect(ws, rnd.random() * ws[-1])]
            if x != 'E' or allow_end: return x
        # fall back to unigram
        ks, ws = self.cum[(g, ())]
        return ks[bisect.bisect(ws, rnd.random() * ws[-1])]
    def corpus(self, T, rnd, maxlen=20):
        out = []
        for meta, s in T:
            g = self.group(meta); h = ('S',) * self.k; seq = []
            if self.end:
                while len(seq) < maxlen:
                    x = self.draw(g, h, rnd, allow_end=len(seq) >= 1)
                    if x == 'E': break
                    seq.append(x); h = (h + (x,))[1:]
                if not seq: seq = [self.draw(g, ('S',) * self.k, rnd, False)]
            else:
                for _ in range(len(s)):
                    x = self.draw(g, h, rnd, allow_end=False); seq.append(x); h = (h + (x,))[1:]
            out.append((meta, tuple(seq)))
        return out

def s366_generator(level, fit_sites=BIG):
    """S366 M* slot-grammar generator fitted on Mohenjo-daro + Harappa at the given merge level. Returns (gen, meta_fn)."""
    sys.argv = ['x', '--seqkey', level, '--n', '1']
    import importlib
    if 'strat_adequacy' in sys.modules: SA = importlib.reload(sys.modules['strat_adequacy'])
    else: SA = importlib.import_module('strat_adequacy')
    # faster sampling: cache cumulative weights per history
    def generate(self, rng, maxlen=14):
        out = []; hist = [SA.Markov.START] * self.k
        if not hasattr(self, '_cum'): self._cum = {}
        while len(out) < maxlen:
            hk = tuple(hist[-self.k:]) if self.k else ()
            c = self._cum.get(hk)
            if c is None:
                keys, wts = self.dist(hist); c = (keys, list(itertools.accumulate(wts))); self._cum[hk] = c
            keys, cw = c; v = keys[bisect.bisect(cw, rng.random() * cw[-1])]
            if v == SA.Markov.END: break
            if v == SA.Markov.NEW:
                self.fresh += 1; out.append(-self.fresh); hist = (hist + [v])[-max(self.k, 1):]; continue
            out.append(v); hist = (hist + [v])[-max(self.k, 1):]
        return tuple(out)
    SA.Markov.generate = generate
    data = [x for x in SA.ALL if x[0] in fit_sites]
    model = SA.SlotModel(data, mech=['textreuse', 'closerdep', 'type', 'open', 'site'])
    return SA, model

def s366_corpus(SA, model, T, rnd):
    meta = [(m['site'], SA.cls_of({'type': m['type']})) for m, _ in T]
    g = model.generate_corpus(meta, rnd)
    return [(m, tuple(x for x in s if isinstance(x, int))) for (m, _), (_, _, s) in zip(T, g)]

# ---------------------------------------------------------------- statistics (each takes list of (meta, seq) -> float)
def strip_suf(s, suf):
    t = list(s)
    while len(t) > 1 and t[-1] in suf: t.pop()
    return t

_BT = {}
def binom_tail(k, n):
    """P(X >= k), X ~ Bin(n, 1/2), cached"""
    key = (k, n)
    if key not in _BT: _BT[key] = sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n
    return _BT[key]

def mk(sets):
    """build the statistic library for a sign-set dictionary (so claims translate to M numbers for IM77)."""
    S = sets; op, mark, suf, cl = S['OP'], S['MARK'], S['SUF'], S['CL']
    def tok(T, X): return [(s, i) for _, s in T if len(s) >= 2 for i, x in enumerate(s) if x in X]
    def init(X):
        def f(T):
            t = tok(T, X); return sum(i == 0 for s, i in t) / len(t) if t else float('nan')
        return f
    def second(X):
        def f(T):
            t = tok(T, X); return sum(i == 1 for s, i in t) / len(t) if t else float('nan')
        return f
    def final(X, strip=True):
        def f(T):
            n = k = 0
            for _, s in T:
                if len(s) < 2: continue
                t = strip_suf(s, suf) if strip else list(s)
                for i, x in enumerate(s):
                    if x in X: n += 1; k += (i == len(t) - 1)
            return k / n if n else float('nan')
        return f
    def repeat(X):
        """texts carrying X twice or more, per 1000 texts with X"""
        def f(T):
            w = [s for _, s in T if any(x in X for x in s)]
            return 1000 * sum(1 for s in w if sum(x in X for x in s) >= 2) / len(w) if w else float('nan')
        return f
    def cooc(X, Y, mindist=1):
        """share of texts with X that also carry Y at distance >= mindist (mindist 2 = non-adjacent only)"""
        def f(T):
            w = k = 0
            for _, s in T:
                px = [i for i, x in enumerate(s) if x in X]; py = [i for i, x in enumerate(s) if x in Y]
                if not px: continue
                w += 1
                if any(abs(i - j) >= mindist for i in px for j in py if i != j): k += 1
            return k / w if w else float('nan')
        return f
    def before(X, Y, mindist=1):
        """among texts with X and Y (distance >= mindist), share where X precedes Y (first X vs first Y)"""
        def f(T):
            a = b = 0
            for _, s in T:
                px = [i for i, x in enumerate(s) if x in X]; py = [i for i, x in enumerate(s) if x in Y]
                pr = [(i, j) for i in px for j in py if abs(i - j) >= mindist and i != j]
                if not pr: continue
                if all(i < j for i, j in pr): a += 1
                elif all(i > j for i, j in pr): b += 1
            return a / (a + b) if a + b else float('nan')
        return f
    def nxt(X, Y):
        def f(T):
            n = k = 0
            for _, s in T:
                for i in range(len(s) - 1):
                    if s[i] in X: n += 1; k += s[i + 1] in Y
            return k / n if n else float('nan')
        return f
    def prv(X, Y):
        """share of Y tokens whose left neighbour is in X"""
        def f(T):
            n = k = 0
            for _, s in T:
                for i in range(len(s)):
                    if s[i] in Y: n += 1; k += (i > 0 and s[i - 1] in X)
            return k / n if n else float('nan')
        return f
    def at2(X, Y):
        """share of Y tokens with X exactly two positions before and NOT adjacent (fish + 1 sign + arrow)"""
        def f(T):
            n = k = 0
            for _, s in T:
                for i in range(len(s)):
                    if s[i] in Y: n += 1; k += (i >= 2 and s[i - 2] in X and s[i - 1] not in X)
            return k / n if n else float('nan')
        return f
    def trigram(A, B, C_):
        """texts carrying the ordered run A B C (adjacent) per 1000 texts"""
        def f(T):
            return 1000 * sum(1 for _, s in T if any(s[i] in A and s[i + 1] in B and s[i + 2] in C_ for i in range(len(s) - 2))) / len(T)
        return f
    def numfix(X):
        """fixity of the numeral before X: share of numeral-preceded X tokens taking the modal numeral"""
        def f(T):
            c = collections.Counter()
            for _, s in T:
                for i in range(1, len(s)):
                    if s[i] in X and s[i - 1] in S['NUM']: c[s[i - 1]] += 1
            n = sum(c.values()); return c.most_common(1)[0][1] / n if n else float('nan')
        return f
    def numvar(X):
        """distinct numerals before X (counted item = many)"""
        def f(T):
            c = set()
            for _, s in T:
                for i in range(1, len(s)):
                    if s[i] in X and s[i - 1] in S['NUM']: c.add(s[i - 1])
            return len(c)
        return f
    def two_of(X):
        """share of texts (>=2 signs) with two or more distinct members of X"""
        def f(T):
            w = [s for _, s in T if len(s) >= 2]
            return sum(1 for s in w if len(set(s) & X) >= 2) / len(w)
        return f
    def two_tokens_nonadj(X):
        """share of texts with two X tokens that are not adjacent"""
        def f(T):
            w = [s for _, s in T if len(s) >= 2]
            def ok(s):
                p = [i for i, x in enumerate(s) if x in X]
                return any(j - i >= 2 for i in p for j in p if j > i)
            return sum(ok(s) for s in w) / len(w)
        return f
    def norepeat(T):
        """share of texts (>=3) with any sign repeated non-adjacently"""
        w = [s for _, s in T if len(s) >= 3]
        def rep(s):
            pos = collections.defaultdict(list)
            for i, x in enumerate(s): pos[x].append(i)
            return any(j - i >= 2 for p in pos.values() for i in p for j in p if j > i)
        return sum(rep(s) for s in w) / len(w)
    def midinit(X):
        """share of X tokens standing first in the middle (after an opener and its marker, if any), texts >= 3"""
        def f(T):
            n = k = 0
            for _, s in T:
                if len(s) < 3: continue
                i = 0
                if s[0] in S['OP5']:
                    i = 1
                    if len(s) > 1 and s[1] in mark: i = 2
                for j, x in enumerate(s):
                    if x in X: n += 1; k += (j == i)
            return k / n if n else float('nan')
        return f
    def midfinal(X):
        """share of X tokens standing last before the closer (texts >= 3, suffix stripped, closer present)"""
        def f(T):
            n = k = 0
            for _, s in T:
                if len(s) < 3: continue
                t = strip_suf(s, suf)
                e = len(t) - 2 if t[-1] in cl else len(t) - 1
                for j, x in enumerate(s):
                    if x in X: n += 1; k += (j == e)
            return k / n if n else float('nan')
        return f
    def closer_paradigm(T):
        """S289 count: signs (n>=15) >=40% final (suffix stripped) with jar co-occurrence O/E <= 0.5"""
        texts = [s for _, s in T if len(s) >= 2]
        tokc = collections.Counter(x for s in texts for x in s)
        fin = collections.Counter(); jw = collections.Counter(); je = collections.defaultdict(float)
        bylen = collections.defaultdict(list)
        jar = S['JAR']
        for s in texts: bylen[len(s)].append(bool(set(s) & jar))
        jr = {L: sum(v) / len(v) for L, v in bylen.items()}
        for s in texts:
            t = strip_suf(s, suf); fin[t[-1]] += 1
            for x in set(s):
                if set(s) & jar: jw[x] += 1
                je[x] += jr[len(s)]
        return sum(1 for x, n in tokc.items() if n >= 15 and x not in jar and x not in suf and fin[x] / n >= 0.4 and (jw[x] / je[x] if je[x] else 9) <= 0.5)
    def frame_opener_first(T):
        w = [s for _, s in T if len(s) >= 2]; return sum(s[0] in op for s in w) / len(w)
    def frame_closer_last(T):
        w = [s for _, s in T if len(s) >= 2]; return sum(strip_suf(s, suf)[-1] in cl for s in w) / len(w)
    def frame_conn_initial(T):
        w = [s for _, s in T if len(s) >= 2]; n = sum(sum(x in mark for x in s) for s in w)
        return sum(s[0] in mark for s in w) / n if n else float('nan')
    def frame_suffix_after_closer(T):
        n = k = 0
        for _, s in T:
            for i in range(1, len(s)):
                if s[i] in suf: n += 1; k += s[i - 1] in cl or s[i - 1] in suf
        return k / n if n else float('nan')
    def length_sd(T):
        return st.pstdev([len(s) for _, s in T])
    def anagram_share(T):
        g = collections.defaultdict(list)
        for _, s in T:
            if len(s) >= 3: g[tuple(sorted(s))].append(s)
        pairs = diff = 0
        for v in g.values():
            for i in range(len(v)):
                for j in range(i + 1, len(v)): pairs += 1; diff += v[i] != v[j]
        return diff / pairs if pairs else float('nan')
    def fixed_pair_share(T, minco=5):
        ab = collections.Counter()
        for _, s in T:
            seen = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    if s[i] != s[j]: seen.add((s[i], s[j]))
            for p in seen: ab[p] += 1
        done = set(); ps = []
        for (a, b), n in ab.items():
            if (b, a) in done: continue
            done.add((a, b)); m = ab[(b, a)]
            if n + m < minco: continue
            ps.append(binom_tail(max(n, m), n + m))
        ps.sort(); M = len(ps); fx = 0
        for i, p in enumerate(ps):
            if p <= 0.05 * (i + 1) / M: fx = i + 1
        return fx / M if M else float('nan')
    def fixed_pair_share_nonadj(T, minco=5):
        """same, but counting only pairs at distance >= 2 (removes the bigram part)"""
        ab = collections.Counter()
        for _, s in T:
            seen = set()
            for i in range(len(s)):
                for j in range(i + 2, len(s)):
                    if s[i] != s[j]: seen.add((s[i], s[j]))
            for p in seen: ab[p] += 1
        done = set(); ps = []
        for (a, b), n in ab.items():
            if (b, a) in done: continue
            done.add((a, b)); m = ab[(b, a)]
            if n + m < minco: continue
            ps.append(binom_tail(max(n, m), n + m))
        ps.sort(); M = len(ps); fx = 0
        for i, p in enumerate(ps):
            if p <= 0.05 * (i + 1) / M: fx = i + 1
        return fx / M if M else float('nan')
    def longrange_pairs(T, mind=2, minexp=3.0, z=3.0):
        """count of sign pairs attracting (O > E, z >= 3) at distance >= 2, E from independent positions within the same text set"""
        texts = [s for _, s in T if len(s) >= 3]
        tokc = collections.Counter(x for s in texts for x in s); N = sum(tokc.values())
        obs = collections.Counter()
        for s in texts:
            seen = set()
            for i in range(len(s)):
                for j in range(i + mind, len(s)):
                    if s[i] != s[j]: seen.add(tuple(sorted((s[i], s[j]))))
            for p in seen: obs[p] += 1
        # expected: pairs at distance >= 2 drawn by frequency
        slots = sum(max(0, len(s) - mind) * (len(s) - mind + 1) / 2 for s in texts)
        hits = 0
        for (a, b), o in obs.items():
            e = 2 * slots * tokc[a] * tokc[b] / N / N
            if e >= minexp and (o - e) / math.sqrt(e) >= z: hits += 1
        return hits
    return dict(init=init, second=second, final=final, repeat=repeat, cooc=cooc, before=before, nxt=nxt, prv=prv, at2=at2,
                trigram=trigram, numfix=numfix, numvar=numvar, two_of=two_of, two_tokens_nonadj=two_tokens_nonadj, norepeat=norepeat,
                midinit=midinit, midfinal=midfinal, closer_paradigm=closer_paradigm, frame_opener_first=frame_opener_first,
                frame_closer_last=frame_closer_last, frame_conn_initial=frame_conn_initial, frame_suffix_after_closer=frame_suffix_after_closer,
                length_sd=length_sd, anagram_share=anagram_share, fixed_pair_share=fixed_pair_share,
                fixed_pair_share_nonadj=fixed_pair_share_nonadj, longrange_pairs=longrange_pairs)

def w_sets():
    return {'OP': set(OP), 'OP5': set(OP5), 'MARK': set(MARK), 'SUF': set(SUF), 'CL': set(CL), 'NUM': set(NUM), 'JAR': {740}}

def to_m_partial(signs, br):
    out = set()
    for w in signs:
        if w in br: out |= br[w]
    return out

def m_sets(br):
    g = lambda ws: to_m_partial(ws, br)
    return {'OP': g(OP), 'OP5': g(OP5), 'MARK': g(MARK), 'SUF': g(SUF), 'CL': g(CL), 'NUM': g(NUM), 'JAR': {342}}

def summarize(obs, null):
    null = [x for x in null if x == x]
    if not null or obs != obs: return {'obs': obs, 'med': float('nan'), 'lo': float('nan'), 'hi': float('nan'), 'inside': None, 'dir': ''}
    null.sort(); n = len(null)
    lo = null[int(0.025 * n)]; hi = null[min(n - 1, int(0.975 * n))]; med = null[n // 2]
    inside = lo <= obs <= hi
    d = '' if inside else ('above' if obs > hi else 'below')
    # also the share of null values at least as extreme
    return {'obs': obs, 'med': med, 'lo': lo, 'hi': hi, 'inside': inside, 'dir': d}

def fmt(x):
    if x is None: return '-'
    if isinstance(x, float):
        if x != x: return 'nan'
        return f'{x:.3f}' if abs(x) < 100 else f'{x:.0f}'
    return str(x)
