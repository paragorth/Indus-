"""S-DARK-74: WHAT IS THE SUBSTRING RECURRENCE?
S-DARK-63.4: 10-12% of 2-4-sign Indus middles recur inside a longer middle (~2x shuffle); every logographic given-name
list sits at or below its shuffle, phonetic lists are positive. Three readings:
  (A) short forms / hypocoristics (full designation + abbreviation; phonetic name lists),
  (B) hierarchical register (a house / office designation reused as the stem of its members' designations),
  (C) Markov artefact (S-DARK-41: Markov-1 reproduces whole-text nesting).
Cycle 1: every containment pair (short middle inside a longer middle; distinct middles from objects deduplicated one per
         site x object type x text), by length pair (2-in-3, 2-in-4, 3-in-4, 2-in-5+, 3-in-5+, 4-in-5+), vs
         (a) global token shuffle, (b) within-slot shuffle (position x length kept), (c) Markov-1 and (d) Markov-2
         middles (MLE chains trained on the distinct middles, sampled at EXACT length by backward DP, same number of
         distinct middles per length), (e) frequency-matched iid random strings.
Cycle 2: position of the contained middle (prefix / suffix / infix) vs the chains; the signs that extend a short middle;
         comparators: Japanese / Vietnamese / ancient Chinese and the phonetic lists (loop63 / loop56 corpora),
         length-matched to the Indus distribution, each with its own shuffle.
Cycle 3: outside facts: do short and long objects of a pair share site, area, room, closer (head), emblem, object type
         more than (n1) a random same-length partner and (n2) a same-length partner that shares a sign with the short
         middle but does not contain it; is the short one older (Mohenjo-daro depth, Harappa period)?
Cycle 4: held-out sites (not Mohenjo-daro / Harappa) and the 324 IM77-only texts: do NEW middles recur inside / contain
         reference middles (MD+H Wells, or all Wells) more than chain-generated middles of the same length do?
Usage: python3 tools/dark_loop74.py <1|2|3|4> <seq_raw|seq_strong|seq_all> [nnull]
"""
import sys, os, json, random, collections, math, csv
_CY = int(sys.argv[1]); _LV = sys.argv[2]; _NN = int(sys.argv[3]) if len(sys.argv) > 3 else 200
sys.argv = ['x', '0']; sys.path.insert(0, 'tools')
import dark_loop56 as L56
CY, LV, NN = _CY, _LV, _NN
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
def save(tag):
    open(f'data/derived/dark/loop74_c{CY}_{LV}{tag}.txt', 'w').write('\n'.join(OUT) + '\n')
def q(v, p):
    v = sorted(x for x in v if x == x); return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')
def mean(v):
    v = [x for x in v if x == x]; return sum(v) / len(v) if v else float('nan')
def fm(s): return '-'.join(str(a) for a in s)
def ub(k, n): return f'{k}/{n}' if k else f'0/{n} (<{3 / n:.3f})' if n else '0/0'

# ------------------------------------------------------------------ data
def wells_objects(level=LV):
    return L56.load_indus(level)
def distinct(objs, field='mid', minlen=2):
    return sorted(set(o[field] for o in objs if len(o[field]) >= minlen))

# ------------------------------------------------------------------ containment
LP = [(2, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)]   # b = 5 means 5+
def lb(b): return min(b, 5)
def contain_pairs(names, ref=None):
    """pairs (s, t, offset) with s a proper contiguous substring of t; s from names, t from ref (default names)."""
    ref = names if ref is None else ref
    S = set(names); out = []
    for t in set(ref):
        L = len(t); seen = set()
        for l in range(2, L):
            for i in range(L - l + 1):
                s = t[i:i + l]
                if s in S and s not in seen:
                    seen.add(s); out.append((s, t, i))
    return out
def cstats(names, ref=None):
    pr = contain_pairs(names, ref)
    by = collections.defaultdict(set); np_ = collections.Counter(); pos = collections.Counter()
    for s, t, i in pr:
        k = (len(s), lb(len(t))); by[k].add(s); np_[k] += 1
        pos[(k, 'pre' if i == 0 else 'suf' if i + len(s) == len(t) else 'inf')] += 1
    nlen = collections.Counter(len(n) for n in names)
    cont = set(s for s, _, _ in pr)
    d = dict(share=len(cont) / len(names) if names else float('nan'), npairs=len(pr))
    for k in LP:
        d[f'K{k[0]}{k[1]}'] = len(by[k]); d[f'P{k[0]}{k[1]}'] = np_[k]
        d[f'f{k[0]}{k[1]}'] = len(by[k]) / nlen[k[0]] if nlen[k[0]] else float('nan')
    for a in (2, 3, 4):
        d[f'sh{a}'] = sum(1 for s in cont if len(s) == a) / nlen[a] if nlen[a] else float('nan')
    tot = sum(pos.values())
    for w in ('pre', 'suf', 'inf'):
        d[w] = sum(v for (k, ww), v in pos.items() if ww == w)
    d['pos'] = pos
    return d

# ------------------------------------------------------------------ nulls
def shuf_global(names, r):
    return L56.shuffle_names(names, r, 'shuf')
def shuf_slot(names, r):
    """within-slot shuffle: tokens permuted among names of the same length at the same position."""
    byL = collections.defaultdict(list)
    for i, n in enumerate(names): byL[len(n)].append(i)
    out = [None] * len(names)
    for L, idx in byL.items():
        cols = [[names[i][p] for i in idx] for p in range(L)]
        for c in cols: r.shuffle(c)
        for j, i in enumerate(idx): out[i] = tuple(cols[p][j] for p in range(L))
    return out
def freq_rand(names, r):
    return L56.shuffle_names(names, r, 'freq')
BOS, EOS = '^', '$'
class Chain:
    """order-k MLE Markov chain on the given strings, exact-length sampling by backward DP."""
    def __init__(self, strings, k):
        self.k = k; self.T = collections.defaultdict(collections.Counter)
        for s in strings:
            st = (BOS,) * k
            for a in list(s) + [EOS]:
                self.T[st][a] += 1; st = (st + (a,))[-k:]
        self.tot = {st: sum(c.values()) for st, c in self.T.items()}
        self.memo = {}
    def beta(self, st, m):
        """prob. of emitting exactly m more symbols then EOS from state st."""
        key = (st, m)
        if key in self.memo: return self.memo[key]
        c = self.T.get(st)
        if not c: v = 0.0
        elif m == 0: v = c.get(EOS, 0) / self.tot[st]
        else:
            v = sum(n / self.tot[st] * self.beta((st + (a,))[-self.k:], m - 1) for a, n in c.items() if a != EOS)
        self.memo[key] = v; return v
    def sample(self, L, r):
        st = (BOS,) * self.k; out = []
        if not hasattr(self, 'cc'): self.cc = {}
        import bisect
        for p in range(L):
            rem = L - p - 1
            key = (st, rem)
            if key not in self.cc:
                cand = [(a, n / self.tot[st] * self.beta((st + (a,))[-self.k:], rem)) for a, n in self.T[st].items() if a != EOS]
                cum = []; acc = 0
                for _, w in cand: acc += w; cum.append(acc)
                self.cc[key] = ([a for a, _ in cand], cum)
            al, cum = self.cc[key]
            if not cum or cum[-1] <= 0: return None
            a = al[min(len(al) - 1, bisect.bisect_left(cum, r.random() * cum[-1]))]
            out.append(a); st = (st + (a,))[-self.k:]
        return tuple(out)
class SChain:
    """interpolated Markov-1: Q(y|x) = lam * P_MLE(y|x) + (1 - lam) * P_uni(y), y over signs + EOS (start likewise);
    lam fitted by 2-fold held-out log-likelihood (grid), then the chain is refit on all strings. Exact-length sampling."""
    def __init__(self, strings, r=None, lam=None):
        import numpy as np
        self.np = np
        r = r or random.Random(1)
        if lam is None:
            idx = list(range(len(strings))); r.shuffle(idx); h = len(idx) // 2
            A = [strings[i] for i in idx[:h]]; B = [strings[i] for i in idx[h:]]
            best = None
            for l in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                ll = self._ll(A, B, l) + self._ll(B, A, l)
                if best is None or ll > best[0]: best = (ll, l)
            lam = best[1]
        self.lam = lam; self._fit(strings)
    @staticmethod
    def _counts(S):
        T = collections.defaultdict(collections.Counter); U = collections.Counter()
        for s in S:
            prev = BOS
            for a in list(s) + [EOS]:
                T[prev][a] += 1; U[a] += 1; prev = a
        return T, U
    def _ll(self, train, test, lam):
        T, U = self._counts(train); V = set(U) | set(a for s in test for a in s) | {EOS}
        tu = sum(U.values()) + len(V)
        ll = 0.0
        for s in test:
            prev = BOS
            for a in list(s) + [EOS]:
                pu = (U[a] + 1) / tu; c = T.get(prev); pm = c[a] / sum(c.values()) if c else 0.0
                ll += math.log(lam * pm + (1 - lam) * pu); prev = a
        return ll
    def _fit(self, S):
        np = self.np
        T, U = self._counts(S); syms = sorted(set(U) - {EOS}, key=str)
        self.syms = syms; ix = {a: i for i, a in enumerate(syms)}; K = len(syms)
        tu = sum(U.values())
        self.pu = np.array([U[a] / tu for a in syms]); self.pue = U[EOS] / tu
        def row(c):
            t = sum(c.values()); v = np.zeros(K)
            for a, n in c.items():
                if a != EOS: v[ix[a]] = n / t
            return v, c.get(EOS, 0) / t
        M = np.zeros((K, K)); E = np.zeros(K)
        for a in syms:
            if a in T: M[ix[a]], E[ix[a]] = row(T[a])
        self.M = self.lam * M + (1 - self.lam) * self.pu[None, :]
        self.E = self.lam * E + (1 - self.lam) * self.pue
        s0, _ = row(T[BOS]); self.S = self.lam * s0 + (1 - self.lam) * self.pu
        self.beta = [self.E]
    def _b(self, m):
        while len(self.beta) <= m: self.beta.append(self.M @ self.beta[-1])
        return self.beta[m]
    def sample(self, L, r):
        np = self.np; w = self.S * self._b(L - 1); out = []
        for p in range(L):
            c = np.cumsum(w); i = int(np.searchsorted(c, r.random() * c[-1])); i = min(i, len(c) - 1)
            out.append(self.syms[i])
            if p < L - 1: w = self.M[i] * self._b(L - p - 2)
        return tuple(out)
def chain_names(names, k, r, chain=None, lens=None):
    ch = chain or (SChain(names, r) if k == 's' else Chain(names, k))
    need = collections.Counter(len(n) for n in names) if lens is None else collections.Counter(lens)
    out = []; short = 0
    for L, c in need.items():
        got = set(); tries = 0
        while len(got) < c and tries < 60 * c:
            s = ch.sample(L, r); tries += 1
            if s is not None: got.add(s)
        out += sorted(got); short += c - len(got)
    return out, short

NULLS = ['shuffle', 'slot', 'markov1', 'markov2', 'freqrand']
def run_nulls(names, nn, keys, ref_fn=None, seed=74):
    res = {m: [] for m in NULLS}; shortfall = collections.Counter()
    ch1 = Chain(names, 1); ch2 = Chain(names, 2)
    for b in range(nn):
        r = random.Random(seed * 1000 + b)
        for m in NULLS:
            if m in ('markov1', 'markov2') and b >= max(20, nn // 2): continue
            if m == 'shuffle': x = shuf_global(names, r)
            elif m == 'slot': x = shuf_slot(names, r)
            elif m == 'freqrand': x = freq_rand(names, r)
            else:
                x, sh = chain_names(names, 1 if m == 'markov1' else 2, r, ch1 if m == 'markov1' else ch2); shortfall[m] += sh
            x = sorted(set(x))
            res[m].append(cstats(x) if ref_fn is None else ref_fn(x))
    return res, shortfall
def zline(obs, nulls, k):
    v = [d[k] for d in nulls]; mu = mean(v)
    sd = (sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 if v else float('nan')
    p = (sum(1 for x in v if x >= obs) + 1) / (len(v) + 1)
    rat = obs / mu if mu else float('inf')
    return f'{mu:.3f} [{q(v, .025):.3f},{q(v, .975):.3f}] x{rat:.2f} p={p:.3f}', obs - mu, p

# ================================================================== cycle 1
if CY == 1:
    objs = wells_objects()
    for field in ('mid', 'name'):
        names = distinct(objs, field)
        lens = collections.Counter(len(n) for n in names)
        P(f'\n##### S-DARK-74 cycle 1, {LV}, field={field} ({"NAME+COUNT, loop63 middle" if field == "mid" else "NAME only, numerals removed"}): {len(names)} distinct middles >= 2; lengths {sorted(lens.items())}')
        obs = cstats(names)
        res, sf = run_nulls(names, NN, None)
        P(f'  nulls: {NN} shuffles / slot shuffles / freq-random, {max(20, NN // 2)} Markov-1 / Markov-2 draws (MLE, exact length); chain shortfall (distinct strings not reached) {dict(sf)}')
        keys = ['share', 'npairs'] + [f'f{a}{b}' for a, b in LP] + [f'P{a}{b}' for a, b in LP]
        P(f'  {"stat":8s} {"obs":>7s} | ' + ' | '.join(f'{m}' for m in NULLS))
        for k in keys:
            line = f'  {k:8s} {obs[k]:7.3f} | ' if isinstance(obs[k], float) else f'  {k:8s} {obs[k]:7d} | '
            line += ' | '.join(zline(obs[k], res[m], k)[0] for m in NULLS)
            P(line)
        if field == 'mid':
            # pair listing
            site = collections.defaultdict(collections.Counter)
            for o in objs:
                if len(o['mid']) >= 2: site[o['mid']][(o['site'], o['ot'])] += 1
            pr = contain_pairs(names)
            P(f'\n  containment pairs: {len(pr)}; distinct short middles contained {len(set(s for s, _, _ in pr))}; distinct containers {len(set(t for _, t, _ in pr))}')
            hub = collections.Counter(s for s, _, _ in pr)
            P('  short middles contained in most longer middles: ' + '; '.join(f'{fm(s)} in {n} (objs {sum(site[s].values())})' for s, n in hub.most_common(15)))
            with open(f'data/derived/dark/loop74_c1_pairs_{LV}.txt', 'w') as f:
                f.write('short\tlong\toffset\tposition\tshort_site_x_type\tlong_site_x_type\n')
                for s, t, i in sorted(pr, key=lambda x: (len(x[0]), len(x[1]), x[0], x[1])):
                    w = 'prefix' if i == 0 else 'suffix' if i + len(s) == len(t) else 'infix'
                    f.write(f'{fm(s)}\t{fm(t)}\t{i}\t{w}\t' + ','.join(f'{a}/{b}:{n}' for (a, b), n in site[s].items()) + '\t' + ','.join(f'{a}/{b}:{n}' for (a, b), n in site[t].items()) + '\n')
            # frame-sign residue: share of pairs whose extension contains a numeral or a frame-class sign
            FR = L56.NUM | L56.OPEN | L56.MARK | L56.MJAR | L56.SUF | set(L56.CL)
            ext = []
            for s, t, i in pr: ext.append(tuple(t[:i]) + tuple(t[i + len(s):]))
            fr = sum(1 for e in ext if any(a in FR for a in e)) / len(ext) if ext else float('nan')
            P(f'  extension (long minus short) contains a numeral / frame-class sign in {fr:.3f} of pairs')
    save('')

# ================================================================== cycle 2
if CY == 2:
    objs = wells_objects()
    names = distinct(objs, 'mid')
    FR = L56.NUM | L56.OPEN | L56.MARK | L56.MJAR | L56.SUF | set(L56.CL)
    def posd(d):
        out = {}
        for k in LP:
            tot = sum(d['pos'].get((k, w), 0) for w in ('pre', 'suf', 'inf'))
            out[k] = (tot, d['pos'].get((k, 'pre'), 0), d['pos'].get((k, 'suf'), 0), d['pos'].get((k, 'inf'), 0))
        tot = d['pre'] + d['suf'] + d['inf']
        out['all'] = (tot, d['pre'], d['suf'], d['inf'])
        return out
    def pfmt(t): return f'n={t[0]} pre {t[1] / t[0]:.2f} suf {t[2] / t[0]:.2f} inf {t[3] / t[0]:.2f}' if t[0] else 'n=0'
    def agg(lst):
        """pooled position shares across null draws, by length pair."""
        out = {}
        for k in LP + ['all']:
            T = [posd(d)[k] for d in lst]; tot = sum(t[0] for t in T)
            out[k] = (tot, sum(t[1] for t in T), sum(t[2] for t in T), sum(t[3] for t in T))
        return out
    def pref_index(t):
        """(prefix - suffix) / (prefix + suffix)."""
        return (t[1] - t[2]) / (t[1] + t[2]) if t[1] + t[2] else float('nan')
    P(f'##### S-DARK-74 cycle 2, {LV}: position of the contained middle (prefix / suffix / infix) and the extension')
    for field in ('mid', 'name'):
        nm = distinct(objs, field)
        obs = cstats(nm); po = posd(obs)
        res, _ = run_nulls(nm, NN, None, seed=742)
        P(f'\n  Indus {field} ({len(nm)} distinct middles):')
        for k in LP + ['all']:
            line = f'    {str(k):8s} obs {pfmt(po[k])} PI {pref_index(po[k]):+.2f}'
            for m in ('slot', 'markov1', 'markov2'):
                a = agg(res[m])[k]; line += f' | {m} {pfmt(a)} PI {pref_index(a):+.2f}'
            P(line)
        # per-draw null distribution of the prefix index (all pairs, short length 2-3, b<=4)
        for m in ('slot', 'markov1', 'markov2'):
            v = []
            for d in res[m]:
                pdd = posd(d); t = tuple(sum(pdd[k][j] for k in [(2, 3), (2, 4), (3, 4)]) for j in range(4)); v.append(pref_index(t))
            t0 = tuple(sum(po[k][j] for k in [(2, 3), (2, 4), (3, 4)]) for j in range(4))
            o = pref_index(t0)
            P(f'    prefix index (2-4 pairs) obs {o:+.3f} vs {m} {mean(v):+.3f} [{q(v, .025):+.3f},{q(v, .975):+.3f}] p(two-sided)={min(1, 2 * min(sum(1 for x in v if x >= o) + 1, sum(1 for x in v if x <= o) + 1) / (len(v) + 1)):.3f}')
        if field == 'mid':
            pr = contain_pairs(nm)
            ext_pre = collections.Counter(); ext_suf = collections.Counter()
            for s, t, i in pr:
                if i == 0: ext_pre[t[len(s)]] += 1           # sign right after the short middle (prefix pairs)
                if i + len(s) == len(t): ext_suf[t[i - 1]] += 1  # sign right before it (suffix pairs)
            el = collections.Counter(a for n in nm for a in n)
            P('    signs appended after a contained PREFIX: ' + ', '.join(f'{a}:{n}{"*" if a in FR else ""}' for a, n in ext_pre.most_common(12)) + f'  (* frame/numeral; frame share {sum(n for a, n in ext_pre.items() if a in FR) / max(1, sum(ext_pre.values())):.2f})')
            P('    signs prepended before a contained SUFFIX: ' + ', '.join(f'{a}:{n}{"*" if a in FR else ""}' for a, n in ext_suf.most_common(12)) + f'  (frame share {sum(n for a, n in ext_suf.items() if a in FR) / max(1, sum(ext_suf.values())):.2f})')
            P(f'    frame/numeral share of all middle tokens: {sum(n for a, n in el.items() if a in FR) / sum(el.values()):.2f}')
            # are extensions concentrated? top-3 extension share vs top-3 token share
            for lab, E in (('prefix-append', ext_pre), ('suffix-prepend', ext_suf)):
                tot = sum(E.values()); t3 = sum(n for _, n in E.most_common(3)) / tot if tot else float('nan')
                P(f'    {lab}: {tot} pairs, {len(E)} distinct extension signs, top-3 share {t3:.2f}; middle-token top-3 share {sum(n for _, n in el.most_common(3)) / sum(el.values()):.2f}')
    # comparators
    base = names; L0 = [len(n) for n in base]; NDC = 10
    ind_m1 = mean([d['share'] for d in run_nulls(names, 20, None, seed=745)[0]['markov1']])
    P(f'\n  Indus {LV} mid: substring share {cstats(names)["share"]:.3f}, Markov-1 {ind_m1:.3f}, obs/M1 {cstats(names)["share"] / ind_m1:.2f}')
    P(f'\n  comparators, length-matched to the Indus {LV} middle lengths (n={len(base)}), 20 draws; each with its own slot shuffle')
    C63 = 'data/derived/dark/loop63_corpora/'
    lists = [('jp_given', C63), ('jp_person_given', C63), ('vi_given', C63), ('cn_ancient', C63), ('cn_given', C63), ('ko_given', C63)] + [(x, L56.CORP) for x in ['ur3_names_dedup', 'ob_names_dedup', 'linb_personnel_dedup', 'latin_names_dedup']]
    for nmx, d in lists:
        fn = d + nmx + '.jsonl'
        if not os.path.exists(fn): P(f'    {nmx}: missing'); continue
        pool = sorted(set(tuple(json.loads(l)['seq']) for l in open(fn)))
        pool = [s for s in pool if len(s) >= 2]
        obsd = []; nulld = []; shorts = []; m1 = []; m2 = []
        for b in range(NDC):
            r = random.Random(7400 + b)
            sub, sh = L56.length_match(pool, L0, len(base), r); shorts.append(sh); sub = sorted(set(sub))
            obsd.append(cstats(sub)); nulld.append(cstats(sorted(set(shuf_slot(sub, r)))))
            m1.append(cstats(sorted(set(chain_names(sub, 1, r)[0])))); m2.append(cstats(sorted(set(chain_names(sub, 2, r)[0]))))
        A = agg(obsd); N = agg(nulld); A1 = agg(m1)
        ex = mean([o['share'] - n['share'] for o, n in zip(obsd, nulld)])
        sh_o = mean([o['share'] for o in obsd])
        P(f'    {nmx:22s} short {mean(shorts):.0f}; substring share {sh_o:.3f} | slot-null {mean([o["share"] for o in nulld]):.3f} (excess {ex:+.3f}) | Markov-1 {mean([o["share"] for o in m1]):.3f} (obs/M1 {sh_o / max(1e-9, mean([o["share"] for o in m1])):.2f}) | Markov-2 {mean([o["share"] for o in m2]):.3f} (obs/M2 {sh_o / max(1e-9, mean([o["share"] for o in m2])):.2f})')
        P(f'        positions: all pairs {pfmt(A["all"])} PI {pref_index(A["all"]):+.2f} | slot-null {pfmt(N["all"])} PI {pref_index(N["all"]):+.2f} | Markov-1 {pfmt(A1["all"])} PI {pref_index(A1["all"]):+.2f}; 2-in-3 {pfmt(A[(2, 3)])}; 2-in-4 {pfmt(A[(2, 4)])}; 3-in-4 {pfmt(A[(3, 4)])}')
    save('')

# ================================================================== cycle 3
if CY == 3:
    objs = wells_objects()
    # outside fields
    C = {r['cisi']: r for r in L56.C}
    depth = {}
    for r in csv.DictReader(open('data/raw/inscriptions.csv')):
        d = r['depth'].strip()
        if d in ('- -', '-', '') or r['cisi'] in depth: continue
        try:
            v = float(d.split()[0]); u = d.split()[1] if len(d.split()) > 1 else 'ft'
            depth[r['cisi']] = abs(v) * (0.3048 if u.startswith('ft') else 1.0)
        except Exception: pass
    def hper(t):
        t = (t or '').replace('Period', '').strip()
        m = {'3A': 1, '3B': 2, '3B/C': 2.5, '3C': 3, '3C-1': 3, '3C-2': 3.2, '3C-3': 3.4, '3C-4': 3.6, '3': 2, '4': 4, '5': 5}
        return m.get(t)
    def bad(v): return v is None or v.strip() in ('-', '--', '- -', '', 'None')
    for o in objs:
        r = C[o['cisi']]
        o['area'] = None if bad(r['area-section']) else (o['site'], r['area-section'].strip())
        o['room'] = None if bad(r['room-grid']) else (o['site'], r['area-section'].strip(), r['block-house'].strip(), r['room-grid'].strip())
        s = (r['symbol'] or '').strip(); o['emb'] = None if s in ('', '-', 'None') else s.split(':')[0]
        o['age'] = None
        if o['site'] == 'Mohenjo-daro' and o['cisi'] in depth: o['age'] = depth[o['cisi']]          # deeper = older
        if o['site'] == 'Harappa':
            h = hper(r['time']);
            if h is not None: o['age'] = -h                                                                 # earlier period = older
    names = distinct(objs, 'mid')
    by = collections.defaultdict(list)
    for o in objs:
        if len(o['mid']) >= 2: by[o['mid']].append(o)
    byL = collections.defaultdict(list)
    for n in names: byL[len(n)].append(n)
    pr = contain_pairs(names)
    FEAT = [('site', lambda a, b: a['site'] == b['site']),
            ('area', lambda a, b: (a['area'] == b['area']) if a['area'] and b['area'] else None),
            ('room', lambda a, b: (a['room'] == b['room']) if a['room'] and b['room'] else None),
            ('closer', lambda a, b: (a['closer'] == b['closer']) if a['closer'] is not None or b['closer'] is not None else None),
            ('emblem', lambda a, b: (a['emb'] == b['emb']) if a['emb'] and b['emb'] else None),
            ('objtype', lambda a, b: a['ot'] == b['ot']),
            ('short_older', lambda a, b: (a['age'] > b['age']) if a['site'] == b['site'] and a['age'] is not None and b['age'] is not None and a['age'] != b['age'] else None)]
    def pairscore(s, t):
        """per (short, long) middle pair: mean over object pairs of each feature (None = not scorable)."""
        out = {}
        for f, fn in FEAT:
            v = [fn(a, b) for a in by[s] for b in by[t]]; v = [x for x in v if x is not None]
            out[f] = sum(v) / len(v) if v else None
        return out
    def summarize(pairs):
        out = {}
        for f, _ in FEAT:
            v = [ps[f] for ps in pairs if ps[f] is not None]; out[f] = (mean(v), len(v))
        return out
    obs = summarize([pairscore(s, t) for s, t, _ in pr])
    shared = lambda s, t: bool(set(s) & set(t))
    def nullpairs(r, mode):
        res = []
        for s, t, _ in pr:
            cand = byL[len(t)]
            for _ in range(200):
                u = r.choice(cand)
                if u == t or s in [u[i:i + len(s)] for i in range(len(u) - len(s) + 1)]: continue
                if mode == 'share' and not shared(s, u): continue
                break
            else: continue
            res.append(pairscore(s, u))
        return summarize(res)
    P(f'##### S-DARK-74 cycle 3, {LV}: outside facts for {len(pr)} containment pairs (distinct middles; object pairs averaged per middle pair)')
    P('  n1 = long middle replaced by a random distinct middle of the same length not containing the short one; n2 = same, but sharing >= 1 sign with the short one')
    NUL = {'n1': [], 'n2': []}
    for b in range(NN):
        r = random.Random(743 + b)
        NUL['n1'].append(nullpairs(r, 'any')); NUL['n2'].append(nullpairs(r, 'share'))
    for f, _ in FEAT:
        o, n = obs[f]; line = f'  {f:12s} obs {o:.3f} (scorable pairs {n})'
        for m in ('n1', 'n2'):
            v = [x[f][0] for x in NUL[m]]; mu = mean(v)
            p = (sum(1 for x in v if x >= o) + 1) / (len(v) + 1)
            line += f' | {m} {mu:.3f} [{q(v, .025):.3f},{q(v, .975):.3f}] x{o / mu if mu else float("nan"):.2f} p_hi={p:.3f}'
        P(line)
    # object-type direction: short on tablet, long on seal?
    ot = collections.Counter(); otn = collections.Counter()
    for s, t, _ in pr:
        for a in by[s]:
            for b in by[t]: ot[(a['ot'], b['ot'])] += 1
    tot = sum(ot.values())
    P('  object-type cross-tab (short obj, long obj), object pairs: ' + ', '.join(f'{k[0]}->{k[1]} {v} ({v / tot:.2f})' for k, v in ot.most_common(8)))
    alltype = collections.Counter(len(o['mid']) >= 3 and o['ot'] for o in objs if len(o['mid']) >= 2)
    for Lx in (2, 3, 4):
        c = collections.Counter(o['ot'] for o in objs if len(o['mid']) == Lx)
        P(f'  object types of all length-{Lx} middles: ' + ', '.join(f'{k} {v}' for k, v in c.most_common(5)))
    # site concentration of the containment network: share of pairs where all objects of both middles are MD/H
    mdh = sum(1 for s, t, _ in pr if all(o['big'] for o in by[s] + by[t])) / len(pr)
    P(f'  pairs resting only on Mohenjo-daro + Harappa objects: {mdh:.2f}')
    json.dump(dict(obs=obs), open(f'data/derived/dark/loop74_c3_{LV}.json', 'w'), indent=1)
    save('')

# ================================================================== cycle 4
if CY == 4:
    objs = wells_objects()
    QUAL = L56.build_qual(objs)
    # merge maps for IM77 W numbers
    AL = json.load(open('data/derived/sign_allographs_levels.json'))['merges']
    lv_ok = {'seq_raw': set(), 'seq_strong': {'strong'}, 'seq_all': {'strong', 'probable'}}[LV]
    MM = {m['form']: m['into'] for m in AL if m['level'] in lv_ok}
    def mm(a):
        seen = 0
        while a in MM and seen < 10: a = MM[a]; seen += 1
        return a
    L27 = json.load(open('data/derived/dark/loop27_sets.json'))
    newk = set((a, b) for a, b in L27['new'])
    M2W = {}
    for w, ms in L56.BR.items():
        for m in ms: M2W.setdefault(m, []).append(int(w))
    for p in L56.PROP['proposals']: M2W.setdefault(p['M'], []).append(p['W'])
    M2W = {m: min(v) for m, v in M2W.items()}
    rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    byk = collections.defaultdict(list)
    for r in rows:
        k = (r['text_no'], r['side'])
        if k in newk and r['signs_clean'].strip(): byk[k].append(r)
    new = []; unb = 0; tot = 0
    for k, rs in byk.items():
        rs.sort(key=lambda r: int(r['line']))
        ms = [int(x) for r in rs for x in r['signs_clean'].split() if x.strip() and x != '0']
        seq = []
        for m in ms:
            tot += 1
            if m in M2W: seq.append(mm(M2W[m]))
            else: unb += 1; seq.append(10000 + m)
        if len(seq) < 2: continue
        site = {'Mohenjodaro': 'Mohenjo-daro'}.get(rs[0]['site'], rs[0]['site'])
        lab = L56.parse(seq, QUAL)
        mid = tuple(a for a, l in zip(seq, lab) if l in ('NAME', 'COUNT'))
        new.append(dict(id=k, site=site, seq=seq, mid=mid))
    P(f'##### S-DARK-74 cycle 4, {LV}: held-out material. IM77-only texts {len(new)} (>= 2 signs; unbridged M tokens {unb}/{tot}, kept opaque, cannot match)')
    def cross(X, R):
        """X = target middles, R = reference middles: share of X (len 2-4) inside a longer R middle; share of X (len >= 3) containing a shorter R middle."""
        Rs = set(R); subsR = set()
        for t in R:
            for l in range(2, len(t)):
                for i in range(len(t) - l + 1): subsR.add(t[i:i + l])
        x24 = [x for x in X if 2 <= len(x) <= 4]; x3 = [x for x in X if len(x) >= 3]
        a = sum(1 for x in x24 if x in subsR) / len(x24) if x24 else float('nan')
        def has(x):
            return any(x[i:i + l] in Rs for l in range(2, len(x)) for i in range(len(x) - l + 1))
        b = sum(1 for x in x3 if has(x)) / len(x3) if x3 else float('nan')
        return dict(inside=a, contains=b, n24=len(x24), n3=len(x3), kin=sum(1 for x in x24 if x in subsR), kc=sum(1 for x in x3 if has(x)))
    def test(label, X, R, nn=NN):
        X = sorted(set(x for x in X if len(x) >= 2)); R = sorted(set(R))
        obs = cross(X, R)
        lens = [len(x) for x in X]; POOLR = [a for n in R for a in n]
        ch1 = Chain(R, 1); ch2 = Chain(R, 2); chs = SChain(R, random.Random(3))
        res = collections.defaultdict(list)
        for b in range(nn):
            r = random.Random(7440 + b)
            res['slot(X)'].append(cross(sorted(set(shuf_slot(X, r))), R))
            res['freqrand(R)'].append(cross(sorted(set(tuple(r.choice(POOLR) for _ in range(L)) for L in lens)), R))
            if b < max(20, nn // 2):
                res['markov1(R)'].append(cross(chain_names(X, 1, r, ch1, lens)[0], R))
                res['markov2(R)'].append(cross(chain_names(X, 2, r, ch2, lens)[0], R))
                res['calib(R)'].append(cross(chain_names(X, 's', r, chs, lens)[0], R))
        P(f'\n  {label}: X {len(X)} distinct middles (len 2-4: {obs["n24"]}, >=3: {obs["n3"]}), reference R {len(R)}')
        out = {}
        for k, kk in (('inside', 'kin'), ('contains', 'kc')):
            line = f'    {k:9s} obs {obs[k]:.3f} ({obs[kk]})'
            for m, v in res.items():
                vv = [d[k] for d in v]; mu = mean(vv); p = (sum(1 for x in vv if x >= obs[k]) + 1) / (len(vv) + 1)
                line += f' | {m} {mu:.3f} [{q(vv, .025):.3f},{q(vv, .975):.3f}] x{obs[k] / mu if mu else float("nan"):.2f} p={p:.3f}'
                out[(k, m)] = (obs[k], mu, p)
            P(line)
        return out
    MDH = [o['mid'] for o in objs if o['big'] and len(o['mid']) >= 2]
    HO = [o['mid'] for o in objs if not o['big'] and len(o['mid']) >= 2]
    ALLW = [o['mid'] for o in objs if len(o['mid']) >= 2]
    newm = [o['mid'] for o in new]
    R1 = test('held-out Wells sites (not MD/H) vs MD+H reference', HO, MDH)
    R2 = test('IM77-only texts vs all-Wells reference', newm, ALLW)
    R3 = test('IM77-only texts from sites other than MD/H vs MD+H reference', [o['mid'] for o in new if o['site'] not in ('Mohenjo-daro', 'Harappa')], MDH)
    # within held-out sites: its own substring share vs its own nulls
    hn = sorted(set(HO))
    obs = cstats(hn); res, _ = run_nulls(hn, NN, None, seed=744)
    P(f'\n  within held-out Wells sites only: {len(hn)} distinct middles; share {obs["share"]:.3f} ({obs["npairs"]} pairs) | ' + ' | '.join(f'{m} ' + zline(obs['share'], res[m], 'share')[0] for m in NULLS))
    # examples of IM77-only middles inside reference middles
    subsR = collections.defaultdict(list)
    for t in set(ALLW):
        for l in range(2, len(t)):
            for i in range(len(t) - l + 1): subsR[t[i:i + l]].append(t)
    ex = [(o['id'], o['site'], o['mid'], subsR[o['mid']][:3]) for o in new if 2 <= len(o['mid']) <= 4 and o['mid'] in subsR]
    P(f'  IM77-only middles found inside Wells middles: {len(ex)}; e.g. ' + '; '.join(f'{i[0]}:{s} {fm(m)} in {",".join(fm(t) for t in ts)}' for i, s, m, ts in ex[:8]))
    save('')

# ================================================================== cycle 5 (= cycle 1b): calibrated chain + frame strip
if CY == 5:
    """The MLE chains regenerate the training bigrams (on a one-copy-per-name list most bigram types occur once), so they
    inflate the substring share even for lists with zero excess. Calibrated chain = interpolated Markov-1 with the mixing
    weight fitted by held-out likelihood (SChain). Run on Indus (mid, name, frame-stripped) and on every comparator."""
    FR = L56.NUM | L56.OPEN | L56.MARK | L56.MJAR | L56.SUF | set(L56.CL)
    objs = wells_objects()
    def strip(n): return tuple(a for a in n if a not in FR)
    sets = [('mid', distinct(objs, 'mid')), ('name', distinct(objs, 'name')),
            ('stripped', sorted(set(strip(o['mid']) for o in objs if len(strip(o['mid'])) >= 2)))]
    P(f'##### S-DARK-74 cycle 5 (1b), {LV}: substring share vs a CALIBRATED chain (interpolated Markov-1, lambda by 2-fold held-out likelihood), {NN} draws')
    def block(label, names, nn):
        r0 = random.Random(75); ch = SChain(names, r0)
        obs = cstats(names); sh = []; ms = []; sl = []
        for b in range(nn):
            r = random.Random(7500 + b)
            ms.append(cstats(sorted(set(chain_names(names, 's', r, ch)[0]))))
            sl.append(cstats(sorted(set(shuf_slot(names, r)))))
        out = dict(lam=ch.lam, obs=obs['share'], chain=mean([d['share'] for d in ms]), slot=mean([d['share'] for d in sl]))
        line = f'  {label:28s} n={len(names)} lambda={ch.lam:.2f} share {obs["share"]:.3f} | slot ' + zline(obs['share'], sl, 'share')[0] + ' | chain ' + zline(obs['share'], ms, 'share')[0]
        for k in ('f23', 'f24', 'f34'):
            line += f' | {k} {obs[k]:.3f} vs chain ' + zline(obs[k], ms, k)[0].split(' p=')[0]
        P(line)
        return out
    R = {}
    for lab, nm in sets: R['indus_' + lab] = block(f'Indus {lab}', nm, NN)
    # frame-only extension pairs
    pr = contain_pairs(sets[0][1])
    fo = [(s, t) for s, t, i in pr if all(a in FR for a in t[:i] + t[i + len(s):])]
    P(f'  Indus mid: containment pairs whose extension is ONLY frame/numeral signs: {len(fo)}/{len(pr)} = {len(fo) / len(pr):.2f}')
    if LV == 'seq_raw':
        base = sets[0][1]; L0 = [len(n) for n in base]
        C63 = 'data/derived/dark/loop63_corpora/'
        lists = [('jp_given', C63), ('jp_person_given', C63), ('vi_given', C63), ('cn_ancient', C63), ('ko_given', C63)] + [(x, L56.CORP) for x in ['ur3_names_dedup', 'ob_names_dedup', 'linb_personnel_dedup', 'latin_names_dedup']]
        for nmx, d in lists:
            pool = sorted(set(tuple(json.loads(l)['seq']) for l in open(d + nmx + '.jsonl'))); pool = [s for s in pool if len(s) >= 2]
            vals = []
            for b in range(5):
                r = random.Random(7600 + b); sub = sorted(set(L56.length_match(pool, L0, len(base), r)[0]))
                vals.append(block(f'{nmx} draw {b}', sub, max(5, NN // 10)))
            R[nmx] = {k: mean([v[k] for v in vals]) for k in vals[0]}
            P(f'  == {nmx}: lambda {R[nmx]["lam"]:.2f}, share {R[nmx]["obs"]:.3f}, chain {R[nmx]["chain"]:.3f} (obs/chain {R[nmx]["obs"] / max(1e-9, R[nmx]["chain"]):.2f}), slot {R[nmx]["slot"]:.3f}')
    json.dump(R, open(f'data/derived/dark/loop74_c5_{LV}.json', 'w'), indent=1)
    save('')

# ================================================================== cycle 6 (= 3b/4b): local or global? closer by position
if CY == 6:
    objs = wells_objects()
    P(f'##### S-DARK-74 cycle 6 (3b/4b), {LV}: is the excess over the calibrated chain LOCAL (within a site) or GLOBAL (across sites)?')
    def within(label, names, nn):
        ch = SChain(names, random.Random(76)); obs = cstats(names); ms = []
        for b in range(nn): ms.append(cstats(sorted(set(chain_names(names, 's', random.Random(7700 + b), ch)[0]))))
        z = zline(obs['share'], ms, 'share')
        P(f'  WITHIN {label:34s} n={len(names)} lambda={ch.lam:.2f} share {obs["share"]:.3f} ({obs["npairs"]} pairs) | calib chain {z[0]}')
        return obs['share'], z[1]
    def crossc(label, X, R, nn):
        X = sorted(set(X)); R = sorted(set(R)); ch = SChain(R, random.Random(77)); lens = [len(x) for x in X]
        subsR = set(t[i:i + l] for t in R for l in range(2, len(t)) for i in range(len(t) - l + 1)); RS = set(R)
        def st(Y):
            y24 = [y for y in Y if 2 <= len(y) <= 4]; y3 = [y for y in Y if len(y) >= 3]
            a = sum(1 for y in y24 if y in subsR) / max(1, len(y24))
            b = sum(1 for y in y3 if any(y[i:i + l] in RS for l in range(2, len(y)) for i in range(len(y) - l + 1))) / max(1, len(y3))
            return dict(inside=a, contains=b)
        o = st(X); ms = [st(chain_names(X, 's', random.Random(7800 + b), ch, lens)[0]) for b in range(nn)]
        P(f'  CROSS  {label:34s} X {len(X)} R {len(R)} lambda={ch.lam:.2f} | inside {o["inside"]:.3f} vs calib ' + zline(o['inside'], ms, 'inside')[0] + f' | contains {o["contains"]:.3f} vs calib ' + zline(o['contains'], ms, 'contains')[0])
    MD = distinct([o for o in objs if o['site'] == 'Mohenjo-daro'], 'mid'); HA = distinct([o for o in objs if o['site'] == 'Harappa'], 'mid')
    HO = distinct([o for o in objs if not o['big']], 'mid')
    within('all Wells', distinct(objs, 'mid'), NN)
    within('Mohenjo-daro only', MD, NN); within('Harappa only', HA, NN); within('held-out sites only (pooled)', HO, NN)
    within('seals only', distinct([o for o in objs if o['ot'] == 'seal'], 'mid'), NN)
    within('tablets only', distinct([o for o in objs if o['ot'] == 'tablet'], 'mid'), NN)
    crossc('Harappa X vs Mohenjo-daro R', HA, MD, NN); crossc('Mohenjo-daro X vs Harappa R', MD, HA, NN)
    crossc('held-out X vs MD+H R', HO, MD + HA, NN)
    # closer sharing by position, with a partner matched on the long middle's last (or first) sign
    by = collections.defaultdict(list)
    for o in objs:
        if len(o['mid']) >= 2: by[o['mid']].append(o)
    names = distinct(objs, 'mid'); pr = contain_pairs(names)
    byLlast = collections.defaultdict(list); byLfirst = collections.defaultdict(list)
    for n in names: byLlast[(len(n), n[-1])].append(n); byLfirst[(len(n), n[0])].append(n)
    def cs(s, t, feat):
        v = []
        for a in by[s]:
            for b in by[t]:
                if feat == 'closer':
                    if a['closer'] is None and b['closer'] is None: continue
                    v.append(a['closer'] == b['closer'])
                else: v.append(a['site'] == b['site'])
        return sum(v) / len(v) if v else None
    for feat in ('closer', 'site'):
        for w in ('prefix', 'suffix', 'infix'):
            sel = [(s, t) for s, t, i in pr if (w == 'prefix' and i == 0) or (w == 'suffix' and i + len(s) == len(t)) or (w == 'infix' and 0 < i and i + len(s) < len(t))]
            ob = mean([x for x in (cs(s, t, feat) for s, t in sel) if x is not None])
            for mt, pool in (('same last sign', byLlast), ('same first sign', byLfirst)):
                nv = []
                for b in range(NN):
                    r = random.Random(7900 + b); v = []
                    for s, t in sel:
                        c = [u for u in pool[(len(t), t[-1] if mt == 'same last sign' else t[0])] if u != t and s not in [u[k:k + len(s)] for k in range(len(u) - len(s) + 1)]]
                        if not c: continue
                        x = cs(s, r.choice(c), feat)
                        if x is not None: v.append(x)
                    nv.append(mean(v))
                p = (sum(1 for x in nv if x >= ob) + 1) / (len(nv) + 1)
                P(f'  {feat:6s} {w:6s} pairs {len(sel):3d}: obs {ob:.3f} | partner of same length with {mt} (not containing the short one) {mean(nv):.3f} [{q(nv, .025):.3f},{q(nv, .975):.3f}] x{ob / mean(nv) if mean(nv) else float("nan"):.2f} p={p:.3f}')
    save('')
