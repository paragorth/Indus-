"""pe45 THE NAMES ARE A POPULATION THAT BREEDS.  Shared code: naming-grammar simulator, summary
statistics, per-sign features.

A naming grammar theta says how a population of individuals gets its names:
  free elements    Zipf vocabulary (Vf signs, exponent a)
  god-like signs   ng constants; a name carries one with prob p_god at slot god_slot
  lineage markers  n_lin lineages share nm = mshare*n_lin marker signs; a member carries the
                   lineage marker with prob p_mark at slot mark_slot
  inheritance      3 generations, fan-out fan; a child copies one free element of its father's
                   name with prob p_inh (same slot, or a new slot with prob shift)
  tablets          a tablet lists n individuals; with prob kin_frac they come from k ~ 1+Pois(lpt-1)
                   lineages (a kin group), else from the whole population
  tablet pool      with prob p_pool each free element as written on a tablet is replaced by one of
                   s_pool tablet-local signs (the pe7 'local pool' alternative to kinship)
Sign classes in simulations: 0 free, 1 god, 2 marker.
"""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    _o.environ[_v] = '1'
import os, json, math, random, collections, bisect
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe45_ckpt')
G = 3

PNAMES = ['lVf', 'a', 'ng', 'p_god', 'god_slot', 'a_god', 'ln_lin', 'mshare', 'p_mark', 'mark_slot', 'fan',
          'p_inh', 'shift', 'kin_frac', 'lpt', 'p_pool', 's_pool']


def prior(rng):
    th = dict(lVf=rng.uniform(math.log(50), math.log(3000)), a=rng.uniform(0.5, 1.8), ng=rng.randint(0, 16),
              p_god=rng.uniform(0, 0.7), god_slot=rng.randint(0, 2), a_god=rng.uniform(0, 2),
              ln_lin=rng.uniform(math.log(5), math.log(1000)), mshare=rng.uniform(0.05, 1),
              p_mark=rng.uniform(0, 0.9), mark_slot=rng.randint(0, 2), fan=rng.randint(1, 6),
              p_inh=rng.uniform(0, 0.9), shift=rng.uniform(0, 1), kin_frac=rng.uniform(0, 1),
              lpt=rng.uniform(1, 6), p_pool=(rng.uniform(0, 0.9) if rng.random() < 0.5 else 0.0),
              s_pool=rng.randint(2, 40))
    if th['ng'] == 0:
        th['p_god'] = 0.0
    return th


def _cdf(n, a):
    w = 1.0 / np.arange(1, n + 1) ** a
    c = np.cumsum(w)
    return (c / c[-1]).tolist()


def _pois(lam, rng):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _slot(mode, L, slots, rng):
    if mode == 0:
        p = 0
    elif mode == 1:
        p = L - 1
    else:
        p = rng.randrange(L)
    if slots[p] == -1:
        return p
    e = [i for i in range(L) if slots[i] == -1]
    return rng.choice(e) if e else -1


def simulate(th, sizes, lens, rng):
    """sizes: names per tablet; lens: empirical name lengths.  Returns (corpus, cls, kinflag)."""
    Vf = max(5, int(round(math.exp(th['lVf']))))
    ng = int(th['ng'])
    n_lin = max(1, int(round(math.exp(th['ln_lin']))))
    nm = max(1, int(round(th['mshare'] * n_lin)))
    fan = int(th['fan'])
    cf = _cdf(Vf, th['a'])
    cg = _cdf(ng, th['a_god']) if ng else None
    mk = [rng.randrange(nm) for _ in range(n_lin)]
    GOD0, MK0 = Vf, Vf + ng
    cache = {}
    p_god, p_mark, p_inh, shift = th['p_god'], th['p_mark'], th['p_inh'], th['shift']
    gs, ms = int(th['god_slot']), int(th['mark_slot'])
    rnd = rng.random

    def rr(n):
        return int(rnd() * n)

    def free():
        return bisect.bisect_left(cf, rnd())

    def name(l, g, i):
        k = (l, g, i)
        v = cache.get(k)
        if v is not None:
            return v
        L = lens[rr(len(lens))]
        s = [-1] * L
        if g > 0 and rnd() < p_inh:
            F = name(l, g - 1, i // fan)
            fr = [(j, e) for j, e in enumerate(F) if e < Vf]
            if fr:
                j, e = fr[rr(len(fr))]
                t = j if (j < L and rnd() >= shift) else rr(L)
                s[t] = e
        if ng and rnd() < p_god:
            p = _slot(gs, L, s, rng)
            if p >= 0:
                s[p] = GOD0 + bisect.bisect_left(cg, rnd())
        if rnd() < p_mark:
            p = _slot(ms, L, s, rng)
            if p >= 0:
                s[p] = MK0 + mk[l]
        for j in range(L):
            if s[j] == -1:
                s[j] = free()
        v = tuple(s)
        cache[k] = v
        return v

    C, kin = [], []
    p_pool, s_pool = th['p_pool'], int(th['s_pool'])
    kf, lpt = th['kin_frac'], th['lpt']
    for n in sizes:
        isk = rnd() < kf
        if isk:
            k = 1 + _pois(max(0.0, lpt - 1), rng)
            lins = [rr(n_lin) for _ in range(k)]
        tab = []
        pool = [free() for _ in range(s_pool)] if p_pool > 0 else None
        for _ in range(n):
            l = lins[rr(len(lins))] if isk else rr(n_lin)
            g = rr(G)
            i = rr(fan ** g)
            nm_ = name(l, g, i)
            if pool is not None:
                nm_ = tuple(pool[rr(s_pool)] if (e < Vf and rnd() < p_pool) else e for e in nm_)
            tab.append(nm_)
        C.append(tab)
        kin.append(isk)
    cls = {}
    for t in C:
        for nmx in t:
            for e in nmx:
                cls[e] = 0 if e < Vf else (1 if e < MK0 else 2)
    return C, cls, kin


# ---------------------------------------------------------------- statistics
SA = ['ttr', 'hapax', 'top1', 'top10', 'top50', 'init10', 'fin10', 'rep', 'recur', 'sh_same', 'sh_cross',
      'rsh_same', 'rsh_cross']
SB = ['fix', 'pos_same', 'pos_init', 'clq_mean', 'clq_big', 'cotravel', 'disp', 'deg', 'trans', 'giant',
      'top5_init', 'top5_fin']
SNAMES = SA + SB


def encode(names_by_tab):
    ix = {}
    C = []
    for t in names_by_tab:
        C.append([tuple(ix.setdefault(x, len(ix)) for x in n) for n in t])
    return C, {v: k for k, v in ix.items()}


def stats(C, seed=0):
    rng = random.Random(seed)
    freq = collections.Counter()
    ini, fin = collections.Counter(), collections.Counter()
    tabs_of = collections.defaultdict(set)
    names_tab = collections.defaultdict(set)
    rep = 0
    allnames = []
    for ti, t in enumerate(C):
        for n in t:
            freq.update(n)
            ini[n[0]] += 1
            fin[n[-1]] += 1
            rep += len(set(n)) < len(n)
            for e in n:
                tabs_of[e].add(ti)
            names_tab[n].add(ti)
            allnames.append((ti, n))
    tok = sum(freq.values())
    nn = len(allnames)
    srt = [c for _, c in freq.most_common()]
    out = {}
    out['ttr'] = len(freq) / tok
    out['hapax'] = sum(1 for c in srt if c == 1) / len(srt)
    out['top1'] = srt[0] / tok
    out['top10'] = sum(srt[:10]) / tok
    out['top50'] = sum(srt[:50]) / tok
    out['init10'] = sum(c for _, c in ini.most_common(10)) / nn
    out['fin10'] = sum(c for _, c in fin.most_common(10)) / nn
    out['rep'] = rep / nn
    out['recur'] = sum(1 for v in names_tab.values() if len(v) >= 2) / len(names_tab)
    rare = {e for e, c in freq.items() if 2 <= c <= 8}
    # same-tablet pairs
    sh = rsh = npair = 0
    ps = pi = pev = 0
    clq, big, nbig = [], 0, 0
    top10 = {e for e, _ in freq.most_common(10)}
    for t in C:
        u = list(dict.fromkeys(t))
        for i in range(len(u)):
            a = u[i]
            sa = set(a)
            for j in range(i + 1, len(u)):
                b = u[j]
                com = sa.intersection(b)
                npair += 1
                if com:
                    sh += 1
                    if com & rare:
                        rsh += 1
                    for e in com:
                        pev += 1
                        ia, ib = a.index(e), b.index(e)
                        ps += ia == ib
                        pi += (ia == 0 and ib == 0)
        if len(u) >= 4:
            nbig += 1
            cc = collections.Counter(e for n in u for e in set(n))
            fl = False
            for e, c in cc.items():
                if c >= 2:
                    clq.append(c / len(u))
                    if e not in top10 and c / len(u) >= 0.5:
                        fl = True
            big += fl
    out['sh_same'] = sh / max(1, npair)
    out['rsh_same'] = rsh / max(1, npair)
    # cross pairs
    csh = crsh = cn = 0
    for _ in range(4000):
        x, y = allnames[int(rng.random() * nn)], allnames[int(rng.random() * nn)]
        if x[0] == y[0] or x[1] == y[1]:
            continue
        com = set(x[1]).intersection(y[1])
        cn += 1
        csh += bool(com)
        crsh += bool(com & rare)
    out['sh_cross'] = csh / max(1, cn)
    out['rsh_cross'] = crsh / max(1, cn)
    # held-out block
    fx = [max(ini[e], fin[e]) / c for e, c in freq.items() if c >= 10]
    out['fix'] = float(np.mean(fx)) if fx else 0.5
    out['pos_same'] = ps / max(1, pev)
    out['pos_init'] = pi / max(1, pev)
    out['clq_mean'] = float(np.mean(clq)) if clq else 0.0
    out['clq_big'] = big / max(1, nbig)
    pair_tabs = collections.defaultdict(int)
    for t in C:
        rs = sorted({e for n in t for e in n if e in rare})
        if len(rs) > 40:
            rs = rng.sample(rs, 40)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                pair_tabs[(rs[i], rs[j])] += 1
    out['cotravel'] = (sum(1 for v in pair_tabs.values() if v >= 2) / len(pair_tabs)) if pair_tabs else 0.0
    dp = [len(tabs_of[e]) / c for e, c in freq.items() if c >= 5]
    out['disp'] = float(np.mean(dp)) if dp else 1.0
    # network of distinct names linked by rare signs
    dn = list(names_tab.keys())
    by = collections.defaultdict(list)
    for k, n in enumerate(dn):
        for e in set(n):
            if e in rare:
                by[e].append(k)
    adj = collections.defaultdict(set)
    for e, ks in by.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                adj[ks[i]].add(ks[j])
                adj[ks[j]].add(ks[i])
    out['deg'] = sum(len(v) for v in adj.values()) / len(dn)
    tri = trip = 0
    nodes = [k for k in adj if len(adj[k]) >= 2]
    for k in (rng.sample(nodes, 300) if len(nodes) > 300 else nodes):
        nb = list(adj[k])
        for _ in range(10):
            x, y = rng.sample(nb, 2)
            trip += 1
            tri += y in adj[x]
    out['trans'] = tri / max(1, trip)
    seen, best = set(), 0
    for s in adj:
        if s in seen:
            continue
        st, sz = [s], 0
        seen.add(s)
        while st:
            u = st.pop()
            sz += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        best = max(best, sz)
    out['giant'] = best / len(dn)
    t5 = [e for e, _ in freq.most_common(5)]
    out['top5_init'] = float(np.mean([ini[e] / freq[e] for e in t5]))
    out['top5_fin'] = float(np.mean([fin[e] / freq[e] for e in t5]))
    return np.array([out[k] for k in SNAMES])


# ---------------------------------------------------------------- per-sign features
FNAMES = ['lfreq', 'rank_q', 'disp', 'init', 'fin', 'mid', 'slot_ent', 'partner_div', 'clump', 'cotab_rare',
          'recur_share', 'name_len']


def sign_features(C, minfreq=3):
    freq = collections.Counter()
    ini, fin = collections.Counter(), collections.Counter()
    tabs = collections.defaultdict(collections.Counter)
    partners = collections.defaultdict(set)
    nl = collections.defaultdict(list)
    names_tab = collections.defaultdict(set)
    for ti, t in enumerate(C):
        for n in t:
            names_tab[n].add(ti)
    for ti, t in enumerate(C):
        for n in t:
            for j, e in enumerate(n):
                freq[e] += 1
                tabs[e][ti] += 1
                nl[e].append(len(n))
                if j == 0:
                    ini[e] += 1
                if j == len(n) - 1:
                    fin[e] += 1
                for f in n:
                    if f != e:
                        partners[e].add(f)
    rare = {e for e, c in freq.items() if 2 <= c <= 8}
    rtab = collections.defaultdict(set)
    for ti, t in enumerate(C):
        for n in t:
            for e in n:
                if e in rare:
                    rtab[ti].add(e)
    rankd = {e: r for r, (e, _) in enumerate(freq.most_common())}
    nt = len(freq)
    recur_tok = collections.Counter()
    for n, ts in names_tab.items():
        if len(ts) >= 2:
            for e in n:
                recur_tok[e] += 1
    keys, X = [], []
    for e, c in freq.items():
        if c < minfreq:
            continue
        i_, f_ = ini[e] / c, fin[e] / c
        m_ = max(0.0, 1 - i_ - f_)
        p = np.array([i_, f_, m_]) + 1e-9
        ent = float(-(p * np.log(p)).sum())
        ts = tabs[e]
        clump = sum(v * v for v in ts.values()) / (c * c)
        co = [len(rtab[ti] - {e}) for ti in ts]
        X.append([math.log(c), rankd[e] / nt, len(ts) / c, i_, f_, m_, ent, len(partners[e]) / c, clump,
                  float(np.mean(co)), recur_tok[e] / c, float(np.mean(nl[e]))])
        keys.append(e)
    return keys, np.array(X)


def load_corpora():
    return json.load(open(os.path.join(CK, 'corpora.json')))
