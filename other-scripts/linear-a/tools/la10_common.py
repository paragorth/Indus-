#!/usr/bin/env python3
"""LA-10 shared code: BREED the language. Random artificial languages, each a handful of sound-structure rules, generate a
vocabulary, spell it in a CV syllabary by Linear B rules, and are scored against a target fingerprint (la5 statistics plus
word length, affix and inventory statistics). A real-coded genetic algorithm evolves the rule sets toward the target.

Sign alphabet: 13 consonant series ('' = pure vowel, P T D K Q M N S Z R W J; H is written as a pure vowel, as in Linear B)
x 5 vowels.  A sign = series index * 5 + vowel index.  Variant signs (RA2, PA3 ...) fold onto their base series.
"""
import json, os, re, math, random, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la5_common as C

V2 = bool(os.environ.get('LA10_V2'))   # model v2 (cycle 2+): consonant stems, obligatory affixes, dissimilation, kober stat
SER = ['', 'P', 'T', 'D', 'K', 'Q', 'M', 'N', 'S', 'Z', 'R', 'W', 'J']
VOW = ['A', 'E', 'I', 'O', 'U']
NS = len(SER) * 5
SIDX = {s: i for i, s in enumerate(SER)}
RN = {SIDX['R'], SIDX['N']}
SIGRE = re.compile(r'^(D|J|K|M|N|P|Q|R|S|T|W|Z)?([AEIOU])[23]?$')

def enc(sign):
    m = SIGRE.match(sign)
    if not m: return None
    return SIDX[m.group(1) or ''] * 5 + VOW.index(m.group(2))

def encode_types(types):
    out = set()
    for t in types:
        e = [enc(a) for a in t]
        if len(e) >= 2 and all(x is not None for x in e): out.add(tuple(e))
    return sorted(out)

def la_types():
    return encode_types(set(w for _, _, w in C.words_of(C.la_docs())))

def lb_types():
    return encode_types(set(w for _, _, w in C.words_of(C.lb_docs())))

def name(t):
    return '-'.join(SER[a // 5] + VOW[a % 5] for a in t)

# ------------------------------------------------------------------ fingerprint
def pcl(i, L):
    if i == 0: return 0
    if i == L - 1: return 4
    if i == 1: return 1
    if i == L - 2: return 3
    return 2

STAT_NAMES = (['cecho', 'vecho_RN', 'vecho_oth', 'dbl', 'gap_exP', 'fin10_ex', 'init10_ex', 'pos', 'pre', 'suf', 'vinit', 'vmed',
               'len2', 'len3', 'len4', 'len5p'] + ['v_' + v for v in VOW] + ['fv_' + v for v in VOW] +
              ['s_' + (s or 'V') for s in SER])
if V2: STAT_NAMES = STAT_NAMES[:16] + ['kober'] + STAT_NAMES[16:]
CORE = set(STAT_NAMES[:17] if V2 else STAT_NAMES[:16])

def fingerprint(types):
    """analytic position-class-shuffle expectations (exact under independence of slots), so the score is deterministic."""
    n = len(types)
    pc = [[0.0] * NS for _ in range(5)]; pcn = [0] * 5
    pairn = collections.Counter()
    oc = ocn = ov_rn = ov_rn_n = ov_o = ov_o_n = od = npairs = 0
    ini = collections.Counter(); fin = collections.Counter(); uni = collections.Counter()
    lens = collections.Counter(); vinit = 0; vmed = vmed_n = 0
    S = set(types); pre = suf = 0
    for t in types:
        L = len(t); lens[min(L, 5)] += 1
        ini[t[0]] += 1; fin[t[-1]] += 1
        vinit += t[0] < 5
        if L >= 3:
            pre += t[1:] in S; suf += t[:-1] in S
        cls = [pcl(i, L) for i in range(L)]
        for i, a in enumerate(t):
            pc[cls[i]][a] += 1; pcn[cls[i]] += 1; uni[a] += 1
            if i: vmed_n += 1; vmed += a < 5
        for i in range(L - 1):
            x, y = t[i], t[i + 1]; pairn[(cls[i], cls[i + 1])] += 1; npairs += 1
            if x >= 5 and y >= 5: ocn += 1; oc += (x // 5 == y // 5)
            if y // 5 in RN: ov_rn_n += 1; ov_rn += (x % 5 == y % 5)
            else: ov_o_n += 1; ov_o += (x % 5 == y % 5)
            od += x == y
    P = [[v / pcn[c] if pcn[c] else 0.0 for v in pc[c]] for c in range(5)]
    # per class: series marginal (consonantal), vowel marginal, vowel marginal restricted to RN series
    def sermarg(p):
        m = [0.0] * len(SER)
        for a, v in enumerate(p): m[a // 5] += v
        return m
    def vowmarg(p, keep=None):
        m = [0.0] * 5
        for a, v in enumerate(p):
            if keep is None or keep(a): m[a % 5] += v
        return m
    SM = [sermarg(P[c]) for c in range(5)]
    VM = [vowmarg(P[c]) for c in range(5)]
    VRN = [vowmarg(P[c], lambda a: a // 5 in RN) for c in range(5)]
    VO = [vowmarg(P[c], lambda a: a // 5 not in RN) for c in range(5)]
    ec_same = ec_n = erns = ernn = eos = eon = ed = 0.0
    for (c1, c2), k in pairn.items():
        q1 = 1 - SM[c1][0]; q2 = 1 - SM[c2][0]
        ec_n += k * q1 * q2; ec_same += k * sum(SM[c1][s] * SM[c2][s] for s in range(1, len(SER)))
        ernn += k * sum(VRN[c2]); erns += k * sum(VM[c1][v] * VRN[c2][v] for v in range(5))
        eon += k * sum(VO[c2]); eos += k * sum(VM[c1][v] * VO[c2][v] for v in range(5))
        ed += k * sum(P[c1][a] * P[c2][a] for a in range(NS))
    rat = lambda o, on, e, en: (o / on) / (e / en) if on and en and e else 1.0
    f = {}
    f['cecho'] = rat(oc, ocn, ec_same, ec_n)
    f['vecho_RN'] = rat(ov_rn, ov_rn_n, erns, ernn)
    f['vecho_oth'] = rat(ov_o, ov_o_n, eos, eon)
    f['dbl'] = (od / npairs) / (ed / npairs) if npairs and ed else 1.0
    # phonotactic gaps among the 15 commonest signs, minus the expected share under slot independence
    top = [a for a, _ in uni.most_common(15)]
    adj = set((t[i], t[i + 1]) for t in types for i in range(len(t) - 1))
    obs_gap = sum(1 for a in top for b in top if (a, b) not in adj) / (len(top) ** 2)
    eg = 0.0
    for a in top:
        for b in top:
            lp = 0.0
            for (c1, c2), k in pairn.items():
                pp = P[c1][a] * P[c2][b]
                if pp >= 1: lp = -1e9; break
                lp += k * math.log1p(-pp)
            eg += math.exp(lp)
    f['gap_exP'] = obs_gap - eg / (len(top) ** 2)
    N = sum(uni.values())
    topu = sum(v for _, v in uni.most_common(10)) / N
    f['fin10_ex'] = sum(v for _, v in fin.most_common(10)) / n - topu
    f['init10_ex'] = sum(v for _, v in ini.most_common(10)) / n - topu
    pv = [abs(ini[a] - fin[a]) / m for a, m in uni.items() if m >= 5]
    f['pos'] = sum(pv) / len(pv) if pv else 0.0
    f['pre'] = pre / n; f['suf'] = suf / n
    f['vinit'] = vinit / n; f['vmed'] = vmed / vmed_n if vmed_n else 0.0
    for L in (2, 3, 4): f['len%d' % L] = lens[L] / n
    f['len5p'] = lens[5] / n
    vm = [0] * 5; fv = [0] * 5; sm = [0] * len(SER)
    for a, m in uni.items(): vm[a % 5] += m; sm[a // 5] += m
    for a, m in fin.items(): fv[a % 5] += m
    for i, v in enumerate(VOW): f['v_' + v] = vm[i] / N; f['fv_' + v] = fv[i] / n
    for i, s in enumerate(SER): f['s_' + (s or 'V')] = sm[i] / N
    if V2: f['kober'] = kober(types)
    return f

def kober(types):
    """type pairs (>= 3 signs) differing only in the final sign: share whose finals share the consonant series (pure
    vowels count as a series), over the share for random pairs from the same pool of alternating finals (la5 LA-5.3d).
    No pairs -> 1."""
    by = collections.defaultdict(set)
    for t in types:
        if len(t) >= 3: by[t[:-1]].add(t[-1])
    alt = [(x, y) for fs in by.values() for x in fs for y in fs if x < y]
    if len(alt) < 3: return 1.0
    o = sum(1 for x, y in alt if x // 5 == y // 5) / len(alt)
    pool = collections.Counter(z for p in alt for z in p); T = sum(pool.values())
    same = collections.Counter()
    for a, c in pool.items(): same[a // 5] += c
    e = (sum(c * c for c in same.values()) - sum(c * c for c in pool.values())) / (T * T - sum(c * c for c in pool.values()))
    return o / e if e > 0 else 1.0

def calib(types, rnd, nd=40):
    """target value (full set) and sampling SD from half-samples (m = n/2, without replacement = bootstrap SD of full n)."""
    full = fingerprint(types); acc = collections.defaultdict(list); m = len(types) // 2
    for _ in range(nd):
        for k, v in fingerprint(rnd.sample(types, m)).items(): acc[k].append(v)
    sd = {}
    for k, v in acc.items():
        mu = sum(v) / len(v); sd[k] = max((sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5, 0.004)
    return full, sd

def score(f, target, sd, wdist=0.3):
    """sum of squared z over statistics; generated and target noise both count (sqrt 2); inventory shares weighted 0.3."""
    s = 0.0
    for k in STAT_NAMES:
        z = (f[k] - target[k]) / (sd[k] * 1.4142)
        s += (1.0 if k in CORE else wdist) * z * z
    return s

# ------------------------------------------------------------------ genome
# consonant categories: written series (H written as pure vowel); place classes for OCP-place
CCAT = ['P', 'T', 'D', 'K', 'Q', 'M', 'N', 'S', 'Z', 'R', 'W', 'J', 'H']
PLACE = {'P': 'lab', 'M': 'lab', 'W': 'lab', 'T': 'cor', 'D': 'cor', 'N': 'cor', 'S': 'cor', 'Z': 'cor', 'R': 'cor',
         'K': 'dor', 'Q': 'dor', 'J': 'pal', 'H': 'lar'}
OBSTR = {'P', 'T', 'D', 'K', 'Q', 'Z'}
FRONT = {1, 2}; BACK = {3, 4}   # e i | o u ; a neutral

# rule genes: (name, lo, hi, null value)  -- the GA works on u in [0,1]
RULES = [
    ('onset0_init', 0.0, 0.7, None),   # P(word-initial syllable has no onset)
    ('onset0_med', 0.0, 0.4, 0.0),     # P(medial syllable has no onset: hiatus)
    ('cluster', 0.0, 0.5, 0.0),        # P(onset is obstruent + R/N/W cluster)
    ('coda_med', 0.0, 0.6, 0.0),       # P(non-final syllable has a coda)
    ('coda_obs', 0.0, 1.0, None),      # share of medial codas that are obstruents (written with a dead vowel)
    ('coda_fin', 0.0, 1.0, None),      # P(final coda): INVISIBLE in LB spelling (built-in unrecoverable gene)
    ('root_len', 1.0, 4.0, None),      # mean syllables per root
    ('harm_copy', -1.0 if (V2 or os.environ.get('LA10_DISSIM')) else 0.0, 1.0, 0.0),
                                       # >0: P(next vowel copies previous vowel) (total harmony / echo vowels);
                                       # <0 (cycle 2, LA10_DISSIM=1): P(next vowel must DIFFER from previous) (dissimilation)
    ('harm_fb', 0.0, 1.0, 0.0),        # P(next vowel restricted to the front/back class of the previous)
    ('ocp_id', -1.0, 1.0, 0.0),        # >0: P(reject same consonant category as previous); <0: P(copy it)
    ('ocp_place', 0.0, 1.0, 0.0),      # P(reject same place of articulation as previous consonant)
    ('pre_p', 0.0, 1.0 if V2 else 0.7, 0.0),          # P(word carries a prefix)
    ('pre_n', 1.0, 8.0, None),         # prefix inventory size
    ('suf_p', 0.0, 1.0 if V2 else 0.7, 0.0),          # P(word carries a suffix)
    ('suf_n', 1.0, 8.0, None),         # suffix inventory size
    ('aff_2syl', 0.0, 1.0, None),      # P(an affix has 2 syllables rather than 1)
    ('aff_v', 0.0, 1.0, None),         # P(a 1-syllable affix is a bare vowel)
    ('reuse', 0.0, 0.8, 0.0),          # P(a word reuses an existing root)
    ('fv_str', 0.0, 1.0, 0.0),         # strength of the final-vowel restriction
] + ([('stem_c', 0.0, 1.0, 0.0)] if V2 else [])  # v2: P(root ends in a consonant; resyllabified onto a V-initial suffix)
NUIS = [('cw_' + c, -3.0, 3.0) for c in CCAT] + [('vw_' + v, -3.0, 3.0) for v in VOW] + [('fw_' + v, -3.0, 3.0) for v in VOW]
GENES = [(n, lo, hi) for n, lo, hi, _ in RULES] + NUIS
NG = len(GENES)
GI = {g[0]: i for i, g in enumerate(GENES)}
NULL = {n: z for n, _, _, z in RULES if z is not None}

def decode(u):
    return {n: lo + (hi - lo) * min(1.0, max(0.0, x)) for (n, lo, hi), x in zip(GENES, u)}

def encode(d):
    u = []
    for n, lo, hi in GENES: u.append((d[n] - lo) / (hi - lo))
    return u

class Lang:
    def __init__(self, g, spelling='LB'):
        self.g = g; self.spelling = spelling
        cw = [math.exp(g['cw_' + c]) if g['cw_' + c] > -2.7 else 0.0 for c in CCAT]
        if sum(cw) == 0: cw[0] = 1.0
        self.cons = [c for c, w in zip(CCAT, cw) if w > 0]; self.cw = [w for w in cw if w > 0]
        vw = [math.exp(g['vw_' + v]) if g['vw_' + v] > -2.7 else 0.0 for v in VOW]
        if sum(vw) == 0: vw[0] = 1.0
        self.vw = vw
        fw = [math.exp(g['fw_' + v]) for v in VOW]
        s = g['fv_str']; tv = sum(vw); tf = sum(fw)
        self.fvw = [(1 - s) * a / tv + s * b / tf * (1.0 if a > 0 else 0.0) for a, b in zip(vw, fw)]
        if sum(self.fvw) == 0: self.fvw = vw
        self.obs = [c for c in self.cons if c in OBSTR] or self.cons
        self.obw = [w for c, w in zip(self.cons, self.cw) if c in OBSTR] or self.cw
        self.son = [c for c in self.cons if c not in OBSTR] or self.cons
        self.sow = [w for c, w in zip(self.cons, self.cw) if c not in OBSTR] or self.cw
        self.c2 = [c for c in self.cons if c in ('R', 'N', 'W')]

    # --- phonology: a syllable = (onset tuple, vowel index, coda tuple)
    def vowel(self, r, prev, final):
        w = self.fvw if final else self.vw
        if prev is not None:
            hc = self.g['harm_copy']
            if hc > 0 and r.random() < hc and w[prev] > 0: return prev
            if hc < 0 and r.random() < -hc:
                ww = [x if i != prev else 0.0 for i, x in enumerate(w)]
                if sum(ww) > 0: w = ww
            if r.random() < self.g['harm_fb'] and prev != 0:
                cl = FRONT if prev in FRONT else BACK
                ww = [x if (i in cl or i == 0) else 0.0 for i, x in enumerate(w)]
                if sum(ww) > 0: w = ww
        return r.choices(range(5), w)[0]

    def consonant(self, r, prev, pool=None, wts=None):
        pool = pool or self.cons; wts = wts or self.cw
        oi = self.g['ocp_id']
        if prev is not None:
            if oi < 0 and r.random() < -oi and prev in pool: return prev
            for _ in range(8):
                c = r.choices(pool, wts)[0]
                if oi > 0 and c == prev and r.random() < oi: continue
                if c != prev and PLACE[c] == PLACE[prev] and r.random() < self.g['ocp_place']: continue
                return c
            return c
        return r.choices(pool, wts)[0]

    def syllables(self, r, n, initial, final, state):
        out = []
        for i in range(n):
            first = initial and i == 0; last = final and i == n - 1
            on = ()
            if r.random() >= (self.g['onset0_init'] if first else self.g['onset0_med']):
                if self.c2 and r.random() < self.g['cluster']:
                    c1 = self.consonant(r, state['c'], self.obs, self.obw); c2 = r.choice(self.c2); on = (c1, c2); state['c'] = c2
                else:
                    c = self.consonant(r, state['c']); on = (c,); state['c'] = c
            v = self.vowel(r, state['v'], last); state['v'] = v
            co = ()
            if last:
                if r.random() < self.g['coda_fin']: co = (r.choices(self.son, self.sow)[0],)
            elif r.random() < self.g['coda_med']:
                if r.random() < self.g['coda_obs']: c = self.consonant(r, state['c'], self.obs, self.obw)
                else: c = self.consonant(r, state['c'], self.son, self.sow)
                co = (c,); state['c'] = c
            out.append((on, v, co))
        return out

    def affixes(self, r, k):
        out = []
        for _ in range(k):
            ns = 2 if r.random() < self.g['aff_2syl'] else 1
            st = {'c': None, 'v': None}
            if ns == 1 and r.random() < self.g['aff_v']:
                out.append([((), r.choices(range(5), self.vw)[0], ())])
            else:
                s = self.syllables(r, ns, False, False, st)
                out.append([(on if on else (r.choices(self.cons, self.cw)[0],), v, ()) for on, v, _ in s])
        return out

    def spell(self, syl):
        """Linear B spelling: onset clusters with the syllable's vowel (ko-no-so); medial obstruent codas with the NEXT
        vowel (dead vowel), medial sonorant / s codas omitted, final codas omitted, h not written."""
        out = []
        for i, (on, v, co) in enumerate(syl):
            for c in on: out.append(SIDX['' if c == 'H' else c] * 5 + v)
            if not on: out.append(v)
            if co and i < len(syl) - 1:
                c = co[0]; nv = syl[i + 1][1]
                if self.spelling == 'LB':
                    if c in OBSTR: out.append(SIDX[c] * 5 + nv)
                elif self.spelling == 'full':   # every medial coda written with the next vowel
                    out.append(SIDX['' if c == 'H' else c] * 5 + nv)
                elif self.spelling == 'prev':   # dead vowel copies the PREVIOUS vowel
                    if c in OBSTR: out.append(SIDX[c] * 5 + v)
                elif self.spelling == 'drop':   # all codas omitted
                    pass
        return tuple(out)

    def vocabulary(self, n, seed):
        r = random.Random(seed); g = self.g
        pre = self.affixes(r, int(round(g['pre_n']))); suf = self.affixes(r, int(round(g['suf_n'])))
        roots = []; out = set(); tries = 0
        while len(out) < n and tries < n * 6:
            tries += 1
            if roots and r.random() < g['reuse']: root = r.choice(roots)
            else:
                L = 1 + min(4, self._pois(r, g['root_len'] - 1)); root = None
            hp = r.random() < g['pre_p']; hs = r.random() < g['suf_p']
            if root is None:
                st = {'c': None, 'v': None}
                root = self.syllables(r, L, not hp, not hs, st)
                if V2 and r.random() < g['stem_c']:
                    on, v, co = root[-1]; root[-1] = (on, v, (self.consonant(r, st['c']),))
                roots.append(root)
            sx = r.choice(suf) if hs and suf else []
            if V2 and sx and root[-1][2] and not sx[0][0]:   # stem consonant + V-initial suffix -> CV
                on, v, co = root[-1]; sx = [((co[0],), sx[0][1], sx[0][2])] + sx[1:]; root = root[:-1] + [(on, v, ())]
            w = (r.choice(pre) if hp and pre else []) + list(root) + sx
            t = self.spell(w)
            if len(t) >= 2: out.add(t)
        return sorted(out)

    @staticmethod
    def _pois(r, lam):
        L = math.exp(-max(lam, 0.0)); k = 0; p = 1.0
        while True:
            p *= r.random()
            if p <= L: return k
            k += 1

NREP = 3
def evaluate(u, target, sd, n, seeds, spelling='LB', nrep=None):
    """Gaussian synthetic likelihood (Wood 2010): each seed -> NREP vocabularies of the SAME rule set (different lexicon
    and affix draws); per statistic z^2 = (model mean - target)^2 / (target sampling var + model between-lexicon var),
    plus log(var ratio) so a rule set cannot win by being erratic. Inventory shares weight 0.3. Mean over seeds."""
    nrep = nrep or NREP
    g = decode(u); L = Lang(g, spelling)
    sc = []
    for s in seeds:
        fs = []
        for k in range(nrep):
            v = L.vocabulary(n, s * 7 + k)
            if len(v) < n * 0.5: break
            fs.append(fingerprint(v))
        if len(fs) < nrep: sc.append(1e6); continue
        tot = 0.0
        for k in STAT_NAMES:
            xs = [f[k] for f in fs]; mu = sum(xs) / nrep; vm = sum((x - mu) ** 2 for x in xs) / (nrep - 1)
            vt = sd[k] ** 2; V = vt * (1 + 1.0 / nrep) + vm
            tot += (1.0 if k in CORE else WDIST) * ((mu - target[k]) ** 2 / V + math.log(V / vt))
        sc.append(tot)
    return sum(sc) / len(sc)
WDIST = 0.3

# ------------------------------------------------------------------ GA
def ga(target, sd, n, seed, pop=160, gens=120, ckpt=None, spelling='LB', fixed=None, log=None):
    """real-coded GA: tournament 3, blend crossover, gaussian mutation (sigma 0.08, rate 0.15), elitism 8 (elites
    re-scored each generation with fresh vocabulary seeds and averaged: noisy-fitness GA). fixed = {gene: u} clamps."""
    rnd = random.Random(seed)
    def clamp(u):
        u = [min(1.0, max(0.0, x)) for x in u]
        if fixed:
            for k, v in fixed.items(): u[GI[k]] = v
        return u
    state = None
    if ckpt and os.path.exists(ckpt):
        state = json.load(open(ckpt))
    if state:
        P = state['P']; F = state['F']; NE = state['NE']; g0 = state['gen']; nevals = state['nevals']; hist = state['hist']
        rnd.seed(seed * 1000 + g0)
    else:
        P = [clamp([rnd.random() for _ in range(NG)]) for _ in range(pop)]
        F = [evaluate(u, target, sd, n, [rnd.randrange(10 ** 9)], spelling) for u in P]; NE = [1] * pop
        g0 = 0; nevals = pop; hist = []
    for gen in range(g0, gens):
        order = sorted(range(len(P)), key=lambda i: F[i])
        el = order[:8]
        newP = [P[i] for i in el]; newF = []; newNE = []
        for i in el:  # re-score elites (running mean)
            f = evaluate(P[i], target, sd, n, [rnd.randrange(10 ** 9)], spelling); nevals += 1
            newF.append((F[i] * NE[i] + f) / (NE[i] + 1)); newNE.append(NE[i] + 1)
        def tour():
            return min(rnd.sample(range(len(P)), 3), key=lambda i: F[i])
        while len(newP) < pop:
            a = P[tour()]; b = P[tour()]
            if rnd.random() < 0.8:
                child = [x + (rnd.random() * 1.5 - 0.25) * (y - x) for x, y in zip(a, b)]
            else: child = list(a)
            child = [x + rnd.gauss(0, 0.08) if rnd.random() < 0.15 else x for x in child]
            if rnd.random() < 0.05: j = rnd.randrange(NG); child[j] = rnd.random()
            child = clamp(child)
            newP.append(child); newF.append(evaluate(child, target, sd, n, [rnd.randrange(10 ** 9)], spelling)); newNE.append(1); nevals += 1
        P, F, NE = newP, newF, newNE
        best = min(range(len(P)), key=lambda i: F[i] if NE[i] > 1 else 1e9)
        hist.append([gen, min(F), F[best], nevals])
        if log: log(f'gen {gen} best {min(F):.1f} elite {F[best]:.1f} evals {nevals}')
        if ckpt and (gen % 5 == 4 or gen == gens - 1):
            json.dump(dict(P=P, F=F, NE=NE, gen=gen + 1, nevals=nevals, hist=hist), open(ckpt + '.tmp', 'w')); os.replace(ckpt + '.tmp', ckpt)
    return P, F, NE, nevals, hist

def ablate(u, target, sd, n, reps=12, seed0=777, spelling='LB'):
    """score of the genome vs the same genome with each rule set to its null; mean over reps fixed vocab seeds
    (common random numbers)."""
    seeds = [seed0 + i for i in range(reps)]
    base = evaluate(u, target, sd, n, seeds, spelling)
    out = {'_base': base}
    def setnull(uu, names):
        uu = list(uu)
        for nm in names:
            lo, hi = GENES[GI[nm]][1:]; uu[GI[nm]] = (NULL[nm] - lo) / (hi - lo)
        return uu
    groups = {'harmony': ['harm_copy', 'harm_fb'], 'harm_copy': ['harm_copy'], 'harm_fb': ['harm_fb'],
              'ocp': ['ocp_id', 'ocp_place'], 'ocp_id': ['ocp_id'], 'ocp_place': ['ocp_place'],
              'clusters': ['cluster'], 'codas': ['coda_med'], 'hiatus': ['onset0_med'],
              'prefixing': ['pre_p'], 'suffixing': ['suf_p'], 'reuse': ['reuse'], 'final_vowel': ['fv_str']}
    for k, names in groups.items():
        out[k] = evaluate(setnull(u, names), target, sd, n, seeds, spelling) - base
    # swap prefixing and suffixing
    uu = list(u)
    for a, b in (('pre_p', 'suf_p'), ('pre_n', 'suf_n')):
        uu[GI[a]], uu[GI[b]] = u[GI[b]], u[GI[a]]
    out['swap_pre_suf'] = evaluate(uu, target, sd, n, seeds, spelling) - base
    # noise: same genome, a disjoint seed set
    out['_noise'] = evaluate(u, target, sd, n, [seed0 + 100 + i for i in range(reps)], spelling) - base
    return out
