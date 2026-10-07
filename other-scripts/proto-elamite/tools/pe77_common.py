"""pe77: simulate forgers.  Thousands of random tablet forgers of differing power are fitted on one third of
the tablets (A); detector feature models are fitted on a second third (B); real held-out tablets (C) are
compared with the forgeries.  A feature that even strong forgers (forgeries the full detector can hardly
tell from real tablets) fail to reproduce is 'unforgeable'.

Tablet format (both corpora): {'id', 'lines': [(surf, signs tuple, nums tuple of (int, code))]}
surf: 'o' obverse, 'r' anything else.
"""
import os, sys, json, math, random, collections, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe2_common

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe77_ckpt')
os.makedirs(CKPT, exist_ok=True)

# ---------------------------------------------------------------- value maps (as pe42_common, floats)
PE_MAPS = [
    {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000, 'N51': 120},          # decimal reading
    {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N51': 120},           # sexagesimal
    {'N39C': 1 / 120, 'N30D': 1 / 60, 'N30C': 1 / 30, 'N24': 1 / 10, 'N39B': 1 / 5, 'N01': 1, 'N14': 6,
     'N45': 60, 'N34': 180, 'N48': 1800},                                             # capacity
]
PC_MAPS = [
    {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N50': 36000},
    {'N01': 1, 'N14': 10, 'N34': 60, 'N51': 120, 'N48': 7200},
    {'N01': 1, 'N14': 6, 'N45': 60, 'N34': 180, 'N48': 1800},
    {'N39A': 1 / 5, 'N39B': 1 / 5, 'N24': 1, 'N01': 5, 'N14': 30, 'N45': 300, 'N34': 1800},  # rough SZE
]

# A-priori calibration labels for proto-cuneiform (fixed before any PC run; conventional readings)
PC_WORLD = {'SZE', 'UDU', 'U8', 'UDUNITA', 'MASZ', 'AB2', 'GU4', 'SZAH2', 'KU6', 'SUHUR', 'GAR', 'KASZ',
            'DUG', 'GA', 'I3', 'SIG2', 'TUG2', 'KU3', 'GAN2', 'ZIZ2', 'U2', 'MUSZEN', 'SAL', 'KUR', 'SAG',
            'ERIM', 'NINDA', 'GU7', 'SZE3', 'KISZ', 'GURUSZ', 'BA'}
PC_CONV = {'EN', 'SANGA', 'NUN', 'GAL', 'NAM2', 'AN', 'E2', 'PAP', 'UNUG', 'URU', 'KI', 'DUB', 'U4', 'ME',
           'PA', 'SUKKAL', 'KISAL', 'MUSZ3', 'SZU', 'NE', 'DU', 'A', 'BU', 'SI', 'TUR', 'IB', 'NI', 'DA',
           'RAD', 'APIN', 'SZITA', 'HI', 'TE', 'AB', 'BAR'}


def _norm_lines(raw_lines, signfilter):
    out = []
    for l in raw_lines:
        sg = tuple(common.base(s) for s in l['signs'] if signfilter(s))
        nums = tuple((int(n), common.norm_code(c)) for n, c in l['numerals'] if isinstance(n, int))
        if not sg and not nums:
            continue
        out.append(('o' if l['surface'] == 'obverse' else 'r', sg, nums))
    return out


def load_corpus(name):
    if name == 'PE':
        T = common.load()
        out = [{'id': t['id'], 'lines': _norm_lines(t['lines'], common.is_sign)} for t in T]
        maps = PE_MAPS
    elif name == 'PC':
        T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
        out = [{'id': t['id'], 'lines': _norm_lines(t['lines'], lambda s: s not in ('x', 'X') and not s.startswith('x'))}
               for t in T]
        maps = PC_MAPS
    else:
        raise ValueError(name)
    out = [t for t in out if len(t['lines']) >= 2 and any(l[2] for l in t['lines'])]
    return out, maps


# ---------------------------------------------------------------- planted world
def planted_world(seed=0, n_tab=1000, hidden=True):
    """Synthetic archive.  WORLD: 200 owners with a hidden herd size; an owner's entry carries his name
    (2-3 name signs) + class sign and a count ~ Poisson(herd * class factor); a closing total is the true sum.
    CONVENTION: header sign sets the number of entries; class signs are drawn by entry position.
    hidden=False: counts drawn without the owner's herd (no world tie between name and number)."""
    rng = random.Random(seed)
    syl = ['S%02d' % i for i in range(40)]
    cls = ['K%d' % i for i in range(6)]
    hdr = ['H%d' % i for i in range(5)]
    owners = []
    for o in range(200):
        nm = tuple(rng.choice(syl) for _ in range(rng.choice([2, 2, 3])))
        herd = math.exp(rng.gauss(2.5, 1.0))
        owners.append((nm, herd))
    cf = [1.0, 0.5, 2.0, 0.3, 1.5, 0.8]
    T = []
    for i in range(n_tab):
        h = rng.randrange(5)
        n_e = max(1, int(rng.gauss(2 + 2 * h, 1)))
        lines = [('o', (hdr[h],), ())]
        tot = 0
        for j in range(n_e):
            o = owners[rng.randrange(200)] if rng.random() < 0.8 else owners[rng.randrange(20)]
            k = min(5, j // 2 + (rng.random() < 0.3))
            lam = (o[1] if hidden else math.exp(rng.gauss(2.5, 1.0))) * cf[k]
            v = max(1, np.random.default_rng(rng.randrange(1 << 30)).poisson(lam))
            tot += v
            lines.append(('o', o[0] + (cls[k],), to_nums(v)))
        if rng.random() < 0.6:
            lines.append(('r', ('TOT',), to_nums(tot)))
        T.append({'id': 'W%04d' % i, 'lines': lines})
    return T, [{'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}]


def to_nums(v, m=None):
    m = m or [('N48', 3600), ('N45', 600), ('N34', 60), ('N14', 10), ('N01', 1)]
    out = []
    for c, u in m:
        k = int(v // u)
        if k:
            out.append((k, c))
            v -= k * u
    return tuple(out) if out else ((1, 'N01'),)


# ---------------------------------------------------------------- numbers
def value(nums, maps):
    if not nums:
        return None
    for m in maps:
        if all(c in m for _, c in nums):
            return sum(n * m[c] for n, c in nums)
    return None


def values_all(nums, maps):
    out = []
    for m in maps:
        if nums and all(c in m for _, c in nums):
            out.append(round(sum(n * m[c] for n, c in nums), 6))
        else:
            out.append(None)
    return out


def sysof(nums):
    return pe2_common.system_of([list(x) for x in nums]) if nums else None


def vbin(v):
    if v is None or v <= 0:
        return -1
    return min(9, max(0, int(math.log2(v) + 3)))


# ---------------------------------------------------------------- structure helpers
def split_tablet(t):
    """header signs (first line, signs and no numbers), entries [(signs, nums, idx)], numeric lines idx."""
    L = t['lines']
    hdr = L[0][1] if L and L[0][1] and not L[0][2] else ()
    ents = [(l[1], l[2], i) for i, l in enumerate(L) if l[1] and l[2]]
    return hdr, ents


# ---------------------------------------------------------------- feature models (fit on B)
class FeatModel:
    def __init__(self, B, maps, K=40, seed=0):
        self.maps = maps
        tabfreq = collections.Counter()
        sys_by = collections.defaultdict(collections.Counter)
        bin_by = collections.defaultdict(collections.Counter)
        str_by = collections.defaultdict(list)
        sys_all = collections.Counter()
        bin_all = collections.Counter()
        cooc = collections.Counter()
        strings = collections.Counter()
        ntab = len(B)
        for t in B:
            hdr, ents = split_tablet(t)
            ss = set(s for l in t['lines'] for s in l[1])
            tabfreq.update(ss)
            ssl = sorted(ss)
            for a in range(len(ssl)):
                for b in range(a + 1, len(ssl)):
                    cooc[(ssl[a], ssl[b])] += 1
            for sg, nums, _ in ents:
                sy = sysof(nums)
                vb = vbin(value(nums, maps))
                sys_all[sy] += 1
                bin_all[vb] += 1
                for s in set(sg):
                    sys_by[s][sy] += 1
                    bin_by[s][vb] += 1
                key = sg[:-1] if len(sg) > 1 else sg
                v = value(nums, maps)
                if v and v > 0:
                    str_by[key].append(math.log(v))
                strings[sg] += 1
        self.top = [s for s, _ in tabfreq.most_common(K)]
        self.tabfreq, self.ntab, self.cooc = tabfreq, ntab, cooc
        self.sys_by, self.bin_by, self.sys_all, self.bin_all = sys_by, bin_by, sys_all, bin_all
        self.str_mu = {k: (np.mean(v), len(v)) for k, v in str_by.items() if len(v) >= 2}
        allv = [x for v in str_by.values() for x in v]
        self.gmu, self.gsd = (np.mean(allv), np.std(allv) + 1e-6) if allv else (0, 1)
        self.strings = strings
        self.names = self._names()

    def _names(self):
        g = ['ST_nlines', 'ST_nent', 'ST_hdr', 'ST_rev', 'ST_elen', 'ST_len1', 'ST_numless',
             'NUM_round', 'NUM_one', 'NUM_rep', 'NUM_logmax', 'NUM_nsys', 'NUM_mono', 'NUM_five',
             'BIND_all', 'MAG_all', 'MAGSTR', 'AR_tot', 'AR_last', 'AR_block', 'COH_pmi', 'REP_str', 'NOV_str']
        for s in self.top:
            g += ['P_' + s, 'I_' + s, 'F_' + s, 'H_' + s, 'BIND_' + s, 'MAG_' + s]
        return g

    def _lp(self, ctr, key, tot_ctr, alpha=0.5):
        n = sum(ctr.values())
        k = max(1, len(tot_ctr))
        p0 = (tot_ctr.get(key, 0) + 1) / (sum(tot_ctr.values()) + k)
        return math.log((ctr.get(key, 0) + alpha * k * p0) / (n + alpha * k))

    def feats(self, t):
        maps = self.maps
        L = t['lines']
        hdr, ents = split_tablet(t)
        nl = len(L)
        f = {}
        f['ST_nlines'] = math.log(nl)
        f['ST_nent'] = math.log1p(len(ents))
        f['ST_hdr'] = 1.0 if hdr else 0.0
        f['ST_rev'] = sum(1 for l in L if l[0] == 'r') / nl
        f['ST_elen'] = np.mean([len(e[0]) for e in ents]) if ents else np.nan
        f['ST_len1'] = np.mean([len(e[0]) == 1 for e in ents]) if ents else np.nan
        f['ST_numless'] = sum(1 for l in L if not l[2]) / nl
        nums = [e[1] for e in ents]
        if nums:
            vals = [value(n, maps) for n in nums]
            f['NUM_round'] = np.mean([len(n) == 1 for n in nums])
            f['NUM_one'] = np.mean([n == ((1, 'N01'),) for n in nums])
            cn = collections.Counter(nums)
            f['NUM_rep'] = np.mean([cn[n] > 1 for n in nums])
            vv = [v for v in vals if v]
            f['NUM_logmax'] = math.log(max(vv)) if vv else np.nan
            f['NUM_nsys'] = len(set(sysof(n) for n in nums))
            if len(vv) >= 3:
                r = np.corrcoef(np.arange(len(vv)), np.argsort(np.argsort(vv)))[0, 1]
                f['NUM_mono'] = 0.0 if np.isnan(r) else r
            else:
                f['NUM_mono'] = np.nan
            u = [sum(k for k, c in n if c == 'N01') for n in nums]
            f['NUM_five'] = np.mean([x % 5 == 0 for x in u])
            bl = [self._lp(self.sys_by.get(e[0][-1], {}), sysof(e[1]), self.sys_all) for e in ents]
            f['BIND_all'] = np.mean(bl)
            ml = [self._lp(self.bin_by.get(e[0][-1], {}), vbin(value(e[1], maps)), self.bin_all) for e in ents]
            f['MAG_all'] = np.mean(ml)
            ms = []
            for sg, n, _ in ents:
                key = sg[:-1] if len(sg) > 1 else sg
                v = value(n, maps)
                if key in self.str_mu and v and v > 0:
                    ms.append(-((math.log(v) - self.str_mu[key][0]) / self.gsd) ** 2)
            f['MAGSTR'] = np.mean(ms) if ms else np.nan
        else:
            for k in ['NUM_round', 'NUM_one', 'NUM_rep', 'NUM_logmax', 'NUM_nsys', 'NUM_mono', 'NUM_five',
                      'BIND_all', 'MAG_all', 'MAGSTR']:
                f[k] = np.nan
        # arithmetic: totals under any value map
        numl = [(i, l) for i, l in enumerate(L) if l[2]]
        tot = last = block = 0.0
        if len(numl) >= 3:
            V = [values_all(l[2], maps) for _, l in numl]
            for mi in range(len(maps)):
                col = [v[mi] for v in V]
                for j in range(len(col)):
                    if col[j] is None:
                        continue
                    others = [c for k, c in enumerate(col) if k != j]
                    if all(c is not None for c in others) and abs(sum(others) - col[j]) < 1e-6:
                        tot = 1.0
                        if j == len(col) - 1:
                            last = 1.0
                    # contiguous preceding block of >= 2 lines
                    s = 0.0
                    for k in range(j - 1, -1, -1):
                        if col[k] is None:
                            break
                        s += col[k]
                        if j - k >= 2 and abs(s - col[j]) < 1e-6:
                            block = 1.0
                            break
            f['AR_tot'], f['AR_last'], f['AR_block'] = tot, last, block
        else:
            f['AR_tot'] = f['AR_last'] = f['AR_block'] = np.nan
        ss = sorted(set(s for l in L for s in l[1]))
        pm = []
        N = self.ntab
        for a in range(len(ss)):
            fa = self.tabfreq.get(ss[a], 0)
            if fa < 3:
                continue
            for b in range(a + 1, len(ss)):
                fb = self.tabfreq.get(ss[b], 0)
                if fb < 3:
                    continue
                c = self.cooc.get((ss[a], ss[b]), 0)
                pm.append(math.log((c + 0.5) * N / (fa * fb)))
        f['COH_pmi'] = np.mean(pm) if pm else np.nan
        strs = [e[0] for e in ents]
        cs = collections.Counter(strs)
        f['REP_str'] = np.mean([cs[s] > 1 for s in strs]) if strs else np.nan
        ls = [s for s in strs if len(s) >= 2]
        f['NOV_str'] = np.mean([self.strings.get(s, 0) == 0 for s in ls]) if ls else np.nan
        sset = set(ss)
        inits = set(e[0][0] for e in ents)
        fins = set(e[0][-1] for e in ents)
        hs = set(hdr)
        for s in self.top:
            f['P_' + s] = 1.0 if s in sset else 0.0
            f['I_' + s] = 1.0 if s in inits else 0.0
            f['F_' + s] = 1.0 if s in fins else 0.0
            f['H_' + s] = 1.0 if s in hs else 0.0
            es = [e for e in ents if s in e[0]]
            if es:
                f['BIND_' + s] = np.mean([self._lp(self.sys_by.get(s, {}), sysof(e[1]), self.sys_all) for e in es])
                f['MAG_' + s] = np.mean([self._lp(self.bin_by.get(s, {}), vbin(value(e[1], maps)), self.bin_all)
                                         for e in es])
            else:
                f['BIND_' + s] = f['MAG_' + s] = np.nan
        return np.array([f[k] for k in self.names], float)


def family(name):
    if name.startswith('ST_'):
        return 'STRUCT'
    if name.startswith('NUM_'):
        return 'NUMFORM'
    if name.startswith('AR_'):
        return 'ARITH'
    if name in ('BIND_all', 'MAG_all', 'MAGSTR') or name.startswith('BIND_') or name.startswith('MAG_'):
        return 'BIND'
    if name.startswith('COH') or name.startswith('REP') or name.startswith('NOV'):
        return 'TABLET'
    return {'P': 'PRES', 'I': 'SLOT', 'F': 'SLOT', 'H': 'HDR'}[name.split('_')[0]]


# ---------------------------------------------------------------- forgers
class Pools:
    def __init__(self, A):
        self.A = A
        self.ents = []           # (signs, nums)
        self.hdrs = []
        self.sig_uni = collections.Counter()
        self.nlines = []
        self.elen = []
        self.nums_by_fin = collections.defaultdict(list)
        self.nums_by_sys = collections.defaultdict(list)
        self.nums = []
        self.big = collections.defaultdict(collections.Counter)    # Markov counts order 1 and 2
        self.tri = collections.defaultdict(collections.Counter)
        self.rev_share = []
        self.num_only = []
        for t in A:
            hdr, ents = split_tablet(t)
            if hdr:
                self.hdrs.append(hdr)
            self.nlines.append(len(ents))
            self.rev_share.append(sum(1 for l in t['lines'] if l[0] == 'r') / len(t['lines']))
            for l in t['lines']:
                self.sig_uni.update(l[1])
                if l[2] and not l[1]:
                    self.num_only.append(l[2])
            for sg, n, _ in ents:
                self.ents.append((sg, n))
                self.elen.append(len(sg))
                self.nums_by_fin[sg[-1]].append(n)
                self.nums_by_sys[sysof(n)].append(n)
                self.nums.append(n)
                seq = ('<',) + sg + ('>',)
                for k in range(1, len(seq)):
                    self.big[seq[k - 1]][seq[k]] += 1
                    if k >= 2:
                        self.tri[(seq[k - 2], seq[k - 1])][seq[k]] += 1
        self.has_hdr = len(self.hdrs) / max(1, len(A))
        self.sigs = list(self.sig_uni)
        self.sw = np.array([self.sig_uni[s] for s in self.sigs], float)
        self.sw /= self.sw.sum()
        self._cum = {k: (list(v), np.cumsum(list(v.values()))) for k, v in self.big.items()}
        self._cum3 = {k: (list(v), np.cumsum(list(v.values()))) for k, v in self.tri.items()}
        self.has_total = []

    def draw(self, rng, cum):
        keys, c = cum
        return keys[int(np.searchsorted(c, rng.random() * c[-1], side='right'))]

    def markov(self, rng, order):
        seq = ['<']
        for _ in range(12):
            if order == 2 and len(seq) >= 2 and (seq[-2], seq[-1]) in self._cum3 and rng.random() < 0.9:
                nx = self.draw(rng, self._cum3[(seq[-2], seq[-1])])
            elif order >= 1:
                nx = self.draw(rng, self._cum[seq[-1]])
            else:
                nx = self.sigs[int(np.searchsorted(np.cumsum(self.sw), rng.random()))] if len(seq) < 1 + rng.choice(self.elen) else '>'
            if nx == '>':
                break
            seq.append(nx)
        if len(seq) == 1:
            seq.append(self.sigs[int(np.searchsorted(np.cumsum(self.sw), rng.random()))])
        return tuple(seq[1:])

    def num_for(self, rng, sg, cond):
        if cond and rng.random() < cond and self.nums_by_fin.get(sg[-1]):
            return rng.choice(self.nums_by_fin[sg[-1]])
        return rng.choice(self.nums)


def random_forger(rng):
    """Sample a forger specification."""
    if rng.random() < 0.3:
        return {'base': 'scratch', 'order': rng.choice([0, 1, 2]), 'strings': rng.random() < 0.4,
                'cond': rng.random(), 'total': rng.random() < 0.5}
    ops = rng.sample(['entry_swap', 'num_swap', 'num_within', 'sign_swap', 'string_swap', 'header_swap',
                      'drop_dup', 'num_jitter'], rng.choice([1, 1, 2, 3]))
    return {'base': 'edit', 'ops': {o: math.exp(rng.uniform(math.log(0.05), 0)) for o in ops},
            'cond': rng.random(), 'fix_total': rng.random() < 0.3}


def _fix_totals(t, maps):
    """If a numeric line was the sum of the others (any map) in the source, recompute it after editing."""
    return t


def forge(spec, P, rng, n, maps):
    out = []
    A = P.A
    for _ in range(n):
        if spec['base'] == 'scratch':
            m = rng.choice(P.nlines) or 1
            lines = []
            if rng.random() < P.has_hdr and P.hdrs:
                lines.append(('o', rng.choice(P.hdrs), ()))
            tot = []
            nrev = int(round(rng.choice(P.rev_share) * m))
            for j in range(m):
                sg = rng.choice(P.ents)[0] if spec['strings'] else P.markov(rng, spec['order'])
                nm = P.num_for(rng, sg, spec['cond'])
                tot.append(value(nm, maps))
                lines.append(('o' if j < m - nrev else 'r', sg, nm))
            if spec['total'] and all(v is not None for v in tot) and len(tot) >= 2 and rng.random() < 0.5:
                v = sum(tot)
                if abs(v - round(v)) < 1e-9:
                    lines.append(('r', (), to_nums(int(round(v)))))
            elif P.num_only and rng.random() < 0.3:
                lines.append(('r', (), rng.choice(P.num_only)))
            if len(lines) < 2:
                lines.append(('o', P.markov(rng, 1), rng.choice(P.nums)))
            out.append({'id': 'F', 'lines': lines})
            continue
        src = rng.choice(A)
        L = list(src['lines'])
        # remember total structure of source
        numl = [i for i, l in enumerate(L) if l[2]]
        tot_idx = None
        if spec.get('fix_total') and len(numl) >= 3:
            V = [value(L[i][2], maps) for i in numl]
            if all(v is not None for v in V) and abs(sum(V[:-1]) - V[-1]) < 1e-6:
                tot_idx = numl[-1]
        ops = spec['ops']
        cond = spec['cond']
        ent_idx = [i for i, l in enumerate(L) if l[1] and l[2] and i != tot_idx]
        for op, p in ops.items():
            if op == 'entry_swap':
                for i in ent_idx:
                    if rng.random() < p:
                        sg, nm = rng.choice(P.ents)
                        L[i] = (L[i][0], sg, nm)
            elif op == 'num_swap':
                for i in ent_idx:
                    if rng.random() < p:
                        if rng.random() < cond:
                            pool = P.nums_by_sys.get(sysof(L[i][2])) or P.nums
                        else:
                            pool = P.nums
                        L[i] = (L[i][0], L[i][1], rng.choice(pool))
            elif op == 'num_within':
                if rng.random() < p and len(ent_idx) >= 2:
                    nm = [L[i][2] for i in ent_idx]
                    rng.shuffle(nm)
                    for i, x in zip(ent_idx, nm):
                        L[i] = (L[i][0], L[i][1], x)
            elif op == 'sign_swap':
                cs = np.cumsum(P.sw)
                for i in range(len(L)):
                    if L[i][1]:
                        sg = tuple(P.sigs[int(np.searchsorted(cs, rng.random()))] if rng.random() < p else s
                                   for s in L[i][1])
                        L[i] = (L[i][0], sg, L[i][2])
            elif op == 'string_swap':
                for i in ent_idx:
                    if rng.random() < p:
                        L[i] = (L[i][0], rng.choice(P.ents)[0], L[i][2])
            elif op == 'header_swap':
                if L and L[0][1] and not L[0][2] and rng.random() < p and P.hdrs:
                    L[0] = ('o', rng.choice(P.hdrs), ())
            elif op == 'drop_dup':
                new = []
                for i, l in enumerate(L):
                    r = rng.random()
                    if i in ent_idx and r < p / 2 and len(L) > 2:
                        continue
                    new.append(l)
                    if i in ent_idx and r > 1 - p / 2:
                        sg, nm = rng.choice(P.ents)
                        new.append((l[0], sg, nm))
                if tot_idx is not None:
                    tot_line = L[tot_idx]
                    new = [l for l in new if l is not tot_line] + [tot_line]
                    tot_idx = len(new) - 1
                L = new
                ent_idx = [i for i, l in enumerate(L) if l[1] and l[2] and i != tot_idx]
            elif op == 'num_jitter':
                for i in ent_idx:
                    if rng.random() < p:
                        nm = list(L[i][2])
                        k = rng.randrange(len(nm))
                        nm[k] = (max(1, nm[k][0] + rng.choice([-1, 1])), nm[k][1])
                        L[i] = (L[i][0], L[i][1], tuple(nm))
        if tot_idx is not None:
            numl = [i for i, l in enumerate(L) if l[2] and i != tot_idx]
            V = [value(L[i][2], maps) for i in numl]
            if all(v is not None for v in V) and V:
                v = sum(V)
                if abs(v - round(v)) < 1e-9 and v >= 1:
                    L[tot_idx] = (L[tot_idx][0], L[tot_idx][1], to_nums(int(round(v))))
        if len(L) < 2:
            L.append(('o', rng.choice(P.ents)[0], rng.choice(P.nums)))
        out.append({'id': 'F', 'lines': L})
    return out


# ---------------------------------------------------------------- detector
def auc(x, y):
    """AUC of score x for label y (1 real, 0 forged); NaN-aware (rows with NaN dropped)."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 < 8 or n0 < 8:
        return np.nan
    from scipy.stats import rankdata
    ranks = rankdata(x)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def full_detector(X, y, rng_seed=0):
    """2-fold logistic regression on imputed, standardised features; AUC."""
    from sklearn.linear_model import LogisticRegression
    X = X.copy()
    mu = np.nanmean(X, 0)
    mu = np.where(np.isnan(mu), 0, mu)
    nanmask = np.isnan(X)
    X[nanmask] = np.take(mu, np.where(nanmask)[1])
    X = np.hstack([X, nanmask.astype(float)])
    sd = X.std(0)
    keep = sd > 1e-9
    X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    r = np.random.default_rng(rng_seed)
    idx = r.permutation(len(y))
    half = len(y) // 2
    sc = np.zeros(len(y))
    for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        m = LogisticRegression(C=0.1, max_iter=300)
        m.fit(X[a], y[a])
        sc[b] = m.decision_function(X[b])
    return auc(sc, y.astype(float))


def run_forger(T, maps, spec, seed, shuffle_labels=False, K=40):
    """One forger: 3-way split, fit pools on A, feature model on B, compare C with forgeries."""
    rng = random.Random(seed)
    idx = list(range(len(T)))
    rng.shuffle(idx)
    n3 = len(T) // 3
    A = [T[i] for i in idx[:n3]]
    B = [T[i] for i in idx[n3:2 * n3]]
    C = [T[i] for i in idx[2 * n3:]]
    P = Pools(A)
    FM = FeatModel(B, maps, K=K)
    F = forge(spec, P, rng, len(C), maps)
    XR = np.array([FM.feats(t) for t in C])
    XF = np.array([FM.feats(t) for t in F])
    X = np.vstack([XR, XF])
    y = np.r_[np.ones(len(C)), np.zeros(len(F))]
    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(y)
    A_all = full_detector(X, y, seed)
    d = np.array([auc(X[:, j], y) for j in range(X.shape[1])])
    ok = ~np.isnan(X)
    n1 = (ok & (y[:, None] == 1)).sum(0).astype(float)
    n0 = (ok & (y[:, None] == 0)).sum(0).astype(float)
    se = np.sqrt((n1 + n0 + 1) / (12 * np.maximum(n1, 1) * np.maximum(n0, 1)))
    z = (d - 0.5) / se
    return FM.names, A_all, d, z


def sha_list(obj):
    s = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode()).hexdigest()


def plant_magstr(T, frac=1.0, seed=0, jitter=0.3, names_only=True):
    """PE-shaped plant for MAGSTR power: for a share `frac` of entry keys (signs minus the final sign) that occur
    in 2-20 entries and have >= 2 signs (name-like), every occurrence gets the same numeral group (one of its own, chosen at random), with
    probability `jitter` moved by one unit.  Everything else (strings, systems, layout, repetition rate) is real."""
    rng = random.Random(seed)
    occ = collections.defaultdict(list)
    for ti, t in enumerate(T):
        for li, l in enumerate(t['lines']):
            if l[1] and l[2]:
                key = l[1][:-1] if len(l[1]) > 1 else l[1]
                occ[key].append((ti, li))
    T2 = [{'id': t['id'], 'lines': list(t['lines'])} for t in T]
    nk = 0
    for key, oc in occ.items():
        if len(oc) < 2 or len(oc) > 20 or (names_only and len(key) < 2) or rng.random() >= frac:
            continue
        nk += 1
        ti, li = rng.choice(oc)
        ref = T[ti]['lines'][li][2]
        for ti, li in oc:
            nm = list(ref)
            if rng.random() < jitter:
                k = rng.randrange(len(nm))
                nm[k] = (max(1, nm[k][0] + rng.choice([-1, 1])), nm[k][1])
            l = T2[ti]['lines'][li]
            T2[ti]['lines'][li] = (l[0], l[1], tuple(nm))
    return T2, nk
