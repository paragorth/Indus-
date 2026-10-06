"""pe59 generative grammar of whole tablets + simpler models, all scored the same way.

Sign stream (per clean line, left to right, with end-of-line token):
  S0 unigram | S1 Witten-Bell bigram | S2 S1 + tablet cache (name pool) | S3 S2 + learned line-initial slot
  G  S3 + reading components: class signs by tablet system (MEASURED/COUNTED given the tablet-type posterior),
     allotment line after a PERSON line, outpost lexicon at non-Susa sites, header roles.
Numerals (each clean entry): P(class) * P(notation | class)
  N0 marginal | N1 | final sign | N2 N1 + tablet cache (number system follows the tablet)
  G  N2 + tablet-type latent via the reading's roles (class), log-normal tablet scale shifted by the pe52
     weights (value under the tablet system), M288 rate rule, fixed-allotment copy.
Totals: N2 vs G (= sum of entries in the tablet system).
Every probability on training tablets is leave-one-tablet-out (own counts subtracted).
"""
import math
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize

from pe59_lib import (ncls, nkey, unkey, value, canon, line_role, header_role, tablet_type, CLS, LROLES, HROLES,
                      AMB)

EOS = '</l>'
BOS = '<l>'
LOG2 = math.log(2)


def sig(x):
    return 1 / (1 + math.exp(-x))


class Grammar:
    def __init__(self, train, roles, maps, corpus='PE'):
        self.train = train
        self.roles = roles
        self.capmap, self.cntmaps = maps
        self.corpus = corpus
        self.cnt_main = 'sex2' if corpus == 'PE' else 'S'
        self._count()

    # ----------------------------------------------------------- counting
    def _own(self, t):
        o = {'uni': Counter(), 'bi': Counter(), 'ctx': Counter(), 'first': Counter(),
             'cls': Counter(), 'fscls': Counter(), 'fs': Counter(), 'key': Counter(), 'fskey': Counter(),
             'sign_tau': Counter(), 'hrole': None, 'tau': tablet_type(t), 'lrole_tau': Counter(),
             'clsrole': Counter(), 'logv': [], 'tsigns': set(), 'xl': Counter(), 'xlc': Counter(), 'site': Counter()}
        nonsusa = self.corpus == 'PE' and not t['site'].startswith('Susa')
        prevfs = None
        for l in t['lines']:
            if l['role'] == 'E':
                if l['signs'] and l['clean'] and prevfs is not None:
                    o['xl'][(prevfs, l['signs'][0])] += 1; o['xlc'][prevfs] += 1
                prevfs = l['signs'][-1] if l['signs'] else 'BARE'
            if nonsusa and l['clean']:
                o['site'].update(l['signs'])
            if l['signs'] and l['clean'] and l['role'] in 'HETN':
                seq = [BOS] + l['signs'] + [EOS]
                for a, b in zip(seq, seq[1:]):
                    o['bi'][(a, b)] += 1
                    o['ctx'][a] += 1
                    o['uni'][b] += 1
                o['first'][(l['role'], l['signs'][0])] += 1
            o['tsigns'] |= set(l['signs'])
            if l['role'] == 'E' and l['numclean']:
                fs = l['signs'][-1] if l['signs'] else 'BARE'
                c = ncls(l['nums']); k = nkey(l['nums'])
                o['cls'][c] += 1; o['fscls'][(fs, c)] += 1; o['fs'][fs] += 1
                o['key'][(c, k)] += 1; o['fskey'][(fs, k)] += 1
        return o

    def _count(self):
        self.own = {}
        self.uni = Counter(); self.bi = Counter(); self.ctx = Counter(); self.first = Counter()
        self.cls = Counter(); self.fscls = Counter(); self.fs = Counter(); self.key = Counter(); self.fskey = Counter()
        self.xl = Counter(); self.xlc = Counter(); self.site = Counter()
        for t in self.train:
            o = self._own(t)
            self.own[t['id']] = o
            for nm in ('uni', 'bi', 'ctx', 'first', 'cls', 'fscls', 'fs', 'key', 'fskey', 'xl', 'xlc', 'site'):
                getattr(self, nm).update(o[nm])
        self.T = Counter(a for (a, b) in self.bi)
        self.vocab = set(self.uni)
        self.Nuni = sum(self.uni.values())
        self.Ncls = sum(self.cls.values())
        self.firstN = Counter()
        for (r, s), n in self.first.items():
            self.firstN[r] += n
        self.clskeyN = Counter()
        for (c, k), n in self.key.items():
            self.clskeyN[c] += n
        self.inv = defaultdict(set)
        for (c, k) in self.key:
            for p in k.split('|'):
                self.inv[c].add(p.split(':')[0])
        self._fit_tau()
        self._value_tables()
        self._role_sign_tables()

    # ----------------------------------------------------------- tablet type classifier (reading roles)
    def tau_features(self, t, upto=None, with_nums=True):
        """Evidence before line index `upto` (None = whole tablet). Returns (hrole, Counter lroles, Counter ncls, site)."""
        h = 'NONE'
        for l in t['lines']:
            if l['role'] == 'H' and l['signs']:
                h = header_role(l['signs'][0], self.roles); break
        lr = Counter(); nc = Counter()
        for i, l in enumerate(t['lines']):
            if upto is not None and i >= upto:
                break
            if l['role'] == 'E':
                lr[line_role(l['signs'], self.roles)] += 1
                if with_nums and l['numclean']:
                    nc[ncls(l['nums'])] += 1
        site = 'Susa' if (t['site'].startswith('Susa') or self.corpus != 'PE') else 'other'
        return h, lr, nc, site

    def _fit_tau(self):
        H = {tau: Counter() for tau in ('CAPT', 'CNTT')}
        LR = {tau: Counter() for tau in ('CAPT', 'CNTT')}
        NC = {tau: Counter() for tau in ('CAPT', 'CNTT')}
        ST = {tau: Counter() for tau in ('CAPT', 'CNTT')}
        n = Counter()
        CR = defaultdict(Counter)  # (tau, lrole) -> cls
        feats = []
        for t in self.train:
            tau = tablet_type(t)
            if tau is None:
                continue
            h, lr, nc, site = self.tau_features(t)
            H[tau][h] += 1; LR[tau].update(lr); NC[tau].update(nc); ST[tau][site] += 1; n[tau] += 1
            for l in t['lines']:
                if l['role'] == 'E' and l['numclean']:
                    CR[(tau, line_role(l['signs'], self.roles))][ncls(l['nums'])] += 1
            feats.append((t['id'], tau, h, lr, site))
        self.tauH, self.tauLR, self.tauNC, self.tauST, self.taun, self.tauCR = H, LR, NC, ST, n, CR
        self.tau_feats = feats
        self.theta = [1.0, 1.0]

        def nll(th):
            s = 0
            for tid, tau, h, lr, site in feats:
                p = self.tau_post_raw(h, lr, Counter(), site, th, own=(tau, h, lr, site))
                s -= math.log(max(1e-12, p if tau == 'CAPT' else 1 - p))
            return s
        best = None
        for a in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            v = nll([a, 1.0])
            if best is None or v < best[0]:
                best = (v, a)
        self.theta = [best[1], 1.0]

    def tau_post_raw(self, h, lr, nc, site, th, own=None):
        """P(CAPT | evidence). `own` = (tau, h, lr, site) of a training tablet to subtract (leave-one-out)."""
        lo = 0.0
        nC, nN = self.taun['CAPT'], self.taun['CNTT']
        if own:
            tau0 = own[0]
            nC -= tau0 == 'CAPT'; nN -= tau0 == 'CNTT'
        lo += math.log((nC + 1) / (nN + 1))

        def ratio(tabC, tabN, x, k, sub):
            cC = tabC[x] - (sub if own and own[0] == 'CAPT' else 0)
            cN = tabN[x] - (sub if own and own[0] == 'CNTT' else 0)
            tC = sum(tabC.values()) - (k if own and own[0] == 'CAPT' else 0)
            tN = sum(tabN.values()) - (k if own and own[0] == 'CNTT' else 0)
            return math.log((cC + 0.5) / (tC + 0.5 * 12)) - math.log((cN + 0.5) / (tN + 0.5 * 12))
        lo += ratio(self.tauH['CAPT'], self.tauH['CNTT'], h, 1, 1)
        lo += ratio(self.tauST['CAPT'], self.tauST['CNTT'], site, 1, 1)
        if own:
            olr = own[2]; ok = sum(olr.values())
        for r, c in lr.items():
            sub = own[2][r] if own else 0
            lo += th[0] * c * ratio(self.tauLR['CAPT'], self.tauLR['CNTT'], r, ok if own else 0, sub)
        for r, c in nc.items():
            if r == 'CAP':
                return 1.0
            lo += th[1] * c * (math.log((self.tauNC['CAPT'][r] + 0.5) / (sum(self.tauNC['CAPT'].values()) + 3))
                               - math.log((self.tauNC['CNTT'][r] + 0.5) / (sum(self.tauNC['CNTT'].values()) + 3)))
        return sig(max(-30, min(30, lo)))

    def tau_post(self, t, upto=None, with_nums=True, loto=False):
        h, lr, nc, site = self.tau_features(t, upto, with_nums)
        own = None
        if loto:
            tau = tablet_type(t)
            if tau is not None:
                h0, lr0, _, s0 = self.tau_features(t)
                own = (tau, h0, lr0, s0)
        return self.tau_post_raw(h, lr, nc, site, self.theta, own)

    def p_cls_tau(self, c, tau, lrole):
        tab = self.tauCR[(tau, lrole)]
        allc = Counter()
        for (tt, r), cc in self.tauCR.items():
            if tt == tau:
                allc.update(cc)
        p0 = (allc[c] + 0.5) / (sum(allc.values()) + 3)
        return (tab[c] + 5 * p0) / (sum(tab.values()) + 5)

    # ----------------------------------------------------------- values
    def logv(self, k, tau):
        nums = unkey(k)
        c = ncls(nums)
        if c == 'CAP':
            v = value(nums, self.capmap)
        elif c == 'AMB':
            v = value(nums, self.capmap if tau == 'CAPT' else self.cntmaps[self.cnt_main])
        elif c in ('FRAC', 'BIS'):
            v = value(nums, self.cntmaps[self.cnt_main])
        else:
            v = None
        return math.log(v) if v else None

    def _value_tables(self):
        self.V = {}
        for c in ('CAP', 'AMB', 'FRAC', 'BIS'):
            keys = [k for (cc, k) in self.key if cc == c and self.logv(k, 'CAPT') is not None and self.logv(k, 'CNTT') is not None]
            idx = {k: i for i, k in enumerate(keys)}
            L = {tau: np.array([self.logv(k, tau) for k in keys]) for tau in ('CAPT', 'CNTT')}
            C = np.array([self.key[(c, k)] for k in keys], dtype=float)
            self.V[c] = (keys, idx, L, C)
        # global mean log value per (cls, lrole, tau)
        acc = defaultdict(list)
        for t in self.train:
            tau = tablet_type(t) or 'CNTT'
            for l in t['lines']:
                if l['role'] == 'E' and l['numclean']:
                    c = ncls(l['nums']); k = nkey(l['nums'])
                    lv = self.logv(k, tau)
                    if lv is not None:
                        acc[(c, line_role(l['signs'], self.roles), tau)].append(lv - self.wsum(l['signs']))
                        acc[(c, '*', tau)].append(lv - self.wsum(l['signs']))
        self.mu = {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= 3}
        self.capval = {}
        for c in ('CAP', 'AMB'):
            for kk in self.V[c][0]:
                self.capval[kk] = value(unkey(kk), self.capmap)

    def wsum(self, signs):
        return sum(self.roles['weights'].get(s, 0.0) for s in set(signs))

    def _role_sign_tables(self):
        """Within-role sign frequencies (train) for the class component and header roles."""
        S = self.roles['sets']
        self.role_final = {}
        for r in ('MEASURED', 'COUNTED'):
            cnt = Counter()
            for t in self.train:
                for l in t['lines']:
                    if l['role'] == 'E' and l['signs'] and l['clean']:
                        for s in l['signs']:
                            if s in S[r]:
                                cnt[s] += 1
            tot = sum(cnt.values())
            self.role_final[r] = {s: (cnt[s] + 0.5) / (tot + 0.5 * len(S[r])) for s in S[r]} if S[r] else {}
        self.out_set = sorted(S['OUT_Yahya'])

    # ----------------------------------------------------------- sign probabilities
    def p_uni(self, w, o):
        c = self.uni[w] - (o['uni'][w] if o else 0)
        N = self.Nuni - (sum(o['uni'].values()) if o else 0)
        return (c + 0.5) / (N + 0.5 * (len(self.vocab) + 1))

    def p_bi(self, p, w, o):
        cp = self.ctx[p] - (o['ctx'][p] if o else 0)
        pu = self.p_uni(w, o)
        if cp <= 0:
            return pu
        cpw = self.bi[(p, w)] - (o['bi'][(p, w)] if o else 0)
        Tp = self.T[p]
        if o:
            Tp -= sum(1 for (a, b), n in o['bi'].items() if a == p and self.bi[(a, b)] == n)
        Tp = max(Tp, 1)
        return (cpw + Tp * pu) / (cp + Tp)

    def p_first(self, role, w, o):
        r = role if role in ('H', 'E') else 'E'
        c = self.first[(r, w)] - (o['first'][(r, w)] if o else 0)
        N = self.firstN[r] - (sum(n for (rr, s), n in o['first'].items() if rr == r) if o else 0)
        return (c + 0.1 * self.p_uni(w, o) * 50) / (N + 5)

    def sign_rows(self, t, loto):
        """One row per scored sign token with the component probabilities that do not depend on mixture weights."""
        o = self.own.get(t['id']) if loto else None
        rows = []
        cache = Counter()
        lines = t['lines']
        prev_role = None
        prevfs = None
        nonsusa = self.corpus == 'PE' and not t['site'].startswith('Susa')
        Nsite = sum(self.site.values()) - (sum(o['site'].values()) if o else 0)
        for i, l in enumerate(lines):
            if l['signs'] and l['clean'] and l['role'] in 'HETN':
                ptau = self.tau_post(t, upto=i, loto=loto)
                seq = l['signs'] + [EOS]
                p = BOS
                for j, w in enumerate(seq):
                    n = sum(cache.values())
                    r = {'s1': self.p_bi(p, w, o), 'uni': self.p_uni(w, o), 'nc': n,
                         'pc': (cache[w] / n) if (n and w != EOS) else 0.0,
                         'pos0': j == 0, 'pf': self.p_first(l['role'], w, o) if (j == 0 and w != EOS) else 0.0,
                         'E': l['role'] == 'E', 'H': l['role'] == 'H'}
                    pu = r['uni']
                    r['a_xl'] = j == 0 and l['role'] == 'E' and prevfs is not None
                    r['pxl'] = 0.0
                    if r['a_xl'] and w != EOS:
                        cx = self.xl[(prevfs, w)] - (o['xl'][(prevfs, w)] if o else 0)
                        nx = self.xlc[prevfs] - (o['xlc'][prevfs] if o else 0)
                        r['pxl'] = (cx + 2 * pu) / (nx + 2)
                    r['a_site'] = nonsusa
                    r['psite'] = 0.0
                    if nonsusa and w != EOS:
                        cs = self.site[w] - (o['site'][w] if o else 0)
                        r['psite'] = (cs + 0.5) / (Nsite + 0.5 * (len(self.vocab) + 1))
                    # reading components
                    pcls = 0.0
                    if l['role'] == 'E' and w != EOS:
                        pM = self.role_final['MEASURED'].get(w, 0.0)
                        pC = self.role_final['COUNTED'].get(w, 0.0)
                        pcls = ptau * pM + (1 - ptau) * pC
                    r['pcls'] = pcls
                    r['pallot'] = 1.0 if (j == 0 and l['role'] == 'E' and prev_role == 'PERSON'
                                          and self.roles['allot_sign'] and w == self.roles['allot_sign']) else 0.0
                    r['a_allot'] = j == 0 and l['role'] == 'E' and prev_role == 'PERSON' and bool(self.roles['allot_sign'])
                    r['a_out'] = (not t['site'].startswith('Susa')) and self.corpus == 'PE' and bool(self.out_set)
                    r['pout'] = (1.0 / len(self.out_set)) if (r['a_out'] and w in self.out_set) else 0.0
                    r['phdr'] = 0.0
                    if j == 0 and l['role'] == 'H' and w != EOS:
                        hr = header_role(w, self.roles)
                        r['phdr'] = 0.0 if hr == 'H_X' else self._p_hdr(w, hr)
                    rows.append(r)
                    if w != EOS:
                        cache[w] += 1
                    p = w
                cache_ok = True
            else:
                for s in l['signs']:
                    cache[s] += 1
            if l['role'] == 'E':
                prev_role = line_role(l['signs'], self.roles)
                prevfs = l['signs'][-1] if l['signs'] else 'BARE'
        return rows

    def _p_hdr(self, w, hr):
        """P(header role) * P(sign | header role): a proper (sub-normalised) distribution over header signs."""
        if not hasattr(self, '_hdr'):
            self._hdr = defaultdict(Counter)
            for t in self.train:
                for l in t['lines']:
                    if l['role'] == 'H' and l['signs'] and l['clean']:
                        h = header_role(l['signs'][0], self.roles)
                        self._hdr[h][l['signs'][0]] += 1
            self._hdrN = sum(sum(c.values()) for c in self._hdr.values())
        c = self._hdr[hr]
        ph = (sum(c.values()) + 0.5) / (self._hdrN + 2.5)
        return ph * (c[w] + 0.5) / (sum(c.values()) + 0.5 * max(1, len(c)))

    @staticmethod
    def sign_arrays(rows):
        keys = ['s1', 'uni', 'nc', 'pc', 'pf', 'pcls', 'pallot', 'pout', 'phdr', 'pxl', 'psite']
        A = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
        for k in ('pos0', 'E', 'H', 'a_allot', 'a_out', 'a_xl', 'a_site'):
            A[k] = np.array([r[k] for r in rows], dtype=bool)
        return A

    NSP = 10

    @staticmethod
    def sign_mix(A, par, level):
        """par = [kappa, beta, m0H, m0E, mX, mS, mu_cls, mu_allot, mu_out, mu_hdr].
        S3 = data-only extras (line-role-aware first slot, cross-line bigram, site lexicon); G adds the reading's roles."""
        if level == 'S0':
            return A['uni']
        if level == 'S1':
            return A['s1']
        k = par[0] * A['nc'] / (A['nc'] + par[1])
        p = (1 - k) * A['s1'] + k * A['pc']
        if level == 'S2':
            return p
        mH = par[2] * (A['pos0'] & A['H']); mE = par[3] * (A['pos0'] & A['E'])
        mX = par[4] * A['a_xl']; mS = par[5] * A['a_site']
        p = (1 - mH - mE - mX - mS) * p + mH * A['pf'] + mE * A['pf'] + mX * A['pxl'] + mS * A['psite']
        if level == 'S3':
            return p
        mc = par[6] * A['E']
        ma = par[7] * A['a_allot']
        mo = par[8] * A['a_out']
        mh = par[9] * (A['pos0'] & A['H'])
        rest = 1 - mc - ma - mo - mh
        return rest * p + mc * A['pcls'] + ma * A['pallot'] + mo * A['pout'] + mh * A['phdr']

    def fit_signs(self):
        rows = []
        for t in self.train:
            rows += self.sign_rows(t, loto=True)
        A = self.sign_arrays(rows)
        self._sA = A

        def tr(x):
            return [sig(x[0]), math.exp(x[1])] + [sig(v) for v in x[2:]]

        def f(x, level):
            return -np.log(np.maximum(self.sign_mix(A, tr(x), level), 1e-300)).sum()
        x = np.array([-1.0, 1.0, -3, -3, -3, -3, -9, -9, -9, -9])
        r = minimize(lambda z: f(np.r_[z, x[2:]], 'S2'), x[:2], method='Nelder-Mead', options={'maxiter': 400})
        x[:2] = r.x
        r = minimize(lambda z: f(np.r_[z, x[6:]], 'S3'), x[:6], method='Nelder-Mead', options={'maxiter': 1500})
        x[:6] = r.x
        self.sign_par_S3 = tr(x)
        xg = x.copy(); xg[6:] = -4
        r = minimize(lambda z: f(np.r_[x[:6], z], 'G'), xg[6:], method='Nelder-Mead', options={'maxiter': 1500})
        xg[6:] = r.x
        r = minimize(lambda z: f(z, 'G'), xg, method='Nelder-Mead', options={'maxiter': 2000})
        xg = r.x
        self.sign_par_G = tr(xg)
        self.sign_train_bits = {lv: f(x if lv != 'G' else xg, lv) / LOG2 / len(rows) for lv in ('S0', 'S1', 'S2', 'S3', 'G')}
        return self.sign_train_bits

    def score_signs(self, tabs):
        rows = []
        per = []
        for t in tabs:
            r = self.sign_rows(t, loto=False)
            per.append((t['id'], len(r)))
            rows += r
        A = self.sign_arrays(rows)
        out = {}
        for lv in ('S0', 'S1', 'S2', 'S3', 'G'):
            par = self.sign_par_G if lv == 'G' else self.sign_par_S3
            out[lv] = -np.log2(np.maximum(self.sign_mix(A, par, lv), 1e-300))
        return out, per

    # ----------------------------------------------------------- numeral probabilities
    def p_cls0(self, c, o):
        return (self.cls[c] - (o['cls'][c] if o else 0) + 1) / (self.Ncls - (sum(o['cls'].values()) if o else 0) + 6)

    def p_cls1(self, c, fs, o, a1):
        cf = self.fscls[(fs, c)] - (o['fscls'][(fs, c)] if o else 0)
        nf = self.fs[fs] - (o['fs'][fs] if o else 0)
        return (cf + a1 * self.p_cls0(c, o)) / (nf + a1)

    def p_new(self, c, k):
        inv = max(1, len(self.inv.get(c, ())) or 6)
        parts = k.split('|')
        p = 0.5 ** len(parts) * math.factorial(len(parts))
        for q in parts:
            code, n = q.split(':')
            try:
                n = int(Fr(n))
            except Exception:
                n = 1
            p *= (1.0 / inv) * 0.5 ** max(1, n)
        return p

    def p_key0(self, c, k, o):
        ck = self.key[(c, k)] - (o['key'][(c, k)] if o else 0)
        nc = self.clskeyN[c] - (sum(n for (cc, kk), n in o['key'].items() if cc == c) if o else 0)
        return (ck + 2.0 * self.p_new(c, k)) / (nc + 2.0)

    def p_key1(self, c, k, fs, o, a2):
        cfk = self.fskey[(fs, k)] - (o['fskey'][(fs, k)] if o else 0)
        cfc = self.fscls[(fs, c)] - (o['fscls'][(fs, c)] if o else 0)
        return (cfk + a2 * self.p_key0(c, k, o)) / (cfc + a2)

    def rate_cands(self, k_units):
        out = set()
        for mult in (60, 120):
            v = Fr(mult * k_units)
            ck = canon(v, {c: x for c, x in self.capmap.items() if c not in ('N45', 'N34', 'N48')})
            if ck:
                out.add(nkey(ck))
            for kk, vv in self.capval.items():
                if vv == v:
                    out.add(kk)
        return out

    def num_rows(self, t, loto, total=False):
        """Rows for each clean entry (or the total line) with everything needed by the numeral models."""
        o = self.own.get(t['id']) if loto else None
        rows = []
        tc = Counter(); tk = Counter()
        prev = []  # (cls, key, logv by tau, lrole, signs)
        last_by_fs = {}
        prev_line = None
        lines = t['lines']
        ents_all_clean = all(l['numclean'] for l in lines if l['role'] == 'E')
        for i, l in enumerate(lines):
            if l['role'] not in ('E', 'T'):
                continue
            want = (l['role'] == 'E' and not total) or (l['role'] == 'T' and total)
            if want and l['numclean']:
                fs = l['signs'][-1] if l['signs'] else 'BARE'
                c = ncls(l['nums']); k = nkey(l['nums'])
                lrole = line_role(l['signs'], self.roles)
                ptau = self.tau_post(t, upto=i, loto=loto)
                r = {'id': t['id'], 'line': i, 'fs': fs, 'cls': c, 'key': k, 'lrole': lrole, 'ptau': ptau,
                     'tc': dict(tc), 'nt': sum(tc.values()), 'tk': tk[k], 'ntc': tc[c],
                     'p0c': self.p_cls0(c, o), 'p0k': self.p_key0(c, k, o)}
                r['p1c'] = {a: self.p_cls1(c, fs, o, a) for a in (0.5, 2, 8, 32)}
                r['p1k'] = {a: self.p_key1(c, k, fs, o, a) for a in (0.5, 2, 8, 32, 128)}
                r['ptc'] = ptau * self.p_cls_tau(c, 'CAPT', lrole) + (1 - ptau) * self.p_cls_tau(c, 'CNTT', lrole)
                # cache-backoff needs P1(cls) and P1(key) for all classes of the cache: store normalisers implicitly
                # LN component inputs
                w = self.wsum(l['signs'])
                ln = None
                if c in self.V and k in self.V[c][1]:
                    keys, idx, L, C = self.V[c]
                    Cadj = C.copy()
                    if o:
                        for (cc, kk), n in o['key'].items():
                            if cc == c and kk in idx:
                                Cadj[idx[kk]] -= n
                    if Cadj[idx[k]] > 0:
                        ms = {}
                        for tau in ('CAPT', 'CNTT'):
                            pv = [pp[2][tau] - pp[5] for pp in prev if pp[0] == c and pp[2][tau] is not None]
                            if pv:
                                ms[tau] = (float(np.mean(pv)) + w, True)
                            else:
                                mu = self.mu.get((c, lrole, tau), self.mu.get((c, '*', tau), 0.0))
                                ms[tau] = (mu + w, False)
                        ln = (c, idx[k], Cadj, ms)
                r['ln'] = ln
                # rate rule (allotment sign after a count line)
                r['rate'] = None
                if self.roles['allot_sign'] and fs == self.roles['allot_sign'] and prev_line is not None:
                    pl = prev_line
                    if pl['numclean'] and ncls(pl['nums']) == 'AMB' and pl['signs'] and pl['signs'][-1] != fs:
                        ku = value(pl['nums'], self.cntmaps[self.cnt_main])
                        if ku and ku <= 60:
                            cands = self.rate_cands(ku)
                            wts = {kk: (self.key[(ncls(unkey(kk)), kk)] + 1) for kk in cands}
                            Z = sum(wts.values())
                            r['rate'] = (wts.get(k, 0) / Z) if Z else 0.0
                # fixed-allotment copy
                r['copy'] = None
                if fs != 'BARE' and fs in last_by_fs:
                    r['copy'] = 1.0 if last_by_fs[fs] == k else 0.0
                # totals: sum component
                r['sum'] = None
                if total and ents_all_clean:
                    r['sum'] = self.sum_prob(t, k, ptau=self.tau_post(t, upto=i, loto=loto))
                # cache P1 for all cache classes/keys (to normalise the cache-backoff exactly)
                r['cache_c'] = {}
                rows.append(r)
            if l['role'] == 'E' and l['numclean']:
                c = ncls(l['nums']); k = nkey(l['nums'])
                tc[c] += 1; tk[k] += 1
                prev.append((c, k, {tau: self.logv(k, tau) for tau in ('CAPT', 'CNTT')},
                             line_role(l['signs'], self.roles), l['signs'], self.wsum(l['signs'])))
                if l['signs']:
                    last_by_fs[l['signs'][-1]] = k
            if l['role'] == 'E':
                prev_line = l
        return rows

    def sum_prob(self, t, k, ptau):
        ents = [l for l in t['lines'] if l['role'] == 'E' and l['numclean']]
        if len(ents) < 2:
            return None
        out = 0.0
        for tau, pt in (('CAPT', ptau), ('CNTT', 1 - ptau)):
            maps = [self.capmap] if tau == 'CAPT' else list(self.cntmaps.values())
            cands = set()
            for m in maps:
                vals = [value(l['nums'], m) for l in ents]
                if any(v is None for v in vals):
                    continue
                S = sum(vals)
                if S <= 0:
                    continue
                ck = canon(S, m)
                if ck:
                    cands.add(nkey(ck))
                if tau == 'CAPT':
                    ck2 = canon(S, {c: v for c, v in m.items() if c in AMB})
                    if ck2:
                        cands.add(nkey(ck2))
            if cands:
                out += pt * (1.0 / len(cands) if k in cands else 0.0)
        return out

    def closes(self, t, tau):
        """Does the written total equal the entry sum under the reading's maps for tablet type tau?"""
        ents = [l for l in t['lines'] if l['role'] == 'E' and l['numclean']]
        tot = [l for l in t['lines'] if l['role'] == 'T' and l['numclean']]
        if len(ents) < 2 or not tot or not all(l['numclean'] for l in t['lines'] if l['role'] == 'E'):
            return None
        maps = [self.capmap] if tau == 'CAPT' else list(self.cntmaps.values())
        for m in maps:
            vals = [value(l['nums'], m) for l in ents]
            T = value(tot[0]['nums'], m)
            if T is None or any(v is None for v in vals):
                continue
            if sum(vals) == T:
                return True
        return False

    # ----------------------------------------------------------- numeral mixture
    @staticmethod
    def ln_prob(ln, ptau, s_anchor, s_free, V):
        c, ti, Cadj, ms = ln
        keys, idx, L, C = V[c]
        p = 0.0
        for tau, pt in (('CAPT', ptau), ('CNTT', 1 - ptau)):
            if pt <= 1e-9:
                continue
            m, anch = ms[tau]
            s = s_anchor if anch else s_free
            z = np.exp(-0.5 * ((L[tau] - m) / s) ** 2) * Cadj
            Z = z.sum()
            if Z > 0:
                p += pt * z[ti] / Z
        return p

    def num_prob(self, r, level, par):
        """Return (bits_cls, bits_key)."""
        a1, a2, b1, b2 = par['a1'], par['a2'], par['b1'], par['b2']
        p1c = r['p1c'][a1]
        if level == 'N0':
            pc, pk = r['p0c'], r['p0k']
            return -math.log2(pc), -math.log2(pk)
        p1k = r['p1k'][a2]
        if level == 'N1':
            return -math.log2(p1c), -math.log2(p1k)
        lam = par.get('lam', 1.0) if level == 'G' else 1.0
        q = lam * p1c + (1 - lam) * r['ptc']
        pc = (r['tc'].get(r['cls'], 0) + b1 * q) / (r['nt'] + b1)
        pk2 = (r['tk'] + b2 * p1k) / (r['ntc'] + b2)
        if level == 'N2':
            return -math.log2(pc), -math.log2(pk2)
        comps = []
        if r['ln'] is not None or r['cls'] in self.V:
            comps.append(('ln', self.ln_prob(r['ln'], r['ptau'], par['sa'], par['sf'], self.V) if r['ln'] else 0.0))
        if r['rate'] is not None:
            comps.append(('rate', r['rate']))
        if r['copy'] is not None:
            comps.append(('copy', r['copy']))
        if r.get('sum') is not None:
            comps.append(('sum', r['sum']))
        tot = sum(par['pi_' + n] for n, _ in comps)
        pk = (1 - tot) * pk2 + sum(par['pi_' + n] * p for n, p in comps)
        return -math.log2(pc), -math.log2(max(pk, 1e-300))

    def fit_nums(self, total=False):
        rows = []
        for t in self.train:
            rows += self.num_rows(t, loto=True, total=total)
        par = {'a1': 2, 'a2': 2, 'b1': 2.0, 'b2': 2.0, 'lam': 1.0, 'sa': 0.7, 'sf': 1.2,
               'pi_ln': 0.0, 'pi_rate': 0.0, 'pi_copy': 0.0, 'pi_sum': 0.0}

        def tot(level, P):
            s = 0.0
            for r in rows:
                a, b = self.num_prob(r, level, P)
                s += a + b
            return s
        for a in (0.5, 2, 8, 32):
            for a2 in (0.5, 2, 8, 32, 128):
                P = dict(par, a1=a, a2=a2)
                v = tot('N1', P)
                if a == 0.5 and a2 == 0.5 or v < best[0]:
                    best = (v, a, a2)
        par['a1'], par['a2'] = best[1], best[2]
        best = None
        for b1 in (0.5, 1, 2, 4, 8, 16):
            for b2 in (0.25, 0.5, 1, 2, 4, 8, 16, 32):
                P = dict(par, b1=b1, b2=b2)
                v = tot('N2', P)
                if best is None or v < best[0]:
                    best = (v, b1, b2)
        par['b1'], par['b2'] = best[1], best[2]
        self.num_par_N2 = dict(par)
        # G: coordinate ascent on lam, sa, sf, pi_*
        grids = {'lam': [1.0, 0.9, 0.75, 0.5, 0.25, 0.0], 'sa': [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0], 'sf': [0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 4.5],
                 'pi_ln': [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8], 'pi_rate': [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7],
                 'pi_copy': [0, 0.05, 0.1, 0.2, 0.35, 0.5], 'pi_sum': [0, 0.1, 0.25, 0.5, 0.75, 0.9]}
        if not total:
            grids.pop('pi_sum')
        G = dict(par)
        cur = tot('G', G)
        for sweep in range(3):
            for nm, gr in grids.items():
                for v in gr:
                    P = dict(G); P[nm] = v
                    if sum(P[k] for k in P if k.startswith('pi_')) >= 0.95:
                        continue
                    s = tot('G', P)
                    if s < cur - 1e-9:
                        cur, G = s, P
        self.num_par_G = G
        bits = {lv: tot(lv, self.num_par_G if lv == 'G' else self.num_par_N2) / max(1, len(rows))
                for lv in ('N0', 'N1', 'N2', 'G')}
        return bits, len(rows)

    def score_nums(self, tabs, total=False, parG=None):
        rows = []
        for t in tabs:
            rows += self.num_rows(t, loto=False, total=total)
        out = {}
        for lv in ('N0', 'N1', 'N2', 'G'):
            P = (parG or self.num_par_G) if lv == 'G' else self.num_par_N2
            out[lv] = np.array([self.num_prob(r, lv, P) for r in rows])
        return out, rows
