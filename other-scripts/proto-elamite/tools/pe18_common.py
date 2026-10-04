"""pe18 'the number system is an accent': shared data for all cycles.

Tablet records with: region (SUSA / PLAT / other), fine number-system label per numeric line,
seal flag (from pe_raw.atf), non-numeral token streams (base signs, variant forms), header,
format vector, publication volume.  No sign readings from anyone are used.
"""
import json, os, re, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, C_CODES, B_CODES, FRAC_CODES  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe18_ckpt')
os.makedirs(CK, exist_ok=True)

PLATEAU = ('Tepe Yahya', 'Malyan', 'Tepe Sialk', 'Tepe Sofalin', 'Ozbaki', 'Shahr-i Sokhta')
SYSTEMS = ['C', 'C@', 'B', 'S@', 'N23', 'FRAC', 'DEC', 'SEX', 'AMB']


def region(p):
    if p.startswith('Susa'):
        return 'SUSA'
    if any(k in p for k in PLATEAU):
        return 'PLAT'
    return 'OTHER'


def site(p):
    for k in PLATEAU:
        if k in p:
            return k
    return 'Susa' if p.startswith('Susa') else (p or 'none')


def line_system(nums):
    codes = [c for _, c in nums]
    if not codes:
        return None
    bases = {c.split('@')[0] for c in codes}
    at = any('@' in c for c in codes)
    if bases & C_CODES:
        return 'C@' if at else 'C'
    if bases & B_CODES:
        return 'B'
    if at:
        return 'S@'
    if 'N23' in bases:
        return 'N23'
    if bases & FRAC_CODES:
        return 'FRAC'
    d = collections.Counter()
    for n, c in nums:
        if isinstance(n, int):
            d[c] += n
    if d.get('N14', 0) >= 6 or 'N45' in bases:
        return 'DEC'
    if 'N34' in bases or 'N48' in bases:
        return 'SEX'
    return 'AMB'


def seal_map():
    """P-number -> (sealed flag, list of seal ids) from the raw ATF."""
    out = {}
    cur = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&P'):
            cur = raw[1:8]
            out[cur] = [False, []]
            continue
        if cur and re.search(r'seal', raw, re.I) and not raw.startswith('@object'):
            if raw.startswith('&'):
                continue
            out[cur][0] = True
            out[cur][1] += re.findall(r'PES\d+', raw)
    return out


def tablets():
    T = load()
    S = seal_map()
    fin = collections.Counter()
    for t in T:
        for l in t['lines']:
            sg = [s for s in l['signs'] if is_sign(s)]
            if sg and l['numerals']:
                fin[base(sg[-1])] += 1
    CLASS = {s for s, c in fin.items() if c >= 30}
    out = []
    for t in T:
        if t['object_type'] != 'tablet':
            continue
        lines = t['lines']
        sysl = [line_system(l['numerals']) for l in lines if l['numerals']]
        sysc = collections.Counter(s for s in sysl if s)
        toks, forms, nocls = [], [], []
        ent_len, single, n_ent, finals = [], 0, 0, []
        for i, l in enumerate(lines):
            sg = [s for s in l['signs'] if is_sign(s)]
            for s in sg:
                b = base(s)
                toks.append(b)
                if b not in CLASS:
                    nocls.append(b)
                if '~' in s and not s.startswith('|'):
                    forms.append((b, s))
            if sg and l['numerals']:
                n_ent += 1
                ent_len.append(len(sg))
                single += len(sg) == 1
                finals.append(base(sg[-1]))
        hdr = None
        if lines and not lines[0]['numerals']:
            h = [base(s) for s in lines[0]['signs'] if is_sign(s)]
            hdr = h[0] if h else None
        surf = {l['surface'] for l in lines}
        cols = {(l['surface'], l['column']) for l in lines}
        off_num = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        fmt = [np.log1p(len(lines)), np.mean(ent_len) if ent_len else 0.0,
               single / n_ent if n_ent else 0.0, len(cols), float('reverse' in surf),
               float(len(off_num) >= 1), (len(sysl) / len(lines)) if lines else 0.0,
               np.mean([l['damaged'] for l in lines]) if lines else 0.0,
               float(hdr is not None)]
        vol = re.sub(r',.*', '', t['designation']).strip()
        sm = S.get(t['id'], [False, []])
        out.append({'id': t['id'], 'prov': t['provenience'], 'region': region(t['provenience']),
                    'site': site(t['provenience']), 'vol': vol, 'n_lines': len(lines),
                    'n_num': len(sysl), 'n_ent': n_ent, 'sys': dict(sysc), 'toks': toks,
                    'nocls': nocls, 'finals': finals,
                    'dom': collections.Counter(finals).most_common(1)[0][0] if finals else 'none', 'forms': forms, 'hdr': hdr, 'fmt': fmt,
                    'sealed': sm[0], 'seals': sm[1], 'cap': bool(sysc.get('C', 0) + sysc.get('C@', 0))})
    return out, CLASS


def size_bin(t):
    n = t['n_num']
    return 0 if n <= 2 else 1 if n <= 5 else 2 if n <= 10 else 3


def strata_perm(labels, strata, rng):
    lab = np.array(labels).copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        lab[idx] = lab[rng.permutation(idx)]
    return lab


import math
A = 0.5
FAM = ['toks', 'nocls', 'forms', 'hdr', 'fmt']


def items(t, f):
    if f == 'hdr':
        return [t['hdr']] if t['hdr'] else []
    if f == 'forms':
        return t['forms']
    return t[f]


class Model:
    def __init__(self, plat, susa):
        self.pc, self.sc, self.st = {}, {}, {}
        for f in ['toks', 'nocls', 'hdr']:
            self.pc[f] = collections.Counter(x for t in plat for x in items(t, f))
            self.sc[f] = collections.Counter(x for t in susa for x in items(t, f))
        # forms: conditional on base
        self.pf = collections.Counter(x for t in plat for x in t['forms'])
        self.pb = collections.Counter(b for t in plat for b, _ in t['forms'])
        self.sf = collections.Counter(x for t in susa for x in t['forms'])
        self.sb = collections.Counter(b for t in susa for b, _ in t['forms'])
        self.nforms = collections.Counter(b for (b, _) in set(self.sf) | set(self.pf))
        F = np.array([t['fmt'] for t in susa]); P = np.array([t['fmt'] for t in plat])
        mu, sd = F.mean(0), F.std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Fz, Pz = (F - mu) / sd, (P - mu) / sd
        self.fm, self.fs = Fz.mean(0), Fz.std(0) + 0.1
        self.pm, self.ps = Pz.mean(0), Pz.std(0) + 0.1
        self.V = {f: len(set(self.pc[f]) | set(self.sc[f])) + 1 for f in self.pc}

    def score(self, t, f, loo=False):
        if f == 'fmt':
            z = (np.array(t['fmt']) - self.mu) / self.sd
            lp = -0.5 * (((z - self.pm) / self.ps) ** 2) - np.log(self.ps)
            ls = -0.5 * (((z - self.fm) / self.fs) ** 2) - np.log(self.fs)
            return float((lp - ls).mean())
        xs = items(t, f)
        if not xs:
            return np.nan
        if f == 'forms':
            own = collections.Counter(xs) if loo else collections.Counter()
            ownb = collections.Counter(b for b, _ in xs) if loo else collections.Counter()
            v = []
            for b, s in xs:
                k = self.nforms[b] + 1
                pp = (self.pf[(b, s)] + A) / (self.pb[b] + A * k)
                ps = (self.sf[(b, s)] - own[(b, s)] + A) / (self.sb[b] - ownb[b] + A * k)
                v.append(math.log(pp / ps))
            return float(np.mean(v))
        pc, sc, V = self.pc[f], self.sc[f], self.V[f]
        Np, Ns = sum(pc.values()), sum(sc.values())
        own = collections.Counter(xs) if loo else collections.Counter()
        n_own = len(xs) if loo else 0
        v = [math.log((pc[x] + A) / (Np + A * V)) - math.log((sc[x] - own[x] + A) / (Ns - n_own + A * V)) for x in xs]
        return float(np.mean(v))


def scores(model, tabs, loo):
    S = np.array([[model.score(t, f, loo) for f in FAM] for t in tabs])
    return S


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def combine(S, ref):
    """z-score each family on Susa reference rows; average available."""
    mu = np.nanmean(ref, 0); sd = np.nanstd(ref, 0) + 1e-9
    Z = (S - mu) / sd
    return np.nanmean(Z, 1)



def contrast(score, lab, elig, strata, nperm=5000, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    sc = score[elig]; lb = lab[elig]; st = strata[elig]
    ok = ~np.isnan(sc); sc, lb, st = sc[ok], lb[ok], st[ok]
    if lb.sum() < 5 or (~lb).sum() < 5:
        return None
    obs = sc[lb].mean() - sc[~lb].mean()
    d = obs / (sc.std() + 1e-9)
    nul = np.array([(lambda L: sc[L].mean() - sc[~L].mean())(strata_perm(lb, st, rng)) for _ in range(nperm)])
    z = (obs - nul.mean()) / (nul.std() + 1e-12)
    p = float((1 + (np.abs(nul - nul.mean()) >= abs(obs - nul.mean())).sum()) / (nperm + 1))
    return {'n1': int(lb.sum()), 'n0': int((~lb).sum()), 'diff': round(float(obs), 4), 'd': round(float(d), 3),
            'z': round(float(z), 2), 'p': round(p, 4)}



def label_sets(rows):
    L = {}
    dec = np.array(['DEC' in t['sys'] and 'SEX' not in t['sys'] for t in rows])
    sex = np.array(['SEX' in t['sys'] and 'DEC' not in t['sys'] for t in rows])
    L['DEC_vs_SEX'] = (dec, dec | sex)
    for s in ['C@', 'B', 'S@', 'N23', 'FRAC', 'C', 'SEX']:
        has = np.array([s in t['sys'] for t in rows])
        L[s + '_vs_rest'] = (has, np.array([t['n_num'] > 0 for t in rows]))
    return L


