"""LA-35: THE NUMBER BENDS THE WORD. Does a word's ending depend on the number that follows it?
Entries = word (+ optional logograms) + number. Number class: '1', '2', '3+' (counts only)."""
import json, os, sys, random
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la35_ckpt'); os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def ncls(v):
    return '1' if v == 1 else ('2' if v == 2 else '3+')


def la_entries():
    """list of dict(doc, site, word(tuple), v, cls, com). Integer part >= 1 only."""
    c = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for r in c:
        toks = [t for t in r['tokens'] if t['t'] not in ('nl', 'div')]
        sec = 'NONE'
        for i, t in enumerate(toks):
            if t['t'] == 'logo': sec = t['v'].split('+')[0]
            if t['t'] != 'word': continue
            j = i + 1; com = None
            while j < len(toks) and toks[j]['t'] == 'logo':
                com = toks[j]['v'].split('+')[0]; j += 1
            if j < len(toks) and toks[j]['t'] == 'num' and toks[j]['v'] >= 1:
                v = toks[j]['v']
                w = tuple(s for s in t['s'])
                if any(s in ('[?]', '?') for s in w): continue
                out.append(dict(doc=r['id'], site=r['id'][:2], word=w, v=v, cls=ncls(v),
                                com=com or sec, frac=bool(toks[j]['frac'])))
    return out


def lb_entries():
    """LB count entries: word, optional logograms, then a bare number (no measure unit)."""
    import la14_common as C
    rows = []
    for k, line in enumerate(open(os.path.join(D, 'damos_items.jsonl'))):
        x = json.loads(line)
        if not x.get('content'): continue
        toks = []
        for ln in x['content'].split('\n'):
            for raw in ln.split():
                if raw.startswith('.') or raw in (',', '/', '//', ':'): toks.append(('X', '')); continue
                t = C._clean_lb(raw)
                if not t or '.' in t: toks.append(('X', '')); continue
                if t.isdigit(): toks.append(('N', int(t))); continue
                if t in C.LB_UNITS: toks.append(('U', t)); continue
                if any(ch.islower() for ch in t):
                    import re
                    if not re.fullmatch(r'[a-z0-9*\-]+', t): toks.append(('X', '')); continue
                    s = tuple(p.upper() for p in t.split('-') if p)
                    toks.append(('W', s)); continue
                toks.append(('L', t.split('+')[0]))
            toks.append(('X', ''))
        head = x['heading']
        sec = 'NONE'
        for i, (ty, v) in enumerate(toks):
            if ty == 'L': sec = v
            if ty != 'W': continue
            j = i + 1; com = None
            while j < len(toks) and toks[j][0] == 'L':
                com = toks[j][1]; j += 1
            if j < len(toks) and toks[j][0] == 'N' and toks[j][1] >= 1:
                if j + 1 < len(toks) and toks[j + 1][0] == 'U': continue
                rows.append(dict(doc=head, site=head[:2], word=v, v=toks[j][1], cls=ncls(toks[j][1]),
                                 com=com or sec, frac=False))
    return rows

import numpy as np
import zlib

SCHEMES = {
    'K3': lambda v: 0 if v == 1 else (1 if v == 2 else 2),
    'K2': lambda v: 0 if v == 1 else 1,
    'RND': lambda v: 1 if (v >= 10 and v % 10 == 0) else 0,
}


def definitions(nrand=30):
    defs = []
    for n in (1, 2, 3):
        for ek in ('E1', 'E2', 'R'):
            defs.append(('pre%d_%s' % (n, ek), n, ek, None))
    for s in range(nrand):
        defs.append(('rcut%02d' % s, None, 'R', s))
    return defs


def split_word(w, n, ek, seed):
    L = len(w)
    if seed is not None:
        if L < 2: return None
        n = 1 + zlib.crc32(('%d|' % seed + '-'.join(w)).encode()) % (L - 1)
    if L <= n: return None
    stem = w[:n]
    if ek == 'E1': end = w[-1:]
    elif ek == 'E2': end = w[-2:] if L - n >= 2 else ('#',) + w[-1:]
    else: end = w[n:]
    return '-'.join(stem), '-'.join(end)


class Design:
    """Fixed (family, ending) structure for one definition; class vector varies."""
    def __init__(self, E, n, ek, seed):
        fam, end, idx = [], [], []
        for i, e in enumerate(E):
            r = split_word(e['word'], n, ek, seed)
            if r: fam.append(r[0]); end.append(r[1]); idx.append(i)
        # keep families with >= 2 distinct endings
        ends = defaultdict(set)
        for f, x in zip(fam, end): ends[f].add(x)
        keep = [k for k, f in enumerate(fam) if len(ends[f]) >= 2]
        self.idx = np.array([idx[k] for k in keep], dtype=int)
        fl = sorted({fam[k] for k in keep}); el = sorted({end[k] for k in keep})
        self.fams, self.ends = fl, el
        fi = {f: i for i, f in enumerate(fl)}; ei = {x: i for i, x in enumerate(el)}
        self.f = np.array([fi[fam[k]] for k in keep], dtype=int)
        self.e = np.array([ei[end[k]] for k in keep], dtype=int)
        self.nf, self.ne = len(fl), len(el)
        self.fe = self.f * self.ne + self.e
        # endings that occur in >= 2 families (for the cross-stem consistency stat)
        pairs = {(a, b) for a, b in zip(self.f, self.e)}
        cnt = Counter(b for a, b in pairs)
        self.multi = np.array([cnt[j] >= 2 for j in range(self.ne)], dtype=bool)

    def stats(self, cls, K):
        """cls: int array over ALL entries. Returns (G_cond, CMH_consistency, n)."""
        if len(self.idx) < 4: return 0.0, 0.0, len(self.idx)
        c = cls[self.idx]
        nf, ne = self.nf, self.ne
        O = np.zeros((nf, ne, K)); np.add.at(O, (self.f, self.e, c), 1)
        nfe = O.sum(2); nfc = O.sum(1); nfn = nfe.sum(1)
        Ex = nfe[:, :, None] * nfc[:, None, :] / np.maximum(nfn, 1)[:, None, None]
        m = O > 0
        G = 2 * float((O[m] * np.log(O[m] / Ex[m])).sum())
        # CMH-like: per (ending, class), sum over families of O-E, hypergeometric variance
        nn = nfn[:, None, None]
        V = nfe[:, :, None] * nfc[:, None, :] * (nn - nfe[:, :, None]) * (nn - nfc[:, None, :]) / np.maximum(nn ** 2 * (nn - 1), 1)
        dO = (O - Ex).sum(0); dV = V.sum(0)
        ok = (dV > 1e-9) & self.multi[:, None]
        cmh = float((dO[ok] ** 2 / dV[ok]).sum())
        return G, cmh, len(self.idx)

    def ending_table(self, cls, K):
        c = cls[self.idx]
        O = np.zeros((self.nf, self.ne, K)); np.add.at(O, (self.f, self.e, c), 1)
        nfe = O.sum(2); nfc = O.sum(1); nfn = nfe.sum(1)
        Ex = nfe[:, :, None] * nfc[:, None, :] / np.maximum(nfn, 1)[:, None, None]
        nn = nfn[:, None, None]
        V = nfe[:, :, None] * nfc[:, None, :] * (nn - nfe[:, :, None]) * (nn - nfc[:, None, :]) / np.maximum(nn ** 2 * (nn - 1), 1)
        return O.sum(0), (O - Ex).sum(0), V.sum(0)


def strata(E, key):
    g = defaultdict(list)
    for i, e in enumerate(E): g[key(e)].append(i)
    return [np.array(v) for v in g.values() if len(v) > 1]


def permute_vals(vals, groups, rng):
    out = vals.copy()
    for g in groups:
        out[g] = vals[rng.permutation(g)]
    return out
