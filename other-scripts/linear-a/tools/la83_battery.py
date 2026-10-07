#!/usr/bin/env python3
"""la83 step 3: fixed battery of established Linear A results (A and B grades), run on any corpus_ra-style
token list (v1 = corpus_ra.json, v2 = corpus_ra_v2.json, or a perturbed copy), always on version 'rd'.

Definitions are those of FINDINGS.md and the la79 re-grade (tools/la79_claims.py, la79_c1.eval_split with
train = test = all documents, i.e. la79's 'full' row) unless stated:
 B1 KU-RO = total (FINDINGS 2a; la71_totals rule): exact closures among testable KU-RO sections vs totals
    shuffled across sections.  pass: P < 0.01.  sign: exact > null mean.
 B2a fraction writing order (FINDINGS 'Writing order', A): pair instances in compound fractions read in the
    fixed order JE L E J Y A F H B K L6 L2 L4; reversals vs letters shuffled within each compound.
    pass: share in order >= 0.9 and P(null reversals <= obs) < 0.01.  sign: share > null share.
 B2b fraction order learned (la79 frac_order, Copeland, 2,000 random orders): pass pct >= 0.95.  sign: agree > 0.5.
 B3 affixing, aggregate (la79 pre_all / suf_all, A): pass z >= 2 on both sides.  sign: both z > 0.
 B4 -ME (la79 suf_ME, B): pass z >= 2 and pct >= 0.9.  sign: z > 0.
 B5 commodity order (la79 comm_order, B): pass pct >= 0.95.  sign: agree > 0.5.
 B6 sites as separate administrations (la58 B; descriptive form of la58's 'every cross-site pair shares
    less than random mixing'): mean Jaccard of word types (2+ signs) over pairs of the six site groups,
    real vs 200 shuffles of site labels over documents.  pass: real below 95 % of shuffles.  sign: real < null mean.
 B7 Linear B shared words (la79 lb_shared, A; extra): pass z >= 2.  sign: z > 0.
 Marginal control M1 (not a result): I- prefix (la79 pre_I, z 1.69, graded C-), pass z >= 1.645; it sits on
    its threshold, so it must flip under any real perturbation (shows the perturbations have bite).
"""
import json, os, sys, math, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la71_parse as P
import la71_totals as TT
import la79_claims as K
from la15_common import _logo

SITE2G = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}
FIXED_FR = ['JE', 'L', 'E', 'J', 'Y', 'A', 'F', 'H', 'B', 'K', 'L6', 'L2', 'L4']
FR_ITEMS = ['JE', 'L', 'E', 'J', 'Y', 'A', 'F', 'H', 'B', 'K', 'L6', 'L2', 'L4', 'D', 'W', 'X', 'L3', 'DD']
_LEX = None


def norm(s): return s.replace('₂', '2').replace('₃', '3')


def docs79(C, status=('read', 'damaged')):
    """la79_common.load on an in-memory corpus_ra-style list (no publication year needed)."""
    docs = []
    for d in C:
        if d.get('superseded_by'): continue
        toks = [t for t in d['tokens'] if t['t'] not in ('nl', 'div') and t.get('st') in status]
        words, logos, fracs = [], [], []; cur = None
        for i, t in enumerate(toks):
            if t['t'] == 'word':
                syl = tuple(norm(s) for s in t['s'] if not _logo(s))
                if syl:
                    nxt = toks[i + 1]['t'] if i + 1 < len(toks) else 'end'
                    words.append(dict(s=syl, read=t['st'] == 'read', nxt=nxt))
            elif t['t'] == 'logo':
                v = t.get('v', '').split('+')[0].rstrip('[') if t.get('v') else ''
                logos.append(v); cur = v
            elif t['t'] == 'num':
                fr = t.get('frac') or []
                if fr: fracs.append((tuple(fr), cur))
        docs.append(dict(id=d['id'], site=d['site'] or '?', g=SITE2G.get(d['site'], 'OTH'), words=words,
                         logos=logos, fracs=fracs))
    return docs


def b1_kuro(Crd, nnull=20000, seed=71):
    rng = random.Random(seed)
    ku = [s for s in TT.sections(Crd, 'KU-RO') if s['entries'] and s['tot'] and not s['bad']]
    sums = [sum(e['v'] for e in s['entries']) for s in ku]; tots = [s['tot']['v'] for s in ku]
    obs = sum(a == b for a, b in zip(sums, tots)); null = []
    for _ in range(nnull):
        p = tots[:]; rng.shuffle(p); null.append(sum(a == b for a, b in zip(sums, p)))
    Pv = (sum(n >= obs for n in null) + 1) / (nnull + 1); nm = sum(null) / len(null)
    return dict(exact=obs, testable=len(ku), null=nm, P=Pv, passed=Pv < 0.01, sign=obs > nm)


def b2a_fixed(Crd, nnull=2000, seed=2):
    rk = {x: i for i, x in enumerate(FIXED_FR)}
    comps = [t['frac'] for d in Crd if not d.get('superseded_by') for t in d['tokens']
             if t['t'] == 'num' and len(t.get('frac') or []) >= 2]
    def count(cs):
        ok = rev = 0
        for f in cs:
            f = [x for x in f if x in rk]
            for i in range(len(f)):
                for j in range(i + 1, len(f)):
                    if f[i] == f[j]: continue
                    if rk[f[i]] < rk[f[j]]: ok += 1
                    else: rev += 1
        return ok, rev
    ok, rev = count(comps); n = ok + rev
    rng = random.Random(seed); nulls = []
    for _ in range(nnull):
        cs = [rng.sample(list(f), len(f)) for f in comps]; nulls.append(count(cs)[1])
    Pv = (sum(r <= rev for r in nulls) + 1) / (nnull + 1)
    share = ok / n if n else float('nan'); nshare = 1 - (sum(nulls) / len(nulls)) / n if n else float('nan')
    return dict(compounds=len(comps), pairs=n, reversals=rev, null_rev=sum(nulls) / len(nulls), share=share,
                P=Pv, passed=(share >= 0.9 and Pv < 0.01), sign=share > nshare)


def battery(C, fast=False, with_lb=True, seed=0, extra=None):
    """C: corpus_ra-style list (all statuses).  Returns dict of results."""
    Crd = P.version([d for d in C if not d.get('superseded_by')], 'rd')
    D = docs79(C)
    r = {}
    r['B1_kuro'] = b1_kuro(Crd, nnull=2000 if fast else 20000, seed=71 + seed)
    r['B2a_frac_fixed'] = b2a_fixed(Crd, nnull=500 if fast else 2000, seed=2 + seed)
    fo = K.order_eval([K.first_order(f, FR_ITEMS) for d in D for f, _ in d['fracs'] if len(set(f)) > 1],
                      [K.first_order(f, FR_ITEMS) for d in D for f, _ in d['fracs'] if len(set(f)) > 1],
                      FR_ITEMS, n_dec=500 if fast else 2000, seed=4 + seed)
    r['B2b_frac_learned'] = dict(agree=fo['agree'], n=fo['n'], pct=fo['pct'], passed=fo['pct'] >= 0.95, sign=fo['agree'] > 0.5)
    pa = K.affix_all_z(D, 'pre', nperm=20 if fast else 40, seed=6 + seed)
    sa = K.affix_all_z(D, 'suf', nperm=20 if fast else 40, seed=6 + seed)
    r['B3_affix_all'] = dict(pre=pa, suf=sa, passed=pa >= 2 and sa >= 2, sign=pa > 0 and sa > 0)
    zs = K.affix_z(D, 'suf'); me = zs.get('ME')
    pct = float((np.array([v for k, v in zs.items() if k != 'ME']) < me).mean()) if me is not None else None
    r['B4_ME'] = dict(z=me, pct=pct, passed=(me is not None and me >= 2 and pct >= 0.9), sign=(me or 0) > 0)
    co = K.order_eval([K.first_order(d['logos'], K.BIG) for d in D], [K.first_order(d['logos'], K.BIG) for d in D],
                      K.BIG, n_dec=500 if fast else 2000, seed=4 + seed)
    r['B5_comm_order'] = dict(agree=co['agree'], n=co['n'], pct=co['pct'], passed=co['pct'] >= 0.95, sign=co['agree'] > 0.5)
    r['B6_sites'] = b6_sites(D, nshuf=100 if fast else 200, seed=58 + seed)
    if with_lb:
        global _LEX
        if _LEX is None: _LEX = K.lb_lexicon()
        z, obs = K.lb_z(D, _LEX, nperm=50 if fast else 100, seed=3 + seed)
        r['B7_lb_shared'] = dict(z=z, obs=obs, passed=z >= 2, sign=z > 0)
    zp = K.affix_z(D, 'pre'); zi = zp.get('I')
    r['M1_prefix_I'] = dict(z=zi, passed=(zi or 0) >= 1.645, sign=(zi or 0) > 0)
    if extra:
        for X in extra:
            r['C_prefix_' + X] = dict(z=zp.get(X), passed=(zp.get(X) or 0) >= 2, sign=(zp.get(X) or 0) > 0)
    return r


def b6_sites(D, nshuf=200, seed=58):
    G = ['HT', 'KH', 'ZA', 'PH', 'KN', 'OTH']
    dt = [(d['g'], {w['s'] for w in d['words'] if len(w['s']) >= 2}) for d in D]
    dt = [x for x in dt if x[1]]
    def stat(labels):
        T = {g: set() for g in G}
        for g, (_, s) in zip(labels, dt): T[g] |= s
        v = [len(T[a] & T[b]) / len(T[a] | T[b]) for i, a in enumerate(G) for b in G[i + 1:] if T[a] and T[b]]
        return float(np.mean(v))
    lab = [g for g, _ in dt]; real = stat(lab)
    rng = random.Random(seed); null = []
    for _ in range(nshuf):
        l = lab[:]; rng.shuffle(l); null.append(stat(l))
    pb = float(np.mean([n > real for n in null]))
    return dict(real=real, null=float(np.mean(null)), share_null_above=pb, passed=pb >= 0.95, sign=real < np.mean(null))


if __name__ == '__main__':
    path = sys.argv[1]
    C = json.load(open(path))
    print(json.dumps(battery(C, fast='--fast' in sys.argv), default=float, indent=1))
