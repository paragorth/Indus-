#!/usr/bin/env python3
"""LA-43 cycle 3: sound-aware edit-distance neighbours across sites and across time.

A word at a held-out site (or in a post-1985 publication) that is new to the training part is linked to every
training word that differs from it by one substituted sign (same length). A link is a 'sound link' under a value
map when the two substituted signs share a consonant (row) or a vowel (column).
  S1  share of cross-part links that are sound links (rows / columns separately and together)
  S2  role agreement: a word's role is what follows it on its document (number, logogram, another word, nothing)
      plus its position (first word of the document or not). Are sound-linked pairs more alike in role than
      the other links? (difference of agreement rates)  -- an outside, non-phonetic check of the variants
  S3  morphology: links at the LAST sign only (ending alternations) that keep the column vs keep the row.
True map vs 10,000 relabelings per tier (R2a, R2b, R3). Splits: LOSO folds (as cycle 2) and TIME (GORILA -> post).
Controls: LB at LA size (5 LOSO draws, as cycle 2) with true values; LB full KN -> PY; role-shuffled LA (roles
permuted among tokens inside each document) must kill S2; within-word sign shuffle of the test part kills S1/S3 links.
"""
import sys, os, json, re, hashlib, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la43_common as M
import la43_c2 as C2
import la32_common as C32
from multiprocessing import Pool

TIERS = ['R2a', 'R2b', 'R3']


def norm(s):
    return s.replace('₂', '2').replace('₃', '3')


def la_role_units():
    """LA word tokens with role: next token kind and first-in-document flag."""
    src = M.L15._pub_source()
    out = []
    for d in C32.la_corpus():
        toks = [t for t in d['tokens'] if t['t'] not in ('nl', 'div')]
        first = True
        for i, t in enumerate(toks):
            if t['t'] != 'word' or len(t['s']) < 2:
                if t['t'] == 'word':
                    first = False
                continue
            w = tuple(norm(s) for s in t['s'])
            nx = toks[i + 1] if i + 1 < len(toks) else None
            if nx is None:
                r = 'end'
            elif nx['t'] == 'num':
                r = 'num'
            elif nx['t'] == 'word' and (len(nx['s']) == 1 or M.L38.cv_of(norm(nx['s'][0])) is None):
                r = 'logo'
            elif nx['t'] == 'word':
                r = 'word'
            else:
                r = nx['t']
            out.append(dict(w=w, doc=d['id'], site=d['site'], pub=src.get(d['id'], 'blank'),
                            role=r + ('/1' if first else '/n'), sup=d['support']))
            first = False
    return out


def lb_role_units():
    """LB word tokens with role from the DAMOS line text (next token: number, logogram, word, end)."""
    out = []
    for l in open(os.path.join(M.LA, 'data', 'damos_items.jsonl')):
        d = json.loads(l)
        if not d.get('content'):
            continue
        h = d['heading']; site = h.split()[0]
        if site not in ('KN', 'PY'):
            continue
        series = (site + (h.split()[1] if len(h.split()) > 1 else ''))[:4]
        txt = re.sub(r'⟦[^⟧]*⟧', ' ', d['content'])
        toks = [t for t in re.split(r'[\s,./|]+', txt) if t]
        first = True
        for i, tok in enumerate(toks):
            if '̣' in tok or not C32.WRD.match(tok):
                continue
            sg = tuple(tok.split('-'))
            if not all(s in C32.LB_VAL for s in sg):
                continue
            nx = toks[i + 1] if i + 1 < len(toks) else None
            if nx is None:
                r = 'end'
            elif re.fullmatch(r'\d+', nx):
                r = 'num'
            elif C32.WRD.match(nx):
                r = 'word'
            else:
                r = 'logo'
            out.append(dict(w=tuple(s.upper() for s in sg), doc=h, site=site, series=series, role=r + ('/1' if first else '/n')))
            first = False
    return out


def links(train, test):
    """one-substitution links (test new type, train type): arrays of sign pair, last-position flag, role agreement."""
    trt = collections.defaultdict(collections.Counter)
    for u in train:
        trt[u['w']][u['role']] += 1
    tet = collections.defaultdict(collections.Counter)
    for u in test:
        tet[u['w']][u['role']] += 1
    idx = collections.defaultdict(list)
    for w in trt:
        for p in range(len(w)):
            idx[(len(w), p, w[:p] + w[p + 1:])].append(w)
    L = []
    for w in tet:
        if w in trt:
            continue
        for p in range(len(w)):
            for v in idx.get((len(w), p, w[:p] + w[p + 1:]), []):
                a, b = w[p], v[p]
                ra, rb = tet[w], trt[v]
                # role agreement: probability two random tokens share role
                na, nb = sum(ra.values()), sum(rb.values())
                agree = sum(ra[k] * rb[k] for k in ra) / (na * nb)
                L.append((a, b, int(p == len(w) - 1), agree, w, v))
    return L


class LinkSet:
    def __init__(self, Ls, signs, values=None):
        self.grid = G = M.Grid(signs, values)
        keep = [l for l in Ls if l[0] in G.ix and l[1] in G.ix]
        self.a = np.array([G.ix[l[0]] for l in keep]); self.b = np.array([G.ix[l[1]] for l in keep])
        self.last = np.array([l[2] for l in keep]); self.agree = np.array([l[3] for l in keep])
        self.n = len(keep)
        pos = {s: k for k, s in enumerate(G.valued)}
        self.va = np.array([pos.get(i, -1) for i in self.a]); self.vb = np.array([pos.get(i, -1) for i in self.b])
        self.ok = (self.va >= 0) & (self.vb >= 0)

    def stats(self, Cn, Vn):
        """Cn, Vn (N, nvalued) -> dict of (N,) arrays."""
        ok = self.ok; va = self.va[ok]; vb = self.vb[ok]; ag = self.agree[ok]; la = self.last[ok]
        sr = Cn[:, va] == Cn[:, vb]; sc = Vn[:, va] == Vn[:, vb]; snd = sr | sc
        out = {}
        out['S1row'] = sr.mean(1); out['S1col'] = sc.mean(1); out['S1'] = snd.mean(1)
        n1 = snd.sum(1); n0 = ok.sum() - n1
        out['S2'] = (snd @ ag) / np.maximum(n1, 1) - ((~snd) @ ag) / np.maximum(n0, 1)
        if la.sum():
            out['S3col'] = (sc[:, la == 1]).mean(1); out['S3row'] = (sr[:, la == 1]).mean(1)
        return out


def evaluate(tag, Ls, signs, N, seed, values=None):
    X = LinkSet(Ls, signs, values); G = X.grid
    obs = X.stats(G.C0[None], G.V0[None])
    res = dict(tag=tag, nlinks=int(X.ok.sum()), nlast=int(X.last[X.ok].sum()), tiers={})
    rng = np.random.default_rng(seed)
    for t in TIERS:
        Cn, Vn = M.L38.relabel(G, t, N, rng)
        nl = X.stats(Cn, Vn)
        res['tiers'][t] = {k: dict(obs=float(obs[k][0]), null=float(nl[k].mean()), z=float((obs[k][0] - nl[k].mean()) / (nl[k].std() + 1e-12)),
                                   p=float(((nl[k] >= obs[k][0] - 1e-12).sum() + 1) / (N + 1))) for k in obs}
    return res


def la_splits():
    U = la_role_units()
    out = {}
    g = lambda u: u['site'] if u['site'] in C2.SITES else 'other'
    Ls = []
    for f in C2.SITES + ['other']:
        Ls += links([u for u in U if g(u) != f], [u for u in U if g(u) == f])
    out['LA_LOSO'] = Ls
    out['LA_TIME'] = links([u for u in U if u['pub'].startswith('G')], [u for u in U if u['pub'] == 'post'])
    # role-shuffle control: roles permuted among the word tokens of each document
    rng = np.random.default_rng(91)
    by = collections.defaultdict(list)
    for u in U:
        by[u['doc']].append(u)
    V = []
    for d, us in by.items():
        r = [u['role'] for u in us]; rng.shuffle(r)
        V += [dict(u, role=x) for u, x in zip(us, r)]
    Ls = []
    for f in C2.SITES + ['other']:
        Ls += links([u for u in V if g(u) != f], [u for u in V if g(u) == f])
    out['LA_LOSO_roleshuf'] = Ls
    # sign-shuffle control: signs shuffled inside each held-out word
    W = [dict(u, w=tuple(np.random.default_rng(hash(u['w']) % 2**32).permutation(u['w']))) for u in U]
    Ls = []
    for f in C2.SITES + ['other']:
        Ls += links([u for u in U if g(u) != f], [u for u in W if g(u) == f])
    out['LA_LOSO_signshuf'] = Ls
    return out, sorted({s for u in U for s in u['w']})


def lb_splits():
    U = lb_role_units()
    out = {}
    out['LB_KNPY'] = links([u for u in U if u['site'] == 'KN'], [u for u in U if u['site'] == 'PY'])
    for d in range(5):
        D = M.draw_docs(U, 5558, 300 + d)
        grp = collections.Counter()
        for u in D:
            grp[u['series'][:3]] += len(u['w'])
        size = np.zeros(6); asg = {}
        for k, n in grp.most_common():
            i = int(np.argmin(size)); asg[k] = i; size[i] += n
        Ls = []
        for f in range(6):
            Ls += links([u for u in D if asg[u['series'][:3]] != f], [u for u in D if asg[u['series'][:3]] == f])
        out[f'LB_LOSO_{d}'] = Ls
    return out, sorted({s for u in U for s in u['w']})


def job(a):
    tag, Ls, signs, N, seed = a
    return evaluate(tag, Ls, signs, N, seed)


def main():
    la, las = la_splits(); lb, lbs = lb_splits()
    fz = os.path.join(M.CK, 'c3_freeze.json')
    F = {k: hashlib.sha256(json.dumps(sorted(['-'.join(l[4]) + '|' + '-'.join(l[5]) for l in v])).encode()).hexdigest()[:16]
         for k, v in list(la.items()) + list(lb.items())}
    json.dump(F, open(fz, 'w'), indent=1)
    print('frozen', hashlib.sha256(open(fz, 'rb').read()).hexdigest()[:16], flush=True)
    jobs = [(k, v, lbs, 2000, 50 + i) for i, (k, v) in enumerate(lb.items())]
    jobs += [(k, v, las, 10000, 10 + i) for i, (k, v) in enumerate(la.items())]
    out = []
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            json.dump(out, open(os.path.join(M.CK, 'c3.json'), 'w'))
            line = [f"{r['tag']} links {r['nlinks']} last {r['nlast']}"]
            for t, d in r['tiers'].items():
                line.append(t + ': ' + ', '.join(f"{k} {v['obs']:.3f}/{v['null']:.3f} z{v['z']:+.1f} P{v['p']:.3g}" for k, v in d.items()))
                print(' | '.join(line), flush=True); line = ['   ']


if __name__ == '__main__':
    main()
