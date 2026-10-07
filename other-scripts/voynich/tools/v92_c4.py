"""v92 cycle 4 (K3'): TRANSPLANT BY SPELLING - are the page-recurring frames arbitrary labels or the page's own spelling?
A name of a thing is an arbitrary label: knowing how the rest of the page is spelled should not tell which page a set of
recurring names came from (beyond section). A spelling mood writes the recurring words in the same glyphs as the rest of
the page. For each random VIEW (glyph features: unigram / bigram / positional / merged classes; unit word or frame; rarity
band) every page's recurring units R_p are scored against the glyph profile of every same-group page q, built from q's
words with all tokens of q's own recurring units removed. Matching accuracy = mean normalised rank of the true page
(0.5 = chance). Train folios select, top 20 tested once on held-out folios (z against 200 within-group relabellings).
Per-frame 'orphan' score = how badly a recurring frame fits its own page: the candidate names."""
import os, sys, json, time, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v92_lib as L

NH = int(os.environ.get('V92_NH4', '1500'))
TOP = 20


def random_view(rng, glyphs):
    kind = str(rng.choice(['uni', 'bi', 'pos', 'uni+pos']))
    merge = None
    if rng.random() < 0.3:
        k = int(rng.integers(4, 12)); merge = {g: int(rng.integers(k)) for g in glyphs}
    return dict(kind=kind, merge=merge, unit=str(rng.choice(['word', 'frame'])), lo=int(rng.choice([2, 3])),
                hi=int(rng.choice([10, 20, 50])), mrec=int(rng.choice([2, 2, 3])), skip_ends=bool(rng.random() < 0.3),
                lam=float(rng.choice([5.0, 20.0, 100.0])))


def featf(v):
    m = v['merge']; kind = v['kind']; se = v['skip_ends']
    def f(w):
        if m is not None: w = ''.join(chr(65 + m.get(c, 0)) for c in w if c != '.')
        else: w = w.replace('.', '')
        if se and len(w) > 2: w = w[1:-1]
        out = []
        if kind in ('uni', 'uni+pos'): out += list(w)
        if kind == 'bi': out += [a + b for a, b in zip('^' + w, w + '$')]
        if kind in ('pos', 'uni+pos'):
            out += ['%s@%d' % (c, i) for i, c in enumerate(w[:3])] + ['%s@-%d' % (c, i) for i, c in enumerate(w[::-1][:3])]
        return out
    return f


def prepare(pages, v, halfsel):
    uf = (lambda w: w) if v['unit'] == 'word' else L.frame
    ff = featf(v)
    cnt = Counter(uf(w) for p in pages for l in p['lines'] for w in l['w'])
    grp = {}
    P = []
    for p in pages:
        if L.split_half(p['id']) != halfsel: continue
        occ = defaultdict(set)
        for li, l in enumerate(p['lines']):
            for w in l['w']: occ[uf(w)].add(li)
        R = [u for u, s in occ.items() if len(s) >= v['mrec'] and v['lo'] <= cnt[u] <= v['hi']]
        if not R: continue
        Rs = set(R)
        rest = Counter(x for l in p['lines'] for w in l['w'] if uf(w) not in Rs for x in ff(w))
        rf = [Counter(ff(u)) for u in R]
        P.append(dict(o=len(P) + 0 * 1, pi=pages.index(p), g='%s|%s|%s' % (p['sec'], p.get('lang', '-'), p.get('hand', '-')), rest=rest, R=R, rf=rf, id=p['id']))
    return P


def match(P, lam, rng=None, R=0, far=0):
    """mean normalised rank of the true page; null = R relabellings of R-sets within group."""
    byg = defaultdict(list)
    for i, x in enumerate(P): byg[x['g']].append(i)
    feats = sorted({f for x in P for f in x['rest']} | {f for x in P for c in x['rf'] for f in c})
    fi = {f: i for i, f in enumerate(feats)}; F = len(feats)
    def vec(c):
        a = np.zeros(F)
        for k, n in c.items(): a[fi[k]] += n
        return a
    rest = np.array([vec(x['rest']) for x in P])
    rvec = np.array([sum((vec(c) for c in x['rf']), np.zeros(F)) for x in P])
    ranks = []; nulls = [[] for _ in range(R)]
    per_unit = []
    for g, idx in byg.items():
        if len(idx) < 3: continue
        G = rest[idx].sum(0) + 1.0; G /= G.sum()
        LR = np.log((rest[idx] + lam * G) / (rest[idx].sum(1, keepdims=True) + lam)) - np.log(G)   # page log-ratio profiles
        S = rvec[idx] @ LR.T                                                                      # S[i, j]: R-set i on page j
        n = len(idx)
        pos = np.array([P[i]['pi'] for i in idx])
        OK = np.abs(pos[:, None] - pos[None, :]) > far
        np.fill_diagonal(OK, True)
        for a in range(n):
            ok = OK[a].copy(); ok[a] = False
            if ok.sum() < 2: continue
            ranks.append((S[a][ok] < S[a, a]).mean())
        for r in range(R):
            perm = rng.permutation(n)
            for a in range(n):
                ok = OK[a].copy(); ok[a] = False
                if ok.sum() < 2: continue
                b = perm[a]
                nulls[r].append((S[a][ok] < S[a, b]).mean() if b != a else 0.5)
        for a in range(n):
            for c in P[idx[a]]['rf']:
                s = vec(c) @ LR.T
                per_unit.append((P[idx[a]]['id'], float((s < s[a]).sum() / (n - 1))))
    acc = float(np.mean(ranks)) if ranks else 0.5
    out = dict(acc=acc, n=len(ranks))
    if R:
        nm = np.array([np.mean(x) for x in nulls])
        out['z'] = float((acc - nm.mean()) / max(nm.std(ddof=1), 1e-6))
    out['units'] = per_unit
    return out


def run(name):
    out_p = os.path.join(L.CK, 'c4_%s.json' % name)
    if os.path.exists(out_p): return name, 0.0
    if os.environ.get('V92_PHASE') == 'A' and os.path.exists(os.path.join(L.CK, 'c4A_%s.json' % name)): return name, 0.0
    if os.environ.get('V92_PHASE') == 'B':
        A = json.load(open(os.path.join(L.CK, 'c4A_%s.json' % name)))
        return test_phase(name, A)
    t0 = time.time()
    pages = L.corpus(name)
    glyphs = sorted({c for p in pages for l in p['lines'] for w in l['w'] for c in w})
    rng = np.random.default_rng(940)
    rows = []
    for h in range(NH):
        v = random_view(rng, glyphs)
        P = prepare(pages, v, 0)
        if len(P) < 20: continue
        m = match(P, v['lam'])
        rows.append(dict(v=v, acc_tr=m['acc'], n_tr=m['n']))
    rows.sort(key=lambda r: -r['acc_tr'])
    if os.environ.get('V92_PHASE') == 'A':
        json.dump(dict(name=name, nh=len(rows), sel=rows[:TOP], acc_tr=[r['acc_tr'] for r in rows]),
                  open(os.path.join(L.CK, 'c4A_%s.json' % name), 'w'), default=str)
        return name, time.time() - t0
    top = []
    for r in rows[:TOP]:
        P = prepare(pages, r['v'], 1)
        m = match(P, r['v']['lam'], np.random.default_rng(7), R=200)
        mf = match(P, r['v']['lam'], np.random.default_rng(8), R=200, far=4)
        r.update(acc_te=m['acc'], z_te=m['z'], n_te=m['n'], acc_far=mf['acc'], z_far=mf['z'], n_far=mf['n'])
        top.append(r)
    # fixed reference view (frame, unigram, count 2-20) both halves, with per-unit fit
    ref = {}
    v0 = dict(kind='uni', merge=None, unit='frame', lo=2, hi=20, mrec=2, skip_ends=False, lam=20.0)
    for h in (0, 1):
        m = match(prepare(pages, v0, h), 20.0, np.random.default_rng(11), R=200)
        mf = match(prepare(pages, v0, h), 20.0, np.random.default_rng(12), R=200, far=4)
        ref['h%d' % h] = dict(acc=m['acc'], z=m['z'], n=m['n'], acc_far=mf['acc'], z_far=mf['z'], n_far=mf['n'],
                              units=m['units'])
    res = dict(name=name, nh=len(rows), top=top, ref=ref, acc_tr=[r['acc_tr'] for r in rows], secs=time.time() - t0)
    json.dump(res, open(out_p, 'w'), default=str)
    return name, res['secs']


def test_phase(name, A):
    t0 = time.time()
    pages = L.corpus(name)
    top = []
    for r in A['sel']:
        P = prepare(pages, r['v'], 1)
        m = match(P, r['v']['lam'], np.random.default_rng(7), R=200)
        mf = match(P, r['v']['lam'], np.random.default_rng(8), R=200, far=4)
        r.update(acc_te=m['acc'], z_te=m['z'], n_te=m['n'], acc_far=mf['acc'], z_far=mf['z'], n_far=mf['n'])
        top.append(r)
    ref = {}
    v0 = dict(kind='uni', merge=None, unit='frame', lo=2, hi=20, mrec=2, skip_ends=False, lam=20.0)
    for h in (0, 1):
        m = match(prepare(pages, v0, h), 20.0, np.random.default_rng(11), R=200)
        mf = match(prepare(pages, v0, h), 20.0, np.random.default_rng(12), R=200, far=4)
        ref['h%d' % h] = dict(acc=m['acc'], z=m['z'], n=m['n'], acc_far=mf['acc'], z_far=mf['z'], n_far=mf['n'], units=m['units'])
    res = dict(name=name, nh=A['nh'], top=top, ref=ref, acc_tr=A['acc_tr'], secs=time.time() - t0)
    json.dump(res, open(os.path.join(L.CK, 'c4_%s.json' % name), 'w'), default=str)
    return name, res['secs']


NAMES = ['ZL3b', 'P_KONRAD', 'G_GM', 'G_GM2', 'P_CIRCA', 'IT2a', 'G_LX', 'P_APIC', 'G_SEED', 'P_HYGIN', 'GC2a', 'G_SELF', 'P_CULP']

if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or NAMES
    for n in names: L.corpus(n)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
