"""v40 cycle 2: massive random search of LAYOUT / PEN hypotheses for each twin choice.
Base = SCRIBE + CTX + FRAME logistic (page-grouped out-of-fold). Each hypothesis = one binary feature:
  ABOVE  any glyph of a random set G in line (l + dl), dl in {-2,-1,+1}, offsets off+lo .. off+hi
  POS    random threshold on a random layout variable (x, offset, to-end, word index, words to end, line in
         paragraph, line in page, line length, paragraph-first/last, recto), optionally AND a second one
  PEN    previous choice of the same pair in line / previous line equals A or B; alternation
Discovery = odd-numbered pages (gain by page-grouped CV inside discovery), top 25 re-tested on the even pages
(fit on discovery, evaluate on confirmation) against 200 y-permutations within (frame x page) strata; base model includes page identity.
Search-wide null: the same search on 4 y-permuted copies -> max-gain threshold.
Planted: k/t resampled from the base model + 'tall glyph above -> t' (s = 0.2) must be found and confirmed."""
import sys, os, json, time
import numpy as np
from scipy.sparse import hstack
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v40_lib import *
from v40_cycle1 import vpairs, same_above
from v40_cycle1 import perm_within as _pw


def perm_within(toks, y, rng):
    return _pw(toks, y, rng, keys=('frame', 'page'))

NH = int(os.environ.get('V40_NH', 3000))
VARS = ['xrel', 'off', 'toend', 'wi', 'wfe', 'lip', 'lpage', 'linelen', 'ps', 'pe', 'recto', 'jw']


def neigh_matrix(pages, toks, gid, dl, W=8):
    M = np.full((len(toks), 2 * W + 1), -1, dtype=np.int16)
    for i, t in enumerate(toks):
        li = t['line'] + dl
        L = pages[t['page']]['lines']
        if li < 0 or li >= len(L):
            continue
        g = L[li]['g']
        for d in range(-W, W + 1):
            j = t['off'] + d
            if 0 <= j < len(g):
                M[i, d + W] = gid.get(g[j], -1)
    return M


def make_hyps(rng, inventory, tallids, pairids, n):
    H = []
    for _ in range(n):
        r = rng.random()
        if r < 0.5:
            u = rng.random()
            if u < 0.25:
                G = tuple(tallids)
            elif u < 0.4:
                G = tuple(pairids[rng.integers(2)])
            else:
                k = int(rng.integers(1, 6))
                G = tuple(sorted(rng.choice(inventory, size=k, replace=False).tolist()))
            dl = int(rng.choice([-1, -1, -1, -2, 1]))
            lo = int(rng.integers(-6, 5)); hi = lo + int(rng.integers(0, 4))
            H.append(('ABOVE', G, dl, lo, min(hi, 8)))
        elif r < 0.88:
            v = VARS[rng.integers(len(VARS))]; q = float(rng.uniform(0.1, 0.9)); sgn = int(rng.choice([-1, 1]))
            if rng.random() < 0.4:
                v2 = VARS[rng.integers(len(VARS))]; q2 = float(rng.uniform(0.1, 0.9)); s2 = int(rng.choice([-1, 1]))
                H.append(('POS2', v, q, sgn, v2, q2, s2))
            else:
                H.append(('POS', v, q, sgn))
        else:
            H.append(('PEN', ['line=B', 'line=A', 'prev=B', 'prev=A', 'alt'][rng.integers(5)]))
    return H


class Data:
    def __init__(self, pages, toks, gid, tallset):
        self.toks = toks
        self.N = {dl: neigh_matrix(pages, toks, gid, dl) for dl in (-2, -1, 1)}
        self.V = {}
        for v in VARS:
            if v == 'wfe':
                a = np.array([t['nw'] - t['wi'] - 1 for t in toks], float)
            else:
                a = np.array([t[v] for t in toks], float)
            self.V[v] = a + np.random.default_rng(1).random(len(a)) * 1e-6

    def feat(self, h, y=None, pen=None):
        if h[0] == 'ABOVE':
            _, G, dl, lo, hi = h
            M = self.N[dl][:, lo + 8: hi + 9]
            return np.isin(M, np.array(G)).any(1).astype(float)
        if h[0] in ('POS', 'POS2'):
            v, q, s = h[1], h[2], h[3]
            a = self.V[v]; x = (s * (a - np.quantile(a, q)) > 0)
            if h[0] == 'POS2':
                b = self.V[h[4]]; x &= (h[6] * (b - np.quantile(b, h[5])) > 0)
            return x.astype(float)
        pl, pp = pen
        return {'line=B': pl > 0, 'line=A': pl < 0, 'prev=B': pp > 0, 'prev=A': pp < 0,
                'alt': pl != 0}[h[1]].astype(float) if h[1] != 'alt' else (pl != 0).astype(float)


def base_p(toks, y, groups):
    X = hstack([onehot(family_cols(toks, f)) for f in ('SCRIBE', 'CTX', 'FRAME')]).tocsr()
    p, f = oof_logloss(X, y, groups)
    return p, f


def search(D, toks, y, H, disc, conf, groups):
    """returns discovery gains (array) and p0."""
    p0, _ = base_p(toks, y, groups)
    pen = pen_feats(toks, y)
    # discovery folds by page within disc
    gd = groups[disc]
    ug = np.unique(gd); fold = {g: i % 4 for i, g in enumerate(np.random.default_rng(3).permutation(ug))}
    fd = np.array([fold[g] for g in gd])
    gains = np.zeros(len(H))
    for k, h in enumerate(H):
        x = D.feat(h, y, pen)
        xd = x[disc]
        if xd.std() == 0 or xd.mean() < 0.01 or xd.mean() > 0.99:
            gains[k] = -1; continue
        gains[k] = offset_gain(y[disc], p0[disc], xd, fd)
    return gains, p0, pen


def confirm(D, toks, y, h, p0, pen, disc, conf, rng, nperm=200):
    x = D.feat(h, y, pen)
    z0 = np.log(p0 / (1 - p0))
    def g(yy, xx):
        a, b = _fit_offset(yy[disc], z0[disc], xx[disc]); a0 = _fit_offset(yy[disc], z0[disc], None)[0]
        return (_ll(yy[conf], z0[conf] + a0) - _ll(yy[conf], z0[conf] + a + b * xx[conf])).mean() / LOG2, b
    obs, b = g(y, x)
    # null: permute y within frame x hand strata on the confirmation pages only (x, p0 fixed)
    cidx = np.where(conf)[0]
    strata = defaultdict(list)
    for i in cidx:
        strata[(toks[i]['frame'], toks[i]['page'])].append(i)
    null = []
    for _ in range(nperm):
        y2 = y.copy()
        for idx in strata.values():
            idx = np.array(idx); y2[idx] = y[rng.permutation(idx)]
        null.append(g(y2, x)[0])
    null = np.array(null)
    return obs, b, float((null >= obs).mean()), float((obs - null.mean()) / (null.std() + 1e-12)), float(x.mean())


def run(name, pages, pairs, tall, rng, log, plant=None, nnull=4, ntop=25):
    toks_all = extract(pages, pairs, tall)
    inv = [c for c, _ in Counter(c for P in pages for L in P['lines'] for c in L['g'] if c != ' ').most_common(22)]
    gid = {c: i for i, c in enumerate(inv)}
    res = {}
    for pname in pairs:
        if plant and pname != 'KT':
            continue
        toks = [t for t in toks_all if t['pair'] == pname]
        y = np.array([t['y'] for t in toks])
        if len(y) < 300 or min(y.mean(), 1 - y.mean()) < 0.02:
            continue
        groups = np.array([t['page'] for t in toks])
        D = Data(pages, toks, gid, tall)
        if plant:
            p0, _ = base_p(toks, y, groups)
            ta = above_any(pages, toks, tall, -1, 1)
            y = (rng.random(len(y)) < p0).astype(int)
            y[(ta == 1) & (rng.random(len(y)) < plant)] = 1
            for i, t in enumerate(toks):
                t['y'] = int(y[i])
        disc = groups % 2 == 1; conf = ~disc
        A, B = pairs[pname][0], pairs[pname][1]
        H = make_hyps(rng, list(range(len(inv))), [gid[c] for c in tall if c in gid],
                      ([gid[c] for c in A if c in gid] or [0], [gid[c] for c in B if c in gid] or [0]), NH)
        t0 = time.time()
        gains, p0, pen = search(D, toks, y, H, disc, conf, groups)
        # search-wide null
        nmax = []
        for k in range(nnull):
            y2 = perm_within(toks, y, rng)
            g2, _, _ = search(D, toks, y2, H, disc, conf, groups)
            nmax.append(float(g2.max()))
        thr = max(nmax)
        order = np.argsort(-gains)[:ntop]
        rows = []
        for k in order:
            obs, b, p, z, cov = confirm(D, toks, y, H[k], p0, pen, disc, conf, rng)
            rows.append(dict(h=[str(u) if not isinstance(u, tuple) else [inv[j] for j in u] for u in H[k]],
                             disc=float(gains[k]), conf=obs, b=b, p=p, z=z, cov=cov))
        nconf = sum(1 for r in rows if r['p'] < 0.05 / ntop and r['disc'] > thr)
        res[pname] = dict(n=len(y), best=float(gains.max()), nullmax=nmax, n_above_null=int((gains > thr).sum()),
                          n_confirmed=nconf, top=rows)
        msg = '%s %s n=%d: best disc gain %.4f bits vs search-null max %s; %d hyps above null; %d confirmed (Bonf) [%.0fs]' % (
            name, pname, len(y), gains.max(), ['%.4f' % v for v in nmax], (gains > thr).sum(), nconf, time.time() - t0)
        print(msg, flush=True); log.write(msg + '\n')
        for r in rows[:8]:
            m2 = '   %s disc %.4f conf %.4f b %+.2f p %.3f z %.1f cov %.2f' % (r['h'], r['disc'], r['conf'], r['b'], r['p'], r['z'], r['cov'])
            print(m2, flush=True); log.write(m2 + '\n')
        log.flush()
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    rng = np.random.default_rng(402)
    log = open(os.path.join(CKPT, 'cycle2_%s.log' % which), 'a')
    if which == 'plant':
        out = {s: run('PLANT%.1f' % s, load_voynich('ZL3b'), vpairs(), V_TALL, rng, log, plant=s) for s in (0.1, 0.2)}
    elif which == 'dta':
        dtall = set('bdfhklſtßꝛ') | set('ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ')
        dp = {'SS': (set('s'), set('ſ'), lambda c: 'S'), 'RR': (set('r'), set('ꝛ'), lambda c: 'R')}
        out = run('DTA-Simpl', load_dta(os.path.join(SCR, 'grimmelshausen_simplicissimus_1669.txt')), dp, dtall, rng, log)
    else:
        out = run(which, load_voynich(which), vpairs(), V_TALL, rng, log)
    json.dump(out, open(os.path.join(CKPT, 'cycle2_%s.json' % which), 'w'), indent=1)
