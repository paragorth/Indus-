"""v85 cycle 2: do words shared across hands sit in the same contexts?
For two samples (N tokens each, whole pages) at canon level: shared band types (not top-30, >= 3 occurrences in each).
  CTX_n : neighbour/position profile = line slot (first, second, interior, last) + left and right neighbour identity
          (top-40 pooled types, else OTHER/edge), each block normalised; CM = mean cos(pX(t), pY(t)) minus mean
          cos(pX(t), pY(t')) over up to 5 shared t' of similar frequency (x0.5-2).
  CTX_p : page co-occurrence profile over the shared band types (which other shared words share a page with t).
Null: per-hand v72 junction resynthesis (real words of the hand; next word drawn among words that follow the same
last glyph; first word from the hand's line-initial words) = context made by spelling alone.
Ratios: cross / within for real and for the null.  Usage: VOY_MODE=glyph python3 v85_c2.py [ZL3b|IT2a]"""
import sys, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import v85_lib as L, v72_lib as V

NAME = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'
R = 20


def sample_lines(pages, N, rng, exclude=None):
    idx = [i for i in range(len(pages)) if not exclude or i not in exclude]
    rng.shuffle(idx); lines = []; n = 0; used = set()
    for i in idx:
        if n >= N: break
        used.add(i)
        for l in pages[i]['lines']:
            if n >= N: break
            ws = l['w'][:N - n]; lines.append((i, ws)); n += len(ws)
    return lines, used


def profiles(lines, types, top):
    tix = {t: i for i, t in enumerate(top)}; K = len(top) + 2
    nb = defaultdict(lambda: np.zeros(4 + 2 * K)); pgset = defaultdict(set); pages_types = defaultdict(set)
    for pg, ws in lines:
        n = len(ws)
        for j, w in enumerate(ws):
            if w in types:
                v = nb[w]
                v[0 if j == 0 else 1 if j == 1 else 3 if j == n - 1 else 2] += 1
                lft = ws[j - 1] if j > 0 else None; rgt = ws[j + 1] if j < n - 1 else None
                v[4 + (K - 1 if lft is None else tix.get(lft, K - 2))] += 1
                v[4 + K + (K - 1 if rgt is None else tix.get(rgt, K - 2))] += 1
                pages_types[pg].add(w)
    tl = sorted(types); ti = {t: i for i, t in enumerate(tl)}
    pc = {t: np.zeros(len(tl)) for t in tl}
    for pg, s in pages_types.items():
        for a in s:
            for b in s:
                if a != b: pc[a][ti[b]] += 1
    out = {}
    for t in tl:
        v = nb[t]
        blocks = [v[:4], v[4:4 + K], v[4 + K:]]
        vv = np.concatenate([b / (np.linalg.norm(b) or 1) for b in blocks])
        out[t] = (vv, pc[t])
    return out


def cos(a, b):
    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb_)) if na and nb_ else 0.0


def ctx(lx, ly, rng):
    wx = [w for _, ws in lx for w in ws]; wy = [w for _, ws in ly for w in ws]
    pooled = Counter(wx + wy); top30 = {w for w, _ in pooled.most_common(L.TOPT)}
    top40 = [w for w, _ in pooled.most_common(40)]
    cx, cy = Counter(wx), Counter(wy)
    sh = {t for t in cx if t not in top30 and cx[t] >= 3 and cy[t] >= 3}
    if len(sh) < 6: return None
    PX, PY = profiles(lx, sh, top40), profiles(ly, sh, top40)
    shl = sorted(sh); f = {t: cx[t] + cy[t] for t in shl}
    dn, dp = [], []
    for t in shl:
        cand = [u for u in shl if u != t and 0.5 <= f[u] / f[t] <= 2] or [u for u in shl if u != t]
        alt = rng.sample(cand, min(5, len(cand)))
        dn.append(cos(PX[t][0], PY[t][0]) - np.mean([cos(PX[t][0], PY[u][0]) for u in alt]))
        # page profile: drop t's own coordinate and the alternatives' coordinates are compared on the same axes
        dp.append(cos(PX[t][1], PY[t][1]) - np.mean([cos(PX[t][1], PY[u][1]) for u in alt]))
    return float(np.mean(dn)), float(np.mean(dp)), len(sh)


def measure(X, Y, N, seed):
    rng = random.Random(seed)
    canw = [P for P in (X, Y) if L.ntok(P) >= 1.9 * N]
    C, W = [], []
    for _ in range(R):
        lx, _ = sample_lines(X, N, rng); ly, _ = sample_lines(Y, N, rng)
        r = ctx(lx, ly, rng)
        if r: C.append(r)
        for P in canw:
            a, u = sample_lines(P, N, rng); b, _ = sample_lines(P, N, rng, exclude=u)
            if sum(len(ws) for _, ws in b) >= 0.9 * N:
                r = ctx(a, b, rng)
                if r: W.append(r)
    m = lambda A, i: float(np.mean([a[i] for a in A])) if A else float('nan')
    return dict(cn=m(C, 0), cp=m(C, 1), nsh_c=m(C, 2), wn=m(W, 0), wp=m(W, 1), nsh_w=m(W, 2))


def junc(P, seed):
    return V.gen_junction(P, seed)


def run(job):
    lab, px, py, N, des = job
    X, Y = L.canon(px), L.canon(py)
    real = measure(X, Y, N, 852)
    nul = [measure(junc(X, 31 + i), junc(Y, 41 + i), N, 853 + i) for i in range(2)]
    nm = {k: float(np.nanmean([d[k] for d in nul])) for k in real}
    return dict(label=lab, design=des, N=N, real=real, null=nm)


def fmt(d):
    r, n = d['real'], d['null']
    rat = lambda a, b: a / b if b and b > 0 else float('nan')
    return ('neigh cross %.3f within %.3f (ratio %.2f) | null cross %.3f within %.3f || page cross %.3f within %.3f '
            '(ratio %.2f) | null %.3f / %.3f | shared types %.0f / %.0f') % (
        r['cn'], r['wn'], rat(r['cn'], r['wn']), n['cn'], n['wn'], r['cp'], r['wp'], rat(r['cp'], r['wp']), n['cp'],
        n['wp'], r['nsh_c'], r['nsh_w'])


if __name__ == '__main__':
    import v85_c1 as C1
    C1.NAME = NAME
    jobs = [(lab, px, py, N, des) for lab, px, py, N, des in C1.voy_jobs()]
    if NAME == 'ZL3b':
        for kind, N in [('SAME_CULP', 1100), ('SAME_CULP', 270), ('SAME_MACER', 1100), ('SAME_KONRAD', 1100),
                        ('TWO_CULP_GERARD', 1100), ('TWO_MACER_HILDE', 1100), ('TWO_CULP_CURY', 1100)]:
            px, py = L.ctrl_pair(kind); jobs.append((kind, px, py, N, 'control'))
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=1)
    L.psave('c2_%s.pkl' % NAME, res)
    for d in res:
        print('%-22s N%-5d %s' % (d['label'], d['N'], fmt(d)), flush=True)
