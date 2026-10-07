"""v85 cycle 3: LEXICAL FINGERPRINT.  A dictionary is not just a set of words: it is a pattern of which glyph-typical
shapes are used more or less than the glyph grammar predicts (idiosyncratic favourites and gaps).  Two scribes with
one dictionary share that pattern; private dictionaries do not.  Frozen 7 Oct 2026 before Voynich numbers:
  split   : pages of hand X dealt at random into X1 (N tokens), X2 (N tokens), rest; same for Y.
  ref     : glyph-unit Markov order 2 (add-0.1, word start/end symbols) fitted on rest_X + rest_Y (canon level).
  residual: r_S(w) = log2((c_S(w) + 0.5) / (n_S p_ref(w) + 0.5)) for candidate types w = types with pooled count >= 2
            in X1+X2+Y1+Y2 outside the 30 commonest, plus the 200 likeliest ref-Markov words (gaps).
  within  : Pearson r(X1, X2) and r(Y1, Y2);  cross: mean of the four X-Y pairings.
  null    : the same on per-hand glyph Markov texts (order 2 and 3, fitted on the whole hand) with the same page/line
            shapes.  LE = real - null;  LSR = LE_cross / LE_within.
Planted: PRIVr/SHAREDr (finite dictionary of Markov words, ranks assigned at random = frequency independent of
shape), PRIVm/SHAREDm (ranks by Markov likelihood), fitted to Voynich hands.  Usage: v85_c3.py [ZL3b|IT2a]"""
import sys, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import v85_lib as L

NAME = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'
R = 20


class RefM:
    def __init__(self, words, k=2, a=0.1):
        self.k = k; self.a = a; self.T = defaultdict(Counter); al = set('$')
        for w in words:
            x = '^' * k + w + '$'; al.update(w)
            for j in range(k, len(x)): self.T[x[j - k:j]][x[j]] += 1
        self.al = len(al); self.tot = {c: sum(v.values()) for c, v in self.T.items()}

    def lp(self, w):
        x = '^' * self.k + w + '$'; s = 0.0
        for j in range(self.k, len(x)):
            c = x[j - self.k:j]
            s += math.log2((self.T[c][x[j]] + self.a) / (self.tot.get(c, 0) + self.a * self.al))
        return s

    def top(self, n, rng, draws=20000):
        seen = Counter()
        for _ in range(draws):
            x = '^' * self.k
            while len(x) < 25:
                c = x[-self.k:]; d = self.T.get(c)
                if not d: break
                ks = list(d); ch = rng.choices(ks, weights=[d[q] for q in ks])[0]
                if ch == '$': break
                x += ch
            seen[x[self.k:]] += 1
        return [w for w, _ in seen.most_common(n)]


def split3(pages, N, rng):
    idx = list(range(len(pages))); rng.shuffle(idx)
    parts = [[], []]; n = [0, 0]; rest = []
    for i in idx:
        ws = [w for l in pages[i]['lines'] for w in l['w']]
        j = 0 if n[0] < N else (1 if n[1] < N else None)
        if j is None: rest.extend(ws); continue
        take = ws[:N - n[j]]; parts[j].extend(take); n[j] += len(take); rest.extend(ws[len(take):])
    return parts[0], parts[1], rest


def corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.std() == 0 or b.std() == 0: return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def fingerprint(X, Y, N, rng):
    x1, x2, rx = split3(X, N, rng); y1, y2, ry = split3(Y, N, rng)
    if min(len(x1), len(x2), len(y1), len(y2)) < 0.9 * N or len(rx) + len(ry) < N: return None
    M = RefM(rx + ry)
    S = [x1, x2, y1, y2]; C = [Counter(s) for s in S]
    pooled = sum(C, Counter()); top = {w for w, _ in pooled.most_common(L.TOPT)}
    cand = {w for w, c in pooled.items() if c >= 2 and w not in top} | (set(M.top(200, rng)) - top)
    cand = sorted(cand); lp = np.array([M.lp(w) for w in cand]); p = np.exp2(lp)
    Rr = [np.log2((np.array([c[w] for w in cand]) + 0.5) / (len(s) * p + 0.5)) for c, s in zip(C, S)]
    within = [corr(Rr[0], Rr[1]), corr(Rr[2], Rr[3])]
    cross = [corr(Rr[i], Rr[j]) for i in (0, 1) for j in (2, 3)]
    return float(np.nanmean(within)), float(np.nanmean(cross)), len(cand)


def measure(X, Y, N, seed):
    rng = random.Random(seed); out = []
    for _ in range(R):
        r = fingerprint(X, Y, N, rng)
        if r: out.append(r)
    if not out: return dict(w=float('nan'), c=float('nan'), sw=float('nan'), sc=float('nan'), n=0)
    A = np.array(out)
    return dict(w=float(A[:, 0].mean()), c=float(A[:, 1].mean()), sw=float(A[:, 0].std()), sc=float(A[:, 1].std()),
                n=float(A[:, 2].mean()))


def run(job):
    lab, px, py, N, des, kind = job
    X, Y = L.canon(px), L.canon(py)
    if kind.startswith('PRIV') or kind.startswith('SHARED'):
        mode = kind[-1]
        def dict_of(P, sd):
            _, dic = L.priv_gen(P, sd, 2)
            if mode == 'r': random.Random(sd + 99).shuffle(dic)
            return dic
        if kind.startswith('PRIV'):
            X, _ = L.priv_gen(X, 1, 2, shared_dict=dict_of(X, 11)); Y, _ = L.priv_gen(Y, 2, 2, shared_dict=dict_of(Y, 12))
        else:
            dic = dict_of(X + Y, 13)
            X, _ = L.priv_gen(X, 1, 2, shared_dict=dic); Y, _ = L.priv_gen(Y, 2, 2, shared_dict=dic)
    real = measure(X, Y, N, 853)
    nul = {}
    for k in (2, 3):
        mx, my = L.MK(X, k), L.MK(Y, k)
        nul[k] = measure(mx.gen(X, 61 + k), my.gen(Y, 71 + k), N, 854 + k)
    return dict(label=lab, design=des, kind=kind, N=N, real=real, null=nul)


def fmt(d):
    r = d['real']; s = '%-7s within %.3f+-%.3f cross %.3f+-%.3f (types %.0f)' % (d['kind'], r['w'], r['sw'], r['c'], r['sc'], r['n'])
    for k in (2, 3):
        n = d['null'][k]; lw, lc = r['w'] - n['w'], r['c'] - n['c']
        s += ' | MK%d null w %.3f c %.3f -> LE_w %.3f LE_c %.3f LSR %.2f' % (k, n['w'], n['c'], lw, lc, lc / lw if lw > 0.02 else float('nan'))
    return s


def jobs_for(name):
    import v85_c1 as C1
    C1.NAME = name
    P, G = L.voy_groups(name)
    g = lambda s, h, l='B': G[(s, h, l)]
    J = [('H h1(A) x h2(B)', g('H', '1', 'A'), g('H', '2'), 700, 'same section diff hand (A/B)'),
         ('H h2 x h3+h5', g('H', '2'), g('H', '3') + g('H', '5'), 350, 'same section diff hand'),
         ('h2 H x h2 B', g('H', '2'), g('B', '2'), 700, 'same hand diff section'),
         ('h1 H x h1 P', g('H', '1', 'A'), g('P', '1', 'A'), 650, 'same hand diff section'),
         ('h3 S(A) x h3 S(B)', g('S', '3', 'A'), g('S', '3'), 230, 'same hand same section diff language'),
         ('h2 B x h3 S', g('B', '2'), g('S', '3'), 700, 'diff hand diff section'),
         ('h2 H x h3 S', g('H', '2'), g('S', '3'), 700, 'diff hand diff section'),
         ('h1 H x h3 S', g('H', '1', 'A'), g('S', '3'), 700, 'diff hand diff section (A/B)'),
         ('T h2 x h1+h5', g('T', '2'), g('T', '1', 'A') + g('T', '5'), 160, 'same section diff hand (small)')]
    out = [(a, b, c, d, e, 'real') for a, b, c, d, e in J]
    if name == 'ZL3b':
        for kind in ('PRIVr', 'SHAREDr', 'PRIVm', 'SHAREDm'):
            out.append(('H h1 x h2 planted', g('H', '1', 'A'), g('H', '2'), 700, 'planted', kind))
            out.append(('H h2 x h3+h5 planted', g('H', '2'), g('H', '3') + g('H', '5'), 350, 'planted', kind))
        for kind, N in [('SAME_CULP', 700), ('SAME_CULP', 350), ('SAME_KONRAD', 700), ('SAME_MACER', 700),
                        ('SAME_MACER', 350), ('TWO_CULP_GERARD', 700), ('TWO_MACER_HILDE', 700), ('TWO_MACER_HILDE', 350),
                        ('TWO_CULP_CURY', 700)]:
            px, py = L.ctrl_pair(kind); out.append((kind, px, py, N, 'control', 'real'))
    return out


if __name__ == '__main__':
    J = jobs_for(NAME)
    with Pool(2) as pool:
        res = pool.map(run, J, chunksize=1)
    L.psave('c3_%s.pkl' % NAME, res)
    for d in res:
        print('%-22s N%-4d %s' % (d['label'], d['N'], fmt(d)), flush=True)
