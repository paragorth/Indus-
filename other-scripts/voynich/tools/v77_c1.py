"""v77 cycle 1: kill tests for the five most specific text-only C guesses.
K1 v72 designation code      : stated kill = word-identity link > 0.05 bits/token, or page recurrence above a copy
                               generator, under the frozen extraction E1c (any transcription). Run on ZL, IT and on
                               generator text through the same E1c, held-out leaves.
K2 v71 index-like direction   : within-line word-pair direction, held-out; must exceed fitted generators and must not
                               come only from paragraph-first lines (v72 found arrows only there).
K3 v59 beginnings kept / endings re-encoded A<->B : stated kill = an endings-only B->A map doing no better than
                               random ending maps (learned on half the B pages, scored on the other half);
                               plus a within-A control (do endings carry subject inside A?).
K4 v54 payload frame vs padding : stated kill = the same variant classes from a junction or line-position model
                               without recurring context.
K6 v63 stem+l -> stem+y      : stated kill = it vanishes for pairs matched for position in the line.
"""
import sys, os, json, math, random, collections
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L
import v72_lib as V72

OUT = 'v77_cycle1.txt'
LOG = open(os.path.join(L.CK, 'c1.log'), 'a')


def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


# ------------------------------------------------------------------ bigram machinery
def pairs_of(pages, half, drop_ps=False):
    out = []
    for p, li, l in L.lines_of(pages, half, drop_ps):
        w = l['w']
        out += [(a, b, l['ps']) for a, b in zip(w, w[1:])]
    return out


class Bigram:
    def __init__(self, pages, half, beta=20.0, gamma=3.0):
        self.u = Counter(); self.j = defaultdict(Counter); self.f = defaultdict(Counter)
        for p, li, l in L.lines_of(pages, half):
            w = l['w']; self.u.update(w)
            for a, b in zip(w, w[1:]):
                self.j[a[-1]][b] += 1; self.f[a][b] += 1
        self.N = sum(self.u.values()); self.V = len(self.u) + 1
        self.beta, self.gamma = beta, gamma
        self.jt = {k: sum(v.values()) for k, v in self.j.items()}
        self.ft = {k: sum(v.values()) for k, v in self.f.items()}
        self.jT = {k: len(v) for k, v in self.j.items()}
        self.fT = {k: len(v) for k, v in self.f.items()}

    def pu(self, b):
        return (self.u[b] + 0.5) / (self.N + 0.5 * self.V)

    def pj(self, b, a):
        n = self.jt.get(a[-1], 0); T = self.jT.get(a[-1], 1)   # Witten-Bell
        return (self.j[a[-1]][b] + T * self.pu(b)) / (n + T)

    def pf(self, b, a):
        n = self.ft.get(a, 0); T = self.fT.get(a, 1)
        return (self.f[a][b] + T * self.pj(b, a)) / (n + T)


PIECE_A = {'o', 'r', 's', 'or', 'ar', 'Sor', 'Cor'}
PIECE_B = {'aiin', 'ain', 'aiir', 'air', 'aiiin'}


def k1_k2(pages):
    """identity gain beyond junction (raw surface and E1c payload), page-cache gain, direction."""
    res = {}
    for tag, P in (('raw', pages), ('E1c', V72.extract(pages, V72.RULES['E1c_keepd']))):
        g_all, g_clean, dirs, dirs_body, dirs_j = [], [], [], [], []
        pc = []
        for tr in (0, 1):
            B = Bigram(P, tr)
            te = pairs_of(P, 1 - tr)
            ga = []; gc = []; dd = []; db = []; dj = []
            pm, pj, cl = [], [], []
            for a, b, ps in te:
                n = B.ft.get(a, 0)
                pm.append(B.f[a][b] / n if n else 0.0); pj.append(B.pj(b, a))
                cl.append(a != b and not (a in PIECE_A and b in PIECE_B))
                d = (math.log2(B.pu(a) * B.pf(b, a)) - math.log2(B.pu(b) * B.pf(a, b)))
                dd.append(d)
                if not ps: db.append(d)
                dj.append(math.log2(B.pu(a) * B.pj(b, a)) - math.log2(B.pu(b) * B.pj(a, b)))
            pm = np.array(pm); pj = np.array(pj); cl = np.array(cl)
            # identity gain: lambda * ML word bigram + (1-lambda) * junction model; lambda fitted by EM on one half
            # of the held-out pairs and scored on the other half (cross-fitted, both ways)
            idx = np.arange(len(pm)); fold = idx % 2
            for f in (0, 1):
                tr_, te_ = fold == f, fold != f
                lam = 0.3
                for _ in range(50):
                    r = lam * pm[tr_] / (lam * pm[tr_] + (1 - lam) * pj[tr_]); lam = float(r.mean())
                gain = np.log2(lam * pm[te_] + (1 - lam) * pj[te_]) - np.log2(pj[te_])
                ga.append(gain.mean()); gc.append(gain[cl[te_]].mean())
            g_all.append(np.mean(ga)); g_clean.append(np.mean(gc)); dirs.append(np.mean(dd)); dirs_body.append(np.mean(db))
            dirs_j.append(np.mean(dj))
            # page cache gain (lambda 0.2) over the discovery unigram, held-out pages
            lam = 0.2; gg = []
            for p in P:
                if L.leaf_half(p['id']) == tr: continue
                seen = Counter(); n = 0
                for l in p['lines']:
                    for w in l['w']:
                        base = B.pu(w)
                        if n > 0:
                            gg.append(math.log2(lam * seen[w] / n + (1 - lam) * base) - math.log2(base))
                        seen[w] += 1; n += 1
            pc.append(np.mean(gg))
        res[tag] = dict(id_all=float(np.mean(g_all)), id_clean=float(np.mean(g_clean)), dir=float(np.mean(dirs)),
                        dir_body=float(np.mean(dirs_body)), dir_junc=float(np.mean(dirs_j)), page=float(np.mean(pc)))
    return res


# ------------------------------------------------------------------ K3 A/B subject transfer
def k3(pages, seed=0, nrand=200):
    rng = random.Random(seed)
    def grp(p):
        if p['lang'] == 'A' and p['sec'] == 'H': return 'AH'
        if p['lang'] == 'A' and p['sec'] == 'P': return 'AP'
        if p['lang'] == 'B' and p['sec'] == 'H': return 'BH'
        if p['lang'] == 'B': return 'BO'
        return None
    toks = {p['id']: [w for l in p['lines'] for w in l['w']] for p in pages}
    G = {p['id']: grp(p) for p in pages}
    def prof(ids, f):
        c = Counter(); [c.update(f(w) for w in toks[i]) for i in ids]; return c
    def auc(pos, neg):
        if not pos or not neg: return float('nan')
        s = 0.0
        for x in pos:
            for y in neg: s += 1.0 if x > y else (0.5 if x == y else 0.0)
        return s / (len(pos) * len(neg))
    def score_pages(ids, f, cH, cP):
        nH = sum(cH.values()); nP = sum(cP.values()); V = len(set(cH) | set(cP)) + 1
        out = {}
        for i in ids:
            fs = [f(w) for w in toks[i]]
            out[i] = np.mean([math.log((cH[x] + .5) / (nH + .5 * V)) - math.log((cP[x] + .5) / (nP + .5 * V)) for x in fs])
        return out
    AH = [i for i, g in G.items() if g == 'AH']; AP = [i for i, g in G.items() if g == 'AP']
    BH = [i for i, g in G.items() if g == 'BH']; BO = [i for i, g in G.items() if g == 'BO']
    feats = dict(word=lambda w: w, first3=lambda w: w[:3], last3=lambda w: w[-3:])
    res = {}
    # transfer AUC on all B pages, per feature
    for fn, f in feats.items():
        cH, cP = prof(AH, f), prof(AP, f)
        sc = score_pages(BH + BO, f, cH, cP)
        res['xfer_' + fn] = auc([sc[i] for i in BH], [sc[i] for i in BO])
    # within-A control: A herbal vs A pharma, profiles from one leaf half, scored on the other
    for fn, f in feats.items():
        a = []
        for tr in (0, 1):
            trH = [i for i in AH if L.leaf_half(i) == tr]; trP = [i for i in AP if L.leaf_half(i) == tr]
            teH = [i for i in AH if L.leaf_half(i) != tr]; teP = [i for i in AP if L.leaf_half(i) != tr]
            cH, cP = prof(trH, f), prof(trP, f)
            sc = score_pages(teH + teP, f, cH, cP)
            a.append(auc([sc[i] for i in teH], [sc[i] for i in teP]))
        res['withinA_' + fn] = float(np.mean(a))
    # endings-only B->A maps: ending = last 2 glyph units of words with >= 3 units
    cH, cP = prof(AH, lambda w: w), prof(AP, lambda w: w)
    Aend = [e for e, _ in Counter(w[-2:] for i in AH + AP for w in toks[i] if len(w) >= 3).most_common(40)]
    Bend = [e for e, _ in Counter(w[-2:] for i in BH + BO for w in toks[i] if len(w) >= 3).most_common(30)]
    def mapf(m):
        return lambda w: (w[:-2] + m.get(w[-2:], w[-2:])) if len(w) >= 3 else w
    def xauc(ids_h, ids_o, m):
        f = mapf(m); s = score_pages(ids_h + ids_o, f, cH, cP)
        return auc([s[i] for i in ids_h], [s[i] for i in ids_o])
    learned, rand = [], []
    for tr in (0, 1):
        trH = [i for i in BH if L.leaf_half(i) == tr]; trO = [i for i in BO if L.leaf_half(i) == tr]
        teH = [i for i in BH if L.leaf_half(i) != tr]; teO = [i for i in BO if L.leaf_half(i) != tr]
        m = {}; best = xauc(trH, trO, m)
        for sweep in range(2):
            for e in Bend:
                cand = rng.sample(Aend, 12) + [e]
                for t in cand:
                    m2 = dict(m); m2[e] = t
                    v = xauc(trH, trO, m2)
                    if v > best + 1e-9: best, m = v, m2
        learned.append((xauc(teH, teO, {}), xauc(teH, teO, m), len([k for k in m if m[k] != k])))
        nch = max(1, len([k for k in m if m[k] != k]))
        rr = []
        for r in range(nrand):
            ks = rng.sample(Bend, min(nch, len(Bend)))
            rr.append(xauc(teH, teO, {k: rng.choice(Aend) for k in ks}))
        rand.append(rr)
    res['map_identity_te'] = float(np.mean([x[0] for x in learned]))
    res['map_learned_te'] = float(np.mean([x[1] for x in learned]))
    res['map_nchanged'] = [x[2] for x in learned]
    R = np.array(rand)
    res['map_rand_te_mean'] = float(R.mean()); res['map_rand_te_q95'] = float(np.quantile(R.mean(0), 0.95))
    res['map_pct'] = float(np.mean(R.mean(0) < res['map_learned_te']))
    return res


# ------------------------------------------------------------------ K4 re-rolled slots in recurring contexts
def edit1(x, y):
    """classify a single glyph-unit edit between x and y; None if distance != 1."""
    if x == y or abs(len(x) - len(y)) > 1: return None
    if len(x) == len(y):
        d = [i for i in range(len(x)) if x[i] != y[i]]
        if len(d) != 1: return None
        a, b = sorted((x[d[0]], y[d[0]]))
        if (a, b) == ('C', 'S'): return 'ch~sh'
        if {a, b} == {'e', 'd'}: return 'e~d'
        if {a, b} == {'T', 'd'}: return 'cth~d'
        if a in L.GALL and b in L.GALL: return 'gallows'
        if a in L.FRAME or b in L.FRAME: return 'frame'
        return 'other'
    s, t = (x, y) if len(x) < len(y) else (y, x)
    for i in range(len(t)):
        if t[:i] + t[i + 1:] == s:
            g = t[i]
            if g == 'e': return 'e-run'
            if g == 'q' and i == 0: return 'q-'
            if g in L.FRAME: return 'frame'
            if g in L.GALL: return 'gallows'
            return 'other'
    return None


PAD = {'ch~sh', 'e~d', 'cth~d', 'e-run', 'q-'}


def k4(pages):
    ctx = defaultdict(list); uni = Counter()
    for p, li, l in L.lines_of(pages):
        w = l['w']; uni.update(w)
        for i in range(1, len(w) - 1): ctx[(w[i - 1], w[i + 1])].append(w[i])
    c_in = Counter(); n_in = 0
    for k, xs in ctx.items():
        if len(xs) < 2: continue
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                e = edit1(xs[i], xs[j])
                if e: c_in[e] += 1; n_in += 1
    # context-free: all one-edit type pairs, weighted by token product (what random contexts would bring)
    types = [w for w, c in uni.items() if c >= 2]
    idx = defaultdict(list)
    for w in types:
        for i in range(len(w)): idx[(w[:i], w[i + 1:])].append(w)  # deletion / substitution keys
    c_free = Counter(); seen = set()
    for key, ws in idx.items():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                if (a, b) in seen: continue
                seen.add((a, b)); e = edit1(a, b)
                if e: c_free[e] += uni[a] * uni[b]
    tset = set(types)
    for w in types:  # insertion relation: w vs w minus one glyph
        for i in range(len(w)):
            v = w[:i] + w[i + 1:]
            if v in tset and (v, w) not in seen:
                seen.add((v, w)); e = edit1(v, w)
                if e: c_free[e] += uni[v] * uni[w]
    tin = sum(c_in.values()); tfr = sum(c_free.values())
    sh_in = {k: v / tin for k, v in c_in.items()}; sh_fr = {k: v / tfr for k, v in c_free.items()}
    return dict(n_in=n_in, pad_in=sum(v for k, v in sh_in.items() if k in PAD), pad_free=sum(v for k, v in sh_fr.items() if k in PAD),
                frame_in=sh_in.get('frame', 0), frame_free=sh_fr.get('frame', 0), cls_in=sh_in, cls_free=sh_fr)


# ------------------------------------------------------------------ K6 stem+l before stem+y
def lform(a, b):
    """a = stem + (l | ol | al), b = stem + y, stem non-empty (chol chy, dal dy, chedal chedy)."""
    if len(a) < 2 or a[-1] != 'l' or len(b) < 2 or b[-1] != 'y': return False
    s = b[:-1]
    return a in (s + 'l', s + 'ol', s + 'al')

def k6(pages, nsh=200, seed=0):
    rng = random.Random(seed)
    def count(P, half=None):
        ly = yl = 0
        for p, li, l in L.lines_of(P, half):
            w = l['w']
            for a, b in zip(w, w[1:]):
                if lform(a, b): ly += 1
                elif lform(b, a): yl += 1
        return ly, yl
    def colshuf(P):
        out = []
        for p in P:
            L_ = [list(l['w']) for l in p['lines']]
            mx = max(len(x) for x in L_)
            for k in range(mx):
                rows = [r for r in range(len(L_)) if len(L_[r]) > k]
                vals = [L_[r][k] for r in rows]; rng.shuffle(vals)
                for r, v in zip(rows, vals): L_[r][k] = v
            out.append(dict(p, lines=[dict(l, w=x) for l, x in zip(p['lines'], L_)]))
        return out
    res = {}
    for half in (None, 0, 1):
        ly, yl = count(pages, half)
        diffs = []
        for s in range(nsh if half is None else nsh // 2):
            a, b = count(colshuf(pages), half); diffs.append(a - b)
        diffs = np.array(diffs)
        res['h%s' % half] = dict(ly=ly, yl=yl, null_mean=float(diffs.mean()), null_sd=float(diffs.std()),
                                 z=float((ly - yl - diffs.mean()) / (diffs.std() + 1e-9)))
    return res


def job(arg):
    lab, gen, seed, name = arg
    P = L.voy(name) if gen is None else L.generate(name, gen, seed)
    out = dict(label=lab, k12=k1_k2(P), k4=k4(P), k6=k6(P, nsh=100 if gen else 200))
    if gen is None or gen in ('STACK', 'SELFCIT'):
        out['k3'] = k3(P)
    L.jsave('c1_%s.json' % lab.replace(':', '_'), out)
    log('done', lab)
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [('ZL3b', None, 0, 'ZL3b'), ('IT2a', None, 0, 'IT2a')]
    for g in ('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'):
        for s in (771, 772):
            jobs.append(('ZL:%s:%d' % (g, s), g, s, 'ZL3b'))
        jobs.append(('IT:%s:%d' % (g, 773), g, 773, 'IT2a'))
    with Pool(2) as pool:
        R = list(pool.imap_unordered(job, jobs))
    L.jsave('c1_all.json', R)
