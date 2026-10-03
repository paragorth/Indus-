"""Arrow-in-the-dark loop 5: signs as NUMBERS. Shared engine.
Texts from data/raw/inscriptions.csv (reading order from 'dir.'), merge levels from the corpus
(seq_raw / seq_strong / seq_all maps), numeral values from S204 (strat_numseries.py).
Annealer assigns integer values to the 60 commonest non-numeral signs (+1 'other' slot) so a
text's combined value (sum / product / positional base-b) rank-correlates with an outside quantity.
"""
import csv, json, re, math, random, collections
import numpy as np
from scipy.stats import spearmanr

SHORT = (set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30))) - {2}
TALL = set(range(32, 38)) | {39}
VAL = {**{i: i for i in range(1, 8)}, **{i: i - 10 for i in range(12, 21)},
       **{i: i - 20 for i in range(25, 30)}, **{i: i - 30 for i in range(32, 38)}, 39: 9}
NUMERALS = SHORT | TALL
EXCL = {2, 31}          # opener marker / frame marker, never valued
TRAIN_SITES = {'Mohenjo-daro', 'Harappa'}

def load_merge_maps():
    C = json.load(open('data/derived/merged-corpus-canonical.json'))
    maps = {'seq_raw': {}, 'seq_strong': {}, 'seq_all': {}}
    for x in C:
        for k in ('seq_strong', 'seq_all'):
            for a, b in zip(x['seq_raw'], x[k]):
                maps[k][a] = b
    return maps

def parse_text(t):
    toks = t.strip('+').split('-')
    out = []
    for tk in toks:
        tk = tk.strip()
        if tk == '' or tk in ('0', '000', '00'):
            return None            # lost / illegible sign: drop text
        if '/' in tk or not tk.isdigit():
            return None            # ambiguous reading
        out.append(int(tk))
    return out

def mm(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except Exception:
        return None

def load_rows():
    rows = list(csv.DictReader(open('data/raw/inscriptions.csv')))
    objs = []
    for r in rows:
        s = parse_text(r['text'])
        if not s:
            continue
        if r['dir.'].strip() == 'R/L':
            s = s[::-1]
        objs.append(dict(id=r['id'], obj=r['id'].split('.')[0], site=r['site'], type=r['type'],
                         area=r['area-section'], seq_raw=s,
                         h=mm(r['horizontal(mm)']), v=mm(r['vertical(mm)']), th=mm(r['thickness(mm)'])))
    maps = load_merge_maps()
    for o in objs:
        for k in ('seq_strong', 'seq_all'):
            o[k] = [maps[k].get(a, a) for a in o['seq_raw']]
    return objs

def count_face(seq):
    """Harappa voucher face: one tall numeral, optionally W700. Returns count or None."""
    nums = [a for a in seq if a in TALL]
    rest = [a for a in seq if a not in TALL]
    if len(nums) == 1 and rest in ([], [700]):
        return VAL[nums[0]]
    return None

def top_signs(seqs, n=60):
    c = collections.Counter(a for s in seqs for a in s if a not in NUMERALS and a not in EXCL)
    return [a for a, _ in c.most_common(n)]

class Problem:
    """Texts as count matrices over K slots (60 signs + OTHER); numerals contribute fixed values."""
    def __init__(self, seqs, signs, rule='sum', base=10):
        self.signs = signs; self.K = len(signs) + 1; self.rule = rule; self.base = base
        idx = {a: i for i, a in enumerate(signs)}
        n = len(seqs)
        self.M = np.zeros((n, self.K))            # slot counts (sum) or positional weights
        self.fixed = np.zeros(n)                  # numeral contribution
        self.lens = np.array([len(s) for s in seqs], float)
        for t, s in enumerate(seqs):
            L = len(s)
            for p, a in enumerate(s):
                w = 1.0 if rule != 'pos' else float(base) ** (L - 1 - p)
                if a in NUMERALS:
                    self.fixed[t] += w * (VAL[a] if rule != 'prod' else math.log(VAL[a]))
                elif a in EXCL:
                    continue
                else:
                    self.M[t, idx.get(a, self.K - 1)] += w
    def values(self, assign):
        v = np.asarray(assign, float)
        if self.rule == 'prod':
            v = np.log(np.maximum(v, 1e-9))
        return self.M @ v + self.fixed

def rank(x):
    o = np.argsort(x, kind='stable'); r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float); return r

def fast_spearman(vals, target_rank):
    r = rank(vals)
    if r.std() == 0: return 0.0
    return float(np.corrcoef(r, target_rank)[0, 1])

def anneal(prob, target, vmax=12, steps=4000, seed=0, T0=0.05):
    rng = random.Random(seed)
    K = prob.K
    assign = [rng.randint(1, vmax) for _ in range(K)]
    tr = rank(target)
    cur = prob.values(assign); score = fast_spearman(cur, tr); best = (score, list(assign))
    for it in range(steps):
        T = T0 * (1 - it / steps) + 1e-4
        k = rng.randrange(K); old = assign[k]; new = rng.randint(1, vmax)
        if new == old: continue
        assign[k] = new
        if prob.rule == 'prod':
            delta = prob.M[:, k] * (math.log(new) - math.log(old))
        else:
            delta = prob.M[:, k] * (new - old)
        nv = cur + delta; ns = fast_spearman(nv, tr)
        if ns >= score or rng.random() < math.exp((ns - score) / T):
            cur = nv; score = ns
            if score > best[0]: best = (score, list(assign))
        else:
            assign[k] = old
    return best

def run_arrow(seqs_tr, y_tr, seqs_te, y_te, signs, rule, restarts=30, vmax=12, steps=4000, seed0=0, base=10):
    """Returns dict: train/test score of best-train restart, max test over restarts, stability, assignments."""
    ptr = Problem(seqs_tr, signs, rule, base); pte = Problem(seqs_te, signs, rule, base)
    runs = []
    for r in range(restarts):
        sc, a = anneal(ptr, y_tr, vmax, steps, seed=seed0 + r)
        te = spearmanr(pte.values(a), y_te).correlation if len(y_te) > 3 else float('nan')
        runs.append((sc, te, a))
    runs.sort(key=lambda z: -z[0])
    top = runs[:10]
    stab = []
    for k in range(len(signs)):
        c = collections.Counter(a[k] for _, _, a in top)
        stab.append(c.most_common(1)[0][1] / len(top))
    stab = np.array(stab)
    return dict(train=runs[0][0], test=runs[0][1], test_max=max(z[1] for z in runs),
                test_mean=float(np.mean([z[1] for z in runs])), stable_frac=float((stab > 0.8).mean()),
                stable_signs=[signs[k] for k in range(len(signs)) if stab[k] > 0.8], best_assign=runs[0][2])

def fixed_score(seqs, y, signs, values_by_sign, rule, base=10):
    """Fixed hypothesis: value = given per-sign number (e.g. stroke count)."""
    prob = Problem(seqs, signs, rule, base)
    assign = [values_by_sign.get(a, 1) for a in signs] + [1]
    r = spearmanr(prob.values(assign), y)
    return r.correlation, r.pvalue

# ---------- stroke-count estimate from the lipi font (Zhang-Suen thinning, pure numpy) ----------
def thin(a):
    a = a.copy().astype(np.uint8)
    def nb(a):
        P = np.pad(a, 1)
        return [P[0:-2, 1:-1], P[0:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, 0:-2], P[1:-1, 0:-2], P[0:-2, 0:-2]]
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = nb(a)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(int) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (a == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                a[m] = 0; changed = True
        if not changed:
            return a

def stroke_count(bitmap):
    """Approximate strokes = (endpoints + sum over junction pixels of (degree-2)) / 2, min 1 per component."""
    sk = thin(bitmap)
    P = np.pad(sk, 1)
    p2, p3, p4, p5, p6, p7, p8, p9 = [P[0:-2, 1:-1], P[0:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, 0:-2], P[1:-1, 0:-2], P[0:-2, 0:-2]]
    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    cn = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(int) for i in range(8)) * sk   # crossing number
    ends = int((cn == 1).sum()); junc = int(np.maximum(cn - 2, 0)[sk == 1].sum())
    from scipy.ndimage import label
    ncomp = label(sk, structure=np.ones((3, 3)))[1]
    est = (ends + junc) / 2.0
    return max(ncomp, round(est)) if ncomp else 0

def length_residual(seqs, y):
    """y minus the mean y of texts with the same sign count (pooled), to remove the length confound."""
    L = np.array([len(s) for s in seqs]); y = np.asarray(y, float); out = y.copy()
    for l in np.unique(L):
        m = L == l; out[m] = y[m] - y[m].mean()
    return out

def stroke_table(signs):
    import sys; sys.path.insert(0, 'tools')
    from sign_glyphs import render
    out = {}
    for w in signs:
        b = render(w)
        if b is None: continue
        out[w] = stroke_count(b)
    return out
