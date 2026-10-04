"""v37 EVERY LINE IS A RECORD: shared library.

Corpora are lists of pages {'id','sec','paras':[[line words...]...]} (v21 format; Voynich in glyph units,
one character per unit).  Controls:
  list texts with one record per line: Isidore Etym. X (glossary), Isidore XVII (plants/agriculture entries),
  Apicius De re coquinaria (Latin recipes), Forme of Cury (Middle English recipes, c.1390),
  Culpeper Complete Herbal (herb entries; fields Descript./Place/Time/Government);
  the same list texts as running text cut into 9-word lines (record boundaries ignored);
  running prose: Caesar and Descartes (Latin) cut into 9-word lines;
  Voynich nulls: within-line word shuffle, v26 evolved-scribe forgery; planted key->field schema.
"""
import os, sys, re, json, math, random
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v21_lib import voynich_pages

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v37_ckpt'); os.makedirs(CK, exist_ok=True)
SRC = os.path.join(CK, 'src')
MAXW = 11


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, indent=1, default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def norm(s):
    s = s.lower().replace('þ', 'th').replace('ȝ', 'y').replace('æ', 'ae').replace('œ', 'oe')
    return re.findall(r'[a-zàèéìíòóùú]+', s)


def pages_from_records(recs, per_page=24, maxw=MAXW, minw=4, sec='x'):
    """One record = one line (truncated to maxw words); consecutive records grouped into pages, one paragraph."""
    L = [r[:maxw] for r in recs if len(r) >= minw]
    out = []
    for i in range(0, len(L), per_page):
        chunk = L[i:i + per_page]
        if len(chunk) >= 6: out.append({'id': f'p{i // per_page}', 'sec': sec, 'paras': [chunk]})
    return out


def pages_cut(words, per_line=9, per_page=24, sec='x'):
    lines = [words[i:i + per_line] for i in range(0, len(words) - per_line + 1, per_line)]
    return [{'id': f'p{i // per_page}', 'sec': sec, 'paras': [lines[i:i + per_page]]}
            for i in range(0, len(lines), per_page) if len(lines[i:i + per_page]) >= 6]


# ------------------------------------------------------------------ list-text parsers
def isidore(book):
    d = json.load(open(os.path.join(vlib.DATA, 'derived', 'v19_isidore.json')))
    return [e for e in d[book] if e]


def apicius():
    raw = open(os.path.join(SRC, 'pg16439.txt'), encoding='utf-8').read().split('\n')
    keep = []
    for k, ln in enumerate(raw):
        if k < 240: continue
        if '*** END' in ln: break
        if ln.startswith('    '): continue          # critical apparatus
        if len(re.findall(r'\b[IVXL]+\. ', ln)) >= 2: continue   # chapter index lines
        ln = re.sub(r'\s{3,}\d+\s*$', '', ln)        # margin line numbers
        keep.append(ln)
    t = ' '.join(keep)
    recs = []
    for p in re.split(r'\(\d+\)', t)[1:]:
        p = re.sub(r'\{[^}]*\}|\+\d+\+|\b\d+\.', ' ', p)
        w = norm(p)
        if len(w) >= 4: recs.append(w)
    return recs


def cury():
    t = open(os.path.join(SRC, 'pg8102.txt'), encoding='utf-8').read()
    i = t.find('FOR TO MAKE GRONDEN BENES')
    j = t.find('*** END')
    body = t[i:j]
    recs = []
    for block in re.split(r'\n\s*\n', body):
        bl = block.strip()
        if not bl or bl.startswith('['): continue
        lines = bl.split('\n')
        if re.match(r'^[A-Z0-9 ,.\'\-\[\]&]+\.?\s*$', lines[0]) and len(lines) == 1:
            recs.append(('T', lines[0])); continue
        if lines[0][:1].isupper() and not bl.startswith('['):
            recs.append(('B', bl))
    out = []
    title = None
    for k, s in recs:
        if k == 'T': title = s; continue
        if title is None: continue
        tw = norm(re.sub(r'\[\d+\]|\b[IVXLC]+\.?\s*$', ' ', title))
        bw = norm(re.sub(r'\[\d+\]', ' ', s))
        if bw and tw: out.append(tw + bw)
        title = None
    return out


def culpeper():
    t = open(os.path.join(SRC, 'pg49513.txt'), encoding='utf-8').read()
    i = t.find('ADDER’S TONGUE OR SERPENT'); j = t.find('*** END')
    out = []
    for block in re.split(r'\n\s*\n', t[i:j]):
        if not block.startswith('_'): continue
        w = norm(block.replace('\n', ' '))
        if len(w) >= 4: out.append(w)
    return out


def caesar_words(n=30000):
    return [w for L in vlib.load_ref('Latin-Caesar') for w in L['words']][:n]


def descartes_words(n=30000):
    return [w for L in vlib.load_ref('Latin-Descartes') for w in L['words']][:n]


def flat(recs):
    return [w for r in recs for w in r]


# ------------------------------------------------------------------ Voynich and nulls
def voy(name='ZL3b'):
    return voynich_pages(name)


def forge_v26(C, seed=1):
    import v26_lib
    g = json.load(open(os.path.join(vlib.DATA, 'v26_ckpt', 'eval_V.json')))['winner']
    S = v26_lib.Scribe(C)
    return S.forge(C, g, random.Random(seed))


def shuffle_within(C, rng, keep_first=False):
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                l = list(l)
                if keep_first:
                    r = l[1:]; rng.shuffle(r); l = l[:1] + r
                else: rng.shuffle(l)
                npa.append(l)
            q['paras'].append(npa)
        out.append(q)
    return out


def plant_schema(C, rng, rho=0.5, K=6, pos=(3, 4), by_prefix=False):
    """Key = first unit of the line's first word (top K-1 + other); with prob rho per line, the word at a
    random field position in pos is replaced by a word drawn from the key's own vocabulary block
    (mid-line vocabulary split at random into K blocks; same frequency profile)."""
    first = Counter(l[0][0] for p in C for pa in p['paras'] for l in pa if l)
    top = [u for u, _ in first.most_common(K - 1)]
    kid = lambda w: top.index(w[0]) if w[0] in top else K - 1
    mid = Counter(w for p in C for pa in p['paras'] for l in pa for w in l[1:-1])
    vocab = [w for w, c in mid.items() if c >= 2]
    rng.shuffle(vocab)
    blocks = [vocab[k::K] for k in range(K)]
    if by_prefix:   # blocks = words whose 2-unit prefix falls in a random prefix group (visible to prefix features)
        pre = sorted({w[:2] for w in vocab}); rng.shuffle(pre); grp = {p: i % K for i, p in enumerate(pre)}
        blocks = [[w for w in vocab if grp[w[:2]] == k] for k in range(K)]
    bw = [[mid[w] for w in b] for b in blocks]
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                l = list(l)
                if len(l) > max(pos) + 1 and rng.random() < rho:
                    k = kid(l[0]); j = rng.choice(pos)
                    l[j] = rng.choices(blocks[k], weights=bw[k])[0]
                npa.append(l)
            q['paras'].append(npa)
        out.append(q)
    return out


def split_half(C, seed=37):
    rng = random.Random(seed)
    bysec = defaultdict(list)
    for i, p in enumerate(C): bysec[p['sec']].append(i)
    tr = set()
    for s, ix in sorted(bysec.items()):
        rng.shuffle(ix); tr |= set(ix[:(len(ix) + 1) // 2])
    return [C[i] for i in range(len(C)) if i in tr], [C[i] for i in range(len(C)) if i not in tr]


def control_corpora():
    D = {}
    D['ISX'] = pages_from_records(isidore('X'))
    D['APIC'] = pages_from_records(apicius())
    D['CURY'] = pages_from_records(cury())
    D['CULP'] = pages_from_records(culpeper())
    D['ISX_cut'] = pages_cut(flat(isidore('X')))
    D['APIC_cut'] = pages_cut(flat(apicius()))
    D['CURY_cut'] = pages_cut(flat(cury()))
    D['CULP_cut'] = pages_cut(flat(culpeper())[:22000])
    D['CAES'] = pages_cut(caesar_words())
    D['DESC'] = pages_cut(descartes_words(32000)[2000:])
    return D


def lines_of(C, body_only=False):
    out = []
    for p in C:
        for pa in p['paras']:
            for i, l in enumerate(pa):
                if body_only and i == 0 and len(pa) > 1: continue
                out.append(l)
    return out


# ------------------------------------------------------------------ word classes
class Classer:
    """Joint word class: the word itself if among the top NTOP words, else first unit (top NF) x last unit
    (top NL)."""

    def __init__(self, C, ntop=40, nf=10, nl=5):
        cnt = Counter(w for l in lines_of(C) for w in l)
        self.top = {w for w, _ in cnt.most_common(ntop)}
        fc = Counter(w[0] for w in cnt.elements()); lc = Counter(w[-1] for w in cnt.elements())
        self.F = [u for u, _ in fc.most_common(nf)]; self.L = [u for u, _ in lc.most_common(nl)]
        self.idx = {}

    def __call__(self, w):
        if w in self.top: k = 'W' + w
        else:
            f = w[0] if w[0] in self.F else '#'; l = w[-1] if w[-1] in self.L else '#'
            k = f + '_' + l
        i = self.idx.get(k)
        if i is None: i = self.idx[k] = len(self.idx)
        return i


# ------------------------------------------------------------------ key -> value coupling (cycle 1)
def line_units(C, body_only=True):
    """(page index, paragraph index, line) for all lines; body_only drops paragraph-opening lines."""
    out = []
    for pi, p in enumerate(C):
        for ai, pa in enumerate(p['paras']):
            for i, l in enumerate(pa):
                if body_only and i == 0 and len(pa) > 1: continue
                out.append((pi, ai, l))
    return out


def anchors_for(n, rng):
    if n < 6: return None
    return {'KEY': 0, 'MID': rng.randint(2, n - 3), 'END': n - 1}


def targets(n, a):
    return [j for j in range(1, n - 1) if abs(j - a) >= 2]


def coupling(C, seed=0, alpha=200.0, body_only=True, anchor='joint', nboot=300, nswap=4):
    """Held-out predictive gain (millibits per target word) of target word classes given the anchor word's class,
    two-fold over pages. Targets: interior words at distance >= 2 from the anchor (junction excluded).
    Anchors: KEY = first word, MID = a random interior word, END = last word. Model P(target class | anchor
    class, anchor kind, distance bucket) vs P(target class | anchor kind, distance bucket).
    Swap null: same anchor position taken from other lines of the same paragraph (page if < 4 lines);
    line-specific coupling = real - swap. KEYNESS = LS(KEY) - LS(MID)."""
    rng = random.Random(seed)
    tr, te = split_half(C, seed=37 + seed)
    per = defaultdict(lambda: defaultdict(list))     # page -> key -> per-line mean gains
    def db(x): return min(abs(x), 5)
    for fold, (A, B) in enumerate(((tr, te), (te, tr))):
        cl = Classer(A)
        def acls(w): return cl(w) if anchor == 'joint' else w[0]
        cnt = defaultdict(Counter); marg = defaultdict(Counter)
        for pi, ai, l in line_units(A, body_only):
            P = anchors_for(len(l), rng)
            if not P: continue
            for k, a in P.items():
                ca = acls(l[a])
                for t in targets(len(l), a):
                    ct = cl(l[t]); cnt[(k, db(t - a), ca)][ct] += 1; marg[(k, db(t - a))][ct] += 1
        ncls = len(cl.idx) + 1
        mt = {key: sum(m.values()) for key, m in marg.items()}
        ct_ = {key: sum(c.values()) for key, c in cnt.items()}
        def lp(k, d, ca, ct):
            m = marg[(k, d)]
            p0 = (m[ct] + 0.5) / (mt.get((k, d), 0) + 0.5 * ncls)
            c = cnt.get((k, d, ca)); n = ct_.get((k, d, ca), 0)
            p = ((c[ct] if c else 0) + alpha * p0) / (n + alpha)
            return math.log2(p / p0)
        U = line_units(B, body_only)
        bypara = defaultdict(list); bypage = defaultdict(list)
        for j, (pi, ai, l) in enumerate(U):
            if len(l) >= 6: bypara[(pi, ai)].append(j); bypage[pi].append(j)
        for j, (pi, ai, l) in enumerate(U):
            P = anchors_for(len(l), rng)
            if not P: continue
            pool = [x for x in bypara[(pi, ai)] if x != j]
            if len(pool) < 3: pool = [x for x in bypage[pi] if x != j]
            if not pool: continue
            partners = rng.sample(pool, min(nswap, len(pool)))
            for k, a in P.items():
                T = targets(len(l), a)
                g = np.mean([lp(k, db(t - a), acls(l[a]), cl(l[t])) for t in T])
                sw = []
                for x in partners:
                    o = U[x][2]
                    aa = 0 if k == 'KEY' else (len(o) - 1 if k == 'END' else min(a, len(o) - 3))
                    sw.append(np.mean([lp(k, db(t - a), acls(o[aa]), cl(l[t])) for t in T]))
                per[(fold, pi)][k].append(g); per[(fold, pi)][k + '_sw'].append(np.mean(sw))
    pages = sorted(per)
    def stat(sample):
        m = {}
        for k in ('KEY', 'MID', 'END'):
            r = [g for p in sample for g in per[p][k]]; s = [g for p in sample for g in per[p][k + '_sw']]
            m[k] = 1000 * (np.mean(r) - np.mean(s)); m[k + '_raw'] = 1000 * np.mean(r)
        m['KEYNESS'] = m['KEY'] - m['MID']; m['ENDNESS'] = m['END'] - m['MID']
        return m
    out = stat(pages)
    brng = np.random.default_rng(seed)
    bs = defaultdict(list)
    for b in range(nboot):
        m = stat([pages[i] for i in brng.integers(0, len(pages), len(pages))])
        for k, v in m.items(): bs[k].append(v)
    out['se'] = {k: float(np.std(v)) for k, v in bs.items()}
    out['n_lines'] = sum(len(per[p]['KEY']) for p in pages)
    return out


# ------------------------------------------------------------------ field automaton (cycle 2)
def encode(lines, cl, maxlen=16):
    L = [l[:maxlen] for l in lines if len(l) >= 3]
    N = len(L); T = max(len(l) for l in L)
    X = np.zeros((N, T), np.int64); M = np.zeros((N, T), bool)
    for i, l in enumerate(L):
        for t, w in enumerate(l): X[i, t] = cl(w); M[i, t] = True
    return X, M


def fb(X, M, pi, A, B):
    """Scaled forward-backward on a padded batch. Returns loglik per line, gamma, xi-sum."""
    N, T = X.shape; k = len(pi)
    E = B[:, X].transpose(1, 2, 0)            # N,T,k
    al = np.zeros((N, T, k)); c = np.ones((N, T))
    a = pi[None, :] * E[:, 0]; c[:, 0] = a.sum(1); al[:, 0] = a / c[:, 0:1]
    for t in range(1, T):
        a = (al[:, t - 1] @ A) * E[:, t]
        s = a.sum(1); s = np.where(M[:, t], s, 1.0)
        a = np.where(M[:, t, None], a / s[:, None], al[:, t - 1])
        al[:, t] = a; c[:, t] = s
    ll = np.log(c).sum(1)
    be = np.ones((N, T, k))
    for t in range(T - 2, -1, -1):
        b = (be[:, t + 1] * E[:, t + 1]) @ A.T / c[:, t + 1:t + 2]
        be[:, t] = np.where(M[:, t + 1, None], b, 1.0)
    g = al * be; g /= g.sum(2, keepdims=True); g *= M[:, :, None]
    xi = np.zeros((k, k))
    for t in range(T - 1):
        m = M[:, t + 1]
        if not m.any(): continue
        x = al[m, t][:, :, None] * A[None] * (E[m, t + 1] * be[m, t + 1])[:, None, :] / c[m, t + 1][:, None, None]
        xi += x.sum(0)
    return ll, g, xi


def fit_hmm(X, M, V, k, rng, lr=True, iters=25):
    pi = rng.dirichlet(np.ones(k)); A = rng.dirichlet(np.ones(k), k); B = rng.dirichlet(np.ones(V), k)
    mask = np.triu(np.ones((k, k))) if lr else np.ones((k, k))
    if lr: pi = np.zeros(k); pi[0] = 0.7; pi[1:] = 0.3 / max(k - 1, 1)
    A = A * mask; A /= A.sum(1, keepdims=True)
    for it in range(iters):
        ll, g, xi = fb(X, M, pi, A, B)
        pi = g[:, 0].sum(0) + 1e-3; pi /= pi.sum()
        A = (xi + 1e-3) * mask; A /= A.sum(1, keepdims=True)
        Bc = np.zeros((k, V)) + 0.5
        for s in range(k): np.add.at(Bc[s], X[M], g[:, :, s][M])
        B = Bc / Bc.sum(1, keepdims=True)
    ll, _, _ = fb(X, M, pi, A, B)
    return ll.sum(), (pi, A, B)


def posbin(X, M, Xt, Mt, V):
    def bins(Mm):
        n = Mm.sum(1); T = Mm.shape[1]; t = np.arange(T)[None, :]
        b = np.full(Mm.shape, 3); b[:, 0] = 0
        b = np.where(t == 1, 1, b); b = np.where(t == 2, 2, b)
        b = np.where(t == n[:, None] - 2, 4, b); b = np.where(t == n[:, None] - 1, 5, b)
        return b
    b = bins(M); Bc = np.zeros((6, V)) + 0.5
    np.add.at(Bc, (b[M], X[M]), 1); Bc /= Bc.sum(1, keepdims=True)
    bt = bins(Mt)
    return np.log(Bc[bt[Mt], Xt[Mt]]).sum()


def field_scores(C, seed=0, ks=(2, 3, 4, 6), restarts=8, body_only=True):
    """Held-out log-likelihood (nats -> millibits/word gains over unigram) of: position bins, left-to-right
    field automaton (k states, forward jumps allowed), ergodic HMM; two folds."""
    rng = np.random.default_rng(seed)
    tr, te = split_half(C, seed=37 + seed)
    tot = defaultdict(float); nw = 0
    for A_, B_ in ((tr, te), (te, tr)):
        cl = Classer(A_)
        la = [l for _, _, l in line_units(A_, body_only)]; lb = [l for _, _, l in line_units(B_, body_only)]
        X, M = encode(la, cl); Xt, Mt = encode(lb, cl)
        V = len(cl.idx) + 1
        uni = np.bincount(X[M], minlength=V) + 0.5; uni = uni / uni.sum()
        u = np.log(uni[Xt[Mt]]).sum(); nw += Mt.sum()
        tot['posbin'] += posbin(X, M, Xt, Mt, V) - u
        for k in ks:
            for lr in (True, False):
                best = None
                for r in range(restarts):
                    ll, par = fit_hmm(X, M, V, k, rng, lr=lr)
                    if best is None or ll > best[0]: best = (ll, par)
                llt, _, _ = fb(Xt, Mt, *best[1])
                tot[f'{"LR" if lr else "ER"}{k}'] += llt.sum() - u
    return {k: 1000 * v / nw / math.log(2) for k, v in tot.items()} | {'n_words': int(nw)}


# ------------------------------------------------------------------ lexical echo (key word re-used in its own line)
def echo(C, seed=0, body_only=True, nswap=6, nboot=300, pre=3):
    """Rate at which an interior target word (distance >= 2 from the anchor) shares its first `pre` units with
    the anchor word (or is identical), real minus swap (anchor from other lines of the paragraph/page).
    Per-line rates, page bootstrap. A catalogue whose key is restated in the value (glossary 'alumnus ab
    alendo', recipe 'caboches ... take caboches') shows KEY > MID."""
    rng = random.Random(seed)
    U = line_units(C, body_only)
    bypara = defaultdict(list); bypage = defaultdict(list)
    for j, (pi, ai, l) in enumerate(U):
        if len(l) >= 6: bypara[(pi, ai)].append(j); bypage[pi].append(j)
    def sim(a, b): return float(a == b or (len(a) >= pre and len(b) >= pre and a[:pre] == b[:pre]))
    per = defaultdict(lambda: defaultdict(list))
    for j, (pi, ai, l) in enumerate(U):
        P = anchors_for(len(l), rng)
        if not P: continue
        pool = [x for x in bypara[(pi, ai)] if x != j]
        if len(pool) < 3: pool = [x for x in bypage[pi] if x != j]
        if not pool: continue
        partners = rng.sample(pool, min(nswap, len(pool)))
        for k, a in P.items():
            T = targets(len(l), a)
            per[pi][k].append(np.mean([sim(l[a], l[t]) for t in T]))
            sw = []
            for x in partners:
                o = U[x][2]; aa = 0 if k == 'KEY' else (len(o) - 1 if k == 'END' else min(a, len(o) - 3))
                sw.append(np.mean([sim(o[aa], l[t]) for t in T]))
            per[pi][k + '_sw'].append(np.mean(sw))
    pages = sorted(per)
    def stat(sample):
        m = {}
        for k in ('KEY', 'MID', 'END'):
            r = [g for p in sample for g in per[p][k]]; s = [g for p in sample for g in per[p][k + '_sw']]
            m[k] = 1000 * (np.mean(r) - np.mean(s)); m[k + '_raw'] = 1000 * np.mean(r)
        m['KEYNESS'] = m['KEY'] - m['MID']; m['ENDNESS'] = m['END'] - m['MID']
        return m
    out = stat(pages); brng = np.random.default_rng(seed); bs = defaultdict(list)
    for b in range(nboot):
        m = stat([pages[i] for i in brng.integers(0, len(pages), len(pages))])
        for k, v in m.items(): bs[k].append(v)
    out['se'] = {k: float(np.std(v)) for k, v in bs.items()}
    return out
