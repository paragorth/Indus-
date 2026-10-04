"""v10 shared helpers: the line-initial glyph chain treated as a second text.

chain_paras(name) -> list of paragraphs; each paragraph is a dict
   {folio, fnum, quire, lang, hand, page_idx, para_idx_on_page, lines: [word lists], chain: [first glyph of each line]}
Only 'P' (paragraph) lines, uncertain words dropped (same as v6). Paragraph = para_start..para_end, broken at folio change.
Glyph units: vlib.glyphs (ch, sh, benched gallows merged).

Also: letter n-gram models for Latin / Italian / German, and a chain-text annealer (glyph type -> letter map)."""
import sys, os, math, random, re, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

U = vlib.glyphs
OUT = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'results', 'v10'); os.makedirs(CK, exist_ok=True)

def fnum(f):
    m = re.match(r'f(\d+)([rv])(\d*)', f)
    if not m: return (9999, 0, 0)
    return (int(m.group(1)), 0 if m.group(2) == 'r' else 1, int(m.group(3) or 0))

def chain_paras(name='ZL3b', minlen=2):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    paras, cur, fol = [], None, None
    def close():
        if cur and len(cur['lines']) >= minlen: paras.append(cur)
    for L in lines:
        if L['para_start'] or L['folio'] != fol or cur is None:
            close()
            cur = {'folio': L['folio'], 'quire': L.get('quire'), 'lang': L.get('lang'), 'hand': L.get('hand'),
                   'illus': L.get('illus'), 'lines': [], 'nums': []}
        fol = L['folio']; cur['lines'].append(list(L['words'])); cur['nums'].append(L['n'])
        if L.get('para_end'):
            close(); cur = None
    close()
    folios = sorted({p['folio'] for p in paras}, key=fnum)
    fi = {f: i for i, f in enumerate(folios)}
    cnt = Counter()
    for p in paras:
        p['page_idx'] = fi[p['folio']]; p['para_idx'] = cnt[p['folio']]; cnt[p['folio']] += 1
        p['chain'] = [U(l[0])[0] for l in p['lines']]
    # approximate bifolio: within a quire, leaf i pairs with leaf n-1-i (leaf = folio number)
    byq = defaultdict(set)
    for p in paras: byq[p['quire']].add(fnum(p['folio'])[0])
    bif = {}
    for q, leaves in byq.items():
        L = sorted(leaves); n = len(L)
        for i, x in enumerate(L): bif[(q, x)] = (q, min(i, n - 1 - i))
    for p in paras: p['bifolio'] = bif[(p['quire'], fnum(p['folio'])[0])]
    return paras

def mi(pairs):
    n = len(pairs)
    if n == 0: return 0.0
    a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in ab.items())

def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v)

def zstat(o, xs):
    m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** .5
    p = (1 + sum(x >= o for x in xs)) / (1 + len(xs))
    return {'obs': round(o, 4), 'null': round(m, 4), 'z': round((o - m) / sd, 2) if sd else 0.0, 'p': round(p, 4)}

def perm_body(paras, rng):
    """permute lines 2..n inside each paragraph (keeps the paragraph-first line, every row, every marginal)."""
    out = []
    for p in paras:
        q = dict(p); body = p['lines'][1:]; idx = rng.sample(range(len(body)), len(body))
        q['lines'] = [p['lines'][0]] + [body[i] for i in idx]
        q['chain'] = [p['chain'][0]] + [p['chain'][1:][i] for i in idx]   # permute the (possibly planted) chain with its lines
        out.append(q)
    return out

# ---------------- letter models ----------------
def letters_of(key):
    ws = vlib.words_of(vlib.load_ref(key))
    s = ''.join(ch for w in ws for ch in w)
    tr = str.maketrans('àáâäèéêëìíîïòóôöùúûüçñjvkwß', 'aaaaeeeeiiiioooouuuucniuccs')
    s = s.translate(tr)
    return ''.join(ch for ch in s if 'a' <= ch <= 'z')

LANGS = {'Latin': ('Latin-Caesar', 'Latin-Descartes'), 'Italian': ('Italian-Manzoni', 'Italian-Dante'),
         'German': ('German-Kafka',)}

class Tri:
    """add-k smoothed letter trigram model on a 26-letter alphabet (j->i, v->u, k->c, w->u, so ~22 used)."""
    def __init__(self, text, k=0.1):
        self.A = sorted(set(text)); a = len(self.A)
        c3 = Counter(text[i:i + 3] for i in range(len(text) - 2)); c2 = Counter(text[i:i + 2] for i in range(len(text) - 1))
        self.lp = {}
        for x in self.A:
            for y in self.A:
                d = c2[x + y] + k * a
                for z in self.A: self.lp[x + y + z] = math.log2((c3[x + y + z] + k) / d)
        c1 = Counter(text); n = len(text)
        self.l1 = {x: math.log2(c1[x] / n) for x in self.A}
        self.l2 = {x + y: math.log2((c2[x + y] + k) / (c1[x] + k * a)) for x in self.A for y in self.A}
    def score(self, s):
        if len(s) < 3: return sum(self.l1[c] for c in s)
        return self.l1[s[0]] + self.l2[s[:2]] + sum(self.lp[s[i:i + 3]] for i in range(len(s) - 2))

def models():
    M = {}
    for L, keys in LANGS.items():
        t = ''.join(letters_of(k) for k in keys)
        M[L] = (Tri(t[:-20000]), t[-20000:])
    return M

def anneal_map(seqs, model, iters=20000, seed=0, restarts=3):
    """seqs: list of symbol sequences (chain per paragraph). Find a map symbol -> letter (many-to-one allowed)
    maximising the summed trigram log-prob of the mapped sequences (each sequence scored separately).
    Works on symbol n-gram type counts, so cost per step ~ number of trigram types touching the changed symbol.
    Returns (bits per symbol, map)."""
    rng = random.Random(seed); syms = sorted({s for q in seqs for s in q}); A = model.A
    c1 = Counter(q[0] for q in seqs if q); c2 = Counter((q[0], q[1]) for q in seqs if len(q) > 1)
    c3 = Counter((q[i], q[i + 1], q[i + 2]) for q in seqs for i in range(len(q) - 2))
    n = sum(len(q) for q in seqs); lp, l1, l2 = model.lp, model.l1, model.l2
    touch = defaultdict(lambda: ([], [], []))
    for k in c1: touch[k[0] if isinstance(k, tuple) else k][0].append(k)
    for k in c2:
        for x in set(k): touch[x][1].append(k)
    for k in c3:
        for x in set(k): touch[x][2].append(k)
    def part(m, x):
        a, b, c = touch[x]
        return (sum(c1[k] * l1[m[k]] for k in a) + sum(c2[k] * l2[m[k[0]] + m[k[1]]] for k in b)
                + sum(c3[k] * lp[m[k[0]] + m[k[1]] + m[k[2]]] for k in c))
    def total(m):
        return (sum(v * l1[m[k]] for k, v in c1.items()) + sum(v * l2[m[a] + m[b]] for (a, b), v in c2.items())
                + sum(v * lp[m[a] + m[b] + m[c]] for (a, b, c), v in c3.items()))
    best_all = (-1e18, None)
    for r in range(restarts):
        m = {x: rng.choice(A) for x in syms}; cur = total(m); best = (cur, dict(m)); T0 = n / 50
        for it in range(iters):
            T = T0 * (1 - it / iters) ** 2 + 1e-3
            x = rng.choice(syms); old = m[x]; before = part(m, x); m[x] = rng.choice(A); d = part(m, x) - before
            if d >= 0 or rng.random() < math.exp(d / T): cur += d
            else: m[x] = old
            if cur > best[0]: best = (cur, dict(m))
        if best[0] > best_all[0]: best_all = best
    return best_all[0] / n, best_all[1]
