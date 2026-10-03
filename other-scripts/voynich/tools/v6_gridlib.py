"""v6 'the page is a grid' tests: shared helpers.

A paragraph is a ragged grid: grid[r][c] = word c of line r. Reading orders are orderings of the grid cells.
Predictability along an order = mutual information between features of consecutive cells (excess over nulls).
Nulls: 'full'  = all words of the paragraph permuted across cells (kills every structure, keeps paragraph content);
       'line'  = lines permuted within the paragraph (keeps each row intact, kills vertical alignment);
       'inrow' = words permuted within each line (keeps row content, kills horizontal order)."""
import sys, os, math, random, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, attack_h2_bpe

# ---------------- corpora ----------------
def voynich_paragraphs(name='ZL3b', lang=None):
    lines = vlib.load_voynich(name, drop_uncertain=True)
    paras, cur, fol = [], [], None
    for L in lines:
        if lang and L.get('lang') != lang:
            continue
        if L['para_start'] or L['folio'] != fol:
            if cur: paras.append(cur)
            cur = []
        fol = L['folio']
        cur.append(list(L['words']))
        if L.get('para_end'):
            paras.append(cur); cur = []
    if cur: paras.append(cur)
    return [p for p in paras if len(p) >= 2]

def units_voy(w): return vlib.glyphs(w)
def units_plain(w): return list(w)

def latin_stream():
    ws = []
    for k in ('Latin-Caesar', 'Latin-Descartes'):
        ws += vlib.words_of(vlib.load_ref(k))
    return ws

def encode_verbose(words, seed=7):
    """fixed verbose substitution, each letter -> 1-3 symbols from 12 (lower-case so units = chars)."""
    return [w.lower() for w in attack_h2_bpe.verbose_encrypt(words, seed)]

def shapes_of(paras): return [[len(l) for l in p] for p in paras]

def cells_of(shape): return [(r, c) for r, n in enumerate(shape) for c in range(n)]

# ---------------- reading orders ----------------
def spiral(shape):
    R = len(shape); C = max(shape); have = set(cells_of(shape))
    top, bot, left, right = 0, R - 1, 0, C - 1; out = []
    while top <= bot and left <= right:
        for c in range(left, right + 1): out.append((top, c))
        for r in range(top + 1, bot + 1): out.append((r, right))
        if top < bot:
            for c in range(right - 1, left - 1, -1): out.append((bot, c))
        if left < right:
            for r in range(bot - 1, top, -1): out.append((r, left))
        top += 1; bot -= 1; left += 1; right -= 1
    return [x for x in out if x in have]

ORDERS = {
    'row':      lambda s: sorted(cells_of(s)),
    'col':      lambda s: sorted(cells_of(s), key=lambda x: (x[1], x[0])),
    'diag_dr':  lambda s: sorted(cells_of(s), key=lambda x: (x[1] - x[0], x[0])),   # down-right diagonals
    'diag_dl':  lambda s: sorted(cells_of(s), key=lambda x: (x[0] + x[1], x[0])),   # down-left (anti-)diagonals
    'spiral':   spiral,
    'boustro':  lambda s: [(r, c) for r, n in enumerate(s) for c in (range(n) if r % 2 == 0 else range(n - 1, -1, -1))],
}

def fill(shape, words, order):
    """write a word stream into a grid along a reading order."""
    g = [[None] * n for n in shape]
    for (r, c), w in zip(ORDERS[order](shape), words):
        g[r][c] = w
    return g

def build_control(shapes, stream, order):
    out, i = [], 0
    for s in shapes:
        n = sum(s); g = fill(s, stream[i:i + n], order); i += n
        out.append(g)
    return out

# ---------------- features & MI ----------------
def make_features(paras, units, min_word=10):
    cnt = Counter(w for p in paras for l in p for w in l)
    def F(w):
        u = units(w)
        return {'J1': u[-1], 'J2': u[0], 'W': w if cnt[w] >= min_word else '*' + u[0] + u[-1],
                'P2': ''.join(u[:2]), 'S2': ''.join(u[-2:]), 'id': w}
    return F

def mi(pairs):
    n = len(pairs)
    if n == 0: return 0.0
    a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in ab.items())

FEATS = ['junction', 'first-first', 'last-last', 'word', 'same-word', 'same-prefix2', 'same-suffix2']

def stats_of_pairs(P):
    """P = list of (featdict_a, featdict_b)."""
    n = max(len(P), 1)
    return {
        'junction': mi([(a['J1'], b['J2']) for a, b in P]),
        'first-first': mi([(a['J2'], b['J2']) for a, b in P]),
        'last-last': mi([(a['J1'], b['J1']) for a, b in P]),
        'word': mi([(a['W'], b['W']) for a, b in P]),
        'same-word': sum(a['id'] == b['id'] for a, b in P) / n,
        'same-prefix2': sum(a['P2'] == b['P2'] for a, b in P) / n,
        'same-suffix2': sum(a['S2'] == b['S2'] for a, b in P) / n,
    }

def order_pairs(paras, order, F, skip_first_line=False):
    P = []
    for g in paras:
        shape = [len(l) for l in g]
        seq = ORDERS[order](shape)
        if skip_first_line: seq = [x for x in seq if x[0] > 0]
        ws = [g[r][c] for r, c in seq]
        P += [(F(a), F(b)) for a, b in zip(ws, ws[1:])]
    return P

def null_grid(g, kind, rng):
    if kind == 'full':
        ws = [w for l in g for w in l]; rng.shuffle(ws); out, i = [], 0
        for l in g: out.append(ws[i:i + len(l)]); i += len(l)
        return out
    if kind == 'line':
        out = [list(l) for l in g]; rng.shuffle(out); return out
    if kind == 'inrow':
        return [rng.sample(l, len(l)) for l in g]

def compare_orders(paras, F, orders=tuple(ORDERS), nulls=('full', 'line', 'inrow'), reps=30, seed=0):
    res = {}
    for o in orders:
        obs = stats_of_pairs(order_pairs(paras, o, F)); res[o] = {'obs': obs}
        for k in nulls:
            sims = []
            for s in range(reps):
                rng = random.Random(seed * 1000 + s)
                sims.append(stats_of_pairs(order_pairs([null_grid(g, k, rng) for g in paras], o, F)))
            d = {}
            for f in FEATS:
                xs = [x[f] for x in sims]; m = sum(xs) / len(xs)
                sd = (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** .5
                d[f] = {'ex': obs[f] - m, 'z': (obs[f] - m) / sd if sd > 0 else 0.0}
            res[o][k] = d
    return res
