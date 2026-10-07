"""v96: TOPIC OR KEY?  Shared helpers.

v92 found that the frames recurring on a Voynich page can be put back on their page from the spelling of the page's
other words (0.80-0.82, above every real text). Two worlds fit: W1 the page's content forces page-specific words
(a topic), W2 the writer's page-level spelling habit (a key or mood) with no topic. v96 builds planted worlds of both
kinds from real medieval texts (v89 texts.json, written through the v72 merge code + Voynich-like surface, via v92_lib)
and looks for statistics that tell them apart, without any cipher-key search.

Planted worlds (all from the same five real texts):
  P_X        W1: real page order (topic per page), no key                         (v92 corpus, cached)
  D_X        W0: lines dealt at random across the pages of their section (topic destroyed), no key
  K_X_k      W2: D_X + per-page spelling key number k (random rules and strength)
  PK_X_k     W1+W2: P_X + per-page key k
Generators: G_GM, G_GM2 (glyph page mood), G_SEED (page seed vocabulary), G_LX (lexical page mood), G_SELF.
"""
import os, sys, json, random, hashlib, pickle
from collections import Counter
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v96_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v92_lib as L92

TEXTS = ['KONRAD', 'CIRCA', 'APIC', 'HYGIN', 'CULP']
GENS = ['G_GM', 'G_GM2', 'G_SEED', 'G_LX', 'G_SELF']
VOY = ['ZL3b', 'IT2a', 'GC2a']
NKEY = 3
FRESH = {'F_MACER': 'macer_floridus', 'F_PLINY': 'pliny_nh_lat'}
FRESH_W1 = sorted(FRESH) + ['E_KONRAD', 'E_CIRCA', 'L_KONRAD', 'L_CIRCA']


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def sha_obj(o): return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def half(pid): return L92.split_half(pid)


def deal_lines(pages, seed):
    """every line goes to a random slot among the lines of its section: page line counts and section kept, topic gone."""
    rng = random.Random(seed)
    bysec = {}
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']): bysec.setdefault(p['sec'], []).append((pi, li))
    new = [[None] * len(p['lines']) for p in pages]
    for s, slots in bysec.items():
        src = [pages[pi]['lines'][li] for pi, li in slots]; rng.shuffle(src)
        for (pi, li), l in zip(slots, src): new[pi][li] = dict(l, ps=pages[pi]['lines'][li]['ps'])
    return [dict(p, lines=new[i]) for i, p in enumerate(pages)]


def key_spec(pages, seed):
    """a random page-key design: R context rules a->b (position initial / final / any) among the corpus's frequent glyphs,
    application strength s; each page switches each rule on with probability 1/2."""
    rng = random.Random(seed)
    gc = Counter(c for p in pages for l in p['lines'] for w in l['w'] for c in w)
    top = [g for g, _ in gc.most_common(14)]
    R = rng.randint(6, 16); s = rng.choice([0.3, 0.5, 0.7, 0.9])
    rules = []
    for _ in range(R):
        a = rng.choice(top); b = rng.choice([g for g in top if g != a])
        rules.append((a, b, rng.choice(['init', 'final', 'any'])))
    return dict(R=R, s=s, rules=rules, seed=seed)


def apply_key(pages, spec):
    rng = random.Random(spec['seed'] + 1)
    out = []
    for p in pages:
        on = [r for r in spec['rules'] if rng.random() < 0.5]
        nl = []
        for l in p['lines']:
            ws = []
            for w in l['w']:
                w = list(w)
                for a, b, pos in on:
                    idx = [0] if pos == 'init' else ([len(w) - 1] if pos == 'final' else range(len(w)))
                    for i in idx:
                        if w[i] == a and rng.random() < spec['s']: w[i] = b
                ws.append(''.join(w))
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


def corpus(name):
    """P_X / D_X / K_X_k / PK_X_k / generators / Voynich; suffix _H0 / _H1 = train / held-out folios."""
    if name.endswith(('_H0', '_H1')):
        h = int(name[-1]); return [p for p in corpus(name[:-3]) if half(p['id']) == h]
    if name in VOY or name in GENS or name in ('E_KONRAD', 'E_CIRCA', 'L_KONRAD', 'L_CIRCA') or \
            name.startswith('P_') and name[2:] in TEXTS:
        return L92.corpus(name)
    if name.startswith('F_'):                              # fresh W1 plants (never used to select views)
        c = pload('corp_%s.pkl' % name)
        if c is None:
            i = sorted(FRESH).index(name)
            c = L92.surfaced(L92.entries(FRESH[name]), 9650 + 11 * i, prefix=name[2:4].lower()); psave('corp_%s.pkl' % name, c)
        return c
    c = pload('corp_%s.pkl' % name)
    if c is not None: return c
    parts = name.split('_')
    X = parts[1]; i = TEXTS.index(X)
    if parts[0] == 'D':
        c = deal_lines(L92.corpus('P_' + X), 9600 + i)
    elif parts[0] in ('K', 'PK'):
        k = int(parts[2]); base = corpus('D_' + X) if parts[0] == 'K' else L92.corpus('P_' + X)
        c = apply_key(base, key_spec(base, 9700 + 10 * i + k))
    else:
        raise KeyError(name)
    psave('corp_%s.pkl' % name, c)
    return c


def world(name):
    if name in VOY: return 'VOY'
    if name in GENS: return 'GEN'
    if name in FRESH_W1: return 'W1F'
    return {'P': 'W1', 'D': 'W0', 'K': 'W2', 'PK': 'W12'}[name.split('_')[0]]


def all_names():
    n = []
    for X in TEXTS:
        n += ['P_' + X, 'D_' + X] + ['K_%s_%d' % (X, k) for k in range(NKEY)] + ['PK_%s_%d' % (X, k) for k in range(NKEY)]
    return n + GENS + VOY


def group(p): return '%s|%s|%s' % (p['sec'], p.get('lang', '-'), p.get('hand', '-'))


def featf(v):
    """random glyph view -> feature function of a word."""
    m = v['merge']; kind = v['kind']
    def f(w):
        if m is not None: w = ''.join(chr(65 + m.get(c, 0)) for c in w)
        out = []
        if kind in ('uni', 'uni+pos'): out += list(w)
        if kind == 'bi': out += [a + b for a, b in zip('^' + w, w + '$')]
        if kind in ('pos', 'uni+pos'):
            out += ['%s@%d' % (c, i) for i, c in enumerate(w[:2])] + ['%s@-%d' % (c, i) for i, c in enumerate(w[::-1][:2])]
        return out
    return f


def random_view(rng, glyphs):
    merge = None
    if rng.random() < 0.3:
        k = int(rng.integers(4, 12)); merge = {g: int(rng.integers(k)) for g in glyphs}
    return dict(kind=str(rng.choice(['uni', 'bi', 'pos', 'uni+pos'])), merge=merge,
                lam=float(rng.choice([5.0, 20.0, 100.0])))


def rank_acc(Q, Ref, groups, lam, pos=None, far=0):
    """Q[i]: feature-count vector of a query set from page i; Ref[i]: reference profile of page i.
    Mean normalised rank of the true page among same-group pages (0.5 = chance)."""
    ranks = []
    for g, idx in groups.items():
        if len(idx) < 4: continue
        idx = np.array(idx)
        Rf = Ref[idx]; G = Rf.sum(0) + 1.0; G /= G.sum()
        LR = np.log((Rf + lam * G) / (Rf.sum(1, keepdims=True) + lam)) - np.log(G)
        S = Q[idx] @ LR.T
        n = len(idx)
        for a in range(n):
            if Q[idx[a]].sum() == 0: continue
            ok = np.ones(n, bool); ok[a] = False
            if far and pos is not None: ok &= np.abs(pos[idx] - pos[idx[a]]) > far
            if ok.sum() < 2: continue
            o = S[a][ok]
            ranks.append(((o < S[a, a]).sum() + 0.5 * (o == S[a, a]).sum()) / (n - 1))
    return float(np.mean(ranks)) if ranks else 0.5, len(ranks)


def deal_frac(pages, seed, frac=1.0):
    """a fraction `frac` of the lines (chosen at random) are dealt at random among their own slots within the page group
    (sec|lang|hand); frac = 1 destroys every page-level property (topic and key), frac = 0 keeps the book."""
    rng = random.Random(seed)
    slots = {}
    for pi, p in enumerate(pages):
        g = group(p)
        for li in range(len(p['lines'])):
            if rng.random() < frac: slots.setdefault(g, []).append((pi, li))
    new = [list(p['lines']) for p in pages]
    for g, sl in slots.items():
        src = [pages[pi]['lines'][li] for pi, li in sl]; rng.shuffle(src)
        for (pi, li), l in zip(sl, src): new[pi][li] = dict(l, ps=pages[pi]['lines'][li]['ps'])
    return [dict(p, lines=new[i]) for i, p in enumerate(pages)]


def apply_key_common(pages, spec, cfrac=0.3):
    """the page key applied ONLY to the commonest types (covering cfrac of tokens): a page habit that touches the function
    words and leaves the topic words alone (two independent page factors)."""
    cnt = Counter(w for p in pages for l in p['lines'] for w in l['w']); tot = sum(cnt.values()); acc = 0; C = set()
    for w, n in cnt.most_common():
        if acc >= cfrac * tot: break
        C.add(w); acc += n
    keyed = apply_key(pages, spec)
    return [dict(p, lines=[dict(l, w=[kw if w in C else w for w, kw in zip(l['w'], kl['w'])])
                           for l, kl in zip(p['lines'], kp['lines'])]) for p, kp in zip(pages, keyed)]


def corpus3(name):
    """cycle 3 extra plants: PKC_X_k = W1 + key on common words only."""
    if not name.startswith('PKC_'): return corpus(name)
    c = pload('corp_%s.pkl' % name)
    if c is None:
        X, k = name.split('_')[1], int(name.split('_')[2]); i = TEXTS.index(X)
        base = L92.corpus('P_' + X); c = apply_key_common(base, key_spec(base, 9700 + 10 * i + k)); psave('corp_%s.pkl' % name, c)
    return c
