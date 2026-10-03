"""S-DARK-66 common code: do the designation (middle) elements combine by semantic class?
Texts: data/derived/merged-corpus-canonical.json (older build, S-DARK-23 caution), complete, direction recorded,
>= 2 signs, collapsed to one per distinct text per site x object type (S-DARK-13).  Parser = S310/S331 frame parser
(dark_loop37).  Middle element = NAME or COUNT token (the designation slot of S-DARK-56/58); TITLE (closer-selected
qualifier), frame signs excluded.  Non-adjacent pair = two middle elements at distance >= 2 in the text.
Four classifications of signs, each built WITHOUT looking at co-occurrence of non-adjacent pairs:
  DESC  : keyword classes from the 84 dossier shape descriptions (numeral / human / animal / plant / tool-vessel / geometric)
  WBLOCK: Wells' own shape-family numbering, hundreds block (numerals 1-56 their own class)
  GLYPH : Ward clusters of the glyph_sim_fine similarity profiles, k = 10
  DIST  : agglomerative (Ward) clusters of PPMI adjacent-context vectors (Brown-style distributional classes), k = 10
"""
import sys, json, csv, re, collections, random, math
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import OPEN, MARK, MJAR, SUF, CL, FISH, NUM, learn_qual, make_parser, pval

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
SITES = ('Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro')
BIG = ('Mohenjo-daro', 'Harappa')
FRAME = set(OPEN) | set(MARK) | set(MJAR) | set(SUF) | set(CL)
STROKE_NUMERALS = set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(31, 40)) | {55, 56} | set(NUM)

def otype(t):
    t0 = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t0, 'other')

def sgroup(s): return s if s in SITES else 'other'

def load_wells(level):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    T = []
    for r in C:
        s = r[level]
        if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
        T.append(dict(id=r['cisi'], site=sgroup(r['site']), rawsite=r['site'], ot=otype(r['type']), seq=list(s)))
    seen = set(); out = []
    for t in T:
        k = (t['site'], t['ot'], tuple(t['seq']))
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

def load_wells_die(level):
    """die regime (S-DARK-41): moulded objects (TAB, TAG, POT) one copy per site x type x text, everything else one per cisi x text"""
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    T = []; seen = set()
    for r in C:
        s = r[level]
        if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
        ot = otype(r['type'])
        k = (r['site'], ot, tuple(s)) if ot in ('tablet', 'sealing', 'pot') else (r['cisi'], tuple(s))
        if k in seen: continue
        seen.add(k); T.append(dict(id=r['cisi'], site=sgroup(r['site']), rawsite=r['site'], ot=ot, seq=list(s)))
    return T

def parse_all(T):
    Q = learn_qual([t['seq'] for t in T]); parse = make_parser(Q)
    for t in T:
        lab = parse(t['seq']); t['lab'] = lab
        t['midpos'] = [i for i, l in enumerate(lab) if l in ('NAME', 'COUNT')]
        t['mid'] = [t['seq'][i] for i in t['midpos']]
    return T

# ------------------------------------------------------------------ IM77
MCL = {342: 'C740', 211: 'C520', 12: 'C151', 15: 'C156', 254: 'C527', 60: 'C226', 245: 'C617', 328: 'C700', 66: 'C236'}
MOPEN = {267, 391, 293, 150}; MMARK = {99, 100, 123}; MMJAR = {343, 344, 345, 346}; MSUF = {176, 1}

def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    objs = collections.OrderedDict()
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        key = (r['text_no'], r['side'])
        o = objs.setdefault(key, dict(id=r['text_no'] + '.' + r['side'], site=r['site'], ot=r['object_type'], seq=[]))
        o['seq'].extend(int(x) for x in r['signs_clean'].split() if x != '0')
    smap = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa', 'Lothal': 'Lothal', 'Kalibangan': 'Kalibangan', 'Chanhudaro': 'Chanhu-daro'}
    tmap = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}
    T = []; seen = set()
    for key, o in objs.items():
        if len(o['seq']) < 2: continue
        t = dict(id=o['id'], key=list(key), site=smap.get(o['site'], 'other'), ot=tmap.get(o['ot'], 'other'), seq=o['seq'])
        k = (t['site'], t['ot'], tuple(t['seq']))
        if k in seen: continue
        seen.add(k); T.append(t)
    return T

def parse_im77(T):
    for t in T:
        s = t['seq']; n = len(s); i = 0; j = n
        lab = ['NAME'] * n
        if s[0] in MOPEN:
            lab[0] = 'OPENER'; i = 1
            if n > 1 and s[1] in MMARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 293 and n > 2 and s[2] in MMJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in MSUF and (s[j - 2] in MCL or s[j - 2] in MSUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in MCL:
            lab[j - 1] = 'CLOSER'; j -= 1
            if j - 1 >= i: lab[j - 1] = 'TITLE'; j -= 1   # adjacent qualifier dropped as in Wells (S303 slot)
        t['lab'] = lab
        t['midpos'] = [k for k in range(i, j)]
        t['mid'] = [s[k] for k in t['midpos']]
    return T

def w2m_map():
    B = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    P = json.load(open(DARK + 'bridge_proposals.json'))['proposals']
    m = {}; src = {}
    for w, v in B.items():
        if len(v) == 1: m[int(w)] = v[0]; src[int(w)] = 'bridge'
    for r in P:
        if r['W'] not in m: m[r['W']] = r['M']; src[r['W']] = 'proposal'
    return m, src

def m2w_classes(classes_w, m, src):
    """map a Wells classification to M numbers (one-to-one entries only; ambiguous M -> dropped)"""
    out = collections.defaultdict(set); srcs = {}
    for w, c in classes_w.items():
        if w in m: out[m[w]].add(c); srcs[m[w]] = src[w]
    return {k: next(iter(v)) for k, v in out.items() if len(v) == 1}, srcs

# ------------------------------------------------------------------ classifications
DESC_RULES = [
    ('human', r"\b(person|man|men|people|human|figure|arms?|legs?|knees|heads?|breasts?)\b"),
    ('animal', r"\b(fish|animal|spider|cat|bird|paw|scorpion|insect|whiskers?|beetle|crab|bull|goat|tiger|snake)\b"),
    ('plant', r"\b(tree|leaf|leaves|plant|branch(es|ing)?|garlic|flower|bud|seed|grain)\b"),
    ('tool', r"\b(jar|arrow|spear|pitch?fork|pincer|wheel|bow|staff|comb|teeth|handles?|vessel|pot|box|knife|axe|rays?|yoke|cart|wagon|bowtie|drum)\b"),
    ('geometric', r"\b(stroke|strokes|line|lines|triangles?|diamond|rectangle|square|oval|circle|cross|parenthes[ie]s|u|x|m|lambda|heart|bar|hatch\w*|trapezoid|chevron|outline|lattice|grid|dots?)\b"),
]

def desc_classes():
    D = json.load(open(ROOT + 'data/derived/sign-dossiers-top200.json'))
    out = {}; txt = {}
    for d in D:
        s = d.get('shape', '') or ''
        if not s.strip(): continue
        w = d['glyph']; low = s.lower(); txt[w] = s
        if w in STROKE_NUMERALS and re.search(r'strokes?', low) and not re.search(r'wing|triang|box|square', low):
            out[w] = 'numeral'; continue
        for c, rx in DESC_RULES:
            if re.search(rx, low): out[w] = c; break
    return out, txt

def wblock_classes(signs):
    out = {}
    for w in signs:
        if w in STROKE_NUMERALS or w <= 56: out[w] = 'num'
        else: out[w] = 'B%d' % (w // 100)
    return out

def glyph_classes(signs, k=10):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    S = np.load(ROOT + 'data/derived/glyph_sim_fine.npy'); L = json.load(open(ROOT + 'data/derived/glyph_sim_signs.json'))
    idx = {w: i for i, w in enumerate(L)}
    sg = [w for w in signs if w in idx]
    M = S[np.ix_([idx[w] for w in sg], [idx[w] for w in sg])]
    M = (M + M.T) / 2; np.fill_diagonal(M, 1.0)
    # Ward on each sign's similarity profile (rows of the similarity matrix): balanced clusters, same information
    Z = linkage(M, 'ward')
    lab = fcluster(Z, k, 'maxclust')
    return {w: 'G%d' % l for w, l in zip(sg, lab)}

def dist_classes(T, signs, k=10, mintok=5):
    """Brown-style distributional classes: Ward clustering of PPMI vectors over adjacent left/right contexts in ALL texts"""
    from scipy.cluster.hierarchy import linkage, fcluster
    ctx_counts = collections.Counter(x for t in T for x in t['seq'])
    ctx = [w for w, c in ctx_counts.most_common() if c >= 8][:120]
    cidx = {w: i for i, w in enumerate(ctx)}; nc = len(ctx)
    sidx = {w: i for i, w in enumerate(signs)}
    M = np.zeros((len(signs), 2 * nc + 2))
    for t in T:
        s = t['seq']
        for i, w in enumerate(s):
            if w not in sidx: continue
            l = s[i - 1] if i > 0 else None; r = s[i + 1] if i + 1 < len(s) else None
            if l is None: M[sidx[w], 2 * nc] += 1
            elif l in cidx: M[sidx[w], cidx[l]] += 1
            if r is None: M[sidx[w], 2 * nc + 1] += 1
            elif r in cidx: M[sidx[w], nc + cidx[r]] += 1
    tot = M.sum(); pr = M.sum(1, keepdims=True) / tot; pc = M.sum(0, keepdims=True) / tot
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.log2((M / tot) / (pr @ pc)); P[~np.isfinite(P)] = 0; P[P < 0] = 0
    Z = linkage(P, 'ward'); lab = fcluster(Z, k, 'maxclust')
    return {w: 'D%d' % l for w, l in zip(signs, lab)}

# ------------------------------------------------------------------ pairs
def nonadj_pairs(t, field='midpos'):
    """set of unordered (a, b) element pairs at text distance >= 2, both middle elements"""
    s = t['seq']; pos = t[field]; out = set()
    for ii in range(len(pos)):
        for jj in range(ii + 1, len(pos)):
            i, j = pos[ii], pos[jj]
            if j - i >= 2 and s[i] != s[j]:
                a, b = s[i], s[j]
                out.add((a, b) if a < b else (b, a))
    return out

def all_nonadj_pairs(s):
    if isinstance(s, dict): s = s['seq']
    out = set()
    for i in range(len(s)):
        for j in range(i + 2, len(s)):
            if s[i] != s[j]:
                a, b = s[i], s[j]; out.add((a, b) if a < b else (b, a))
    return out

def class_table(pairs_per_text, cls, classes):
    """count class x class unordered cells over pair sets; pairs with an unclassed sign dropped"""
    ci = {c: i for i, c in enumerate(classes)}; k = len(classes)
    M = np.zeros((k, k)); n_in = 0; n_all = 0
    for ps in pairs_per_text:
        for a, b in ps:
            n_all += 1
            ca, cb = cls.get(a), cls.get(b)
            if ca is None or cb is None: continue
            n_in += 1
            i, j = ci[ca], ci[cb]
            if i > j: i, j = j, i
            M[i, j] += 1
    return M, n_in, n_all

def permute_middles(T, rnd, strata=lambda t: (t['site'], t['ot'], len(t['midpos']))):
    """permute middle elements among texts within strata (positions kept); returns list of new seqs"""
    groups = collections.defaultdict(list)
    for i, t in enumerate(T): groups[strata(t)].append(i)
    new = [list(t['seq']) for t in T]
    for idx in groups.values():
        pool = [x for i in idx for x in T[i]['mid']]
        rnd.shuffle(pool); p = 0
        for i in idx:
            for pos in T[i]['midpos']:
                new[i][pos] = pool[p]; p += 1
    return new

def markov_fit(seqs, order):
    tab = collections.defaultdict(collections.Counter)
    for s in seqs:
        s2 = ['<s>'] * order + list(s) + ['</s>']
        for i in range(order, len(s2)):
            tab[tuple(s2[i - order:i])][s2[i]] += 1
    return tab

def markov_gen(tab, n, rnd, order, uni, backoff):
    """generate exactly n signs (no END), backing off to lower order then unigram"""
    out = []; hist = ['<s>'] * order
    for _ in range(n):
        choice = None
        for o in range(order, 0, -1):
            key = tuple(hist[len(hist) - o:]) if o <= len(hist) else None
            tb = backoff[o]
            c = tb.get(key)
            if c:
                items = [(w, k) for w, k in c.items() if w != '</s>']
                if items:
                    tot = sum(k for _, k in items); r = rnd.random() * tot
                    for w, k in items:
                        r -= k
                        if r <= 0: choice = w; break
                    break
        if choice is None:
            items = list(uni.items()); tot = sum(k for _, k in items); r = rnd.random() * tot
            for w, k in items:
                r -= k
                if r <= 0: choice = w; break
        out.append(choice); hist.append(choice)
    return out

def markov_corpus(T, rnd, order=2):
    """whole-text Markov-order chain fitted per site x object type, lengths kept"""
    groups = collections.defaultdict(list)
    for t in T: groups[(t['site'], t['ot'])].append(t['seq'])
    models = {}
    for g, seqs in groups.items():
        backoff = {o: markov_fit(seqs, o) for o in range(1, order + 1)}
        uni = collections.Counter(x for s in seqs for x in s)
        models[g] = (backoff, uni)
    out = []
    for t in T:
        backoff, uni = models[(t['site'], t['ot'])]
        out.append(markov_gen(None, len(t['seq']), rnd, order, uni, backoff))
    return out

def attraction_pairs(seqs, pairfn, mincount=5, ratio=3.0):
    """S366 statistic: unordered pairs at distance >= 2 with count >= 5 and O/E >= 3 (E from text-presence)"""
    n = len(seqs); has = collections.Counter(); both = collections.Counter()
    for s in seqs:
        for x in set(s): has[x] += 1
        for p in pairfn(s): both[p] += 1
    att = {}
    for (x, y), c in both.items():
        e = has[x] * has[y] / n
        if c >= mincount and c / e >= ratio: att[(x, y)] = (c, e)
    return att, both, has

def lognull(obs, null):
    null = np.asarray(null, float); mu = null.mean(); sd = null.std() + 1e-9
    return (obs - mu) / sd
