"""v49 cycle 3: subject names hidden under variable spelling? 4,000 random spelling views, discovery and held-out.

If entries open with their subject but the scribe spells it differently on re-mention (endings, prefixes, twins),
exact-type recurrence (cycle 1, M1) misses it. A view f maps every word to a key (normalisation level, strip a first
and b last glyphs, collapse e/i runs, delete a random vowel-like set, merge a random glyph pair, keep only an m-glyph
prefix). Statistic per view: for each opening-line off-table token (CANDIDATE), hit = its key occurs in the body
lines of its own paragraph, minus its hit rate in 8 random other-page paragraphs of similar body length. The same
for CONTROL tokens (off-table tokens of the paragraph's second line, checked against lines 3+ of their own paragraph
vs other paragraphs' lines 3+). Score = paired excess(candidates) - excess(controls), z by token-level se.
Discovery on even pages, held-out on odd pages; survivors (top 20 discovery) re-tested held-out.
Positives: GENT (planted terms), BRf (Latin name heads its entry; positional ch->sh twin in first lines makes
re-mentions spell differently). Negatives: GEN0, GENN.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v49_lib import *

FN = 'v49_cycle3.txt'
NV = 4000
VOW = 'oayei'
GL = 'oayeidlrsnmqtTCkpfKPFS'


def random_view(rng):
    v = dict(lev=rng.choice(['raw', 'E3', 'G']), a=rng.choice([0, 0, 1, 2]), b=rng.choice([0, 0, 1, 1, 2, 3]),
             runs=rng.random() < 0.5, dele=''.join(c for c in VOW if rng.random() < 0.25),
             merge=(rng.choice(GL), rng.choice(GL)) if rng.random() < 0.4 else None,
             pre=rng.choice([None, None, 2, 3, 4, 5]))
    return v


def apply_view(v, w):
    w = {'raw': w, 'E3': norm(w), 'G': gnorm(w)}[v['lev']]
    if v['a']: w = w[v['a']:]
    if v['b']: w = w[:-v['b']] if len(w) > v['b'] else ''
    if v['runs']: w = re.sub(r'(e)e+', r'\1', re.sub(r'(i)i+', r'\1', w))
    if v['dele']: w = ''.join(c for c in w if c not in v['dele'])
    if v['merge']: w = w.replace(v['merge'][1], v['merge'][0])
    if v['pre']: w = w[:v['pre']]
    return w if len(w) >= 2 else None


def prepare(name):
    T = tokens_table(name)
    paras = defaultdict(lambda: defaultdict(list))
    for t in T: paras[(t['page'], t['para'])][t['line']].append(t)
    P = []
    for key, lines in paras.items():
        if 0 not in lines or len(lines) < 3: continue
        P.append(dict(page=key[0], pf=[t['w'] for t in lines[0] if t['off'] and t['k'] > 0],
                      l1=[t['w'] for t in lines.get(1, []) if t['off'] and t['k'] > 0],
                      body=[t['w'] for li, l in lines.items() if li >= 1 for t in l],
                      body2=[t['w'] for li, l in lines.items() if li >= 2 for t in l]))
    return P


def setup(P, rng, pages):
    idx = [i for i, p in enumerate(P) if p['page'] in pages]
    lens = np.array([len(P[i]['body']) for i in idx])
    others = {}
    for j, i in enumerate(idx):
        ok = [idx[k] for k in range(len(idx)) if P[idx[k]]['page'] != P[i]['page'] and abs(lens[k] - lens[j]) <= 0.3 * lens[j] + 2]
        if len(ok) < 8: ok = [idx[k] for k in range(len(idx)) if P[idx[k]]['page'] != P[i]['page']]
        others[i] = rng.sample(ok, 8)
    return idx, others


def score(P, idx, others, v, cache):
    """cache: dict with 'types' (list of word types) and per-paragraph type-id arrays in P[i]['_ids']."""
    types = cache['types']
    if 'kid' not in cache:
        kd = {}; kid = np.empty(len(types), np.int64)
        for j, w in enumerate(types):
            kw = apply_view(v, w); kid[j] = -1 if kw is None else kd.setdefault(kw, len(kd))
        cache['kid'] = kid
    kid = cache['kid']
    need = set(idx)
    for i in idx: need.update(others[i])
    ks = {}
    for i in need:
        ks[(i, 'body')] = set(kid[P[i]['_ids']['body']].tolist()) - {-1}
        ks[(i, 'body2')] = set(kid[P[i]['_ids']['body2']].tolist()) - {-1}
    def part(cfld, tfld):
        d = []
        for i in idx:
            for t in P[i]['_ids'][cfld].tolist():
                kw = kid[t]
                if kw < 0: continue
                own = kw in ks[(i, tfld)]
                oth = sum(kw in ks[(o, tfld)] for o in others[i]) / len(others[i])
                d.append(own - oth)
        return np.array(d, float)
    a = part('pf', 'body'); b = part('l1', 'body2')
    if len(a) < 30 or len(b) < 30: return None
    ex = a.mean() - b.mean(); se = math.sqrt(a.var() / len(a) + b.var() / len(b)) + 1e-9
    return float(ex), float(ex / se), float(a.mean()), float(b.mean())


def index_types(P):
    types = sorted({w for p in P for f in ('pf', 'l1', 'body', 'body2') for w in p[f]})
    ti = {w: j for j, w in enumerate(types)}
    for p in P:
        p['_ids'] = {f: np.array([ti[w] for w in p[f]], np.int64) for f in ('pf', 'l1', 'body', 'body2')}
    return types


def run(name):
    rng = random.Random(4930)
    P = prepare(name)
    types = index_types(P)
    pages = sorted(set(p['page'] for p in P)); disc = set(pages[0::2]); held = set(pages[1::2])
    Id, Od = setup(P, rng, disc); Ih, Oh = setup(P, rng, held)
    vr = random.Random(4931)
    views = [random_view(vr) for _ in range(NV)]
    views[0] = dict(lev='raw', a=0, b=0, runs=False, dele='', merge=None, pre=None)   # exact identity
    res = []
    for k, v in enumerate(views):
        s = score(P, Id, Od, v, {'types': types})
        if s is None: continue
        res.append((k, s))
    res.sort(key=lambda x: -x[1][1])
    top = res[:20]
    held_s = [(k, score(P, Ih, Oh, views[k], {'types': types})) for k, _ in top]
    ident_d = score(P, Id, Od, views[0], {'types': types}); ident_h = score(P, Ih, Oh, views[0], {'types': types})
    zd = np.array([s[1] for _, s in res])
    hz = np.array([s[1] for _, s in held_s if s])
    out = dict(n_views=len(res), disc_best=top[0][1], disc_q95=float(np.quantile(zd, 0.95)), disc_frac_z3=float((zd > 3).mean()),
               disc_median=float(np.median(zd)), held_mean_z=float(hz.mean()), held_frac_z2=float((hz > 2).mean()),
               held_best=float(hz.max()), identity_disc=ident_d, identity_held=ident_h,
               top_views=[(views[k], s, h) for (k, s), (_, h) in zip(top[:5], held_s[:5])])
    jsave(f'c3_{name}.json', out)
    return name, out


if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or ['V', 'VI', 'GEN0', 'GENN', 'GENT', 'BRf']
    with Pool(2) as Pp:
        for nm, o in Pp.imap_unordered(run, names):
            print(nm, json.dumps({k: o[k] for k in o if k != 'top_views'}, default=float), flush=True)
            print('  top', json.dumps(o['top_views'][:3], default=float), flush=True)
