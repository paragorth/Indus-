"""v77 cycle 2: kill tests for four open C guesses that need searches.
K5  v56 pages as independent records: stated kill = a real herbal matched to the Voynich type/token ratio scoring
    about 5 on the v56 table test (Voynich z_hit 4-8.5; Brumati 52-60). Brumati's Italian herbal and Culpeper's
    English herbal (names) are given scribe-like one-glyph variation until their TTR equals the Voynich's at the same
    token count; v56_c3.table_test unchanged.
K10 v64 concepts written as whole words: if a small alphabet of whole words is combined in a fixed (Lullian) order,
    alphabet words that share a line must keep one global order. 3,000 random alphabets (5-15 of the top-200 types),
    order learned on discovery leaves, concordance tested on held-out leaves; planted control = 9 whole words inserted
    in sorted triples into 40% of the lines of a fitted generator. Kill: the Voynich's best held-out alphabets no
    better than generator text searched the same way.
K11 v60 complexion stated per page (as a word or a rare sign): sets of 2-4 page features (word types or glyph
    bigrams) with exactly one member per herbal page, beyond independence; 20,000 random sets + greedy, selected on
    discovery pages, scored on held-out pages; plant = one of four complexion words per herbal page (0.7/0.1/0.1/0.1)
    in generator text. Kill: Voynich at generator level when the plant is found.
K12 v73 spelling-only / first-glyph-pointer marking and v67 one marked word per line: 3,000 random spelling-class
    selectors (glyph content, edges, length) and 3,000 random first-glyph pointer tables pick one word per line; the
    stream is scored on held-out leaves for neighbour-page recurrence and recurring ordered item pairs; planted control =
    an ingredient list (Antidotarium Nicolai) one item per line, items spelled as Voynich-form words that contain a
    marker glyph, in generator filler. Stated kill: 'the same null at that power'.
"""
import sys, os, json, math, random, collections
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L

LOG = open(os.path.join(L.CK, 'c2.log'), 'a')


def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


# ------------------------------------------------------------------ K5
def k5_job(arg):
    import v56_lib as S
    import v56_c3 as C3
    name, mu, seed = arg
    rng = random.Random(seed)
    if name.startswith('ZL'):
        C = S.voynich('ZL3b')
    elif name.startswith('BRU'):
        C = C3.brumati()
    else:
        C = S.culpeper(numerals=False)
    if mu > 0:
        alpha = sorted({c for p in C['pages'] for l in p['lines'] for w in l for c in w})
        for p in C['pages']:
            p['lines'] = [[(w[:j] + rng.choice(alpha) + w[j + 1:]) if (rng.random() < mu and len(w) > 1 and (j := rng.randrange(len(w))) is not None) else w
                           for w in l] for l in p['lines']]
    toks = [w for p in C['pages'] for l in p['lines'] for w in l]
    ttr20k = len(set(toks[:20000])) / min(20000, len(toks))
    E = S.build_R(C); TT = S.token_table(C, E)
    out = dict(name=name, mu=mu, ttr20k=ttr20k, ntok=len(toks), npages=len(C['pages']),
               tok_per_page=len(toks) / len(C['pages']))
    for split in range(3):
        trp = S.split_pages(E, seed=split); tok_tr = trp[E['tok_page']]
        m = np.ones(len(TT['words']), bool)
        for key in ('word', 'skel'):
            r = C3.table_test(E, TT, C, m, tok_tr, key=key)
            out['%s|%d' % (key, split)] = dict(z_hit=r['z_hit'], z_spec=r['z_spec'])
    L.jsave('c2_k5_%s_%.3f.json' % (name, mu), out)
    log('K5', json.dumps(out))
    return out


def ttr_probe(name, mu, seed=1):
    import v56_lib as S, v56_c3 as C3
    rng = random.Random(seed)
    C = C3.brumati() if name.startswith('BRU') else (S.culpeper(numerals=False) if name.startswith('CUL') else S.voynich('ZL3b'))
    toks = [w for p in C['pages'] for l in p['lines'] for w in l]
    alpha = sorted({c for w in toks for c in w})
    out = []
    for w in toks[:20000]:
        if rng.random() < mu and len(w) > 1:
            j = rng.randrange(len(w)); w = w[:j] + rng.choice(alpha) + w[j + 1:]
        out.append(w)
    return len(set(out)) / len(out)


# ------------------------------------------------------------------ K10
def k10(pages, n_alpha=2000, seed=0):
    rng = random.Random(seed)
    uni = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    top = [w for w, _ in uni.most_common(200)]
    LN = [(L.leaf_half(p['id']), l['w']) for p in pages for l in p['lines']]
    pos = {0: [], 1: []}
    for h, w in LN:
        d = {}
        for i, x in enumerate(w): d.setdefault(x, i)
        pos[h].append(d)
    ti = {w: i for i, w in enumerate(top)}
    # pair order counts per half: W[h][i, j] = lines where top i precedes top j
    W = {}
    for h in (0, 1):
        M = np.zeros((200, 200))
        for d in pos[h]:
            it = [(i_, ti[x]) for x, i_ in d.items() if x in ti]
            for a in range(len(it)):
                for b in range(len(it)):
                    if it[a][0] < it[b][0]: M[it[a][1], it[b][1]] += 1
        W[h] = M
    best = []
    for k in range(n_alpha):
        K = rng.randint(5, 15); A = rng.sample(range(200), K)
        r = []
        for tr in (0, 1):
            M = W[tr][np.ix_(A, A)]
            s = M.sum(1) - M.sum(0); order = np.argsort(-s); rank = np.empty(K); rank[order] = np.arange(K)
            def conc(Mx):
                c = d = 0.0
                for a in range(K):
                    for b in range(K):
                        if a == b: continue
                        if rank[a] < rank[b]: c += Mx[a, b]
                        else: d += Mx[a, b]
                n = c + d
                return (c / n if n else 0.5), n
            ctr, ntr = conc(M); cte, nte = conc(W[1 - tr][np.ix_(A, A)])
            ztr = (ctr - .5) / math.sqrt(.25 / max(ntr, 1)); zte = (cte - .5) / math.sqrt(.25 / max(nte, 1))
            r.append((ztr, zte, cte, nte))
        best.append((np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), np.mean([x[2] for x in r]), K))
    best.sort(key=lambda x: -x[0])
    top10 = best[:10]
    return dict(sel_train_z=float(np.mean([x[0] for x in top10])), sel_test_z=float(np.mean([x[1] for x in top10])),
                sel_test_conc=float(np.mean([x[2] for x in top10])), all_test_z_med=float(np.median([x[1] for x in best])),
                all_test_z_q99=float(np.quantile([x[1] for x in best], 0.99)))


def plant_lull(pages, seed=5, rate=0.4):
    rng = random.Random(seed)
    uni = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    A = [w for w, _ in uni.most_common(60)][30:39]
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            w = list(l['w'])
            if len(w) >= 4 and rng.random() < rate:
                letters = sorted(rng.sample(range(9), 3)); ps = sorted(rng.sample(range(len(w)), 3))
                for li, pi in zip(letters, ps): w[pi] = A[li]
            nl.append(dict(l, w=w))
        out.append(dict(p, lines=nl))
    return out


# ------------------------------------------------------------------ K11
def k11(pages, n_sets=8000, seed=0):
    rng = random.Random(seed)
    H = [p for p in pages if p['sec'] == 'H']
    feats = []
    for p in H:
        ws = [w for l in p['lines'] for w in l['w']]
        f = set(ws) | {'#' + w[i:i + 2] for w in ws for i in range(len(w) - 1)}
        feats.append(f)
    half = np.array([L.leaf_half(p['id']) for p in H])
    cnt = Counter(x for f in feats for x in f)
    n = len(H)
    pool = [x for x, c in cnt.items() if 0.04 * n <= c <= 0.7 * n]
    pi = {x: i for i, x in enumerate(pool)}
    X = np.zeros((n, len(pool)), bool)
    for r, f in enumerate(feats):
        for x in f:
            if x in pi: X[r, pi[x]] = True
    def excess(cols, rows):
        Y = X[np.ix_(rows, cols)]
        obs = float((Y.sum(1) == 1).mean())
        p = Y.mean(0)
        exp = float(sum(p[i] * np.prod([1 - p[j] for j in range(len(cols)) if j != i]) for i in range(len(cols))))
        return obs - exp, obs
    res = {}
    for tr in (0, 1):
        rtr = np.nonzero(half == tr)[0]; rte = np.nonzero(half != tr)[0]
        cand = []
        for _ in range(n_sets):
            k = rng.randint(2, 4); cols = rng.sample(range(len(pool)), k)
            e, o = excess(cols, rtr); cand.append((e, cols))
        cand.sort(key=lambda x: -x[0])
        refined = []
        for e, cols in cand[:30]:
            cols = list(cols)
            for _ in range(3):
                for slot in range(len(cols)):
                    trial = rng.sample(range(len(pool)), 40)
                    for c in trial:
                        if c in cols: continue
                        c2 = cols[:]; c2[slot] = c
                        e2, _ = excess(c2, rtr)
                        if e2 > e: e, cols = e2, c2
            refined.append((e, cols))
        refined.sort(key=lambda x: -x[0])
        te = [excess(c, rte) for e, c in refined[:10]]
        res['h%d' % tr] = dict(train_excess=float(np.mean([x[0] for x in refined[:10]])),
                               test_excess=float(np.mean([x[0] for x in te])), test_cover=float(np.mean([x[1] for x in te])),
                               top=[[pool[c] for c in refined[0][1]]])
    res['test_excess'] = float(np.mean([res['h0']['test_excess'], res['h1']['test_excess']]))
    res['train_excess'] = float(np.mean([res['h0']['train_excess'], res['h1']['train_excess']]))
    return res


def plant_complexion(pages, seed=7):
    rng = random.Random(seed)
    uni = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    rare = [w for w, c in uni.most_common() if 3 <= c <= 6][:4]
    out = []
    for p in pages:
        if p['sec'] != 'H': out.append(p); continue
        u = rng.random(); w0 = rare[0] if u < .7 else (rare[1] if u < .8 else (rare[2] if u < .9 else rare[3]))
        lines = [dict(l, w=list(l['w'])) for l in p['lines']]
        li = rng.randrange(len(lines)); j = rng.randrange(len(lines[li]['w'])); lines[li]['w'][j] = w0
        out.append(dict(p, lines=lines))
    return out, rare


# ------------------------------------------------------------------ K12
GL1 = ['q', 'o', 'd', 'y', 's', 'k', 't', 'C', 'S', 'p', 'f', 'l', 'r', 'a', 'e', 'i', 'n', 'm', 'g', 'T', 'K', 'P', 'F']


def rand_spelling(rng):
    kind = rng.random()
    g = rng.sample(GL1, 2)
    lo = rng.randint(1, 5); hi = lo + rng.randint(0, 6)
    st = rng.choice(GL1); en = rng.choice(['y', 'n', 'l', 'r', 'm', 's', 'o', 'd', 'g'])
    if kind < 0.25: return ('has', g[0], lo, hi), lambda w: g[0] in w and lo <= len(w) <= hi
    if kind < 0.45: return ('has2', g[0], g[1]), lambda w: g[0] in w and g[1] in w
    if kind < 0.6: return ('start', st, lo, hi), lambda w: w[0] == st and lo <= len(w) <= hi
    if kind < 0.75: return ('end', en, g[0]), lambda w: w[-1] == en and g[0] in w
    if kind < 0.9: return ('startend', st, en), lambda w: w[0] == st and w[-1] == en
    return ('hasnot', g[0], g[1], lo), lambda w: g[0] in w and g[1] not in w and len(w) >= lo


def stream(pages, select):
    """select(line_words, line) -> index or None. Returns per page list of items (page order kept)."""
    out = []
    for p in pages:
        items = []
        for l in p['lines']:
            j = select(l['w'])
            if j is not None: items.append(l['w'][j])
        out.append(items)
    return out


def stream_scores(pages, S, half, rng, nperm=30):
    idx = [i for i, p in enumerate(pages) if L.leaf_half(p['id']) == half]
    sec = [p['sec'] for p in pages]
    sets = [set(s) for s in S]
    nb, far, tot = 0, 0, 0
    for i in idx:
        if not S[i]: continue
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(pages) and sec[j] == sec[i]]
        farj = [j for j in range(i - 12, i - 5) if 0 <= j and sec[j] == sec[i]][:2] + [j for j in range(i + 6, i + 13) if j < len(pages) and sec[j] == sec[i]][:2]
        if not near or not farj: continue
        for w in S[i]:
            tot += 1
            nb += sum(w in sets[j] for j in near) / len(near)
            far += sum(w in sets[j] for j in farj) / len(farj)
    XP = (nb - far) / max(tot, 1)
    # recurring ordered item pairs across pages (consecutive items on a page)
    def pair_rec(SS):
        pp = defaultdict(set)
        for i in idx:
            s = SS[i]
            for a, b in zip(s, s[1:]):
                if a != b: pp[(a, b)].add(i)
        return sum(1 for v in pp.values() if len(v) >= 2)
    obs = pair_rec(S)
    allit = [w for i in idx for w in S[i]]
    nul = []
    for _ in range(nperm):
        rng.shuffle(allit); k = 0; SS = list(S)
        for i in idx:
            SS[i] = allit[k:k + len(S[i])]; k += len(S[i])
        nul.append(pair_rec(SS))
    nul = np.array(nul)
    cov = sum(len(S[i]) for i in idx) / max(1, sum(len(pages[i]['lines']) for i in idx))
    return dict(XP=XP, pair_z=float((obs - nul.mean()) / (nul.std() + 1e-9)), cover=cov)


def k12(pages, n_rules=400, seed=0):
    rng = random.Random(seed)
    rules = []
    for k in range(n_rules):
        desc, pred = rand_spelling(rng)
        rules.append((('S',) + desc, (lambda pr: (lambda ws: next((i for i, w in enumerate(ws) if pr(w)), None)))(pred)))
    for k in range(n_rules):
        tab = {g: rng.choice([0, 1, 2, 3, 4, -1]) for g in GL1}
        def sel(ws, tab=tab):
            j = tab.get(ws[0][0], 1)
            j = len(ws) - 1 if j == -1 else min(j, len(ws) - 1)
            return j
        rules.append((('FG', k), sel))
    res = {}
    scored = {0: [], 1: []}
    for desc, sel in rules:
        S = stream(pages, sel)
        for tr in (0, 1):
            s = stream_scores(pages, S, tr, rng, nperm=0)
            if s['cover'] < 0.25: continue
            scored[tr].append((s['XP'], desc, S))
    for tr in (0, 1):
        sc = sorted(scored[tr], key=lambda x: -x[0])
        te = [stream_scores(pages, S, 1 - tr, rng, nperm=20) for xp, desc, S in sc[:10]]
        res['h%d' % tr] = dict(train_XP=float(np.mean([x[0] for x in sc[:10]])), test_XP=float(np.mean([t['XP'] for t in te])),
                               test_pair_z=float(np.mean([t['pair_z'] for t in te])), test_pair_zmax=float(np.max([t['pair_z'] for t in te])),
                               top=[str(x[1]) for x in sc[:3]], n_rules=len(sc))
    res['test_XP'] = float(np.mean([res['h0']['test_XP'], res['h1']['test_XP']]))
    res['test_pair_z'] = float(np.mean([res['h0']['test_pair_z'], res['h1']['test_pair_z']]))
    return res


def plant_list(pages, seed=11, marker='f'):
    """one ingredient per line (Antidotarium Nicolai order), each item type spelled as a Voynich word type that
    contains the marker glyph (one-word codebook), placed at a random position; filler = generator text."""
    import v73_lib as T
    rng = random.Random(seed)
    items = T.item_stream('antid')
    uni = Counter(w for p in L.voy('ZL3b') for l in p['lines'] for w in l['w'])
    vt = [w for w, _ in uni.most_common() if marker in w]
    it = [w for w, _ in Counter(items).most_common()]
    book = {w: vt[i % len(vt)] for i, w in enumerate(it)}
    k = 0; out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            w = [x.replace(marker, 'k') for x in l['w']]   # filler cleaned of the marker glyph
            if k < len(items):
                w[rng.randrange(len(w))] = book[items[k]]; k += 1
            nl.append(dict(l, w=w))
        out.append(dict(p, lines=nl))
    return out


def job(arg):
    test, lab, gen, seed, name, plant = arg
    if gen is None: P = L.voy(name)
    else: P = L.generate(name, gen, seed)
    extra = {}
    if plant == 'lull': P = plant_lull(P)
    if plant == 'cplx': P, extra['rare'] = plant_complexion(P)
    if plant == 'list': P = plant_list(P)
    if test == 'k10': r = k10(P)
    elif test == 'k11': r = k11(P)
    else: r = k12(P)
    r.update(extra)
    out = dict(test=test, label=lab, plant=plant, res=r)
    L.jsave('c2_%s_%s_%s.json' % (test, lab.replace(':', '_'), plant or 'none'), out)
    log(test, lab, plant, json.dumps(r)[:600])
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    what = sys.argv[1].split(',') if len(sys.argv) > 1 else ['k5', 'k10', 'k11', 'k12']
    jobs = []
    for t in ('k10', 'k11', 'k12'):
        if t not in what: continue
        jobs += [(t, 'ZL3b', None, 0, 'ZL3b', None), (t, 'IT2a', None, 0, 'IT2a', None)]
        for g in ('STACK', 'SELFCIT', 'JUNC', 'MK2'):
            jobs.append((t, 'ZL:%s:771' % g, g, 771, 'ZL3b', None))
        jobs.append((t, 'ZL:STACK:772', 'STACK', 772, 'ZL3b', None))
        pl = dict(k10='lull', k11='cplx', k12='list')[t]
        jobs.append((t, 'ZL:STACK:771', 'STACK', 771, 'ZL3b', pl))
        jobs.append((t, 'ZL:MK2:771', 'MK2', 771, 'ZL3b', pl))
    with Pool(2) as pool:
        R = list(pool.imap_unordered(job, jobs)) if jobs else []
        if 'k5' in what:
            k5jobs = [('ZL', 0.0, 5), ('BRU', 0.0, 5), ('BRU', 0.058, 5), ('BRU', 0.058, 6), ('CUL', 0.0, 5), ('CUL', 0.019, 5),
                      ('BRU', 0.2, 5)]
            R += list(pool.imap_unordered(k5_job, k5jobs))
    L.jsave('c2_all_%s.json' % '_'.join(what), R)


# ------------------------------------------------------------------ K12b: exhaustive single-glyph spelling bank
def k12b(pages, seed=0):
    """same scoring as k12, but the rule bank is exhaustive over simple spelling classes: contains glyph g with
    length in [lo, hi] (23 x 35), starts with g, ends with g; first-glyph pointer tables omitted."""
    rng = random.Random(seed)
    rules = []
    for g in GL1:
        for lo in range(1, 6):
            for hi in range(lo, lo + 7):
                rules.append((('has', g, lo, hi), (lambda g, lo, hi: (lambda ws: next((i for i, w in enumerate(ws) if g in w and lo <= len(w) <= hi), None)))(g, lo, hi)))
        rules.append((('start', g), (lambda g: (lambda ws: next((i for i, w in enumerate(ws) if w[0] == g), None)))(g)))
        rules.append((('end', g), (lambda g: (lambda ws: next((i for i, w in enumerate(ws) if w[-1] == g), None)))(g)))
    res = {}
    scored = {0: [], 1: []}
    for desc, sel in rules:
        S = stream(pages, sel)
        for tr in (0, 1):
            s = stream_scores(pages, S, tr, rng, nperm=0)
            if s['cover'] < 0.25: continue
            scored[tr].append((s['XP'], desc, S))
    for tr in (0, 1):
        sc = sorted(scored[tr], key=lambda x: -x[0])
        te = [stream_scores(pages, S, 1 - tr, rng, nperm=20) for xp, desc, S in sc[:5]]
        res['h%d' % tr] = dict(train_XP=float(np.mean([x[0] for x in sc[:5]])), test_XP=float(np.mean([t['XP'] for t in te])),
                               test_pair_z=float(np.mean([t['pair_z'] for t in te])), test_pair_zmax=float(np.max([t['pair_z'] for t in te])),
                               top=[str(x[1]) for x in sc[:3]], n_rules=len(sc))
    res['test_XP'] = float(np.mean([res['h0']['test_XP'], res['h1']['test_XP']]))
    res['test_pair_z'] = float(np.mean([res['h0']['test_pair_z'], res['h1']['test_pair_z']]))
    return res


def job_b(arg):
    lab, gen, seed, name, plant = arg
    P = L.voy(name) if gen is None else L.generate(name, gen, seed)
    if plant: P = plant_list(P)
    r = k12b(P)
    L.jsave('c2_k12b_%s_%s.json' % (lab.replace(':', '_'), plant or 'none'), dict(label=lab, plant=plant, res=r))
    log('k12b', lab, plant, r['test_XP'], r['test_pair_z'], r['h0']['top'][:1])
    return r


def main_b():
    from multiprocessing import Pool
    jobs = [('ZL:STACK:771', 'STACK', 771, 'ZL3b', 'list'), ('ZL:MK2:771', 'MK2', 771, 'ZL3b', 'list'),
            ('ZL3b', None, 0, 'ZL3b', None), ('IT2a', None, 0, 'IT2a', None), ('ZL:STACK:771', 'STACK', 771, 'ZL3b', None),
            ('ZL:JUNC:771', 'JUNC', 771, 'ZL3b', None), ('ZL:MK2:771', 'MK2', 771, 'ZL3b', None), ('ZL:SELFCIT:771', 'SELFCIT', 771, 'ZL3b', None)]
    with Pool(2) as pool:
        list(pool.imap_unordered(job_b, jobs))
