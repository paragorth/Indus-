"""v54 cycle 2: reduced alphabet from collation (train pages only) -> held-out page-variable information and
word-bigram gain, against random reductions of the same make-up and against reductions learned from line-shuffled
(context-free) near-repeats."""
import sys, os, json, collections, random, math, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
from multiprocessing import Pool

NR = 30
KS = (4, 8)

def null_posshuf(pages, seed=1, key='sec'):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    def slot(k, n): return 'F' if k == 0 else ('L' if k == n - 1 else 'M')
    for p in pages:
        for l in p['lines']:
            for k, w in enumerate(l): by[(p['vars'][key], slot(k, len(l)))].append(w)
    for v in by.values(): rng.shuffle(v)
    it = {k: iter(v) for k, v in by.items()}
    return [dict(p, lines=[[next(it[(p['vars'][key], slot(k, len(l)))]) for k in range(len(l))] for l in p['lines']]) for p in pages]

def ops_from(pages, K, seed=1, min_occ=30):
    """Collation ops over-represented in real witness pairs relative to chance pairs (posshuf null):
    excess = (rate per pair in real - rate per pair in null) * real pairs.  Also returns the chance-pair ops
    (top raw counts in null pairs) as a comparator."""
    toks, pairs, _ = V.families(pages)
    col = V.collate(toks, pairs)
    tn, pn, _ = V.families(null_posshuf(pages, seed))
    coln = V.collate(tn, pn)
    occ = col['occ']
    ok = lambda k: all(occ[c] >= min_occ for c in k[1:])
    ex = sorted(((col['ops'][k] - coln['ops'][k] * col['npair'] / max(coln['npair'], 1), k) for k in set(col['ops']) | set(coln['ops']) if ok(k)), reverse=True)
    ch = sorted(((n, k) for k, n in coln['ops'].items() if ok(k)), reverse=True)
    return [k for e, k in ex[:K]], [k for n, k in ch[:K]], toks, pairs

def exact_after(toks, pairs, red):
    if not pairs: return 0.0
    e = sum(1 for i, j, *_ in pairs if all(red(toks[i + k][2]) == red(toks[j + k][2]) for k in range(3)))
    return e / len(pairs)

def score(train, test, red, vars_):
    tr = V.apply_red(train, red); te = V.apply_red(test, red)
    out = {v: V.nb_info(tr, te, v) for v in vars_}
    out['bigram'] = V.bigram_gain(tr, te)
    out['types'] = len({w for p in tr for l in p['lines'] for w in l})
    return out

def run(job):
    name, pages, vars_, inv = job
    rng = random.Random(54)
    occ = collections.Counter(c for p in pages for l in p['lines'] for w in l for c in w)
    tot = sum(occ.values())
    freq_units = [c for c, n in occ.most_common() if n / tot >= 0.004]
    res = dict(name=name, folds=[])
    for fi, te_idx in enumerate(V.folds(pages, 5, 0)):
        train = [p for i, p in enumerate(pages) if i not in te_idx]; test = [p for i, p in enumerate(pages) if i in te_idx]
        base = score(train, test, lambda w: w, vars_)
        F = dict(base=base, K={})
        for K in KS:
            ops, nops, toks, pairs = ops_from(train, K, seed=fi + 1)
            red = V.make_reduction(ops)
            col = score(train, test, red, vars_)
            nul = score(train, test, V.make_reduction(nops), vars_)
            rnd = []
            ex_r = []
            for r in range(NR):
                ro = V.random_ops(ops, freq_units, rng); rr = V.make_reduction(ro)
                rnd.append(score(train, test, rr, vars_)); ex_r.append(exact_after(toks, pairs, rr))
            show = lambda k: tuple((inv.get(c, c) if inv else c) for c in k)
            F['K'][K] = dict(ops=[show(k) for k in ops], nullops=[show(k) for k in nops], col=col, nul=nul, rnd=rnd,
                             exact_col=exact_after(toks, pairs, red), exact_rnd=statistics.mean(ex_r),
                             exact_base=exact_after(toks, pairs, lambda w: w))
        res['folds'].append(F)
        print(name, fi, json.dumps({K: {v: round(F['K'][K]['col'][v] - statistics.mean(x[v] for x in F['K'][K]['rnd']), 4) for v in list(vars_) + ['bigram']} for K in KS}), flush=True)
    json.dump(res, open(os.path.join(V.CK, 'c2_%s.json' % name), 'w'), ensure_ascii=False)
    return name

if __name__ == '__main__':
    which = sys.argv[1:] or None
    zl = V.voynich('ZL3b'); vv = ('sec', 'lang', 'hand', 'quire')
    bru, M = V.brumati(); inv = {v: k for k, v in M.items()}
    bru0, M0 = V.brumati(plant=False); inv0 = {v: k for k, v in M0.items()}
    jobs = [('ZL', zl, vv, None), ('IT', V.voynich('IT2a'), vv, None),
            ('BRU_planted', bru, ('sec', 'hand'), inv), ('BRU_clean', bru0, ('sec', 'hand'), inv0),
            ('GER_real', V.german(), ('sec', 'pos'), None),
            ('ZL_markov', V.null_markov(zl, 7), vv, None), ('ZL_selfcit', V.null_selfcit(zl, 7), vv, None),
            ('BRU_planted_markov', V.null_markov(bru, 7), ('sec', 'hand'), inv)]
    if which: jobs = [j for j in jobs if j[0] in which]
    with Pool(2) as pool:
        for n in pool.imap_unordered(run, jobs): print('done', n, flush=True)
