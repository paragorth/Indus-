"""v27 cycle 2: the whole deck. Every body line of the book is a card; one open path through all cards is
optimised (simulated annealing, or-opt + swap, C) under the S scorer. Do the best chains stay in their
paragraph / page / section, or cross pages (cards dealt across pages)? Reproducible across restarts and
transcriptions? Positive controls: Latin / Italian herbals (cards from all pages shuffled together);
planted: Voynich F3 resynthesis with a hidden order planted along a random permutation that crosses pages.
"""
import sys, os, json, random, time, itertools
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_lib as L
import v8_lib

HDR = v8_lib.page_headers()
ITERS = int(os.environ.get('V27_ITERS', 300_000_000))
RESTARTS = 4


def build(nm, ver, s):
    C = L.corpus('ZL' if nm == 'PL' else nm)
    rng = random.Random(1000 + s)
    hidden = None
    if ver == 'real': pass
    elif ver == 'wshuf': C = L.word_shuffle(C, rng)
    elif ver == 'F5': C = L.forge(C, 'F5', 100 + s)[0]
    elif ver == 'cross': C = L.cross_page(C, 200 + s)
    elif ver.startswith('plant'):
        # hidden order: body cards of each section in a random order (crossing pages); plant along it
        rj, rl = {'plant60': (0.6, 0.3), 'plant30': (0.3, 0.15)}[ver]
        G, f = L.forge(C, 'F3', 300 + s)
        K = L.cards(G)
        bysec = {}
        for i, c in enumerate(K): bysec.setdefault(c['sec'], []).append(i)
        hidden = []
        for sec, ix in bysec.items():
            rng.shuffle(ix); hidden += list(zip(ix, ix[1:]))
        loc = {}
        for pg in G:
            for qi, pa in enumerate(pg['paras']):
                for li, l in enumerate(pa):
                    if li: loc[len(loc)] = l
        for a, b in hidden:
            la, lb = loc[a], loc[b]
            if rng.random() < rj:
                tab = f._get(('j', K[b]['sec'], 'p1', la[-1][-1]), ('j', K[b]['sec'], 'any', la[-1][-1]))
                if tab: lb[0] = L.V21._draw(rng, tab)
            if rng.random() < rl:
                lb[rng.randrange(len(lb))] = rng.choice(la)
        C = G
    return C, hidden


def metrics(K, M, path, hidden=None):
    N = len(K)
    E = list(zip(path, path[1:]))
    sp = sum(K[a]['page'] == K[b]['page'] for a, b in E) / len(E)
    sq = sum(K[a]['page'] == K[b]['page'] and K[a]['qi'] == K[b]['qi'] for a, b in E) / len(E)
    ss = sum(K[a]['sec'] == K[b]['sec'] for a, b in E) / len(E)
    # random-pair base rates
    from collections import Counter
    pc = Counter(c['page'] for c in K); qc = Counter((c['page'], c['qi']) for c in K); sc = Counter(c['sec'] for c in K)
    base = lambda cnt: sum(v * (v - 1) for v in cnt.values()) / (N * (N - 1))
    bif = lambda c: HDR.get(c['page'], {}).get('B')
    cross = [(a, b) for a, b in E if K[a]['page'] != K[b]['page']]
    sb = sum(1 for a, b in cross if bif(K[a]) and bif(K[a]) == bif(K[b])) / max(1, len(cross))
    # bifolio base among cross-page pairs of the same section
    if bif(K[0]) is not None:
        bc = Counter(bif(c) for c in K)
        sb_base = (sum(v * v for v in bc.values()) - sum(v * v for v in pc.values())) / max(1, N * N - sum(v * v for v in pc.values()))
    else: sb_base = None
    W = {(i, i + 1) for i in range(N - 1) if K[i]['page'] == K[i + 1]['page'] and K[i]['qi'] == K[i + 1]['qi']}
    if hidden is not None: W = set(hidden)
    hit = [e in W for e in E]
    runs, r = [], 0
    for h in hit:
        if h: r += 1
        else:
            if r: runs.append(r)
            r = 0
    if r: runs.append(r)
    pagerun, r = [], 1
    for a, b in E:
        if K[a]['page'] == K[b]['page']: r += 1
        else: pagerun.append(r); r = 1
    pagerun.append(r)
    true_pagerun = []
    r = 1
    for i in range(N - 1):
        if K[i]['page'] == K[i + 1]['page']: r += 1
        else: true_pagerun.append(r); r = 1
    true_pagerun.append(r)
    sc_edges = np.array([M[a, b] for a, b in E])
    return dict(same_page=round(sp, 4), same_page_base=round(base(pc), 4), same_para=round(sq, 4),
                same_para_base=round(base(qc), 5), same_sec=round(ss, 4), same_sec_base=round(base(sc), 4),
                bifolio_cross=round(sb, 4), bifolio_cross_base=None if sb_base is None else round(sb_base, 4),
                written_hits=sum(hit), written_total=len(W), written_chance=round(len(W) / N, 2),
                chain_max=max(runs) if runs else 0, chain_mean=round(float(np.mean(runs)), 2) if runs else 0,
                chains_ge3=sum(1 for x in runs if x >= 3), pagerun_mean=round(float(np.mean(pagerun)), 2),
                pagerun_true_mean=round(float(np.mean(true_pagerun)), 2), edge_mean=round(float(sc_edges.mean()), 3))


def run(job):
    nm, ver, s = job
    fn = os.path.join(L.CK, f'c2_{nm}_{ver}_{s}.json')
    if os.path.exists(fn): return json.load(open(fn))
    t = time.time()
    C, hidden = build(nm, ver, s)
    sc = L.Scorer(C)
    K = L.cards(C)
    M = L.global_matrix(sc, K, 'S')
    N = len(K)
    rng = np.random.RandomState(s)
    off = M[rng.randint(0, N, 300000), rng.randint(0, N, 300000)]
    off = off[off > -1e3]
    W = [(i, i + 1) for i in range(N - 1) if K[i]['page'] == K[i + 1]['page'] and K[i]['qi'] == K[i + 1]['qi']]
    paths, scores = [], []
    for r in range(RESTARTS):
        sco, p = L.global_solve(M, 1000 * s + r, ITERS)
        paths.append(p); scores.append(sco)
    b = int(np.argmax(scores))
    eds = [set(zip(p, p[1:])) for p in paths]
    jac = [len(a & c) / len(a | c) for a, c in itertools.combinations(eds, 2)]
    # an edge found in ALL restarts = a stable link
    stable = set.intersection(*eds)
    out = dict(job=job, N=N, scores=scores, written_score=float(sum(M[a, c] for a, c in W)), n_written=len(W),
               off_mean=float(off.mean()), off_sd=float(off.std()), restart_jaccard=round(float(np.mean(jac)), 4),
               stable_edges=len(stable), m=metrics(K, M, paths[b], hidden),
               stable_same_page=round(sum(K[a]['page'] == K[c]['page'] for a, c in stable) / max(1, len(stable)), 4),
               cids=[list(map(str, K[i]['cid'])) for i in paths[b]], secs=time.time() - t)
    if hidden is not None:
        out['stable_hidden'] = sum(1 for e in stable if e in set(hidden))
    json.dump(out, open(fn, 'w'))
    return out


JOBS = [('LA', 'real', 0), ('IT', 'real', 0), ('ZL', 'real', 0), ('IT2a', 'real', 0),
        ('ZL', 'F5', 1), ('ZL', 'cross', 1), ('ZL', 'wshuf', 1), ('PL', 'plant60', 1), ('PL', 'plant30', 1),
        ('LA', 'F5', 1), ('IT', 'F5', 1)]

if __name__ == '__main__':
    t = time.time()
    with Pool(2) as P:
        res = P.map(run, JOBS, chunksize=1)
    print('done', round(time.time() - t))
    for r in res:
        print(r['job'], r['N'], 'jac', r['restart_jaccard'], 'stable', r['stable_edges'], r.get('stable_hidden'),
              'stable_same_page', r['stable_same_page'], r['m'])
