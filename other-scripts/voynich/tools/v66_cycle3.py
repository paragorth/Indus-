"""v66 cycle 3: does keyword-graph SHAPE carry genre at all? (lexicon-free, text-to-text matching)

Cycles 1-2: every clustered text aligns to the medical concept graph, chronicles included.
Here no concept graph and no lexicon: every text's 40-keyword page co-occurrence graph is
matched against every other's (annealed injective QAP, both directions, best kept). Pair scores are
double-centred (row/column effects = how clustered each graph is) to remove generic clustering.
Test: leave-one-out nearest-neighbour genre recognition (herbal/medical vs other) and a
mean within- vs between-genre gap, both against 2000 genre-label permutations.
Two samples (different units) per text; same-text pairs excluded from neighbours.
Then: where do the Voynich graphs (ZL, IT; herbal-A, bio, stars sections) fall?
"""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v66_lib import *
from multiprocessing import Pool

K = 40
NU = 180
MED = ['culpeper', 'konrad_plants', 'macer', 'circa_fr', 'celsus_lat', 'celsus_eng', 'v21_IT', 'konrad_body', 'v21_LA']
OTH = UNRELATED + ['hyginus']


def graphs():
    T = load_texts()
    G = []
    for name in MED + OTH:
        for s in (1, 2):
            us = sample_units(T[name]['units'], NU, s * 17)
            us, back = opaque(us, s)
            kw, X, B = keyword_graph(us, K=K)
            G.append({'id': '%s#%d' % (name, s), 'text': name, 'genre': 'MED' if name in MED else 'OTH', 'B': B})
    for nm in ('ZL3b', 'IT2a'):
        keys, P, meta = voynich_pages(nm)
        kw, X, B = keyword_graph(P, K=K)
        G.append({'id': nm, 'text': nm, 'genre': 'VMS', 'B': B})
        for sec, test in (('herbal', lambda m: m[0] == 'H'), ('bio', lambda m: m[0] == 'B'), ('stars', lambda m: m[0] == 'S')):
            us = [p for k, p in zip(keys, P) if test(meta[k])]
            if len(us) < 15: continue
            kw, X, B = keyword_graph(us, K=K, )
            G.append({'id': '%s-%s(%d)' % (nm, sec, len(us)), 'text': nm + sec, 'genre': 'VMS', 'B': B})
    return G


G = None


def pair(ij):
    i, j = ij
    a, b = G[i]['B'], G[j]['B']
    s1 = qap(a, b, restarts=12, iters=120000, seed=i * 1000 + j)[0]
    s2 = qap(b, a, restarts=12, iters=120000, seed=j * 1000 + i)[0]
    return i, j, max(s1, s2)


def init(g):
    global G
    G = g


if __name__ == '__main__':
    g = graphs()
    print('graphs', len(g), [x['id'] for x in g], flush=True)
    n = len(g)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    t0 = time.time()
    with Pool(2, initializer=init, initargs=(g,)) as p:
        res = p.map(pair, pairs, chunksize=8)
    S = np.zeros((n, n))
    for i, j, s in res: S[i, j] = S[j, i] = s
    print('pairs done %.0fs' % (time.time() - t0), flush=True)
    off = ~np.eye(n, dtype=bool)
    # double-centre on off-diagonal entries
    r = np.array([S[i, off[i]].mean() for i in range(n)])
    mu = S[off].mean()
    D = S - r[:, None] - r[None, :] + mu
    np.fill_diagonal(D, np.nan)
    ref = [k for k in range(n) if g[k]['genre'] != 'VMS']
    lab = np.array([g[k]['genre'] for k in ref]); txt = [g[k]['text'] for k in ref]

    def stats(lab, M):
        acc, gap_in, gap_out = 0, [], []
        for a, ia in enumerate(ref):
            cand = [(M[ia, ib], b) for b, ib in enumerate(ref) if txt[b] != txt[a]]
            acc += lab[max(cand)[1]] == lab[a]
            for v, b in cand:
                (gap_in if lab[b] == lab[a] else gap_out).append(v)
        return acc / len(ref), np.mean(gap_in) - np.mean(gap_out)

    out = {}
    for nmM, M in (('raw', S), ('centred', D)):
        acc, gap = stats(lab, M)
        rng = np.random.default_rng(0)
        # permute genre labels at the TEXT level (both samples of a text keep one label)
        texts = sorted(set(txt)); tg = {t: lab[txt.index(t)] for t in texts}
        pa, pg = [], []
        for _ in range(2000):
            perm = rng.permutation([tg[t] for t in texts]); mp = dict(zip(texts, perm))
            a2, g2 = stats(np.array([mp[t] for t in txt]), M)
            pa.append(a2); pg.append(g2)
        out[nmM] = {'acc': acc, 'acc_p': float(np.mean(np.array(pa) >= acc)), 'acc_null': float(np.mean(pa)),
                    'gap': float(gap), 'gap_p': float(np.mean(np.array(pg) >= gap))}
        print(nmM, out[nmM], flush=True)
        # Voynich graphs: mean similarity to MED vs OTH
        for k in range(n):
            if g[k]['genre'] != 'VMS': continue
            vm = [M[k, b] for b in ref if g[b]['genre'] == 'MED']; vo = [M[k, b] for b in ref if g[b]['genre'] == 'OTH']
            nn = sorted(((M[k, b], g[b]['id']) for b in range(n) if b != k and g[b]['genre'] != 'VMS'), reverse=True)[:3]
            # same statistic for every reference graph, for a percentile
            ref_d = []
            for a in ref:
                am = [M[a, b] for b in ref if g[b]['genre'] == 'MED' and g[b]['text'] != g[a]['text']]
                ao = [M[a, b] for b in ref if g[b]['genre'] == 'OTH' and g[b]['text'] != g[a]['text']]
                ref_d.append((np.mean(am) - np.mean(ao), g[a]['genre']))
            d = np.mean(vm) - np.mean(vo)
            out.setdefault('vms_' + nmM, {})[g[k]['id']] = {'med_minus_oth': float(d), 'nn': nn,
                'pct_vs_MED': float(np.mean([d > x for x, gg in ref_d if gg == 'MED'])),
                'pct_vs_OTH': float(np.mean([d > x for x, gg in ref_d if gg == 'OTH']))}
            print('  ', g[k]['id'], round(d, 4), out['vms_' + nmM][g[k]['id']]['pct_vs_MED'], out['vms_' + nmM][g[k]['id']]['pct_vs_OTH'], nn, flush=True)
    out['ids'] = [x['id'] for x in g]; out['S'] = S.tolist()
    json.dump(out, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=float)
