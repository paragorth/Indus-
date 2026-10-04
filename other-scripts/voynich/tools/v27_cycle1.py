"""v27 cycle 1: within-paragraph card sequencing. Real corpora, nulls, planted and negative controls.

For every corpus version, every paragraph (head fixed, body lines = cards) is sequenced for the best path
under J, L and S scorers fitted on that version. The solver never sees the written order.
"""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_lib as L

JOBS = []
for nm in ('LA', 'IT', 'ZL', 'IT2a'):
    JOBS.append((nm, 'real', 0))
    for s in (1, 2):
        JOBS += [(nm, 'wshuf', s), (nm, 'wshuf_ends', s), (nm, 'F3', s), (nm, 'F5', s), (nm, 'cross', s)]
for s in (1, 2):
    JOBS += [('ZL', 'plantJ30', s), ('ZL', 'plantJ60', s), ('ZL', 'plantL30', s), ('ZL', 'plantJ15L15', s)]


def build(nm, ver, s):
    C = L.corpus(nm)
    rng = random.Random(1000 + s)
    if ver == 'real': return C
    if ver == 'wshuf': return L.word_shuffle(C, rng)
    if ver == 'wshuf_ends': return L.word_shuffle(C, rng, keep_ends=True)
    if ver in ('F3', 'F5'): return L.forge(C, ver, 100 + s)[0]
    if ver == 'cross': return L.cross_page(C, 200 + s)
    if ver == 'plantJ30': return L.plant(C, 300 + s, rho_j=0.3)
    if ver == 'plantJ60': return L.plant(C, 300 + s, rho_j=0.6)
    if ver == 'plantL30': return L.plant(C, 300 + s, rho_l=0.3)
    if ver == 'plantJ15L15': return L.plant(C, 300 + s, rho_j=0.15, rho_l=0.15)


def run(job):
    nm, ver, s = job
    fn = os.path.join(L.CK, f'c1_{nm}_{ver}_{s}.json')
    if os.path.exists(fn): return json.load(open(fn))
    C = build(nm, ver, s)
    sc = L.Scorer(C)
    out = {'job': job}
    for k in ('J', 'L', 'S'):
        R = L.sequence_corpus(C, sc, k, seed=s * 10 + 1, nr=200, restarts=6, iters=60000)
        out[k] = L.summarise(R)
        if k == 'S' and ver == 'real':
            out['paths'] = [(r['pid'], r['qi'], r['path']) for r in R]
    json.dump(out, open(fn, 'w'))
    return out


if __name__ == '__main__':
    t = time.time()
    with Pool(2) as P:
        res = P.map(run, JOBS, chunksize=1)
    print('done', round(time.time() - t))
    for r in res:
        print(r['job'], {k: {x: r[k][x] for x in ('Wz', 'Gz', 'gapz', 'PE', 'mb_rate', 'adj_rate', 'adj_rate_chance')} for k in ('J', 'L', 'S')})
