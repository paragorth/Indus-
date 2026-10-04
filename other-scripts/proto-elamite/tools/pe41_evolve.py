"""pe41 cycle 2: evolve the economy (not the signs) until its invented script looks like the target.
usage: python3 pe41_evolve.py TARGET GENS POP SEED [workers]
fitness = standardised distance of 12 corpus-shape statistics + alignment cost (both to TARGET).
every evaluated population is logged with its votes on TARGET signs.
"""
import sys, os, json, gzip, time, copy
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import *

CK = os.path.join(DATA, 'pe41_ckpt')
T = None


def target(name):
    if name in ('PE', 'PC'):
        tabs, _ = load_real(name); truth = None
    elif name.startswith('SHUF'):
        pe, _ = load_real('PE'); tabs = shuffle_corpus(pe, np.random.default_rng(100 + int(name[4:]))); truth = None
    elif name.startswith('HELD'):
        s = 900000 + int(name[4:])
        P = Population(sample_params(np.random.default_rng(s)), s)
        tabs, lab = P.write_corpus(); truth = lab
    k, G, A = features(tabs, maxsigns=300)
    tr = [truth.get(x, -1) for x in k] if truth else None
    return dict(name=name, signs=[str(x) for x in k], G=rankq(G), A=A, stats=corpus_stats(tabs), truth=tr)


def init(name):
    global T, SCALE
    T = target(name)
    SCALE = np.array(json.load(open(os.path.join(CK, 'stat_scale.json'))))


# spread of stats across the prior (from cycle-1 populations) for standardisation
SCALE = None


def evaluate(arg):
    seed, p = arg
    try:
        P = Population(p, seed)
        tabs, lab = P.write_corpus()
        k, G, A = features(tabs, maxsigns=400)
        if len(k) < 10:
            return dict(seed=seed, p=p, fit=99.0)
        st = corpus_stats(tabs)
        d = float(np.sqrt(np.mean(((st - T['stats']) / SCALE) ** 2)))
        m, c = align(T['G'], T['A'], rankq(G), A)
        labs = np.array([lab.get(x, -1) for x in k])
        v = [int(labs[j]) if j >= 0 else -1 for j in m]
        return dict(seed=seed, p=p, fit=d + 2.0 * c, sd=d, cost=c, v=v, L=P.L, pol=P.pol, stats=st.tolist())
    except Exception as e:
        return dict(seed=seed, p=p, fit=99.0, err=repr(e))


def mutate(p, rng, rate=0.25):
    q = copy.deepcopy(p); fresh = sample_params(rng)
    for k in q:
        if rng.random() < rate:
            q[k] = fresh[k]
    return q


if __name__ == '__main__':
    name, gens, npop, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    st = json.load(open(os.path.join(CK, 'stat_scale.json')))
    SCALE = np.array(st)
    rng = np.random.default_rng(seed)
    fn = os.path.join(CK, 'evo_%s.jsonl.gz' % name)
    meta = target(name)
    json.dump(dict(signs=meta['signs'], truth=meta['truth'], stats=meta['stats'].tolist()), open(os.path.join(CK, 'evo_%s_meta.json' % name), 'w'))
    t0 = time.time(); sd = seed * 100000
    pop = [(sd + i, sample_params(rng)) for i in range(npop * 2)]
    elite = []
    with gzip.open(fn, 'wt') as f, Pool(w, initializer=init, initargs=(name,)) as pool:
        for g in range(gens):
            res = pool.map(evaluate, pop)
            for r in res:
                r['gen'] = g; f.write(json.dumps(r) + '\n')
            f.flush()
            elite = sorted(elite + res, key=lambda r: r['fit'])[:npop // 2]
            print(name, 'gen', g, 'best', round(elite[0]['fit'], 3), 'median elite', round(float(np.median([e['fit'] for e in elite])), 3), round(time.time() - t0), flush=True)
            sd += 1000
            pop = []
            for i in range(npop):
                par = elite[int(rng.integers(len(elite)))]['p']
                if rng.random() < 0.3:
                    other = elite[int(rng.integers(len(elite)))]['p']
                    par = {k: (par[k] if rng.random() < 0.5 else other[k]) for k in par}
                pop.append((sd + i, mutate(par, rng)))
    print('done', round(time.time() - t0))
