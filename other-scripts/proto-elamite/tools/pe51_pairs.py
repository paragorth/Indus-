"""pe51 cycle 3: PAIR deletion. Which signs cover for each other?

For sign pairs (a, b) that share >= MIN tablets, erase a, b and both, on held-out tablets of a fresh
random model per fold. Interaction I_k = D_ab - D_a - D_b (mean loss change per target, task k,
own slots of a and b excluded everywhere). I > 0: the pair is REDUNDANT (each covers for the other:
losing both hurts more than the two losses added); I < 0: overlapping damage.
Null: (a, R random other tokens of b's count in the same tablets) -> I_null; z = (I - mean)/sd.
usage: python3 pe51_pairs.py CORPUS M0 M1 [TOPN]
"""
import itertools, json, os, random, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_lib as L
from pe51_run import corpus

MIN = 6
R = 4


def tot(r, b, excl, E, ti):
    out = {}
    for key, v in r.items():
        k, c = key
        if key not in b:
            continue
        if k in ('HEAD', 'ENT') and E[ti]['tok'][c] in excl:
            continue
        a = out.setdefault(k, [0.0, 0]); a[0] += v - b[key]; a[1] += 1
    return out


def run(name, m, topn=30):
    fn = os.path.join(L.CK, 'pair_%s_%d.json' % (name, m))
    if os.path.exists(fn):
        return
    t0 = time.time()
    C = corpus(name)
    V = L.Vocab(C)
    E = [L.encode(t, V) for t in C]
    rr = random.Random(9)
    order = list(range(len(C))); rr.shuffle(order)
    fold_of = {ti: k % 5 for k, ti in enumerate(order)}
    fold = m % 5
    tr = [E[i] for i in range(len(E)) if fold_of[i] != fold]
    ho = [i for i in range(len(E)) if fold_of[i] == fold]
    arch = L.random_arch(random.Random(7000 + m))
    model = L.train(tr, V, arch, seed=7000 + m)
    jobs = [(ti, vi, None) for ti in ho for vi in range(len(L.VIEWS))]
    B = {(j[0], j[1]): r for j, r in zip(jobs, L.eval_views(model, E, jobs))}
    signs = [w for w, c in V.count.most_common() if w in V.stoi][:topn]
    if name == 'plant':
        signs = ['d%d' % i for i in range(4)] + ['h%d_%d' % (i, j) for i in range(4) for j in range(3)] + \
            ['t0', 't1'] + ['u%d' % i for i in range(4)] + ['c%d' % i for i in range(6)] + ['n%d' % i for i in range(4)]
    ids = {w: V.stoi[w] for w in signs}
    rng = random.Random(m)
    res = {}
    for a, b in itertools.combinations(signs, 2):
        ia, ib = ids[a], ids[b]
        tabs = [ti for ti in ho if ia in E[ti]['tok'] and ib in E[ti]['tok']]
        if len(tabs) < MIN // 2:
            continue
        djobs, meta = [], []
        for ti in tabs:
            e = E[ti]
            pa = [i for i, x in enumerate(e['tok']) if x == ia]
            pb = [i for i, x in enumerate(e['tok']) if x == ib]
            other = [i for i, x in enumerate(e['tok']) if x not in (0, 2, 3, 4, 5, 6, ia, ib)]
            conds = [('a', pa), ('b', pb), ('ab', pa + pb)]
            for d in range(R):
                q = rng.sample(other, min(len(pb), len(other)))
                conds += [('q%d' % d, q), ('aq%d' % d, pa + q)]
            for cn, dm in conds:
                for vi in range(len(L.VIEWS)):
                    djobs.append((ti, vi, dm)); meta.append((ti, vi, cn))
        outs = L.eval_views(model, E, djobs)
        acc = {}
        for (ti, vi, cn), r in zip(meta, outs):
            for k, (s, n) in tot(r, B[(ti, vi)], (ia, ib), E, ti).items():
                x = acc.setdefault(cn, {}).setdefault(k, [0.0, 0]); x[0] += s; x[1] += n
        res[a + '|' + b] = dict(n=len(tabs), acc=acc)
    json.dump(dict(corpus=name, m=m, arch=arch, res=res, t=time.time() - t0), open(fn, 'w'))
    print(name, m, 'pairs', len(res), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    topn = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    for m in range(int(sys.argv[2]), int(sys.argv[3])):
        run(sys.argv[1], m, topn)
