"""la63 driver: train one random model on 4/5 of a corpus, play the deletion game on the held-out 1/5.

usage: python3 la63_run.py CORPUS M0 M1 [R] [MINF]
CORPUS: la, la_nw (non-logogram tokens re-dealt across the entries of each document), plant,
plant1 (second plant seed), lb, ur3. Writes data/la63_ckpt/del_<corpus>_<m>.json with, per erased
type and loss key: [sum of loss change, n targets, [null sums per draw], [null n per draw]].
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la63_lib as L


def corpus(name):
    if name == 'la':
        return L.build_la()
    if name == 'la_nw':
        return L.null_within_doc(L.build_la(), 7)
    if name == 'plant':
        return L.build_plant(0)
    if name == 'plant1':
        return L.build_plant(1)
    if name == 'dose':
        return L.build_plant_dose()
    if name == 'lb':
        return L.build_lb()
    if name == 'ur3':
        return L.build_ur3()['docs']
    raise ValueError(name)


def folds(C):
    import re
    tabs = sorted({re.sub(r'[ab]$', '', d['id']) for d in C})
    random.Random(9).shuffle(tabs)
    fk = {t: k % 5 for k, t in enumerate(tabs)}
    return {i: fk[re.sub(r'[ab]$', '', d['id'])] for i, d in enumerate(C)}


def run(name, m, R=8, minf=3):
    out_fn = os.path.join(L.CK, 'del_%s_%d.json' % (name, m))
    if os.path.exists(out_fn):
        return
    t0 = time.time()
    C = corpus(name)
    V = L.Vocab(C)
    E = [L.encode(d, V) for d in C]
    fo = folds(C)
    fold = m % 5
    tr = [E[i] for i in range(len(E)) if fo[i] != fold]
    ho = [i for i in range(len(E)) if fo[i] == fold]
    arch = L.random_arch(random.Random(1000 + m))
    model = L.train(tr, V, arch, seed=m)
    t1 = time.time()
    jobs = [(ti, vi, None) for ti in ho for vi in range(len(L.VIEWS))]
    base = L.eval_views(model, E, jobs)
    B = {(j[0], j[1]): r for j, r in zip(jobs, base)}
    base_mean = {}
    for r in base:
        for (k, _), v in r.items():
            a = base_mean.setdefault(k, [0.0, 0]); a[0] += v; a[1] += 1
    types = [w for w, c in V.count.most_common() if c >= minf and w in V.stoi]
    rng = random.Random(500 + m)
    res = {}
    for w in types:
        sid = V.stoi[w]
        tabs = [ti for ti in ho if sid in E[ti]['tok']]
        djobs, meta, ents_of = [], [], {}
        for ti in tabs:
            e = E[ti]
            pos = [i for i, x in enumerate(e['tok']) if x == sid]
            other = [i for i, x in enumerate(e['tok']) if x not in (0, 2, 3, 4, sid)]
            if not other:
                continue
            dels = [pos] + [rng.sample(other, min(len(pos), len(other))) for _ in range(R)]
            for d, dm in enumerate(dels):
                ents_of[(ti, d)] = {e['li'][p] for p in dm}
                for vi in range(len(L.VIEWS)):
                    djobs.append((ti, vi, dm)); meta.append((ti, vi, d))
        if not djobs:
            continue
        outs = L.eval_views(model, E, djobs)
        acc = {}
        for (ti, vi, d), r in zip(meta, outs):
            b = B[(ti, vi)]
            e = E[ti]
            for key, v in r.items():
                k, c = key
                if key not in b:
                    continue
                if e['tok'][c] == sid:      # its own slots (ENT/HEAD/LOGO targets) are never scored
                    continue
                loc = 'L' if e['li'][c] in ents_of[(ti, d)] else 'R'
                a = acc.setdefault(k + '-' + loc, [0.0, 0, [0.0] * R, [0] * R])
                if d == 0:
                    a[0] += v - b[key]; a[1] += 1
                else:
                    a[2][d - 1] += v - b[key]; a[3][d - 1] += 1
        res[w] = dict(acc=acc, ndoc=len({m_[0] for m_ in meta}))
    json.dump(dict(corpus=name, m=m, arch=arch, ntab=len(ho), count={w: V.count[w] for w in types},
                   base={k: v[0] / max(1, v[1]) for k, v in base_mean.items()},
                   res=res, t_train=t1 - t0, t_del=time.time() - t1), open(out_fn + '.tmp', 'w'))
    os.replace(out_fn + '.tmp', out_fn)
    print(name, m, arch['type'], arch['d'], 'base', {k: round(v[0] / max(1, v[1]), 3) for k, v in base_mean.items()},
          'train %.0fs del %.0fs' % (t1 - t0, time.time() - t1), flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    minf = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    for m in range(int(sys.argv[2]), int(sys.argv[3])):
        run(name, m, R=R, minf=minf)
