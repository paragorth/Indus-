"""pe51 driver: train one random model on 4/5 of a corpus, play the deletion game on the held-out 1/5.

usage: python3 pe51_run.py CORPUS M0 M1 [R] [TOPK]
CORPUS in pe, pe_nw (signs shuffled within each line), pe_nx (signs re-dealt across the lines of the
tablet), ur3, pc, plant. Writes data/pe51_ckpt/del_<corpus>_<m>.json with, per sign and task:
[sum of loss change over targets, n targets, [null sums per draw]] separately for tablet halves.
"""
import json, os, random, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_lib as L


def corpus(name):
    if name == 'pe':
        return L.build_pe()
    if name == 'pe_nw':
        return L.null_within_entry(L.build_pe(), 7)
    if name == 'pe_nx':
        return L.null_across_lines(L.build_pe(), 7)
    if name == 'pe_v':
        return L.build_pe_variants()
    if name == 'ur3':
        return L.build_ur3()
    if name == 'pc':
        return L.build_pc()
    if name == 'plant':
        return L.build_plant()
    raise ValueError(name)


def run(name, m, R=5, topk=150, minf=20):
    out_fn = os.path.join(L.CK, 'del_%s_%d.json' % (name, m))
    if os.path.exists(out_fn):
        return
    t0 = time.time()
    C = corpus(name)
    V = L.Vocab(C)
    E = [L.encode(t, V) for t in C]
    half = [zlib.crc32(t['id'].encode()) % 2 for t in C]
    fold = m % 5
    rr = random.Random(9)
    order = list(range(len(C)))
    rr.shuffle(order)
    fold_of = {ti: k % 5 for k, ti in enumerate(order)}
    tr = [E[i] for i in range(len(E)) if fold_of[i] != fold]
    ho = [i for i in range(len(E)) if fold_of[i] == fold]
    arch = L.random_arch(random.Random(1000 + m))
    model = L.train(tr, V, arch, seed=m)
    t1 = time.time()
    # baseline
    jobs = [(ti, vi, None) for ti in ho for vi in range(len(L.VIEWS))]
    base = L.eval_views(model, E, jobs)
    B = {(j[0], j[1]): r for j, r in zip(jobs, base)}
    base_mean = {}
    for r in base:
        for (k, _), v in r.items():
            a = base_mean.setdefault(k, [0.0, 0]); a[0] += v; a[1] += 1
    signs = [w for w, c in V.count.most_common() if c >= minf and w in V.stoi][:topk]
    rng = random.Random(500 + m)
    res = {}
    for w in signs:
        sid = V.stoi[w]
        tabs = [ti for ti in ho if sid in E[ti]['tok']]
        if not tabs:
            continue
        djobs, meta = [], []
        for ti in tabs:
            e = E[ti]
            pos = [i for i, x in enumerate(e['tok']) if x == sid]
            other = [i for i, x in enumerate(e['tok']) if x not in (0, 2, 3, 4, 5, 6, sid)]
            dels = [pos] + [rng.sample(other, min(len(pos), len(other))) for _ in range(R)]
            for d, dm in enumerate(dels):
                for vi in range(len(L.VIEWS)):
                    djobs.append((ti, vi, dm)); meta.append((ti, vi, d))
        outs = L.eval_views(model, E, djobs)
        acc = {}  # half -> task -> [sum_real, n, [sum_null_d]]
        for (ti, vi, d), r in zip(meta, outs):
            b = B[(ti, vi)]
            h = half[ti]
            for key, v in r.items():
                k, c = key
                if k in ('HEAD', 'ENT') and E[ti]['tok'][c] == 2 and False:
                    pass
                if key not in b:
                    continue
                # skip the sign's own slots (targets whose true token is the sign)
                if k in ('HEAD', 'ENT'):
                    mode, pat = L.VIEWS[vi]
                    if E[ti]['tok'][c] == sid:
                        continue
                a = acc.setdefault(h, {}).setdefault(k, [0.0, 0, [0.0] * R])
                if d == 0:
                    a[0] += v - b[key]; a[1] += 1
                else:
                    a[2][d - 1] += v - b[key]
        res[w] = acc
    json.dump(dict(corpus=name, m=m, arch=arch, ntab=len(ho), count={w: V.count[w] for w in signs},
                   base={k: v[0] / max(1, v[1]) for k, v in base_mean.items()},
                   res=res, t_train=t1 - t0, t_del=time.time() - t1), open(out_fn, 'w'))
    print(name, m, arch, 'base', {k: round(v[0] / max(1, v[1]), 3) for k, v in base_mean.items()},
          'train %.0fs del %.0fs' % (t1 - t0, time.time() - t1), flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    topk = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    for m in range(int(sys.argv[2]), int(sys.argv[3])):
        run(name, m, R=R, topk=topk)
