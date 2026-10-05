#!/usr/bin/env python3
"""LA-46 cycle 2: do residues found on half A predict half B better than forgers do?
For each random half split: (1) inside-out contest on A only -> residue set with weights
w = log((R+.5)/(M+.5)) for p < 1e-3; (2) forgers (fresh random architectures) trained on A + the other
B folds forge each B fold; (3) the residue score S(doc) = sum of w over the doc's relations; AUC of
S on real B docs vs forged B docs (forgers that saw everything except the scored doc).
Controls: random weight sets of equal size from A's tested pairs (same families); worlds, shuffles,
planted world, LB, UR3.
usage: la46_c2.py NAME [NAME ...]"""
import sys, json, time, random
from la46_common import *
from la46_c1 import build

N_SPLIT = int(os.environ.get('LA46_NSPLIT', '6'))
N_A = int(os.environ.get('LA46_NA', '40'))
N_B = int(os.environ.get('LA46_NB', '30'))
MODE = os.environ.get('LA46_MODE', 'mean')
PTH = float(os.environ.get('LA46_PTH', '0.01'))


def auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # rank-based with ties
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks.setdefault(k, (i + j + 1) / 2)
        i = j
    rs = sum(ranks[k] for k, (v, l) in enumerate(allv) if l == 1)
    n1, n0 = len(pos), len(neg)
    return (rs - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float('nan')


def score(doc, W):
    return sum(W.get(r, 0.0) for r in relations(doc))


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, 'c2_%s_%s.json' % (MODE, name))
        if os.path.exists(fn):
            continue
        t0 = time.time()
        docs, truth = build(name)
        logf = open(os.path.join(CK, 'c2_%s_%s.log' % (MODE, name)), 'w')

        def log(s):
            logf.write(s + '\n'); logf.flush()
        out = {'name': name, 'splits': []}
        for sp in range(N_SPLIT):
            rng = random.Random(seed('c2split%s%d' % (name, sp)))
            idx = list(range(len(docs))); rng.shuffle(idx)
            A = [docs[i] for i in idx[:len(docs) // 2]]
            B = [docs[i] for i in idx[len(docs) // 2:]]
            real, gE, _, _ = contest(A, N_A, rng)
            res = residues_mean(real, gE, contest.last_gn) if MODE == 'mean' else residues(real, gE)
            W = {}
            fam_ct = collections.Counter()
            for k, R, M, m, p in res:
                if p < PTH:
                    W[k] = math.log((R + .5) / (M + .5)); fam_ct[k[0]] += 1
            # random control weight set: same families, same size, from tested pairs, same weights shuffled
            pool = collections.defaultdict(list)
            for k, R, M, m, p in res:
                if p >= PTH:
                    pool[k[0]].append(k)
            wv = list(W.values()); rng.shuffle(wv)
            Wr = {}
            it = iter(wv)
            for f, c in fam_ct.items():
                for k in rng.sample(pool[f], min(c, len(pool[f]))):
                    Wr[k] = next(it, 1.0)
            # forgers on B folds (trained on A + other B folds)
            folds = 5
            fb = [i % folds for i in range(len(B))]
            forged = []
            for ai in range(N_B):
                a = random_arch(rng)
                for f in range(folds):
                    tr = A + [B[i] for i in range(len(B)) if fb[i] != f]
                    nte = sum(1 for i in range(len(B)) if fb[i] == f)
                    F = Forger(a, tr, random.Random(rng.randrange(1 << 30)))
                    for _ in range(nte):
                        forged.append(F.forge())
            sr = [score(d, W) for d in B]
            sf = [score(d, W) for d in forged]
            rr = [score(d, Wr) for d in B]
            rf = [score(d, Wr) for d in forged]
            # per family AUC
            fam_auc = {}
            for f in fam_ct:
                Wf = {k: v for k, v in W.items() if k[0] == f}
                fam_auc[f] = auc([score(d, Wf) for d in B], [score(d, Wf) for d in forged])
            # which residues fire in real B vs forged B (per-pair held-out excess)
            cr = collections.Counter(); cf = collections.Counter()
            for d in B:
                cr.update(r for r in relations(d) if r in W)
            for d in forged:
                cf.update(r for r in relations(d) if r in W)
            ratio = len(forged) / len(B)
            fire = sorted([[list(k), cr[k], cf[k] / ratio] for k in W], key=lambda x: -(x[1] - x[2]))
            rec = {'split': sp, 'nres': len(W), 'fam_ct': dict(fam_ct), 'auc': auc(sr, sf), 'auc_rand': auc(rr, rf),
                   'fam_auc': fam_auc, 'fire': fire[:60],
                   'mean_real': sum(sr) / len(sr), 'mean_forg': sum(sf) / len(sf)}
            if truth:
                T = set(tuple(t) for t in truth)
                rec['planted_in_W'] = len(T & set(W))
            out['splits'].append(rec)
            log('split %d nres %d auc %.3f rand %.3f %s %.0fs' % (sp, len(W), rec['auc'], rec['auc_rand'],
                                                                 {f: round(v, 3) for f, v in fam_auc.items()},
                                                                 time.time() - t0))
        json.dump(out, open(fn, 'w'))


if __name__ == '__main__':
    main()
