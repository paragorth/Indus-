#!/usr/bin/env python3
"""pe56 cycle 2: do residues found on half A of the tablets fire on unseen half-B tablets more than
forgers that saw everything except the scored tablet?
Per split: contest on A only (N_A random architectures) -> residue set (ensemble-mean p < PTH);
fresh N_B architectures trained on A + the other B folds forge each B tablet REPS times;
transfer ratio per relation / family = real B instances / forged B expectation.
Control set: same families, same size, drawn from A's tested non-residues.
usage: pe56_c2.py NAME [NAME ...]  (names as pe56_c1.build)"""
import sys, json, time, random
from pe56_common import *
from pe56_c1 import build

N_SPLIT = int(os.environ.get('PE56_NSPLIT', '4'))
N_A = int(os.environ.get('PE56_NA', '80'))
N_B = int(os.environ.get('PE56_NB', '60'))
PTH = float(os.environ.get('PE56_PTH', '1e-4'))
REPS = 2


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, 'c2_%s.json' % name)
        if os.path.exists(fn):
            continue
        t0 = time.time()
        docs, truth = build(name)
        freq = freq_set(docs)
        logf = open(os.path.join(CK, 'c2_%s.log' % name), 'w')

        def log(s):
            logf.write(s + '\n'); logf.flush()
        out = {'name': name, 'truth': truth, 'pth': PTH, 'splits': []}
        for sp in range(N_SPLIT):
            rng = random.Random(seed('pe56c2split%s%d' % (name, sp)))
            idx = list(range(len(docs))); rng.shuffle(idx)
            A = [docs[i] for i in idx[:len(docs) // 2]]
            B = [docs[i] for i in idx[len(docs) // 2:]]
            archs = [random_arch(rng) for _ in range(N_A)]
            real, gE, gn = contest(A, archs, rng, freq, reps=REPS)
            res = residues(real, gE, gn)
            W = [k for k, R, E, M, pm, px in res if pm < PTH]
            famc = collections.Counter(k[0] for k in W)
            pool = collections.defaultdict(list)
            for k, R, E, M, pm, px in res:
                if pm >= 0.05:
                    pool[k[0]].append(k)
            ctrl = []
            for f, c in famc.items():
                ctrl += rng.sample(sorted(pool[f]), min(c, len(pool[f])))
            Wset, Cset = set(W), set(ctrl)
            # B side
            folds = 5
            fb = [i % folds for i in range(len(B))]
            realB = rel_counts(B, freq)
            Ef = collections.Counter()
            barchs = [random_arch(rng) for _ in range(N_B)]
            for a in barchs:
                for f in range(folds):
                    tr = A + [B[i] for i in range(len(B)) if fb[i] != f]
                    F = Forger(a, tr, random.Random(rng.randrange(1 << 30)))
                    nte = sum(1 for i in range(len(B)) if fb[i] == f)
                    for _ in range(REPS * nte):
                        for r in relations(F.forge(), freq):
                            if r in Wset or r in Cset:
                                Ef[r] += 1
            scale = 1.0 / (N_B * REPS)
            out['splits'].append({
                'W': [[list(k), realB.get(k, 0), Ef.get(k, 0) * scale] for k in W],
                'C': [[list(k), realB.get(k, 0), Ef.get(k, 0) * scale] for k in ctrl],
                'nA': len(A), 'nB': len(B)})
            sw = sum(realB.get(k, 0) for k in W); ew = sum(Ef.get(k, 0) for k in W) * scale
            sc = sum(realB.get(k, 0) for k in ctrl); ec = sum(Ef.get(k, 0) for k in ctrl) * scale
            log('split %d: residues %d %s; B real/forged %d/%.1f; control %d/%.1f (%.0fs)' % (
                sp, len(W), dict(famc), sw, ew, sc, ec, time.time() - t0))
        out['secs'] = time.time() - t0
        json.dump(out, open(fn, 'w'))


if __name__ == '__main__':
    main()
