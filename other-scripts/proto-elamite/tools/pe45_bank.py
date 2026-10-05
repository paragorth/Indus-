"""pe45 reference bank: random naming grammars from the prior, simulated on a corpus template
(tablet sizes and name lengths of PE or UR3), summary statistics saved in chunks of 1000.
usage: python3 pe45_bank.py PE|UR3 n_chunks seed0"""
import os, sys, json, random, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe45_common import prior, simulate, stats, encode, load_corpora, PNAMES, CK  # noqa


def main(which, nch, seed0):
    D = load_corpora()
    C, _ = encode([t['names'] for t in D[which]])
    sizes = [len(t) for t in C]
    lens = [len(n) for t in C for n in t]
    od = os.path.join(CK, 'bank_' + which)
    os.makedirs(od, exist_ok=True)
    for ch in range(seed0, seed0 + nch):
        fn = os.path.join(od, 'c%05d.npy' % ch)
        if os.path.exists(fn):
            continue
        rng = random.Random(1000003 * ch + (7 if which == 'UR3' else 0))
        rows = []
        t0 = time.time()
        for i in range(1000):
            th = prior(rng)
            Cs, _, _ = simulate(th, sizes, lens, rng)
            s = stats(Cs, seed=i)
            rows.append([th[k] for k in PNAMES] + list(s))
        np.save(fn + '.tmp.npy', np.array(rows, dtype=np.float32))
        os.replace(fn + '.tmp.npy', fn)
        print(which, ch, round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
