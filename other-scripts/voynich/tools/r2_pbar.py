"""R2 'layout given' variant: for each corpus write <name>.pbar = baseline probability of the
line-end token '|' at every position (same KN models, folds and order as r2_data.build).
The C engine in mode 2/3 then removes line-end positions from scoring and renormalises
the baseline over non-'|' symbols, so a program cannot win by predicting line length.
usage: python3 r2_pbar.py name [name...]"""
import os, sys, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_data as RD
from r2_report import load_bin


def main(name):
    out = os.path.join(RD.OUT, name + '.pbar')
    if os.path.exists(out): return
    meta = json.load(open(os.path.join(RD.OUT, name + '.json')))
    D = load_bin(name)
    k, V = meta['kn_order'], meta['V']
    bar = meta['symbols'].index('|')
    X, DOC, SP = D['X'], D['DOC'], D['SP']
    xs, splits = [], []
    for d in range(D['ND']):
        m = DOC == d
        xs.append(X[m].tolist()); splits.append('ABC'[SP[m][0]])
    A = [i for i, s in enumerate(splits) if s == 'A']
    rng = random.Random(7)
    fold = {i: rng.random() < 0.5 for i in A}
    m = RD.KN([xs[i] for i in A], k, V)
    m0 = RD.KN([xs[i] for i in A if fold[i]], k, V)
    m1 = RD.KN([xs[i] for i in A if not fold[i]], k, V)
    pb = []
    for i, s in enumerate(splits):
        mod = m if s != 'A' else (m1 if fold[i] else m0)
        seq = [-1] * k + xs[i]
        pb.extend(mod.p(seq[j - k:j], bar) for j in range(k, len(seq)))
    pb = np.array(pb)
    # sanity: at '|' positions this must equal PB
    chk = np.abs(pb[X == bar] - D['PB'][X == bar]).max()
    print(name, 'check', chk, flush=True)
    pb.astype(np.float64).tofile(out)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        main(n)
