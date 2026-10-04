"""pe7 cycle 1: tree-likeness of name corpora (delta score, CI/RI of the best
parsimony tree, MCMC clade support) vs a shuffled-element null (tokens permuted
across names, lengths kept).  Positive: Ur III / OB father+son pairs, Chinese
full names grouped by surname.  Negative: Linear B personnel names, random
strings with PE sign frequencies.  2 workers, checkpointed per run."""
import sys, random, json, os, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import *  # noqa

CORP = ['PE', 'PE_tab', 'UR3_SEAL', 'OB_SEAL', 'UR3_PAT', 'OB_PAT', 'CN_FULL', 'LINB', 'RAND']
NREP, NNULL, N = 6, 2, 200


def run(job):
    corp, rep, kind = job
    key = 'c1_%s_%d_%s' % (corp, rep, kind)
    r = ckpt(key)
    if r:
        return key, r
    C = corpora()
    rng = random.Random(1000 * rep + 7)
    names, _ = sample_corpus(C, corp, N, rng)
    if kind != 'real':
        rng2 = random.Random(1000 * rep + 13 + int(kind[-1]))
        names = shuffle_elements(names, rng2)
    t0 = time.time()
    out, *_ = tree_stats(names, random.Random(rep * 31 + 5), restarts=6)
    out.pop('_co', None)
    out['rates'] = [float(x) for x in out['rates']]
    out['sec'] = time.time() - t0
    save_ckpt(key, out)
    return key, out


if __name__ == '__main__':
    jobs = [(c, r, k) for r in range(NREP) for c in CORP for k in ['real'] + ['null%d' % i for i in range(NNULL)]]
    res = {}
    with Pool(2) as pool:
        for key, out in pool.imap_unordered(run, jobs):
            res[key] = out
            print(key, round(out['delta'], 4), round(out['ri'], 4), round(out.get('support_mean', 0), 4), round(out['sec']), flush=True)
    json.dump(res, open(os.path.join(DATA, 'pe7_cycle1.json'), 'w'))
