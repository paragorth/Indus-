"""X1: compute the battery on every corpus under a size regime.

usage: python3 x1_run.py REGIME BUDGET NREP [corpus ...]
  REGIME label for the output file; BUDGET tokens per window ('none' = whole corpus, capped at 60,000);
  NREP windows per corpus (contiguous runs of whole texts at random starts).
Output: voynich/data/results/x1_features_<REGIME>.json  {corpus: {'class','gran','ntok_full','reps':[{feat:val}]}}
"""
import sys, os, json, random, warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')
import x1_battery as B

_CACHE = {}


def job(args):
    name, budget, rep = args
    if name not in _CACHE:
        _CACHE[name] = B.CORPORA[name][2]()
    texts = _CACHE[name]
    rng = random.Random(1000 * rep + 7)
    w = B.window(texts, budget, rng)
    return name, rep, B.battery(w, seed=rep), sum(len(t) for t in texts), len(texts)


def main():
    regime = sys.argv[1]
    budget = None if sys.argv[2] == 'none' else int(sys.argv[2])
    nrep = int(sys.argv[3])
    names = sys.argv[4:] or list(B.CORPORA)
    cap = 60000 if budget is None else budget
    jobs = [(n, cap, r) for n in names for r in range(nrep)]
    out = {}
    with Pool(min(8, os.cpu_count() or 2)) as pool:
        for name, rep, F, ntok, ntx in pool.imap_unordered(job, jobs):
            d = out.setdefault(name, {'class': B.CORPORA[name][0], 'gran': B.CORPORA[name][1],
                                      'ntok_full': ntok, 'ntexts_full': ntx, 'reps': []})
            d['reps'].append(F)
            print(regime, name, rep, flush=True)
    path = os.path.join(B.RES, f'x1_features_{regime}.json')
    old = json.load(open(path)) if os.path.exists(path) and sys.argv[4:] else {}
    old.update(out)
    json.dump(old, open(path, 'w'))
    print('wrote', path)


if __name__ == '__main__':
    main()
