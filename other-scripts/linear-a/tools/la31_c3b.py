"""LA-31 cycle 3b: the summer-minus-winter contrast. Statistic = mean partial Mantel r (given km)
of Jun-Sep monthly wind times minus the mean of Nov-Mar, on the ALL layer (all 29 sites).
Null: site permutation (1,000). Planted: words spread under Jul-Aug sailing times (L 16 h) vs
under Dec-Feb times, 10 reps each: does the contrast tell them apart?"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, OUT, CKPT, load_docs, euclid  # noqa
from la31_c3 import partial_r, close  # noqa
from la31_stats import sharing_z  # noqa
from la31_c1 import plant  # noqa

rng = np.random.default_rng(3132)
SUM = (6, 7, 8, 9); WIN = (11, 12, 1, 2, 3)


def contrast(Z, mats, km, iu):
    r = {m: partial_r(Z, mats[m], km, iu) for m in mats}
    return np.mean([r[m] for m in SUM]) - np.mean([r[m] for m in WIN])


def main():
    travel = json.load(open(os.path.join(OUT, 'travel.json')))
    codes = list(SITES); K = len(codes); iu = np.triu_indices(K, 1)
    walk = np.array(travel['walk'], float)
    km = close(euclid(codes))
    raw = {m: np.minimum(walk, np.array(travel[f'windsea_m{m:02d}'], float)) for m in range(1, 13)}
    mats = {m: close((A + A.T) / 2) for m, A in raw.items()}
    lines = []
    Zs = np.load(os.path.join(CKPT, 'c1_Z_all29.npy'))
    for nm, Z in (('ALL', Zs.sum(0)), ('W', Zs[0]), ('S', Zs[1]), ('E', Zs[3])):
        c0 = contrast(Z, mats, km, iu)
        null = []
        for _ in range(int(os.environ.get('NPERM', 1000))):
            q = rng.permutation(K); null.append(contrast(Z[np.ix_(q, q)], mats, km, iu))
        null = np.array(null)
        lines.append(f'real {nm}: summer-winter contrast {c0:+.3f}, site-perm P {(np.sum(null >= c0) + 1) / (len(null) + 1):.3f} (null sd {null.std():.3f})')
        print(lines[-1], flush=True)
    docs = load_docs()
    Tsum = np.median([raw[m] for m in (7, 8)], 0); Twin = np.median([raw[m] for m in (12, 1, 2)], 0)
    for nmT, T in (('summer-planted', Tsum), ('winter-planted', Twin)):
        cs = []
        for _ in range(int(os.environ.get('NREP', 10))):
            pd_ = plant(docs, codes, T, 16, rng)
            Z, _ = sharing_z(pd_, codes, nperm=200, strat=True, rng=rng)
            cs.append(contrast(Z['W'], mats, km, iu))
        lines.append(f'{nmT} (W, L 16 h): contrast mean {np.mean(cs):+.3f} sd {np.std(cs):.3f}; > 0 in {np.mean(np.array(cs) > 0):.2f}')
        print(lines[-1], flush=True)
    open(os.path.join(CKPT, 'c3b_out.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
