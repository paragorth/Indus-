"""LA-31 cycle 2b: power of the downwind test vs corpus size. Planted downwind / symmetric /
upwind spread (etesian sailing times, L = 64 h) on the real 29-site geography with every
document's word count multiplied by k (k = 1, 4, 16). AUC of Delta (downwind vs symmetric)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, OUT, CKPT, load_docs  # noqa
from la31_c2 import gain, incidence, mats  # noqa
from la31_c1 import plant  # noqa

rng = np.random.default_rng(3122)


def auc(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(a[:, None] > b[None]) + 0.5 * np.mean(a[:, None] == b[None]))


def main():
    travel = json.load(open(os.path.join(OUT, 'travel.json')))
    docs = load_docs(); codes = list(SITES)
    P = mats(travel, codes); Tet = P['wind_etesian']; Tsym = (Tet + Tet.T) / 2
    lines = []
    for k in (1, 4, 16):
        dk = [dict(d, words=d['words'] * k) for d in docs]
        res = {}
        for nmT, T in (('down', Tet), ('sym', Tsym), ('up', Tet.T)):
            ds = []
            for rep in range(int(os.environ.get('NREP', 16))):
                pd_ = plant(dk, codes, T, 64, rng, V=4000 * k)
                Y, size = incidence(pd_, codes, 'W')
                g1, _, base = gain(Y, size, Tet); g2, _, _ = gain(Y, size, Tet.T, base)
                ds.append(g1 - g2)
            res[nmT] = ds
        lines.append(f'x{k} corpus: Delta down {np.mean(res["down"]):+.2f}, sym {np.mean(res["sym"]):+.2f}, '
                     f'up {np.mean(res["up"]):+.2f}; AUC down>sym {auc(res["down"], res["sym"]):.2f}, '
                     f'down>up {auc(res["down"], res["up"]):.2f}')
        print(lines[-1], flush=True)
    open(os.path.join(CKPT, 'c2b_out.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
