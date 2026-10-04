"""v30 dialect ladder driver. Usage: python3 v30_ladder.py <tag> [kmax] [n_cand] [ctx] [seeds] [pairs...]

Each pair (X -> Y): X and Y split into disjoint fit / held pages; the rewrite is searched on
fit only and scored on held. Floor F = D(Y-sample, Y_held) of the same size (sampling noise).
closure(k) = (D0 - Dk) / (D0 - F) on held data.
"""
import sys, os, json, time, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v30_lib import run_search, split_pages, distance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, 'data', 'v30_ckpt')
BUD = {'M': (3000, 1500), 'S': (2000, 1400)}

# name: (X corpus, Y corpus, mode, budget, rung)
PAIRS = {
    'N_AA': ('V_A', 'V_A', 'idx', 'M', 'null'),
    'N_BB': ('V_B', 'V_B', 'idx', 'M', 'null'),
    'N_BavBav': ('G_Bav2', 'G_Bav2', 'idx', 'M', 'null'),
    'N_LatLat': ('I_Lat', 'I_Lat', 'idx', 'M', 'null'),
    'N_MA': ('V_A', 'M_A', 'idx', 'M', 'markov'),
    'N_MB': ('V_B', 'M_B', 'idx', 'M', 'markov'),
    'V_AB': ('V_A', 'V_B', 'free', 'M', 'voynich'),
    'V_BA': ('V_B', 'V_A', 'free', 'M', 'voynich'),
    'V_B2B3': ('V_B2', 'V_B3', 'free', 'M', 'voynich'),
    'V_ZLIT': ('V_A', 'V_A_IT', 'idx', 'M', 'voynich'),
    'L_ComCom': ('I_Com1', 'I_Com2', 'free', 'M', 'same work, other scribe'),
    'L_CzSame': ('C_Mod', 'C_Mod2', 'free', 'M', 'same text, same spelling'),
    'L_CzReform': ('C_Mod', 'C_Old', 'free', 'M', 'spelling reform'),
    'L_BavBav': ('G_Bav1', 'G_Bav2', 'free', 'M', 'same dialect, other texts'),
    'L_BavAlem': ('G_Bav1', 'G_Alem', 'free', 'M', 'dialect'),
    'L_BavRip': ('G_Bav1', 'G_Rip', 'free', 'M', 'far dialect'),
    'L_LatIta': ('I_Lat', 'I_Ita', 'free', 'M', 'related language'),
    'U_GerIta': ('G_Bav1', 'I_Ita', 'free', 'M', 'unrelated'),
    'U_LatGer': ('I_Lat', 'G_Bav1', 'free', 'M', 'unrelated'),
    'U_CzGer': ('C_Mod', 'G_Bav1', 'free', 'M', 'unrelated'),
    'U_VLat': ('V_A', 'I_Lat', 'free', 'M', 'unrelated system'),
    'K_Sub': ('G_Bav2', 'K_Sub', 'idx', 'M', 'cipher: full substitution'),
    'K_Swap3': ('G_Bav2', 'K_Swap3', 'idx', 'M', 'cipher: 3 letter swaps'),
    # small budget (herbal B has only 3456 words)
    'S_ABherb': ('V_Aherb', 'V_Bherb', 'free', 'S', 'voynich'),
    'S_AA': ('V_Aherb', 'V_Aherb', 'idx', 'S', 'null'),
    'S_BavAlem': ('G_Bav1', 'G_Alem', 'free', 'S', 'dialect'),
    'S_LatIta': ('I_Lat', 'I_Ita', 'free', 'S', 'related language'),
    'S_ComCom': ('I_Com1', 'I_Com2', 'free', 'S', 'same work, other scribe'),
}

_C = None


def corpora():
    global _C
    if _C is None:
        _C = json.load(open(os.path.join(CK, 'corpora.json')))['corpora']
    return _C


def make_split(name, seed):
    C = corpora()
    xc, yc, mode, bud, _ = PAIRS[name]
    nf, nh = BUD[bud]
    if mode == 'idx':
        rng = random.Random(1000 + seed)
        idx = list(range(min(len(C[xc]), len(C[yc])))); rng.shuffle(idx)
        h = len(idx) // 2
        X = [C[xc][i] for i in idx[:h]]; Y = [C[yc][i] for i in idx[h:]]
    else:
        X, Y = C[xc], C[yc]
    xf, xh, _ = split_pages(X, nf, nh, 2 * seed + 1)
    yf, yh, _ = split_pages(Y, nf, nh, 2 * seed + 2)
    return xf, xh, yf, yh


def job(args):
    tag, name, seed, kmax, ncand, ctx = args
    out = os.path.join(CK, f'{tag}_{name}_s{seed}.json')
    if os.path.exists(out):
        return out
    t = time.time()
    xf, xh, yf, yh = make_split(name, seed)
    floor = distance(yf[:len(yh)], yh)
    floorx = distance(xf[:len(xh)], xh)
    path = run_search(xf, yf, xh, yh, kmax=kmax, n_cand=ncand, seed=seed, ctx_symbols=ctx)
    res = dict(name=name, seed=seed, pair=PAIRS[name], floor=floor, floorx=floorx, path=path,
               n=(len(xf), len(xh), len(yf), len(yh)), secs=time.time() - t)
    json.dump(res, open(out, 'w'), ensure_ascii=False, default=float)
    print(f'{name} s{seed} done {time.time() - t:.0f}s  D0 {path[0]["held"]:.3f} F {floor[0]:.3f} Dk {path[-1]["held"]:.3f} k {path[-1]["k"]}', flush=True)
    return out


if __name__ == '__main__':
    tag = sys.argv[1]; kmax = int(sys.argv[2]); ncand = int(sys.argv[3]); ctx = sys.argv[4] == '1'
    seeds = [int(s) for s in sys.argv[5].split(',')]
    names = sys.argv[6:] or list(PAIRS)
    jobs = [(tag, n, s, kmax, ncand, ctx) for s in seeds for n in names]
    with Pool(2) as p:
        for _ in p.imap_unordered(job, jobs):
            pass
