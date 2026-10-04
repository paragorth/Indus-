"""v30 cycle 3: WHERE does the rewrite live, and which way is it cheap?
Same search restricted to (edge) word-initial/final contexts only, or (free) context-free rules only.
Direction asymmetry: X->Y vs Y->X for the same pair. Usage: python3 v30_c3.py
"""
import sys, os, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v30_lib import run_search, distance, PAD
from v30_ladder import make_split, CK, PAIRS

FAM = {'edge': [(PAD, None), (None, PAD)], 'free': [(None, None)]}


def job(a):
    name, fam, seed = a
    out = os.path.join(CK, f'c3{fam}_{name}_s{seed}.json')
    if os.path.exists(out):
        return
    xf, xh, yf, yh = make_split(name, seed)
    floor = distance(yf[:len(yh)], yh)
    path = run_search(xf, yf, xh, yh, kmax=20, n_cand=150, seed=seed, ctx_pool=FAM[fam], guided=True)
    json.dump(dict(name=name, seed=seed, pair=PAIRS[name], floor=floor, path=path), open(out, 'w'), default=float, ensure_ascii=False)
    print(name, fam, 'done', round(path[0]['held'], 3), round(path[-1]['held'], 3), round(floor[0], 3), flush=True)


if __name__ == '__main__':
    names = ['V_AB', 'V_BA', 'N_AA', 'L_CzReform', 'L_BavAlem', 'L_LatIta', 'L_ComCom', 'L_BavBav', 'K_Swap3']
    jobs = [(n, f, 0) for n in names for f in FAM]
    with Pool(2) as p:
        list(p.imap_unordered(job, jobs))
