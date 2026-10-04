"""v18 cycle 1: THE PEN REMEMBERS on the Voynich text pages (Q20 + f58).

1. Dips under 6,048 detection settings (7 ink measures x content-residualised or not x
   smoothing x 6 detectors x 6 thresholds x 3 min spacings x line-start in/out).
2. Dip spacing.
3. Reset tests: 7 boundary features, analytic z vs page x position-stratum resampling.
   PRIMARY family (fixed before looking): 'clean' detectors (gap1, gap2: darkness of the two
   words at the boundary is not used) on content-residualised ink = 1,008 settings.
   Family-wise: max|z| and a directional reset score (mean over settings of
   (-z_junc - z_bigr - z_rep + z_init)/4) against 24 page-swap surrogates.
4. Planted control: synthetic Voynich-like text (unigram x junction x repeats) whose
   generator resets at canonical dips (20% of them shifted +-1 word to mimic alignment
   error), reset probability 1, 0.5, 0.25, and 0 (null).
Checkpoints: data/results/v18/c1_*.json
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
D = os.path.join(HERE, '..', 'data', 'derived')
RES = os.path.join(HERE, '..', 'data', 'results', 'v18'); os.makedirs(RES, exist_ok=True)
CANON = ('top', True, False, 'gap2', 0.12, 3, 'all')
SIGN = {'junc': -1, 'bigr': -1, 'rep': -1, 'init': 1}


def summarise(ev, P):
    """ev rows: (st, n, [z_real, z_sur1..]); returns family stats."""
    nsur = len(ev[0][2]) - 1
    def mx(j):
        return max(abs(z) for st, n, row in ev for z in row[j].values())
    def score(j):
        return float(np.mean([sum(SIGN[f] * row[j][f] for f in SIGN) / 4 for st, n, row in ev]))
    def featmean(j, f):
        return float(np.mean([row[j][f] for st, n, row in ev]))
    real_max = mx(0); sur_max = [mx(j) for j in range(1, nsur + 1)]
    real_sc = score(0); sur_sc = [score(j) for j in range(1, nsur + 1)]
    fm = {f: (featmean(0, f), float(np.mean([featmean(j, f) for j in range(1, nsur + 1)])),
              float(np.std([featmean(j, f) for j in range(1, nsur + 1)]))) for f in FEATS}
    best = max(((abs(z), f, st, n, z) for st, n, row in ev for f, z in row[0].items()))
    return {'max_abs_z': real_max, 'p_max': (1 + sum(s >= real_max for s in sur_max)) / (1 + nsur),
            'sur_max_median': float(np.median(sur_max)),
            'reset_score': real_sc, 'p_score': (1 + sum(s >= real_sc for s in sur_sc)) / (1 + nsur),
            'sur_score_mean': float(np.mean(sur_score := sur_sc)), 'sur_score_sd': float(np.std(sur_sc)),
            'feature_mean_z': fm, 'best': [best[1], list(best[2]), best[3], best[4]]}


def main():
    t0 = time.time()
    pages = json.load(open(os.path.join(D, 'v18_words.json')))
    ZL = json.load(open(os.path.join(D, 'ZL3b_lines.json')))
    model = [r['words'] for r in ZL if r['ltype'] == 'P']
    C = Corpus(pages, glyphs, model)
    out = {'N': C.N, 'pages': C.npages}
    ck = os.path.join(RES, 'c1_dips.npz')
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True); ds = list(zip([tuple(x) for x in z['st']], z['idx']))
        ds = [(tuple([s[0], s[1] == 'True', s[2] == 'True', s[3], float(s[4]), int(s[5]), s[6]]), i) for s, i in ds]
    else:
        ds = all_dips(C)
        np.savez(ck, st=np.array([[str(v) for v in st] for st, _ in ds]), idx=np.array([i for _, i in ds], dtype=object))
    print('dips', len(ds), round(time.time() - t0), flush=True)
    # spacing
    sp = {}
    for st, idx in ds:
        if st[3] in CLEAN and st[1] and st[2] is False and st[6] == 'all' and st[5] == 1:
            d = np.zeros(C.N, bool); d[idx] = True
            sp[str(st)] = spacing_stats(C, d)
    can = [i for st, i in ds if st == CANON][0]
    d = np.zeros(C.N, bool); d[can] = True
    out['canon'] = {'setting': list(CANON), 'spacing': spacing_stats(C, d)}
    gm = [v['median_gap_words'] for v in sp.values()]
    out['spacing_q'] = {q: float(np.median([v['median_gap_words'] for k, v in sp.items() if ', %s,' % q in k])) for q in QS}
    maps = [page_swap_map(C, k) for k in range(1, C.npages)]
    prim = [x for x in ds if x[0][3] in CLEAN and x[0][1]]
    ev = evaluate(C, prim, maps)
    out['primary'] = summarise(ev, C.npages); print('primary', out['primary'], flush=True)
    evf = evaluate(C, ds, maps)
    out['full'] = summarise(evf, C.npages); print('full', {k: out['full'][k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'best')}, flush=True)
    # circular-shift null for best primary setting
    bf, bst = out['primary']['best'][0], tuple(out['primary']['best'][1])
    bidx = [i for st, i in ds if st == bst][0]
    rng = np.random.default_rng(1)
    zs = []
    for r in range(1000):
        dd = np.zeros(C.N, bool)
        for p in range(C.npages):
            idx = np.where(C.page == p)[0]; n = len(idx)
            loc = np.searchsorted(idx, bidx[(bidx >= idx[0]) & (bidx <= idx[-1])]) if n else []
            sh = rng.integers(5, max(6, n - 5))
            dd[idx[(np.array(loc, int) + sh) % n]] = True
        zs.append(C.zscores(dd)[bf])
    obs = C.zscores(np.isin(np.arange(C.N), bidx))[bf]
    out['best_circular'] = {'feature': bf, 'z': obs, 'p_two_sided': float((1 + np.sum(np.abs(zs) >= abs(obs))) / 1001)}
    json.dump(out, open(os.path.join(RES, 'c1_real.json'), 'w'), indent=1, default=str)
    print('real done', round(time.time() - t0), flush=True)
    # planted
    gen_lines = C.lines_of_text()
    canm = np.zeros(C.N, bool); canm[can] = True
    plant = {}
    for strength in (1.0, 0.5, 0.25, 0.0):
        res = []
        for trial in range(3):
            rng = np.random.default_rng(100 + trial)
            mask = np.zeros(C.N, bool)
            for t in np.where(canm)[0]:
                if rng.random() < strength:
                    u = t + (rng.choice([-1, 1]) if rng.random() < 0.2 else 0)
                    if 0 <= u < C.N:
                        mask[u] = True
            G = Generator(C, gen_lines, seed=200 + trial)
            words = G.text(mask)
            Cs = Corpus(pages, glyphs, model)
            Cs.word = words; Cs.fit_model(Cs.lines_of_text()); Cs.features()
            ev = evaluate(Cs, prim, maps)
            s = summarise(ev, C.npages)
            res.append({k: s[k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'best', 'sur_score_mean', 'sur_score_sd')})
            print('planted', strength, trial, res[-1], flush=True)
        plant[str(strength)] = res
        json.dump(plant, open(os.path.join(RES, 'c1_planted.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main()
