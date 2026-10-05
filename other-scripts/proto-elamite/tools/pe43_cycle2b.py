"""pe43 cycle 2b: is the tree-'terminus post quem' date just tablet size?

Re-runs the T family (1,500 random tree-age hypotheses, top 2% by fit compactness) and scores
(a) AUC Uruk III later, raw and within strata of the number of distinct forms per tablet;
(b) the size-only baseline (number of forms) AUC;
(c) the same with ages drawn on a tree whose parent links are shuffled among forms
    (each derived form gets the parent list of another random derived form) = phylogeny null.
PE: report Spearman of the T date with tablet size and with the PC-passing direction, and freeze
the best order (hash) for later tests.
usage: python3 pe43_cycle2b.py PC|PE|PLANT
"""
import json, os, sys, hashlib
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from pe43_common import *
import pe43_common
from pe43_cycle2 import build, compact, to_ranks, gen_T


def auc(r, lab):
    a = r[lab == 1]; b = r[lab == 0]
    return float(mannwhitneyu(a, b).statistic / (len(a) * len(b))) if len(a) and len(b) else np.nan


def strat_auc(r, lab, size):
    qs = np.quantile(size, [0, .2, .4, .6, .8, 1.0])
    num = den = 0
    for i in range(5):
        m = (size >= qs[i]) & ((size < qs[i + 1]) if i < 4 else (size <= qs[i + 1]))
        a = auc(r[m], lab[m])
        if not np.isnan(a):
            w = (lab[m] == 1).sum() * (lab[m] == 0).sum(); num += a * w; den += w
    return num / den if den else np.nan


def run(T, rng, lab=None, truth=None, nh=1500, shuffle_tree=False):
    F, fidx, X, fitmask, keep_t = build(T, rng)
    if shuffle_tree:
        der = [f for f in F if parents(f)]
        pl = [parents(f) for f in der]
        rng.shuffle(pl)
        mp = dict(zip(der, pl)); der = set(der)
        orig = pe43_common.parents
        def fake(f, _o=orig):
            if f in mp:
                # keep recursion acyclic: a shuffled parent must be shallower than f in the real tree
                return [p if depth(p) < depth(f) else root(p) for p in mp[f] if root(p) != f]
            return _o(f)
        import pe43_cycle2
        pe43_cycle2.parents = fake
    H = []
    for h in range(nh):
        s = gen_T(X, fitmask, F, fidx, rng)
        r = to_ranks(s, rng)
        H.append((compact(r, X, fitmask), r))
    if shuffle_tree:
        import pe43_cycle2
        pe43_cycle2.parents = pe43_common.parents
    H.sort(key=lambda x: x[0])
    top = H[:max(20, nh // 50)]
    size = X[:, fitmask].sum(1).astype(float)
    o = {'n_tab': int(X.shape[0])}
    if lab is not None:
        L = np.array([lab[i] for i in keep_t])
        o['auc_raw'] = [float(np.mean([auc(r, L) for _, r in top])), float(np.std([auc(r, L) for _, r in top]))]
        o['auc_strat'] = [float(np.mean([strat_auc(r, L, size) for _, r in top])), float(np.std([strat_auc(r, L, size) for _, r in top]))]
        o['auc_size_only'] = auc(size + 1e-6 * rng.standard_normal(len(size)), L)
        o['auc_random_hyp'] = float(np.mean([auc(r, L) for _, r in H[::30]]))
    if truth is not None:
        tt = np.array([truth[i] for i in keep_t])
        o['rho_true'] = float(np.mean([spearmanr(r, tt)[0] for _, r in top]))
        o['rho_size_true'] = float(spearmanr(size, tt)[0])
    o['rho_date_size'] = float(np.mean([spearmanr(r, size)[0] for _, r in top]))
    best = top[0][1]
    o['best_hash'] = hashlib.sha256(np.round(best, 6).tobytes()).hexdigest()[:16]
    return o, best, keep_t


def main(name):
    rng = np.random.default_rng(4322)
    lab = truth = None
    if name == 'PC':
        T = load_pc(); lab = [int(t['period'] == 'Uruk III') for t in T]
    elif name == 'PE':
        T = load_pe()
    else:
        T, tr = make_plant(seed=2); truth = [t['t'] for t in T]
    R = {}
    R['tree'], best, kt = run(T, rng, lab, truth)
    print(name, 'tree', json.dumps(R['tree']), flush=True)
    R['tree_shuffled'], _, _ = run(T, rng, lab, truth, shuffle_tree=True)
    print(name, 'tree_shuffled', json.dumps(R['tree_shuffled']), flush=True)
    json.dump(R, open(os.path.join(CK, f'c2b_{name}.json'), 'w'), indent=1)
    if name == 'PE':
        json.dump({'tablets': [T[i]['id'] for i in kt], 'rank': best.tolist()}, open(os.path.join(CK, 'c2b_PE_order.json'), 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
