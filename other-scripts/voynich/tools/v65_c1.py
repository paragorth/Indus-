"""v65 cycle 1: carry-over strength in the current binding, against every physically
allowed arrangement of each quire; controls and nulls.

Per corpus and word class (hash halves 0 / 1; feature = word or payload skeleton):
  seam_adj  mean double-centred tail->head overlap over binding-adjacent pages
            (z against pages shuffled within section x language, 200x)
  arrow     mean of J(a->b) - J(b->a) over binding-adjacent pairs (z against sign flips)
  pct       mean percentile of the binding arrangement among all arrangements of its quire
            (quires with >= 8 arrangements; 0.5 = chance)
  seam|page partial: seam_adj after regressing out whole-page similarity (drift / topic)
"""
import os, sys, json, random, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v65_lib as L

rng = random.Random(65)
quires, leaves = L.structure()
vp, vmeta = L.voynich_pages('ZL3b')
ip, imeta = L.voynich_pages('IT2a')


def corpora():
    C = {}
    C['VOY_ZL'] = (vp, vmeta)
    C['VOY_IT'] = (ip, imeta)
    isid = L.isidore_words(); kon = L.konrad_entries(); mac = L.macer_entries()
    konflow = [w for e in kon for w in e]
    C['CTL_isidore_flow'] = (L.pour(vp, isid[5000:], 'flow', 1, quires), vmeta)
    C['CTL_konrad_flow'] = (L.pour(vp, konflow, 'flow', 2, quires), vmeta)
    C['CTL_konrad_entry'] = (L.pour(vp, kon, 'entry', 3, quires), vmeta)
    C['CTL_macer_entry'] = (L.pour(vp, mac, 'entry', 4, quires), vmeta)
    C['NULL_markov'] = (L.markov_pages(vp, vmeta, 5), vmeta)
    C['NULL_selfcit'] = (L.selfcit_pages(vp, quires, 6), vmeta)
    return C


def analyse(pages, meta, spaces_cache, feat, cls, nl=3, nperm=200):
    keys = sorted(k for k in pages if sum(len(x) for x in pages[k]) >= L.MIN_WORDS and k in meta)
    kidx = {k: i for i, k in enumerate(keys)}
    groups = [str(meta[k]['sec']) + str(meta[k]['lang']) for k in keys]
    Js, Jp = L.carry_matrices(pages, keys, nl=nl, cls=cls, feat=feat)
    Ws = L.centre(Js, groups); Wp = L.centre(Jp, groups)
    # residual seam after page similarity
    m = ~np.eye(len(keys), dtype=bool)
    x = Wp[m]; y = Ws[m]; beta = (x @ y) / (x @ x + 1e-12)
    Wr = Ws - beta * Wp
    spaces = []
    for qn, bifs in quires:
        sp = spaces_cache.get((qn, tuple(keys)))
        if sp is None:
            sp = L.QuireSpace(bifs, kidx); spaces_cache[(qn, tuple(keys))] = sp
        spaces.append((qn, sp))
    adj = []
    for qn, sp in spaces:
        adj += sp.pairs(sp.ident)
    # cross-quire adjacencies in binding order
    order = [kidx[k] for k in L.binding_order(quires) if k in kidx]
    a_idx = np.array([a for a, b in adj]); b_idx = np.array([b for a, b in adj])
    def mean_adj(W):
        return float(W[a_idx, b_idx].mean())
    res = {'n_pages': len(keys), 'n_adj': len(adj), 'beta_seam_on_page': float(beta)}
    for nm, W in (('seam', Ws), ('page', Wp), ('resid', Wr)):
        obs = mean_adj(W)
        # null: pages shuffled within group (relabel W rows/cols by a within-group permutation)
        nulls = []
        g = np.array(groups)
        for _ in range(nperm):
            perm = np.arange(len(keys))
            for x_ in set(groups):
                ix = np.where(g == x_)[0]; perm[ix] = np.random.permutation(ix)
            nulls.append(W[perm[a_idx], perm[b_idx]].mean())
        nulls = np.array(nulls)
        res[nm + '_adj'] = obs; res[nm + '_z'] = float((obs - nulls.mean()) / (nulls.std() + 1e-12))
    # arrow
    Jsn = np.nan_to_num(Js)
    d = Jsn[a_idx, b_idx] - Jsn[b_idx, a_idx]
    flips = np.array([np.mean(d * np.random.choice([-1, 1], len(d))) for _ in range(2000)])
    res['arrow'] = float(d.mean()); res['arrow_z'] = float(d.mean() / (flips.std() + 1e-12))
    # percentile of binding arrangement per quire
    pcts = {}
    for qn, sp in spaces:
        if len(sp.configs) < 8:
            continue
        for nm, W in (('seam', Ws), ('resid', Wr)):
            sc = sp.scores(W)
            pcts.setdefault(nm, []).append(float((sc < sc[sp.ident]).mean() + 0.5 * (sc == sc[sp.ident]).mean()))
    for nm, v in pcts.items():
        res[nm + '_pct_mean'] = float(np.mean(v)); res[nm + '_pct_list'] = [round(x, 3) for x in v]
        # z for mean of uniforms
        res[nm + '_pct_z'] = float((np.mean(v) - 0.5) / (np.sqrt(1 / 12 / len(v))))
    return res


def main():
    np.random.seed(65)
    C = corpora()
    out = {}
    cache = {}
    for name, (pages, meta) in C.items():
        for feat in ('word', 'skel'):
            for cls in (None, 0, 1):
                t = time.time()
                r = analyse(pages, meta, cache, feat, cls)
                key = f'{name}|{feat}|{cls}'
                out[key] = r
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith('list')},
                      f'{time.time()-t:.0f}s', flush=True)
        json.dump(out, open(os.path.join(L.CK, 'c1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
