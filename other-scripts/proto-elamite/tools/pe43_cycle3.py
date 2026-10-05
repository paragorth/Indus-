"""pe43 cycle 3: grow the tree down to twigs, and ask which branch sprouts next.

3A HIERARCHICAL FAMILIES: the cycle-1 random-phylogeny search applied recursively inside each family;
   a split is kept only if its held-out modularity (sub-graph, 30% tablets) beats the 95th percentile of
   200 same-size random partitions of that family (calibrated stopping). Leaves = twig families.
   PC leaves vs lexical lists (pair precision; nulls: random same-size partitions of all roots, and
   random re-partitions INSIDE each top-level family = the refinement must add lexical signal).
   PLANT leaves vs truth (ARI). PE leaves vs contexts (SYS, POS).
3B WHICH BRANCH SPROUTS NEXT: early part -> late part (PC Uruk IV -> III; PLANT t < 0.6 -> t >= 0.6;
   PE random 70% -> 30% tablets, no time). Target: root gains a form not seen in the early part.
   Predictors: log early frequency (baseline); + own vigor (distinct derived forms in early part);
   + family vigor (mean vigor of the root's family, leave-self-out; family = lexical list for PC
   truth check, search family otherwise). 5-fold CV AUC. Null: variant labels shuffled among roots.
usage: python3 pe43_cycle3.py PLANT|PC|PE
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pe43_common import *
from pe43_search import search, consensus


def split_family(A_tr, A_te, nodes, rng, depth, maxd=4, minsz=16):
    if len(nodes) < minsz or depth >= maxd:
        return [nodes]
    sub_tr = A_tr[np.ix_(nodes, nodes)]; sub_te = A_te[np.ix_(nodes, nodes)]
    if sub_tr.sum() == 0 or sub_te.sum() == 0:
        return [nodes]
    keep, _, _ = search(sub_tr, rng, n_rand=800, n_climb=16, n_keep=8, steps=2500)
    lab, _ = consensus(keep, len(nodes))
    if lab.max() == 0:
        return [nodes]
    q = modularity(sub_te, lab)
    nl = [modularity(sub_te, rng.permutation(lab)) for _ in range(200)]
    if q <= np.quantile(nl, 0.95):
        return [nodes]
    out = []
    for k in range(lab.max() + 1):
        out += split_family(A_tr, A_te, [nodes[i] for i in np.nonzero(lab == k)[0]], rng, depth + 1, maxd, minsz)
    return out


def part_A(T, name, lexf=None, truth=None, seed=0, min_tab=4):
    rng = np.random.default_rng(seed)
    n = len(T)
    perm = rng.permutation(n)
    te = set(perm[:int(0.3 * n)].tolist())
    TB = tablet_bases(T)
    cnt = Counter(b for s in TB for b in s)
    roots = sorted(b for b in cnt if cnt[b] >= min_tab)
    idx = {b: i for i, b in enumerate(roots)}
    A_tr = cooc([TB[i] for i in range(n) if i not in te], idx)
    A_te = cooc([TB[i] for i in range(n) if i in te], idx)
    keep, _, _ = search(A_tr, rng)
    top, _ = consensus(keep, len(roots))
    leaves = []
    for k in range(top.max() + 1):
        leaves += split_family(A_tr, A_te, list(np.nonzero(top == k)[0]), rng, 1)
    lab = np.zeros(len(roots), int)
    for k, L in enumerate(leaves):
        lab[L] = k
    o = {'n_roots': len(roots), 'n_top': int(top.max() + 1), 'n_leaves': len(leaves),
         'leaf_sizes': sorted([len(L) for L in leaves], reverse=True)[:30],
         'Q_te_top': float(modularity(A_te, top)), 'Q_te_leaves': float(modularity(A_te, lab))}
    fams = sorted([sorted([roots[i] for i in L], key=lambda b: -cnt[b]) for L in leaves], key=len, reverse=True)
    o['leaves'] = fams
    if lexf:
        tl = [lexf.get(b) for b in roots]
        pp, tot = pair_precision(lab, tl)
        nl = [pair_precision(rng.permutation(lab), tl)[0] for _ in range(1000)]
        # within-top shuffles
        def within_top():
            l2 = lab.copy()
            for k in range(top.max() + 1):
                ii = np.nonzero(top == k)[0]
                l2[ii] = rng.permutation(lab[ii])
            return l2
        nw = [pair_precision(within_top(), tl)[0] for _ in range(1000)]
        o['lex'] = {'pair_prec': pp, 'pairs': tot, 'null_all': [float(np.mean(nl)), float(np.std(nl))],
                    'p_all': float((np.array(nl) >= pp).mean()),
                    'null_within_top': [float(np.mean(nw)), float(np.std(nw))],
                    'p_within_top': float((np.array(nw) >= pp).mean()),
                    'base_rate': pair_precision(np.zeros(len(lab), int), tl)[0]}
        o['lex_comp'] = [(f[:10], dict(Counter(lexf[b] for b in f if b in lexf))) for f in fams if len(f) >= 3][:40]
    if truth:
        tl = [truth['fam'].get(b) for b in roots]
        o['ari_leaves'] = float(ari(list(lab), tl)); o['ari_top'] = float(ari(list(top), tl))
    from pe43_cycle1 import profiles, js_sim
    P = profiles(T, bases, ['SYS', 'POS'])
    for key in ['SYS', 'POS']:
        S = np.full((len(roots), len(roots)), np.nan)
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                S[i, j] = js_sim(P[key][roots[i]], P[key][roots[j]])
        def within(l):
            m = (l[:, None] == l[None, :]); v = S[m & ~np.isnan(S)]
            return float(v.mean()) if len(v) else np.nan
        w = within(lab)
        def within_top():
            l2 = lab.copy()
            for k in range(top.max() + 1):
                ii = np.nonzero(top == k)[0]
                l2[ii] = rng.permutation(lab[ii])
            return l2
        nw = [within(within_top()) for _ in range(200)]
        o['ctx_' + key] = {'within': w, 'null_within_top': [float(np.mean(nw)), float(np.std(nw))],
                           'z': float((w - np.mean(nw)) / (np.std(nw) + 1e-12))}
    # leaf profile summaries: dominant SYS and POS per leaf
    summ = []
    for f in fams:
        if len(f) < 3:
            continue
        cs, cp = Counter(), Counter()
        for b in f:
            cs.update(P['SYS'][b]); cp.update(P['POS'][b])
        ns, npp = sum(cs.values()), sum(cp.values())
        summ.append({'roots': f[:12], 'n': len(f), 'SYS': [(k, round(v / ns, 2)) for k, v in cs.most_common(3)],
                     'POS': [(k, round(v / npp, 2)) for k, v in cp.most_common(3)]})
    o['leaf_summary'] = summ
    fam_of = {roots[i]: int(lab[i]) for i in range(len(roots))}
    return o, fam_of


def part_B(T, early_mask, fam_of, rng, label, vshuffle=False):
    forms_e = Counter(); forms_l = Counter()
    for i, t in enumerate(T):
        for l in t['lines']:
            for f in l['forms']:
                (forms_e if early_mask[i] else forms_l)[f] += 1
    if vshuffle:
        allf = set(forms_e) | set(forms_l)
        rts = sorted({root(f) for f in allf if not f.startswith('|')})
        vm = {f: rts[rng.integers(len(rts))] for f in allf if not f.startswith('|') and is_derived(f)}
        rf = lambda f: [vm[f]] if f in vm else bases(f)
    else:
        rf = bases
    freq = Counter(); vig = defaultdict(set)
    for f, c in forms_e.items():
        for b in rf(f):
            freq[b] += c
            if is_derived(f):
                vig[b].add(f)
    new = defaultdict(int)
    for f in forms_l:
        if f not in forms_e and is_derived(f):
            for b in rf(f):
                new[b] += 1
    R = sorted(b for b in freq if freq[b] >= 2)
    y = np.array([new[b] > 0 for b in R], int)
    x_f = np.log1p([freq[b] for b in R])
    x_v = np.log1p([len(vig[b]) for b in R])
    famv = defaultdict(list)
    for b in R:
        if b in fam_of:
            famv[fam_of[b]].append(len(vig[b]))
    x_fv = []
    for b, v in zip(R, x_v):
        if b in fam_of and len(famv[fam_of[b]]) > 1:
            L = famv[fam_of[b]]
            x_fv.append(np.log1p((sum(L) - len(vig[b])) / (len(L) - 1)))
        else:
            x_fv.append(np.nan)
    x_fv = np.array(x_fv); m = ~np.isnan(x_fv)
    x_fv[~m] = np.nanmean(x_fv) if m.any() else 0
    def cv_auc(X):
        if y.sum() < 5 or (1 - y).sum() < 5:
            return np.nan
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        p = np.zeros(len(y))
        for a, b in skf.split(X, y):
            p[b] = LogisticRegression(max_iter=1000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        return float(roc_auc_score(y, p))
    o = {'n_roots': len(R), 'n_sprout': int(y.sum()),
         'auc_freq': cv_auc(np.c_[x_f]), 'auc_freq_vig': cv_auc(np.c_[x_f, x_v]),
         'auc_freq_famvig': cv_auc(np.c_[x_f, x_fv]), 'auc_all': cv_auc(np.c_[x_f, x_v, x_fv]),
         'fam_cover': float(m.mean())}
    # family-level effect beyond own frequency: permutation of family labels (among roots), 200
    if m.sum() > 20:
        base_auc = o['auc_freq_famvig']
        Rm = [b for b, mm in zip(R, m) if mm]
        nul = []
        for r in range(100):
            labs = [fam_of[b] for b in Rm]; rng.shuffle(labs)
            fo = dict(zip(Rm, labs))
            fv2 = defaultdict(list)
            for b in Rm:
                fv2[fo[b]].append(len(vig[b]))
            xx = []
            for b in R:
                if b in fo and len(fv2[fo[b]]) > 1:
                    L = fv2[fo[b]]; xx.append(np.log1p((sum(L) - len(vig[b])) / (len(L) - 1)))
                else:
                    xx.append(np.nan)
            xx = np.array(xx); xx[np.isnan(xx)] = np.nanmean(xx)
            nul.append(cv_auc(np.c_[x_f, xx]))
        o['famvig_null'] = [float(np.nanmean(nul)), float(np.nanstd(nul))]
        o['famvig_p'] = float((np.array(nul) >= base_auc).mean())
    print(label, json.dumps(o), flush=True)
    return o


def main(name):
    rng = np.random.default_rng(4303)
    lexf = truth = None
    if name == 'PE':
        T = load_pe(); early = rng.random(len(T)) < 0.7
    elif name == 'PC':
        T = load_pc(); lexf = lex_families(); early = np.array([t['period'] == 'Uruk IV' for t in T])
    else:
        T, truth = make_plant(seed=3); early = np.array([t['t'] < 0.6 for t in T])
    R = {}
    R['A'], fam_of = part_A(T, name, lexf, truth)
    print(name, 'A', json.dumps({k: v for k, v in R['A'].items() if k not in ('leaves', 'lex_comp', 'leaf_summary')}), flush=True)
    R['B'] = {'search_fam': part_B(T, early, fam_of, rng, name + ' B searchfam'),
              'vshuf': part_B(T, early, fam_of, rng, name + ' B vshuf', vshuffle=True)}
    if lexf:
        R['B']['lex_fam'] = part_B(T, early, lexf, rng, name + ' B lexfam')
    if truth:
        R['B']['true_fam'] = part_B(T, early, truth['fam'], rng, name + ' B truefam')
    if name == 'PE':
        # second, independent random split for stability
        early2 = np.random.default_rng(9).random(len(T)) < 0.7
        R['B']['split2'] = part_B(T, early2, fam_of, rng, name + ' B split2')
    json.dump(R, open(os.path.join(CK, f'c3_{name}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1])
