"""v73: the v67 kill test for single-word-per-line selectors. For the 20 best discovery rules of cycle 1 on each
target, the item stream's ordered adjacent pairs (item of line l, item of line l+1, same page) are counted:
  REC  = pairs that recur on >= 2 different pages, as z against 200 within-page shuffles of the stream
  ASYM = among unordered pairs seen in both orders or recurring, share seen in one consistent order, minus the
         shuffle mean (lists and texts have a preferred order; copy-and-vary drift does not)."""
import os, sys, json
import numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V, v73_c1 as C1
from v73_c1_report import zscores, PRIM, NULLSET


def pair_stats(pages_tok):
    pp = defaultdict(set); cnt = Counter()
    for pi, s in enumerate(pages_tok):
        for a, b in zip(s, s[1:]):
            pp[(a, b)].add(pi); cnt[(a, b)] += 1
    rec = sum(1 for k, v in pp.items() if len(v) >= 2 and k[0] != k[1])
    tot = 0; cons = 0
    for (a, b), n in cnt.items():
        if a >= b: continue
        m = cnt.get((b, a), 0)
        if n + m >= 2: tot += 1; cons += (n == 0 or m == 0)
    return rec, (cons / tot if tot else np.nan)


def test(C, idx, rng, nperm=200):
    pages = defaultdict(list)
    for l in range(C.nl): pages[C.line_page[l]].append(int(C.tok[idx[l]]))
    pt = [pages[k] for k in sorted(pages)]
    r0, a0 = pair_stats(pt)
    rs, as_ = [], []
    for _ in range(nperm):
        sh = [list(rng.permutation(s)) for s in pt]; r, a = pair_stats(sh); rs.append(r); as_.append(a)
    return dict(REC=r0, RECz=float((r0 - np.mean(rs)) / (np.std(rs) + 1e-9)), ASYM=float(a0 - np.nanmean(as_)),
                ASYMz=float((a0 - np.nanmean(as_)) / (np.nanstd(as_) + 1e-9)))


if __name__ == '__main__':
    T = C1.targets(); bank, names = L.make_bank(C1.NR)
    want = ['V_ZL3b', 'PL_POS2_MK2_VOY', 'PL_AFTER_SELFCIT_NOVEL', 'PL_POS2_SELFCIT_VOY', 'FP_SELFCIT', 'FP_MK2']
    out = {}
    for t in want:
        R = np.load(os.path.join(L.CK, 'S_%s__REAL.npy' % t))
        N = np.stack([np.load(os.path.join(L.CK, 'S_%s__%s.npy' % (t, k))) for k in NULLSET])
        comp = zscores(R, N)[:, :, PRIM].sum(2); order = np.argsort(-comp[:, 0])[:20]
        C = L.Corpus(T[t], t); rng = np.random.default_rng(7)
        res = [test(C, C.select(bank[i]), rng) for i in order]
        rr = [test(C, C.random_select(s), rng) for s in range(5)]
        out[t] = dict(rules=res, random=rr)
        f = lambda xs, k: round(float(np.nanmedian([x[k] for x in xs])), 2)
        print(t, 'top20 median RECz', f(res, 'RECz'), 'ASYMz', f(res, 'ASYMz'), 'max RECz', round(max(x['RECz'] for x in res), 2),
              '| random-word RECz', f(rr, 'RECz'), 'ASYMz', f(rr, 'ASYMz'), flush=True)
    L.jsave('pairs_c1.json', out)
