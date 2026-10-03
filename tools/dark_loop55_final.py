"""Loop 55 final: duplication-matched and site-structured robustness of the cycle-3 top discriminators and the cycle-5
profile classifier. For each sample made with Indus-matched duplication (dup=indus) or Indus site structure (dup=sites),
report the top-6 single statistics next to the natural-duplication values, the L-vs-D AUC on the matched samples, and
the P(L) the raw-feature logit (trained on the natural samples of the other corpora) gives them.
Output: loop55_final_robust.txt (rows go to loop55_final.txt by hand)."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import sys, json, collections, statistics as st, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dark_loop55_common as C
import dark_loop55_c5 as C5
from dark_loop55_c2 import auc

TOP = ['same_prefix_diff_last', 'anagram_rate', 'H_cond_bigram', 'unique_text_share', 'mi_first_last', 'texts_rep3']


def main():
    nat = C5.load([C5.D1], lambda d: d.get('dup') == 'natural')
    corp = [c for c in nat if C.TYPE[c] in 'LD']; lab = {c: 1 if C.TYPE[c] == 'L' else 0 for c in corp}
    feats = C5.usable(nat, 'raw', corp)
    lines = []; P = lines.append
    for dup in ('indus', 'sites'):
        R = C5.load([C5.D1], lambda d, dup=dup: d.get('dup') == dup)
        if not R: continue
        P(f'## dup={dup}: top single statistics (matched mean | natural mean) and raw-logit P(L) (model trained on the natural samples of the other L/D corpora)')
        P('| corpus | class | n samples | distinct share | ' + ' | '.join(TOP) + ' | P(L) matched | P(L) natural (LOCO) |')
        P('|---|---|---|---|' + '---|' * len(TOP) + '---|---|')
        Pn, _ = C5.loco(nat, corp, lab, 'raw', feats, 'logit')
        for c in sorted(R, key=lambda c: (C.TYPE[c], c)):
            tr = [k for k in corp if k != c]
            Xt = np.array([C5.vec(d, 'raw', feats) for k in tr for d in nat[k]]); yt = np.array([lab[k] for k in tr for d in nat[k]])
            p = float(np.mean(C5.fit_predict(Xt, yt, np.array([C5.vec(d, 'raw', feats) for d in R[c]]), 'logit')))
            cells = []
            for k in TOP:
                m = st.mean(d['obs'][k] for d in R[c]); n0 = st.mean(d['obs'][k] for d in nat[c]) if c in nat else float('nan')
                cells.append(f'{C.fmt(float(m))} | {C.fmt(float(n0))}'.replace(' | ', ' / '))
            P(f"| {c} | {C.TYPE[c]} | {len(R[c])} | {st.mean(d['meta']['distinct_share'] for d in R[c]):.2f} | " + ' | '.join(cells) + f" | {p:.2f} | {Pn.get(c, float('nan')):.2f} |")
        Lc = [c for c in R if C.TYPE[c] == 'L']; Dc = [c for c in R if C.TYPE[c] == 'D']
        P('AUC L vs D on matched samples: ' + ', '.join(f"{k} {auc([d['obs'][k] for c in Lc for d in R[c]], [d['obs'][k] for c in Dc for d in R[c]]):.2f}" for k in TOP))
        P('')
    txt = '\n'.join(lines)
    open(os.path.join(C.DARK, 'loop55_final_robust.txt'), 'w').write(txt + '\n'); print(txt)


if __name__ == '__main__':
    main()
