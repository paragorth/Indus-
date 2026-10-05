"""v56 cycle 2: sharper pointer score + MARKED pointers. The cycle-1 residual sum let wrong rules overfit above the true
rule. Here a pointer scores 1 when its target is among the 10 pages its context most resembles (column-centred top-10
indicator; same pairing correction, held-out unseen word types, type-robust z). Selectors: cycle-1 selectors plus
'the word after W' for the 25 most frequent preceding words (medieval references carry a marker: 'vide', 'cap.').
Controls: Culpeper with 'vide' + additive Roman numeral (real cross-reference graph); Voynich with planted 3-digit
base-7 pointers, 2 per page, with and without a marker word ('chol') before them. Nulls: relabelled pages, Markov,
self-citation; Voynich ZL and IT2a."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v56_lib as L
from multiprocessing import Pool

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c2'
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1500
FOCUSED = os.environ.get('V56_FOCUSED') == '1'
SELK = os.environ.get('V56_SELK', 'both')


def corpus(name):
    if name == 'CULP_vide': return L.culpeper(marker=True)
    if name == 'IT2a': return L.voynich('IT2a')
    Z = L.voynich('ZL3b')
    if name == 'ZL': return Z
    if name == 'ZL_markov': return L.markov_corpus(Z)
    if name == 'ZL_selfcit': return L.selfcit_corpus(Z)
    if name == 'ZL_relab': return L.relabel_corpus(Z)
    if name == 'PL_mark2': return L.plant_pointers(Z, L.build_R, rate_per_page=2, marker='chol')
    if name == 'PL2': return L.plant_pointers(Z, L.build_R, rate_per_page=2)
    raise ValueError(name)


def true_rule(C, E, TT):
    """held-out robust z of the planted / Roman rule on its natural selectors (power reference)."""
    tr = L.split_types(TT, 0); out = {}
    if 'digits' in C:
        coef = L.pos_weights(TT, True, 7, True); d = np.zeros(len(TT['glyphs']), np.int64)
        for k, g in enumerate(C['digits']): d[TT['gi'][g]] = k
        sels = [('all', np.ones(len(tr), bool)), ('skel=3', TT['ln_skel'] == 3)]
    else:
        coef = TT['cnt_skel']; d = np.zeros(len(TT['glyphs']), np.int64)
        for c, v in zip('ivxlc', [1, 5, 10, 50, 100]): d[TT['gi'][C['roman'][c]]] = v
        sels = [('all', np.ones(len(tr), bool))]
    prev = np.array(TT['prev'], dtype=object)
    for w in set(prev[i] for i in range(len(prev))):
        pass
    for nm, m in L.marker_selectors(TT, top=25): sels.append((nm, m))
    for nm, m in sels:
        ite = np.where(m & ~tr)[0]
        if len(ite) < 10: continue
        out[nm] = L.score_robust(coef, ite, d, ('page', 'abs_clip', -1), E, TT)
    return dict(sorted(out.items(), key=lambda x: -x[1])[:5])


def run(name):
    log = open(os.path.join(L.CK, '%s_%s.log' % (TAG, name)), 'w')
    C = corpus(name)
    E = L.build_R(C)
    rows, E2, TT, ne, cov = L.search_grid(C, seed=0, log=log, time_budget=BUDGET, sel_kind=SELK, starts=6, E=E, hitk=10, focused=FOCUSED)
    S = L.summarise(rows)
    out = dict(name=name, n_eval=ne, summary=S, rows=rows, coverage=cov)
    for k in ('digits', 'roman'):
        if k in C: out[k] = C[k]
    if 'digits' in C or 'roman' in C: out['true_rule'] = true_rule(C, E2, TT)
    json.dump(out, open(os.path.join(L.CK, '%s_%s.json' % (TAG, name)), 'w'))
    print(name, 'done', file=log, flush=True)
    return name


if __name__ == '__main__':
    which = sys.argv[3].split(',') if len(sys.argv) > 3 else ['PL2', 'CULP_vide', 'PL_mark2', 'ZL', 'ZL_relab', 'IT2a',
                                                              'ZL_markov', 'ZL_selfcit']
    with Pool(2) as P:
        for nm in P.imap_unordered(run, which): print('done', nm, flush=True)
