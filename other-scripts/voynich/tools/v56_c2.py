"""v56 cycle 2: MARKED pointers and the density ladder. Medieval cross-references carry a marker ('vide', 'cap.',
a sign) before the address. Selectors here are 'the word after marker W' for the 40 most frequent preceding
skeletons; the rest is the cycle-1 machinery (additive / alphabetic-numeral / positional addresses, absolute or
relative, page / folio / quire; held-out = unseen word types, cluster-robust z).
Controls: Culpeper with 'vide' + Roman numeral (real cross-reference graph); Voynich with planted positional
pointers after a marker word (2 per page) and without a marker at 2, 6 and 12 per page (density ladder, cycle-1
selectors). Nulls: relabelled pages, Markov, self-citation; Culpeper with names (no pointers)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v56_lib as L
from multiprocessing import Pool

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c2'
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1500


def corpus(name):
    if name == 'CULP_vide': return L.culpeper(marker=True)
    if name == 'CULP_names': return L.culpeper(numerals=False)
    if name == 'IT2a': return L.voynich('IT2a')
    Z = L.voynich('ZL3b')
    if name == 'ZL': return Z
    if name == 'ZL_markov': return L.markov_corpus(Z)
    if name == 'ZL_selfcit': return L.selfcit_corpus(Z)
    if name == 'ZL_relab': return L.relabel_corpus(Z)
    if name == 'PL_mark2': return L.plant_pointers(Z, L.build_R, rate_per_page=2, marker='chol')
    if name.startswith('PL_dens'): return L.plant_pointers(Z, L.build_R, rate_per_page=int(name[7:]))
    raise ValueError(name)


def run(name):
    log = open(os.path.join(L.CK, '%s_%s.log' % (TAG, name)), 'w')
    C = corpus(name)
    kind = 'fixed' if name.startswith('PL_dens') else 'marker'
    rows, E, TT, ne, cov = L.search_grid(C, seed=0, log=log, time_budget=BUDGET, sel_kind=kind, starts=6)
    S = L.summarise(rows)
    out = dict(name=name, n_eval=ne, summary=S, rows=rows, coverage=cov, sel_kind=kind)
    for k in ('digits', 'roman'):
        if k in C: out[k] = C[k]
    if 'truth' in C: out['n_truth'] = len(C['truth'])
    json.dump(out, open(os.path.join(L.CK, '%s_%s.json' % (TAG, name)), 'w'))
    print(name, 'done', file=log, flush=True)
    return name


if __name__ == '__main__':
    which = sys.argv[3].split(',') if len(sys.argv) > 3 else ['CULP_vide', 'PL_mark2', 'ZL', 'ZL_relab', 'IT2a',
                                                              'ZL_markov', 'ZL_selfcit', 'CULP_names', 'PL_dens6', 'PL_dens12']
    with Pool(2) as P:
        for nm in P.imap_unordered(run, which): print('done', nm, flush=True)
