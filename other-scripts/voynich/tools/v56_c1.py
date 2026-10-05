"""v56 cycle 1: massive random pointer-rule search (additive / alphabetic-numeral / positional, absolute or
relative, page / folio / quire addresses) with coordinate ascent and selector refinement on half the pages,
re-test on the other half. Corpora: Voynich ZL, controls (Culpeper with real cross-references as additive Roman
numerals in opaque glyphs + padding; Voynich with a planted positional pointer system), nulls (unit-Markov and
self-citation generators, pages relabelled, Culpeper with names left in place = no numeral pointers)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v56_lib as L
from multiprocessing import Pool

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c1'
NCFG = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 1500
WHICH = sys.argv[4].split(',') if len(sys.argv) > 4 else ['CULP', 'ZL_planted', 'ZL', 'ZL_markov', 'ZL_selfcit',
                                                          'ZL_relab', 'CULP_names', 'IT2a']


def corpus(name):
    if name == 'CULP': return L.culpeper()
    if name == 'CULP_names': return L.culpeper(numerals=False)
    if name == 'IT2a': return L.voynich('IT2a')
    Z = L.voynich('ZL3b')
    if name == 'ZL': return Z
    if name == 'ZL_markov': return L.markov_corpus(Z)
    if name == 'ZL_selfcit': return L.selfcit_corpus(Z)
    if name == 'ZL_relab': return L.relabel_corpus(Z)
    if name == 'ZL_planted': return L.plant_pointers(Z, L.build_R)
    raise ValueError(name)


def run(name):
    log = open(os.path.join(L.CK, '%s_%s.log' % (TAG, name)), 'w')
    C = corpus(name)
    bud = BUDGET if name in ('CULP', 'ZL_planted', 'ZL', 'IT2a') else 0.67 * BUDGET
    rows, E, TT, ne, cov = L.search_grid(C, seed=0, log=log, time_budget=bud, max_cfg=NCFG)
    S = L.summarise(rows)
    out = dict(name=name, n_eval=ne, summary=S, rows=rows, coverage=cov)
    if 'truth' in C: out['n_truth'] = len(C['truth'])
    if 'digits' in C: out['digits'] = C['digits']
    if 'roman' in C: out['roman'] = C['roman']
    json.dump(out, open(os.path.join(L.CK, '%s_%s.json' % (TAG, name)), 'w'))
    print(name, 'done', ne, S['top_te_mean'], S['top_te_max'], file=log, flush=True)
    return name


if __name__ == '__main__':
    with Pool(2) as P:
        for nm in P.imap_unordered(run, WHICH): print('done', nm, flush=True)
