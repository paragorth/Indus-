"""v56 cycle 3b: fairness check for the table test. The controls' 'sections' are 6 coarse positional groups, while
the Voynich residual removes 8 illustrated sections x 2 Currier languages. Recompute the Voynich with the controls'
grouping (6 positional groups, no language) and Brumati with a finer grouping (12 groups)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v56_lib as L, v56_c3 as T
from multiprocessing import Pool


def job(name):
    if name.startswith('ZL'):
        C = L.voynich('ZL3b')
        if name == 'ZL_relab6': C = L.relabel_corpus(C)
        n = len(C['pages'])
        C = dict(C, pages=[dict(p, sec='G%d' % (i * 6 // n), lang='-') for i, p in enumerate(C['pages'])])
    else:
        C = T.brumati(); n = len(C['pages'])
        C = dict(C, pages=[dict(p, sec='G%d' % (i * 12 // n)) for i, p in enumerate(C['pages'])])
    E = L.build_R(C); TT = L.token_table(C, E)
    out = {}
    for s in range(3):
        tok_tr = L.split_pages(E, seed=s)[E['tok_page']]
        for key in ('skel', 'word'):
            out['all|%s|%d' % (key, s)] = T.table_test(E, TT, C, np.ones(len(TT['words']), bool), tok_tr, key=key)
    return name, out


if __name__ == '__main__':
    res = {}
    with Pool(2) as P:
        for nm, o in P.imap_unordered(job, ['ZL_pos6', 'ZL_relab6', 'BRUMATI12']):
            res[nm] = o
            for key in ('skel', 'word'):
                v = [o['all|%s|%d' % (key, s)] for s in range(3)]
                print(nm, key, 'z_hit %.1f z_spec %.1f z_pair %.1f' % tuple(np.mean([x[k] for x in v]) for k in ('z_hit', 'z_spec', 'z_pair')), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c3b.json'), 'w'))
