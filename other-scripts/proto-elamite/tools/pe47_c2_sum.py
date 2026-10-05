"""Summarise pe47 cycle 2 (cross-tablet sum links and printed-twice twins) against nulls."""
import collections, json, sys
import numpy as np

R = json.load(open(sys.argv[1]))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for d in R:
    by[d['name']][d['null']].append(d)
KEYS = ['n_summ_links', 'n_summ_pairs', 'summ_jac', 'summ_entry_names_D', 'n_twins', 'twin_jac', 'n_vtwins', 'vtwin_jac',
        'summ_sameyear', 'summ_samemonth', 'twin_sameyear', 'twin_samemonth', 'vtwin_sameyear', 'vtwin_samemonth',
        'rand_sameyear', 'rand_samemonth', 'rand_jac', 'summ_true_share', 'summ_true_n']
for name, D in by.items():
    print('==', name, {k: len(v) for k, v in D.items()})
    for k in KEYS:
        line = []
        for nl in ('real', 'N1', 'N2'):
            vals = [d.get(k) for d in D.get(nl, []) if d.get(k) is not None]
            if vals:
                line.append('%s %.4g [%.4g-%.4g]' % (nl, np.mean(vals), min(vals), max(vals)))
        if line:
            r = [d.get(k) for d in D.get('real', []) if d.get(k) is not None]
            extra = ''
            for nl in ('N1', 'N2'):
                nv = [d.get(k) for d in D.get(nl, []) if d.get(k) is not None]
                if r and nv:
                    extra += ' p(%s>=real0)=%.3f' % (nl, (1 + sum(x >= r[0] for x in nv)) / (1 + len(nv)))
            print('  %-20s %s%s' % (k, ' | '.join(line), extra))
pe = [d for d in R if d['name'] == 'PE' and d['null'] == 'real']
if pe:
    d = pe[0]
    print('PE links (first 40):')
    for l in d['links'][:40]:
        print('  ', l)
    print('PE twins', d['twins_ids'][:30])
