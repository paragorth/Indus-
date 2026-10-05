"""v61 cycle 3b: line-edge pull by Currier language and by section (is any part of the book sandhi-like?)."""
import json, sys
from multiprocessing import Pool
import v61_lib as L
from v61_c1 import summarize
import v61_edge as E

def job(arg):
    name, key, val, d = arg
    lines = [l for l in L.load_vms(name) if l[key] == val]
    if d == 'P':
        lines = L.reverse_text(lines)
    recs, nh, info = L.scan(lines, n_random=0, seed=1) if False else L.scan(lines, n_random_classes=4000, seed=1, min_tok=10)
    out = {'sub': '%s %s=%s %s' % (name, key, val, d), 'nh': nh, 'n_tok': sum(len(l['words']) for l in lines),
           'n_test3': sum(r['z_test'] > 3 for r in recs), 'mi': L.coupling_mi(lines)}
    for k in ('finL', 'finP', 'both'):
        out[k] = E.index(recs, kind=k, nmin=15)
    L.jsave('c3b_%s_%s_%s_%s.json' % (name, key, val, d), {'out': out, 'recs': recs[:300]})
    return out

if __name__ == '__main__':
    jobs = [(n, 'lang', v, d) for n in ('ZL3b', 'IT2a') for v in ('A', 'B') for d in 'RP']
    jobs += [('ZL3b', 'sec', v, 'R') for v in ('H', 'S', 'B')]
    with Pool(2) as p:
        for o in p.imap_unordered(job, jobs):
            print(json.dumps(o), flush=True)
