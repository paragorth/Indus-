"""v26 report helpers: summarise evolution checkpoints and evaluations (prints; rows are written by hand-checked
calls at the bottom when run with an argument)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v26_lib import *


def random_landscape(name, R=0):
    s = load(f'evo_{name}_r{R}.json')
    if not s: return None
    r = s['random']; a = np.array([e['auc'] for e in r])
    # gene enrichment: mean AUC with the gene on vs off
    enr = {}
    for k in GENES:
        on = [e['auc'] for e in r if (k + '=') in e['g']]
        off = [e['auc'] for e in r if (k + '=') not in e['g']]
        if len(on) >= 5: enr[k] = (np.mean(on) - np.mean(off), len(on))
    return dict(n=len(a), pct=np.percentile(a, [0, 10, 50, 90, 100]).round(3).tolist(),
                best=sorted(r, key=lambda e: e['auc'])[:3], enr=sorted(enr.items(), key=lambda t: t[1][0]))


def trajectory(name, R):
    s = load(f'evo_{name}_r{R}.json')
    if not s: return None
    return s['log'], s['pop'][:3], s['nevals']


if __name__ == '__main__':
    for name in sys.argv[1:]:
        print('==', name)
        print(random_landscape(name))
        for R in range(4):
            t = trajectory(name, R)
            if t:
                log, pop, ne = t
                print('restart', R, 'evals', ne)
                for l in log: print('  gen', l['gen'], round(l['best_auc'], 3), round(l['best_arc'], 3), round(l['med_auc'], 3), l['bits'], l['best'])
                for e in pop: print('  POP', np.round(e['auc'], 3), round(e['arc'], 3), e['bits'], gstr(e['g']))
