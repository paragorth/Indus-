"""pe72 summary of a c1 checkpoint: real v1 / v2 vs twin distributions (mean, 95th pct, share of twins >= real)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe72_lib as L

KEYS = ('full', 'strict', 'dense', 'C3', 'pass', 'fail', 'net', 'C1', 'tab_fail')


def table(fn):
    d = json.load(open(fn))
    rows = {'v1': d['v1'], 'v2': d['v2']}
    out = {'real': rows, 'twins': {}}
    for k, L_ in d['twins'].items():
        out['twins'][k] = {m: {'mean': round(float(np.mean([x[m] for x in L_])), 1),
                               'p95': float(np.percentile([x[m] for x in L_], 95)),
                               'ge_v2': sum(x[m] >= d['v2'][m] for x in L_), 'n': len(L_)} for m in KEYS}
    return d, out


if __name__ == '__main__':
    d, out = table(sys.argv[1])
    print('real v1', {m: d['v1'][m] for m in KEYS}, 'n', d['v1']['n'])
    print('real v2', {m: d['v2'][m] for m in KEYS})
    for k, v in out['twins'].items():
        print(k, {m: (v[m]['mean'], v[m]['ge_v2']) for m in KEYS}, 'n', v['full']['n'])
    print('transitions', {k: len(v) for k, v in d['transitions'].items()})
    print('train v1', {m: d['v1_train'][m] for m in KEYS}, 'n', d['v1_train']['n'])
    print('train v2', {m: d['v2_train'][m] for m in KEYS})
