"""pe59 report helpers: summarise cycle 1 / cycle 2 checkpoints for PE and PC into compact numbers."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe59_lib import CK


def c1(corpus):
    d = json.load(open(os.path.join(CK, 'c1_%s.json' % corpus)))
    h = d['heldout']
    r = {'n_sign': h['n_sign'], 'n_num': h['n_num'], 'n_tot': h['n_tot'],
         'sign': h['sign'], 'num': {k: sum(v) for k, v in h['num'].items()},
         'num_cls': {k: v[0] for k, v in h['num'].items()},
         'tot': {k: sum(v) for k, v in h['tot'].items()} if h['tot'] else {},
         'tau': h['tau_signs'], 'tau_nb': d['tau_nb_allsigns'], 'tau_maj': d['tau_majority'],
         'closure': d['closure'], 'ablation': {k: sum(v) for k, v in d['ablation_num'].items()},
         'rate': d['rate_rule'], 'copy': d['copy_rule'], 'boot_sign': d['boot_sign'], 'boot_num': d['boot_num']}
    for ctl in ('shuffled', 'metro'):
        S = d.get(ctl, [])
        if not S:
            continue
        r[ctl] = {
            'n': len(S),
            'sign_G': [float(np.mean([s['sign']['G'] for s in S])), float(np.min([s['sign']['G'] for s in S]))],
            'num_G': [float(np.mean([sum(s['num']['G']) for s in S])), float(np.min([sum(s['num']['G']) for s in S]))],
            'tot_G': [float(np.mean([sum(s['tot']['G']) for s in S if s['tot']])), float(np.min([sum(s['tot']['G']) for s in S if s['tot']]))] if any(s['tot'] for s in S) else None,
            'tau_acc': [float(np.mean([s['tau_signs']['acc'] for s in S])), float(np.max([s['tau_signs']['acc'] for s in S]))],
            'tau_bits': [float(np.mean([s['tau_signs']['bits'] for s in S])), float(np.min([s['tau_signs']['bits'] for s in S]))],
            'closure_read': [float(np.mean([s['closure']['read'][0] for s in S])), int(np.max([s['closure']['read'][0] for s in S])),
                             S[0]['closure']['read'][1]],
            'n_better_num': sum(sum(s['num']['G']) <= r['num']['G'] for s in S),
            'n_better_sign': sum(s['sign']['G'] <= r['sign']['G'] for s in S),
            'n_better_tot': sum(sum(s['tot']['G']) <= r['tot']['G'] for s in S if s['tot']) if r['tot'] else None,
        }
    if 'refit_no_weights' in d:
        r['no_weights_num'] = sum(d['refit_no_weights']['num']['G'])
    return r


def c2(corpus):
    d = json.load(open(os.path.join(CK, 'c2_%s.json' % corpus)))
    r = {'real': d['real'], 'params_match': d['params_match']}
    for ctl in ('transplant', 'shuffled'):
        S = d[ctl]
        r[ctl] = {k: [float(np.mean([s[k] for s in S])), float(np.max([s[k] for s in S]))]
                  for k in ('C1', 'C2_lines', 'C2_all_ok', 'C3', 'full', 'full_with_roles', 'strict')}
    return r


if __name__ == '__main__':
    what = sys.argv[1]
    corp = sys.argv[2]
    print(json.dumps(c1(corp) if what == 'c1' else c2(corp), indent=1))
