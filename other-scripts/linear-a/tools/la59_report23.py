#!/usr/bin/env python3
"""LA-59 rows for cycles 2 and 3 (loops/la59_cycle2.txt, la59_cycle3.txt)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *
from la59_report1 import truth_eval, zp

def r2():
    rows = []
    for tag in ('LA', 'LBs'):
        p = os.path.join(CK, f'c2_{tag}_mn.json')
        if not os.path.exists(p):
            continue
        o = json.load(open(p)); g = o['real_eb2']
        zk, pk = zp(g, o['null_key']); zr, pr = zp(g, o['null_rewire'])
        rows.append(f"| LA-59.2{tag} | {tag}, syllabary constraint (each sign a distinct C x V state; swaps), 20 sweeps; real = expected best of 2 of 4 restarts; nulls best of 2: 8 shuffled keys, 8 rewired graphs (double-edge swaps, pair weights kept) | gain {g:.4f} (best {o['real'][0]['g']:.4f}); shuffled key {np.mean(o['null_key']):.4f} (z {zk:.2f}, P {pk:.3f}); rewired {np.mean(o['null_rewire']):.4f} (z {zr:.2f}, P {pr:.3f}) | see verdict |")
        for sh in (1.0, 0.5):
            q = [x for x in o['planted'] if x['share'] == sh]
            rows.append(f"| LA-59.2{tag}-plant{sh} | planted realistic syllabary (distinct states from ~n/5+1 consonants x 5 vowels) on {tag} degrees/weight, sound share {sh}, 3 draws; chance C ~1/18, V 1/5 | fit {np.mean([x['g'] for x in q]):.3f} vs true {np.mean([x['g_true'] for x in q]):.3f}; accuracy C {np.mean([x['acc_c'] for x in q]):.2f}, V {np.mean([x['acc_v'] for x in q]):.2f}, place {np.mean([x['acc_place'] for x in q]):.2f}; co-assignment C {np.mean([x['co_c'] for x in q]):.2f}, V {np.mean([x['co_v'] for x in q]):.2f} | - |")
        h = o['heldout']
        rows.append(f"| LA-59.2{tag}-heldout | 3 splits by sign pair; test gain under the train fit; 4 shuffled-key fits per split; 100,000 random injective guesses on train, top 20 vs middle 20 re-scored on test | held-out real {np.mean([x['held'] for x in h]):.4f} ({', '.join(f'{x[chr(104)+chr(101)+chr(108)+chr(100)]:.4f}' for x in h)}); shuffled keys {np.mean([np.mean(x['held_shufkey']) for x in h]):.4f}; random top-20 {np.mean([x['rand_top_held'] for x in h]):.4f} vs middle {np.mean([x['rand_mid_held'] for x in h]):.4f} (train max {np.mean([x['rand_top_train'] for x in h]):.4f}) | - |")
        script = 'LB' if tag.startswith('LB') else 'LA'
        te = truth_eval(signs := o['signs'], o['real'][0]['s'], script)
        lab = 'LB truth (values hidden)' if script == 'LB' else 'OUTSIDE CHECK ONLY: conventional LA labels'
        rows.append(f"| LA-59.2{tag}-truth | {lab}, 2,000 permutations | " + '; '.join(f"{k} {v[0]:.3f} vs {v[1]:.3f} (P {v[2]:.3f})" for k, v in te.items() if not k.startswith('n_')) + " | - |")
    return rows

def r3():
    rows = []
    for tag in ('LA', 'LBs'):
        p = os.path.join(CK, f'c3_{tag}.json')
        if not os.path.exists(p):
            continue
        o = json.load(open(p)); k = o['keys']; nd = o['null_diach']
        rows.append(f"| LA-59.3{tag} | {tag}, unconstrained fit, best of 4 restarts; held-out mean of 3 splits (best of 2). Keys: Miller-Nicely (Shepard), Miller-Nicely (Hubert), Index Diachronica sound changes, flat (identity only); null 6 shuffled diachronica keys | " +
                    '; '.join(f"{n} {k[n]['g']:.4f} / held {np.mean(k[n]['held']):.4f}" for n in ('mn', 'semds', 'diach', 'flat')) +
                    f"; shuffled diach {np.mean([x['g'] for x in nd]):.4f} / held {np.mean([np.mean(x['held']) for x in nd]):.4f} (diach in-sample z {zp(k['diach']['g'], [x['g'] for x in nd])[0]:.2f}, held z {zp(np.mean(k['diach']['held']), [np.mean(x['held']) for x in nd])[0]:.2f}) | see verdict |")
        for mult in (1, 10):
            q = [x for x in o.get('ceiling', []) if x['mult'] == mult]
            if q:
                rows.append(f"| LA-59.3{tag}-ceiling{mult}x | planted realistic syllabary, MN key, sound share 1.0, {mult}x {tag}'s weight, 2 draws; fit from 3 random starts vs annealed from the true states | fit {np.mean([x['g'] for x in q]):.3f}, true {np.mean([x['g_true'] for x in q]):.3f}, oracle start {np.mean([x['g_oracle'] for x in q]):.3f}; accuracy C {np.mean([x['acc_c'] for x in q]):.2f} (oracle start {np.mean([x['oracle_acc_c'] for x in q]):.2f}), V {np.mean([x['acc_v'] for x in q]):.2f}; co-assignment C {np.mean([x['co_c'] for x in q]):.2f}, V {np.mean([x['co_v'] for x in q]):.2f} | - |")
        if tag == 'LBs':
            te = truth_eval(o['signs'], k['diach']['s'], 'LB')
            rows.append(f"| LA-59.3LBs-truth | LB truth vs the diachronica-key fit, 2,000 permutations | " + '; '.join(f"{a} {v[0]:.3f} vs {v[1]:.3f} (P {v[2]:.3f})" for a, v in te.items() if not a.startswith('n_')) + " | - |")
    return rows

if __name__ == '__main__':
    which = sys.argv[1]
    rows = r2() if which == '2' else r3()
    print('\n'.join(rows))
