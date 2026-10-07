"""v90 cycle 1b: re-score the cycle-1 compactness survivors with the de-duplicated recurrence lift
(one event per distinct page x value), since token-level z was anti-conservative (Markov 11/60 calls)."""
import os, sys, json
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
import v90_c1 as C1

def run(name):
    f = os.path.join(L.CK, 'c1_%s.json' % name)
    r = json.load(open(f))
    toks, order = C1.build(name)
    tr, te = L.split_pages(order)
    tei = set(order.index(p) for p in te)
    out = {'name': name, 'schemes': {}}
    for s, d in r['schemes'].items():
        enc = L.Enc(toks, order, int(s))
        rows = []
        for sv in d['surv']:
            prof = []
            for cols in sv['decomp']:
                a = L.recur_lift(enc, enc.wheel_codes(cols), tei)
                comp = [c for c in range(L.NF) if c not in cols]
                b = L.recur_lift(enc, enc.wheel_codes(comp), tei)
                prof.append({'cols': [L.FIELDS[c] for c in cols], 'lift': a['lift'], 'lift_z': a['z'], 'lift_n': a['n'],
                             'clift': b['lift'], 'clift_z': b['z'], 'clift_n': b['n']})
            rows.append({'decomp': sv['decomp'], 'bits': sv['bits'], 'prof': prof})
        w = L.recur_lift(enc, enc.wheel_codes(list(range(L.NF))), tei)
        f23 = L.recur_lift(enc, enc.wheel_codes([2, 3]), tei)
        f23c = L.recur_lift(enc, enc.wheel_codes([0, 1, 4, 5, 6]), tei)
        out['schemes'][s] = {'surv': rows, 'whole': w, 'f2l2': f23, 'f2l2_comp': f23c, 'best_bits': d['best_bits'],
                             'indep_bits': d['indep_bits'], 'whole_bits': d['whole_bits']}
    json.dump(out, open(os.path.join(L.CK, 'c1b_%s.json' % name), 'w'), default=float)
    return name

if __name__ == '__main__':
    names = sys.argv[1:]
    with Pool(2) as P:
        for n in P.imap_unordered(run, names):
            print(n, flush=True)
