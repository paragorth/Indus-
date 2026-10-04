"""v37 cycle 4: rotation null for line schemas. Each line is rotated cyclically by a random offset (words keep
their neighbours except at one seam; the line's bag is kept), so any schema anchored to the line start is
destroyed while local word-to-word structure survives. Alignment = held-out gain lost by rotation, for the
left-to-right field automaton (k 4, 6) and for fixed position bins. Schema beyond edge rules = LR loss - posbin
loss. Usage: python3 v37_cycle4.py NAME [NAME...]"""
import sys, time
from v37_lib import *
from v37_cycle1 import get

def rotate(C, rng):
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                if len(l) > 2:
                    r = rng.randint(1, len(l) - 1); l = l[r:] + l[:r]
                npa.append(list(l))
            q['paras'].append(npa)
        out.append(q)
    return out

if __name__ == '__main__':
  for name in sys.argv[1:]:
    t = time.time(); C = get(name, 0); body = name.startswith('V')
    out = {'real': field_scores(C, seed=0, ks=(4, 6), restarts=6, body_only=body),
           'rot': [field_scores(rotate(C, random.Random(s)), seed=s, ks=(4, 6), restarts=6, body_only=body) for s in (1, 2)]}
    save(f'c4_{name}.json', out)
    r = out['real']; ro = {k: np.mean([x[k] for x in out['rot']]) for k in r}
    lr = max(r['LR4'], r['LR6']); lro = max(ro['LR4'], ro['LR6']); er = max(r['ER4'], r['ER6']); ero = max(ro['ER4'], ro['ER6'])
    print(name, f'{time.time()-t:.0f}s', 'LR real %.1f rot %.1f loss %.1f | ER loss %.1f | posbin real %.1f rot %.1f loss %.1f | schema beyond edges %.1f' %
          (lr, lro, lr - lro, er - ero, r['posbin'], ro['posbin'], r['posbin'] - ro['posbin'], (lr - lro) - (r['posbin'] - ro['posbin'])), flush=True)
