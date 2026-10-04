"""v37 cycle 2: line schema search. Left-to-right field automaton (k = 2,3,4,6 field states, forward jumps
allowed, 8 random restarts each) vs ergodic HMM and fixed position bins; held-out two-fold log-likelihood,
gains over unigram in millibits per word. Order information = real minus interior-shuffled copy (first and last
words kept in place). Usage: python3 v37_cycle2.py NAME [NAME...]"""
import sys, time
from v37_lib import *
from v37_cycle1 import get

def interior_shuffle(C, rng):
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                if len(l) > 3:
                    mid = l[1:-1]; rng.shuffle(mid); l = [l[0]] + mid + [l[-1]]
                npa.append(list(l))
            q['paras'].append(npa)
        out.append(q)
    return out

if __name__ == '__main__':
  for name in sys.argv[1:]:
    t = time.time(); out = {}
    C = get(name, 0); body = name.startswith('V')
    out['real'] = field_scores(C, seed=0, body_only=body)
    out['ishuf'] = [field_scores(interior_shuffle(C, random.Random(s)), seed=s, body_only=body) for s in (1, 2)]
    save(f'c2_{name}.json', out)
    r = out['real']; sh = {k: np.mean([x[k] for x in out['ishuf']]) for k in r}
    bestLR = max((k for k in r if k.startswith('LR')), key=lambda k: r[k]); bestER = max((k for k in r if k.startswith('ER')), key=lambda k: r[k])
    print(name, f'{time.time()-t:.0f}s n={r["n_words"]}', 'real: posbin %.1f %s %.1f %s %.1f' % (r['posbin'], bestLR, r[bestLR], bestER, r[bestER]),
          '| ishuf: posbin %.1f %s %.1f %s %.1f' % (sh['posbin'], bestLR, sh[bestLR], bestER, sh[bestER]),
          '| order info LR %.1f ER %.1f posbin %.1f; LR beyond posbin %.1f (null %.1f)' % (r[bestLR] - sh[bestLR], r[bestER] - sh[bestER], r['posbin'] - sh['posbin'],
          r[bestLR] - r['posbin'], sh[bestLR] - sh['posbin']), flush=True)
