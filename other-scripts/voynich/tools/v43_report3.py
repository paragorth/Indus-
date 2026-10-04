"""v43 cycle 3 report: where the random-classifier survivors put their island."""
import numpy as np
from collections import Counter
import v43_lib as L
surv = L.load('cycle3_surv.json')
for nm, name in (('ZL', 'ZL3b'), ('IT', 'IT2a')):
    P = L.voynich(name); S = L.stream_of_pages(P); cp = [S[i*100+50][1] for i in range(len(S)//100)]
    sig = [d for d in surv if d[nm]['p'] < 0.05]
    c = Counter(); s = Counter(); h = Counter(); q = Counter()
    for d in sig:
        i = d[nm]['i']; pg = [cp[j] for j in range(i, min(i + 12, len(cp)))]
        m = Counter(pg).most_common(1)[0][0]
        c[P[m]['id']] += 1; s[P[m]['sec']] += 1; h[P[m]['hand']] += 1; q[P[m]['quire']] += 1
    n = len(sig)
    print(nm, 'n sig', n, 'of', len(surv), 'sections', {k: round(v/n, 2) for k, v in s.most_common()},
          'quires', {k: round(v/n, 2) for k, v in q.most_common(5)}, 'hands', {k: round(v/n, 2) for k, v in h.most_common()}, 'pages', c.most_common(8))
for nm in ('plant',):
    ins = np.mean([52 <= d[nm]['i'] <= 68 for d in surv]); sg = np.mean([d[nm]['p'] < 0.05 for d in surv])
    print('plant argmax inside', round(ins, 3), 'sig', round(sg, 3))
# held-out by whether the A-island is in BB
P = L.voynich('ZL3b'); sec = {p['id']: p['sec'] for p in P}
held = [d['held'] for d in surv if d['held']['B'] is not None]
bb = [h for h in held if Counter(sec[x] for x in h['pages']).most_common(1)[0][0] == 'BB']
print('held: A-island mainly BB', len(bb), 'of', len(held), 'Bpct>=0.95', round(np.mean([h['Bpct'] >= .95 for h in held]), 3),
      'B above tau', round(np.mean([h['B_above_tau'] for h in held]), 3), 'B mean', round(np.mean([h['B'] for h in held]), 3), 'Brest', round(np.mean([h['Brest'] for h in held]), 3))
taus = [d['tau'] for d in surv]; print('tau median', round(np.median(taus), 3), 'bacc median', round(np.median([d['bacc'] for d in surv]), 3))
for nm in ('ZL', 'IT', 'U_tri', 'plant'):
    print(nm, 'max median', round(np.median([d[nm]['max'] for d in surv]), 3), 'frac max>=tau', round(np.mean([d[nm]['max'] >= d['tau'] for d in surv]), 3))
# vs uniform ceiling per classifier
print('ZL max > U_tri max', round(np.mean([d['ZL']['max'] > d['U_tri']['max'] for d in surv]), 3), 'IT', round(np.mean([d['IT']['max'] > d['U_tri']['max'] for d in surv]), 3),
      'plant', round(np.mean([d['plant']['max'] > d['U_tri']['max'] for d in surv]), 3))
