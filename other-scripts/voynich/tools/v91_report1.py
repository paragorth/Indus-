import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_lib as V
R = {}
for nm in sys.argv[1].split(','):
    p = os.path.join(V.CK, 'c1_%s.pkl' % nm)
    if os.path.exists(p): R[nm] = pickle.load(open(p, 'rb'))
def key(r): return (r[3], r[4], r[5], r[6])
tab = {}
for nm, d in R.items():
    res = d['res']
    sc = np.array([r[0] + r[1] for r in res])
    o = np.argsort(-sc)
    print('==', nm, 'items', d['n_items'], 'masks', len(res), 'max %.1f  p99.9 %.1f  median %.1f' % (sc.max(), np.percentile(sc, 99.9), np.median(sc)))
    for i in o[:8]: print('   %.1f u=%.1f b=%.1f n=%d %s | %s | k%d o%d' % ((sc[i],) + tuple(res[i][1:2]) + tuple(res[i][0:1]) + tuple(res[i][2:])))
    tab[nm] = {key(r): r[0] + r[1] for r in res}
    if d['meta'].get('mask'):
        pr, f, k = d['meta']['mask']
        tgt = (pr, f, k, 0)
        s = tab[nm].get(tgt)
        print('   planted mask', tgt, 'score', s, 'rank', int((sc > s).sum()) + 1 if s is not None else None)
nulls = [n for n in ['N1', 'N2', 'N3'] if n in tab]
if nulls and 'ZL' in tab:
    floor = max(max(tab[n].values()) for n in nulls)
    print('FP floor (max over null searches) %.1f' % floor)
    for real in ['ZL', 'IT']:
        if real not in tab: continue
        surv = [(s, m) for m, s in tab[real].items() if s > floor and all(s > tab[n].get(m, -1e9) for n in nulls)]
        surv.sort(reverse=True)
        print(real, 'survivors above floor and above own mask in every null:', len(surv))
        for s, m in surv[:15]: print('   %.1f' % s, m, ' nulls:', [round(tab[n].get(m, np.nan), 1) for n in nulls])
