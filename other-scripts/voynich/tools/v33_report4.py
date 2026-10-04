"""v33 cycle 4 report: classify the contracted / truncated / recoded / verbose Latin and planted declension texts
against the cycle-2 references; distances to the Voynich cluster; gap ratios for the new sources (cycle-3 job)."""
import sys
from v33_cycle2 import *

if __name__ == '__main__':
    D = loadall()
    cs = sorted(set(corp(n) for n in D))
    refs = [c for c in cs if c.startswith(('L-', 'G-'))]
    queries = [n for n in D if n.startswith(('C-', 'P-decl'))]
    allq = classify(D, COMMON, refs, queries)
    per = {r: classify(D, [r], refs, queries) for r in COMMON}
    # distance to Voynich cluster in the same standardisation
    names, X = vecs(D, COMMON); ix = {n: i for i, n in enumerate(names)}
    R = [ix[n] for n in names if corp(n) in refs]
    Z = (X - X[R].mean(0)) / (X[R].std(0) + 1e-9)
    def dist(a, b):
        return float(np.mean([np.linalg.norm(Z[ix[x]] - Z[ix[y]]) for x in names if corp(x) == a for y in names if corp(y) == b and x != y]))
    out = {}
    for q in sorted(set(corp(n) for n in queries)):
        votes = Counter(per[r][n]['fam'] for r in COMMON for n in queries if corp(n) == q)
        dv = {v: round(dist(q, v), 2) for v in ('V-ZL', 'V-IT', 'V-A', 'V-B')}
        dl = round(dist(q, 'L-la'), 2)
        nr = [allq[n]['order'][:3] for n in queries if corp(n) == q]
        meds = {k: round(med(D, q, k), 2) for k in ('conn', 'zNODF', 'zC', 'zQ', 'cvrow', 'ginicol')}
        out[q] = dict(votes=dict(votes), dV=dv, dLatin=dl, nearest=nr, med=meds)
        print(q, meds, '| votes', dict(votes), '| nearest', nr, '| dV', dv, 'dLa', dl)
    print('ref: V-ZL to own generators 9.4-10.2, V block spread 8.2-10.9; Latin to own generators 21')
    save('c4_report.json', out)
