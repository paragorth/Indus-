"""v81 cycle 2 report: slot inversion (onset vs middle vs coda streams under the same 200 random definitions)."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v81_lib as L
D = pickle.load(open(os.path.join(L.CK, 'c2.pkl'), 'rb'))
S = D['slot']; names = list(S[0])
for feat in ('lag1', 'skip', 'tri', 'nbr', 'page', 'sec'):
    for k in (1, 2, 3):
        ix = [i for i in S if D['defs'][i]['k'] == k]
        line = '%-5s k%d ' % (feat, k)
        for n in names:
            v = {s: np.median([S[i][n][s][feat] for i in ix]) for s in ('on', 'mid', 'coda')}
            line += '%s %+.3f/%+.3f/%+.3f  ' % (n[:7], v['on'], v['mid'], v['coda'])
        print(line)
# share of definitions where the onset slot beats the coda slot on lag1+skip+tri and on page
for n in names:
    a = np.mean([S[i][n]['on']['lag1'] + S[i][n]['on']['skip'] > S[i][n]['coda']['lag1'] + S[i][n]['coda']['skip'] for i in S])
    b = np.mean([S[i][n]['on']['page'] > S[i][n]['coda']['page'] for i in S])
    print('%-12s onset>coda order %.2f page %.2f' % (n, a, b))
