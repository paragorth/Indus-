#!/usr/bin/env python3
"""LA-57: add Linear A ADMINISTRATIVE-only draws to feats.pkl (documents with at least one number; drops libation
tables and other number-free texts, since every known system is administrative).  + 20 S1 / 20 S2 shuffles."""
import os, sys, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C

p = os.path.join(C.CK, 'feats.pkl')
F = pickle.load(open(p, 'rb'))
la = [d for d in C.la_docs() if any(x[0] == 'N' for x in d['toks'])]
print('LA admin docs', len(la), 'tokens', C.ntok(la))
X, types, labs, di = C.features(la)
F[('LA_ADM', 0)] = dict(X=X, types=types, labs=labs, didx=di)
for j in range(20):
    for tag, fn in (('S1', C.shuffle_types), ('S2', C.shuffle_order)):
        d = fn(la, random.Random(C.seed('la57-adm-%s-%d' % (tag, j))))
        X, types, labs, di = C.features(d)
        F[('LA_ADM_' + tag, j)] = dict(X=X, types=types, labs=labs, didx=di)
pickle.dump(F, open(p + '.tmp', 'wb')); os.replace(p + '.tmp', p)
print('saved')
