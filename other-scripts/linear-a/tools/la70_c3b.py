#!/usr/bin/env python3
"""LA-70 cycle 3b: pseudo-outside test as in la60 3a: V2 gated on documents published in GORILA only,
frozen, then used on the administrative documents published after 1985 (not GORILA); vs 200 role shuffles,
50 random readings, EMPTY and la60 v1 PRIOR through the same decoder."""
import json, os, random, statistics as st
from la70_common import *

A = admin_docs(load_la())
tr = [d for d in A if d['pub'].startswith('G')]
te = [d for d in A if not d['pub'].startswith('G')]
R = gate_v2(tr, 'V2-GORILA'); h = hash_v2(R)
rng = random.Random(seed('la70-c3b'))
real = score_v2(tr, te, R)
sh = [score_v2(tr, te, shuffle_v2(R, rng)) for _ in range(200)]
rd = [score_v2(tr, te, random_v2(tr, R, rng)) for _ in range(50)]
P1 = prior_reading(); P1['rules'] = {}
out = dict(hash=h, n_train=len(tr), n_test=len(te), test_ids=[d['id'] for d in te], V2=real, EMPTY=score_v2(tr, te, EMPTY),
           V1=score_v2(tr, te, P1))
for nm, L in (('shuf', sh), ('rand', rd)):
    for k in ('full', 'strict', 'close', 'agree', 'viol'):
        v = [x.get(k, 0) for x in L]
        out['%s_%s' % (nm, k)] = (round(st.mean(v), 2), round((sum(x >= real.get(k, 0) for x in v) + 1) / (len(v) + 1), 3))
acc = acceptor_v2(tr, R)
out['per_doc'] = {d['id']: {k: v for k, v in decode_v2(d, R, acc)[0].items() if k != 'bad_steps'} for d in te}
json.dump(out, open(os.path.join(CK, 'c3b_post1985.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_doc'}))
