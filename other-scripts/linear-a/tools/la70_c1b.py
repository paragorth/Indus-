#!/usr/bin/env python3
"""LA-70 cycle 1b: gate the v2 candidate items on the TRAINING half of the la60 primary split
('la60-main', by tablet, site-stratified) and freeze V2 by sha256 before any held-out document is decoded.
Also gate on each of the 10 la60 splits (to show how stable the train-only gates are)."""
import json, os
from la70_common import *


def main():
    A = admin_docs(load_la())
    tr, te = split(A, 'la60-main')
    RG = gate_v2(tr, 'V2G', strict=True)
    json.dump(dict(roles=RG['roles'], order=RG['order'], rules=RG['rules'], sha256=hash_v2(RG), dropped=RG['dropped']),
              open(os.path.join(CK, 'frozen_V2G.json'), 'w'), ensure_ascii=False, indent=1, sort_keys=True)
    print('FROZEN V2G (strict gate)', hash_v2(RG), sorted(Counter(RG['roles'].values()).items()))
    R = gate_v2(tr, 'V2')
    h = hash_v2(R)
    fz = dict(name='V2', roles=R['roles'], order=R['order'], lib=sorted(R['lib']), site_default=R['site_default'],
              rules=R['rules'], ungated=R['ungated'], frac_values=FRAC_VAL, dropped_by_train_gate=R['dropped'],
              grades={w: CAND[w][1] for w in R['roles']}, glosses={w: CAND[w][2] for w in R['roles']},
              sha256=h, train_docs=len(tr), test_docs=len(te), split='la60-main')
    json.dump(fz, open(os.path.join(CK, 'frozen_V2.json'), 'w'), ensure_ascii=False, indent=1, sort_keys=True)
    json.dump(dict(V2=h, V2G=hash_v2(RG), test_ids=sorted(d['id'] for d in te)), open(os.path.join(CK, 'frozen_hash.json'), 'w'), indent=1)
    print('FROZEN V2', h, 'train', len(tr), 'test', len(te))
    print('roles', len(R['roles']), sorted(Counter(R['roles'].values()).items()))
    print('dropped', R['dropped']); print('ungated', R['ungated']); print('rules', R['rules']); print('order', R['order'])
    stab = Counter(); nsp = 10
    for s in range(nsp):
        tr2, _ = split(A, 'la60-c2-split%d' % s)
        R2 = gate_v2(tr2)
        for w in R2['dropped']:
            stab[w] += 1
    print('dropped per split (of 10):', dict(stab))
    json.dump(dict(dropped_per_split=stab), open(os.path.join(CK, 'c1b_gate_stability.json'), 'w'))


if __name__ == '__main__':
    main()
