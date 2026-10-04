"""pe25 cycle 4b: the cycle-4 scan with an honest per-rule null.

The cycle-4 z (random line subsets, sd ~ 1/sqrt(n)) let 2/20 rules 'replicate'
on a single-hand plant.  Here every one of the top-20 discovery rules is re-tested
on half B with a per-rule permutation null: in each tablet the same number of
lines as the rule selects are drawn at random (40 draws; partner tablets get the
same share), z = (H_rule - mean H_perm)/sd; survive at z < -2.8 (0.05/20 one-sided).
Run on REAL, single-hand plant S1, and the two planted second-hand corpora.
"""
import random, json
import numpy as np
from pe25_common import *
import pe25_cycle4 as c4

OUT = c4.OUT
NP = 40


def score_mask(TT, masks, idx):
    real, nul = [], []
    for i in idx:
        L = TT[i]; m = masks[i]
        P = [t for l, x in zip(L, m) if x for t in l['tok']]
        Q = [t for l, x in zip(L, m) if not x for t in l['tok']]
        if not P or not Q:
            continue
        cc = cells(P, Q, True)
        if not cc:
            continue
        real.extend(cc)
        for j in c4.partners[i]:
            Qj = [t for l, x in zip(TT[j], masks[j]) if not x for t in l['tok']]
            nul.extend(cells(P, Qj, True))
    if len(real) < 15 or not nul:
        return None
    a, a0 = np.mean(real), np.mean(nul)
    return (a - a0) / (1 - a0) if a0 < 1 else None


def perm_test(TT, rule, idx, r):
    pred = lambda l: all(c4.ATOMS[a](l) for a in rule.split('&'))
    masks = [[pred(l) for l in L] for L in TT]
    h = score_mask(TT, masks, idx)
    if h is None:
        return None, None
    need = set(idx) | set(j for i in idx for j in c4.partners[i])
    hs = []
    for _ in range(NP):
        pm = list(masks)
        for i in need:
            m = masks[i]
            k = sum(m)
            if k == 0:
                frac = sum(sum(x) for x in masks) / sum(len(x) for x in masks)
                k = int(round(frac * len(m)))
            s = set(r.sample(range(len(m)), min(k, len(m))))
            pm[i] = [q in s for q in range(len(m))]
        x = score_mask(TT, pm, idx)
        if x is not None:
            hs.append(x)
    hs = np.array(hs)
    return float(h), float((h - hs.mean()) / hs.std())


if __name__ == '__main__':
    prev = json.load(open(os.path.join(CK, 'cycle4.json')))
    import sys
    want = sys.argv[1].split(',')
    corp = {'REAL': c4.T, 'S1': c4.planted('S1', 61), 'PLANT_N14': c4.planted('P', 62, preds=(c4.ATOMS['num_N14'],)),
            'PLANT_LASTOBV': c4.planted('P', 63, preds=(c4.ATOMS['lastobv'],))}
    B = [i for i in range(len(c4.T)) if c4.half[i] == 1]
    out = {}
    for k, TT in corp.items():
        if k not in want:
            continue
        r = random.Random(77)
        res = []
        for t in prev[k]['top']:
            h, p = perm_test(TT, t['rule'], B, r)
            res.append(dict(rule=t['rule'], HB=h, p=p))
        surv = [x for x in res if x['p'] is not None and x['p'] < -2.8]
        out[k] = dict(res=res, surv=surv)
        print(k, 'survivors', [(x['rule'], round(x['HB'], 2), x['p']) for x in surv], flush=True)
        dump('cycle4b_%s.json' % sys.argv[1].replace(',', '_'), out)
    if len(sys.argv) < 3:
        sys.exit()
    out = {}
    for f in os.listdir(CK):
        if f.startswith('cycle4b_'):
            out.update(json.load(open(os.path.join(CK, f))))
    for k, rid in (('S1', 'PE-25.4e'), ('PLANT_N14', 'PE-25.4f'), ('PLANT_LASTOBV', 'PE-25.4g'), ('REAL', 'PE-25.4h')):
        s = out[k]['surv']
        row(OUT, rid, 'Scan re-test with per-rule permutation null (%s): top-20 half-A rules on half B, random same-size line sets within each tablet (40 draws; z of the rule H against the permutation mean and sd), survive at z < -2.8 (one-sided 0.05/20)' % k,
            'survivors %d: %s' % (len(s), '; '.join('%s H %.2f z %.1f' % (x['rule'], x['HB'], x['p']) for x in s) or 'none'),
            {'S1': 'control: must be 0', 'PLANT_N14': 'control: must include num_N14', 'PLANT_LASTOBV': 'control: must include lastobv', 'REAL': 'see 4i'}[k])
