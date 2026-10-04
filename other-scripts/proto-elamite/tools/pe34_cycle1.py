"""pe34 cycle 1: additive transfer test. A modifier's shift (compound minus base)
learned on other bases predicts the compound on a held-out base?
Statistic G = err(base + generic compound shift) - err(base + modifier shift).
Null: modifier labels shuffled among compounds (2,000x). Controls: proto-cuneiform
(ARCH), Linear B (LINB), planted modifiers in PE, shuffled-data calibration."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe34_common import tokens, prep, perm_test, base_test, plant, transfer, CK

EXC = ('X', 'xX')
T = tokens()
res = {}
for c in ['PE', 'ARCH', 'LINB']:
    P = prep(T[c], c)
    real, nm, ns, p = perm_test(P, 2000, seed=1, exclude=EXC)
    rS = perm_test(P, 2000, seed=2, exclude=EXC, stat='S')
    b = base_test(P, 2000)
    res[c] = {'n': real['n'], 'err': {k: real[k] for k in ('base', 'add', 'gen', 'modonly')},
              'G': real['G'], 'G_null': [nm, ns], 'p_G': p, 'S': real['S'], 'p_S': rS[3],
              'base_real': b[0], 'base_null': b[1], 'p_base': b[2]}
    print(c, json.dumps(res[c]), flush=True)
    # calibration: shuffle modifier labels in the DATA, run the full test (200 perms)
    rng = random.Random(7)
    ps = []
    for k in range(20):
        mods = [m for _, _, m in P['comp']]
        keep = [i for i, m in enumerate(mods) if m not in EXC]
        vals = [mods[i] for i in keep]
        rng.shuffle(vals)
        Q = dict(P)
        comp = list(P['comp'])
        for i, v in zip(keep, vals):
            comp[i] = (comp[i][0], comp[i][1], v)
        Q['comp'] = comp
        ps.append(perm_test(Q, 200, seed=100 + k, exclude=EXC)[3])
    res[c]['calib_p'] = ps
    res[c]['calib_fp05'] = sum(x <= 0.05 for x in ps)
    print(c, 'calibration false positives', res[c]['calib_fp05'], '/ 20', flush=True)

# planted modifiers in PE
pl = []
for strength in (0.0, 0.3, 0.6, 0.9):
    for seed in range(3):
        toks, planted, eff = plant(T['PE'], 'PE', seed, strength)
        P = prep(toks, 'PE')
        tg = set(planted)
        real, nm, ns, p = perm_test(P, 500, seed=seed, exclude=EXC, targets=tg)
        pl.append({'strength': strength, 'seed': seed, 'n': real['n'], 'G': real.get('G'),
                   'null': nm, 'p': p})
        print('plant', pl[-1], flush=True)
res['plant_PE'] = pl
json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
