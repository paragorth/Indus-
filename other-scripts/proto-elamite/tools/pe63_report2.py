#!/usr/bin/env python3
"""pe63 cycle 2 report: Ur III truth check of slot roles; PE slot table."""
import os, json, random
from collections import Counter, defaultdict
import numpy as np
import pe63_common as C

res = json.load(open(os.path.join(C.CK, 'c2_res.json')))
rng = random.Random(632)
ROLES = ['ID', 'COM', 'QTY', 'CONST', 'STEP', 'MIX']


def truth_of(slot):
    c = Counter(C.ur3_role(x) for x in slot['members'].values())
    return c.most_common(1)[0][0]


def stat(pairs):
    """P(ID | TIME or PERSON) - P(ID | COMMODITY); and P(COM/QTY | COMMODITY) - P(COM/QTY | TIME/PERSON)"""
    tp = [r for r, t in pairs if t in ('TIME', 'PERSON')]
    cm = [r for r, t in pairs if t == 'COMMODITY']
    a = (np.mean([r == 'ID' for r in tp]) if tp else 0) - (np.mean([r == 'ID' for r in cm]) if cm else 0)
    b = (np.mean([r in ('COM', 'QTY') for r in cm]) if cm else 0) - (np.mean([r in ('COM', 'QTY') for r in tp]) if tp else 0)
    return a, b


out = {}
for corp in ['DR', 'UM']:
    pairs = [(o['role'], truth_of(o)) for a in res[corp]['real'] for o in a['slots'] if o['role'] != 'CONST']
    conf = defaultdict(Counter)
    for r, t in pairs:
        conf[t][r] += 1
    a, b = stat(pairs)
    na, nb = [], []
    for _ in range(2000):
        tt = [t for _, t in pairs]; rng.shuffle(tt)
        x, y = stat(list(zip([r for r, _ in pairs], tt)))
        na.append(x); nb.append(y)
    # variable-slot recall: share of TIME / PERSON / COMMODITY slots that vary at all
    allp = [(o['role'], truth_of(o)) for a_ in res[corp]['real'] for o in a_['slots']]
    rec = {t: round(np.mean([r != 'CONST' for r, tt in allp if tt == t]), 3) for t in ('TIME', 'PERSON', 'COMMODITY', 'OTHER')}
    nn = Counter(t for _, t in allp)
    out[corp] = {'conf': {t: dict(c) for t, c in conf.items()}, 'ID_gap': a, 'p_ID': float(np.mean(np.array(na) >= a)),
                 'COM_gap': b, 'p_COM': float(np.mean(np.array(nb) >= b)), 'vary_recall': rec, 'n_slots': dict(nn)}
    print(corp, json.dumps(out[corp]))

# PE slot table
print()
for a in res['PE']['real']:
    print('DOSSIER ref', a['ref'], 'members', len(set(k for o in a['slots'] for k in o['members'])))
    for o in a['slots']:
        vals = ['%s:%s' % (k[-3:], ('%g' % x['v']) if x['v'] is not None else (x['sys'] or '-')) for k, x in sorted(o['members'].items())]
        print('   slot %2d %-5s core=%s variants=%s %s %s' % (o['slot'], o['role'], ' '.join(o['core']) or '-',
              ' | '.join(o['variants'])[:110], ' '.join(vals)[:120], 'TOTAL' if o.get('total') else ''))
json.dump(out, open(os.path.join(C.CK, 'c2_truth.json'), 'w'))
