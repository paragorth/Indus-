"""pe68 cycle 2b: reading-specific point predictions on the hidden parts of joined tablets.
(1) one-system rule: P(any capacity line among the hidden lines) from the frozen file vs truth;
(2) M288 allotment rule (pe27/pe59 B): a hidden M288 line after a count line n carries 60 N39C x n;
    null = the same pairs with M288 values re-dealt among all corpus count -> M288 pairs."""
import os, sys, json, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe68_lib import *  # noqa
import pe59_lib as P

F = json.load(open(os.path.join(DATA, 'pe68_frozen_join_predictions.json')))
J = json.load(open(os.path.join(CK, 'c1_joins.json')))
U = {t['id']: t for t in corpus('PE')}
raw = {t['id']: t for t in P.build_pe()}
cap, cnt = P.pe_maps()
cm = cnt['sex2']
out = {'system': [], 'm288': []}
# (1)
ll = {'READ': [], 'CACHE': [], 'BASE': []}
base = None
for p in F['predictions']:
    idx = set(J['fragments'][p['tablet']][p['fragment']])
    H = [l for i, l in enumerate(U[p['tablet']]['lines']) if i not in idx]
    truth = int(any(l['cls'] == 'CAP' for l in H))
    if not any(l['cls'] for l in H):
        continue
    row = {'tablet': p['tablet'], 'frag': p['fragment'], 'vk': p['READ']['visible_kind'], 'truth_any_cap': truth}
    for k in ('READ', 'CACHE'):
        q = p[k]['p_any_cap_hidden']
        q = min(max(q, 1e-3), 1 - 1e-3)
        row[k] = q
        ll[k].append(-(truth * math.log2(q) + (1 - truth) * math.log2(1 - q)))
    out['system'].append(row)
for k in ('READ', 'CACHE'):
    print('any-cap bits', k, round(float(np.sum(ll[k])), 2), 'n', len(ll[k]))
disc = [r for r in out['system'] if r['vk'] != 'CAPT']
print('fragments that do not show capacity alone:')
for r in disc:
    print('  ', r)
# (2) M288 pairs in the whole PE corpus (count line followed by an M288 line)
pairs = []
for t in raw.values():
    L = t['lines']
    for i in range(1, len(L)):
        a, b = L[i - 1], L[i]
        if b['signs'] and b['signs'][-1] == 'M288' and b['numclean'] and a['numclean'] and a['nums'] \
                and a['signs'] and a['signs'][-1] != 'M288' and P.ncls(a['nums']) == 'AMB':
            n = P.value(a['nums'], cm)
            m = P.value(b['nums'], cap)
            if n and m:
                pairs.append((t['id'], i, float(n), float(m)))
allm = [m for _, _, _, m in pairs]
hits_all = sum(1 for _, _, n, m in pairs if m == 60 * n)
print('corpus count->M288 pairs %d, at 60 per unit %d (%.2f)' % (len(pairs), hits_all, hits_all / max(1, len(pairs))))
rng = random.Random(seed('pe68-m288'))
hidden_pairs = []
for p in F['predictions']:
    idx = set(J['fragments'][p['tablet']][p['fragment']])
    for (tid, i, n, m) in pairs:
        if tid == p['tablet'] and i not in idx:
            hidden_pairs.append((p['tablet'], p['fragment'], i, n, m, m == 60 * n))
uniq = {(a, c): (n, m, h) for a, b, c, n, m, h in hidden_pairs}
hits = sum(1 for v in uniq.values() if v[2])
nulls = []
for _ in range(10000):
    nulls.append(sum(1 for (n, m, h) in uniq.values() if rng.choice(allm) == 60 * n))
nulls = np.array(nulls)
pv = float(((nulls >= hits).sum() + 1) / (len(nulls) + 1))
print('hidden count->M288 pairs on joined tablets: %d unique lines, %d at 60/unit; null mean %.2f, p %.4f' %
      (len(uniq), hits, nulls.mean(), pv))
for k, v in sorted(uniq.items()):
    print('  ', k, v)
out['m288'] = {'n': len(uniq), 'hits': hits, 'null_mean': float(nulls.mean()), 'p': pv, 'corpus_rate': hits_all / max(1, len(pairs))}
json.dump(out, open(os.path.join(CK, 'c2b.json'), 'w'), indent=1)
