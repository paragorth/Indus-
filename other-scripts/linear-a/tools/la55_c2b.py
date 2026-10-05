#!/usr/bin/env python3
"""LA-55 cycle 2b: 'names grow in proportion'. Deviation = scaling class != LINEAR (from c1, null N1 or N3).
Test: known names (PERSON) deviate less than function words + commodities, with labels permuted inside frequency
bins (so rarer names cannot win by low power). Linear A: la45 ENTRY+RARE vs HEADER+COMMODITY, same test."""
import json, re, ast, collections, sys
import numpy as np
from la55_common import *

rng = np.random.default_rng(seed('la55-c2b'))


def la45_classes():
    la45 = {}
    for line in open(os.path.join(D, 'la45_ckpt', 'c1_report_LA.txt')):
        m = re.match(r'\s+(HEADER|COMMODITY|ENTRY|RARE) (\[.*\])\s*$', line)
        if m:
            for t in ast.literal_eval(m.group(2)):
                w = t[0]
                k = 'L:' + w[2:].split('+')[0] if w.startswith('L:') else ('w:' + w if '-' in w else 's:' + w)
                la45.setdefault(k, m.group(1))
    return la45


def test(types, K, cls, lab, A, B, nperm=20000):
    dev = np.array([c != 'LINEAR' for c in cls])
    lab = np.array([l or '' for l in lab], dtype=object)
    use = np.array([l in A or l in B for l in lab])
    isA = np.array([l in A for l in lab])
    lk = np.log(np.array(K, float))
    edges = np.quantile(lk[use], [0.25, 0.5, 0.75]) if use.sum() > 8 else []
    bins = np.digitize(lk, edges)
    def stat(isA_):
        a = dev[use & isA_].mean() if (use & isA_).any() else np.nan
        b = dev[use & ~isA_].mean() if (use & ~isA_).any() else np.nan
        return b - a          # B deviates more than A
    s0 = stat(isA)
    perms = []
    idx_by_bin = [np.where(use & (bins == b))[0] for b in np.unique(bins[use])]
    for _ in range(nperm):
        x = isA.copy()
        for ix in idx_by_bin:
            x[ix] = rng.permutation(isA[ix])
        perms.append(stat(x))
    perms = np.array(perms)
    p = (np.sum(perms >= s0 - 1e-12) + 1) / (nperm + 1)
    return dict(devA=float(dev[use & isA].mean()), nA=int((use & isA).sum()), devB=float(dev[use & ~isA].mean()),
                nB=int((use & ~isA).sum()), diff=float(s0), p=float(p))


out = {}
for corpus in ('LB', 'LBS', 'UR', 'LA'):
    fn = os.path.join(CK, 'c1_%s.json' % corpus)
    if not os.path.exists(fn):
        continue
    r = json.load(open(fn))
    if corpus == 'LA':
        tr = la45_classes(); A = {'ENTRY', 'RARE'}; B = {'HEADER', 'COMMODITY'}
    elif corpus.startswith('LB'):
        tr = lb_truth_classes(); A = {'PERSON'}; B = {'FUNC', 'COMM'}
    else:
        tr = ur_truth_classes(ur_units(max_docs=12000)); A = {'PERSON'}; B = {'FUNC', 'COMM'}
    lab = [tr.get(t) for t in r['types']]
    for key in ('N1', 'N3'):
        if key not in r: continue
        res = test(r['types'], r['Ktot'], r[key]['cls'], lab, A, B)
        out['%s_%s' % (corpus, key)] = res
        print(corpus, key, 'deviating: %s %.2f (n %d) vs %s %.2f (n %d); diff %.2f, freq-binned perm p %.4f' % (
            '+'.join(sorted(A)), res['devA'], res['nA'], '+'.join(sorted(B)), res['devB'], res['nB'], res['diff'], res['p']))
        if corpus != 'LA':
            res2 = test(r['types'], r['Ktot'], r[key]['cls'], lab, {'PLACE'}, B)
            out['%s_%s_place' % (corpus, key)] = res2
            print('   PLACE vs FUNC+COMM: %.2f (n %d) vs %.2f (n %d), p %.4f' % (res2['devA'], res2['nA'], res2['devB'],
                                                                          res2['nB'], res2['p']))
jdump(out, 'c2b.json')
