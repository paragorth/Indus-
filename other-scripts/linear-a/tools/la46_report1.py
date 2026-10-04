#!/usr/bin/env python3
"""LA-46 cycle-1 report: family-level residue counts vs null corpora, planted recall, LB/UR truth enrichment,
LA residue list."""
import sys, json, glob, collections
from la46_common import *

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c1'
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
MODE = sys.argv[3] if len(sys.argv) > 3 else 'max'
NULLS = ['W1', 'W2', 'SH1', 'SH2']


def load(name):
    fn = os.path.join(CK, '%s_%s.json' % (TAG, name))
    if not os.path.exists(fn):
        return None
    o = json.load(open(fn))
    if MODE == 'min':   # beat the weakest forger group only (liberal; calibrated by the null corpora)
        o['res'] = [[k, R, M, m, pois_sf(R, m)] for k, R, M, m, p in o['res']]
    return o


def fam_rates(o, th):
    t = collections.Counter(); p = collections.Counter()
    for k, R, M, m, pv in o['res']:
        t[k[0]] += 1
        if pv < th:
            p[k[0]] += 1
    return t, p


def main():
    C = {n: load(n) for n in ['LA', 'W1', 'W2', 'SH1', 'SH2', 'PL1', 'PL2', 'LB1', 'LB2', 'UR1', 'UR2']}
    C = {k: v for k, v in C.items() if v}
    print('threshold p <', TH)
    print('%-5s' % 'fam', ' '.join('%11s' % n for n in C))
    rates = {n: fam_rates(o, TH) for n, o in C.items()}
    for f in FAMS:
        print('%-5s' % f, ' '.join('%5d/%-5d' % (rates[n][1][f], rates[n][0][f]) for n in C))
    # null rate per family
    nulls = [n for n in NULLS if n in C]
    nr = {}
    for f in FAMS:
        a = sum(rates[n][1][f] for n in nulls); b = sum(rates[n][0][f] for n in nulls)
        nr[f] = (a + 0.5) / (b + 1)
    print('\nexcess over null rate (obs pass, expected from null rate, FDR):')
    for n in C:
        if n in nulls:
            continue
        row = []
        for f in FAMS:
            obs = rates[n][1][f]; exp = nr[f] * rates[n][0][f]
            row.append('%s %d/%.1f' % (f, obs, exp))
        print('  %-4s' % n, '  '.join(row))
    # planted recall
    for n in ('PL1', 'PL2'):
        if n not in C:
            continue
        o = C[n]; truth = set(tuple(t) for t in o['truth'])
        P = {tuple(k): pv for k, R, M, m, pv in o['res']}
        Rr = {tuple(k): (R, M) for k, R, M, m, pv in o['res']}
        found = [t for t in truth if P.get(t, 1) < TH]
        print('\n%s planted: %d/%d found at p<%g' % (n, len(found), len(truth), TH))
        for t in sorted(truth):
            print('   ', t, 'p=%.2g' % P.get(t, 1), Rr.get(t))
        fp = [k for k, pv in P.items() if pv < TH and k not in truth]
        print('   other passes:', len(fp), fp[:10])
    # LB / UR truth enrichment
    for n in ('LB1', 'LB2', 'UR1', 'UR2'):
        if n not in C:
            continue
        docs = [dict(d, toks=[tuple(x) for x in d['toks']]) for d in
                json.load(open(os.path.join(CK, '%s_%s_docs.json' % (TAG, n))))]
        if n.startswith('LB'):
            from la45_common import lb_truth
            def cv(t):
                return ('T', 'L:' + t[1]) if t[0] == 'L' else ('T', t[1]) if t[0] == 'W' else ('N', 0, 0)
            lab = lb_truth([{'toks': [cv(t) for t in d['toks'] if t[0] != 'NL'], 'site': d['site'],
                             'series': d.get('series', '')} for d in docs])
            lab = {k: v for k, v in lab.items() if not k.startswith('L:')}
        else:
            from la45_common import ur_truth
            lab = ur_truth([{'toks': [('T', t[1]) if t[0] == 'W' else ('N', 0, 0) for t in d['toks']
                                      if t[0] != 'NL'], 'site': d['site']} for d in docs])
        o = C[n]
        tot = collections.Counter(); hit = collections.Counter()
        cls_all = collections.Counter(); cls_res = collections.Counter()
        for k, R, M, m, pv in o['res']:
            if k[0] in ('XS', 'NN', 'SW'):
                continue
            ws = [x for x in (k[1], k[2]) if x in lab]
            c = lab[ws[0]] if ws else 'none'
            cls_all[c] += 1
            if pv < TH:
                cls_res[c] += 1
        nres = sum(cls_res.values()); nall = sum(cls_all.values())
        print('\n%s residues (word families) %d of %d; class share residue vs tested:' % (n, nres, nall))
        for c in sorted(cls_all, key=lambda c: -cls_all[c]):
            print('   %-10s %4d (%.2f) vs %5d (%.2f)' % (c, cls_res[c], cls_res[c] / max(1, nres), cls_all[c],
                                                       cls_all[c] / nall))
        top = sorted([r for r in o['res'] if r[4] < TH], key=lambda r: r[4])[:20]
        for k, R, M, m, pv in top:
            print('     ', k, R, '%.2f' % M, '%.1e' % pv, [lab.get(x, '') for x in k[1:]])
    if 'LA' in C:
        print('\nLA residues p<%g:' % TH)
        top = sorted([r for r in C['LA']['res'] if r[4] < TH], key=lambda r: r[4])
        for k, R, M, m, pv in top:
            print('   ', k, R, '%.2f' % M, '%.1e' % pv)


if __name__ == '__main__':
    main()
