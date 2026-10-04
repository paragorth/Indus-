#!/usr/bin/env python3
"""pe38 report helpers: summarise c1/c2/c3 checkpoints. Usage: python3 pe38_report.py c1|c2|c3"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe38_ckpt')
ROLES = ['NAM', 'PRO', 'COM', 'ANI', 'MEA', 'HDR', 'VRB', 'TOT', 'QUA']


def c1():
    R = json.load(open(os.path.join(CK, 'c1.json')))
    by = collections.defaultdict(list)
    for r in R:
        by[r['kind']].append(r)
    pe = by['PE'][0]
    fr = collections.Counter(v[0] for v in pe['forced'].values())
    print('PE: types', pe['ntypes'], 'forced', len(pe['forced']), dict(fr), 'viol', pe['viol'][:4])
    for k in ('NULLE', 'NULLA'):
        cs = [collections.Counter(v[0] for v in r['forced'].values()) for r in by[k]]
        print(k, {ro: (round(np.mean([c[ro] for c in cs]), 1), max(c[ro] for c in cs)) for ro in ROLES},
              'forced total', [len(r['forced']) for r in by[k]])
    # per-sign: forced role in PE and how often the nulls force the same sign to the same role
    rows = []
    for w, (ro, f, a) in sorted(pe['forced'].items(), key=lambda x: (x[1][0], -pe['nocc'][x[0]])):
        ne = sum(1 for r in by['NULLE'] if r['forced'].get(w, [None])[0] == ro)
        na = sum(1 for r in by['NULLA'] if r['forced'].get(w, [None])[0] == ro)
        rows.append((w, ro, f, a, pe['nocc'][w], ne, na))
    nonnull = [x for x in rows if x[5] == 0 and x[1] not in ('NAM',)]
    print('forced non-NAM roles never forced to that role in entry-shuffle nulls:', len(nonnull))
    for x in nonnull:
        print('  %s %s f%.2f a%.2f n%d nullE %d/4 nullA %d/2' % x)
    print('special roles (VRB/TOT/HDR/QUA/MEA) in PE forced:')
    for x in rows:
        if x[1] in ('VRB', 'TOT', 'HDR', 'MEA', 'QUA'):
            print('  %s %s f%.2f a%.2f n%d nullE %d/4 nullA %d/2' % x)
    for r in by['PLANT']:
        print('PLANT', r['k'], {w: (r['mode'].get(w), r['freq'].get(w), w in r['forced']) for w in list(r['truth']) + ['PLANT-E']},
              'E expected', collections.Counter(r['E_expected']).most_common(3))
    for k in ('PC', 'PCNULL', 'UR', 'URNULL'):
        for r in by[k]:
            s = r['score']
            fr = collections.Counter(v[0] for v in r['forced'].values())
            print(k, r['k'], 'types', r['ntypes'], 'forced', len(r['forced']), 'lab', s['n_lab'], 'forced-lab', s['n_forced_lab'],
                  'strict %.2f all %.2f bestmap %.2f maj %.2f' % (s['strict'], s['acc_all'], s['bestmap'], s['majority']),
                  'recall', {a: '%d/%d' % tuple(b) for a, b in s['recall'].items()}, dict(fr))
        if k in ('PC', 'UR'):
            # forced labelled types across samples
            fl = collections.Counter()
            for r in by[k]:
                for w, a, b in r['score']['forced_lab']:
                    fl[(w, a, b)] += 1
            print('  forced labelled (word, inferred, truth, samples):', sorted(fl.items(), key=lambda x: -x[1])[:30])


def c1g():
    import pe38_common as C
    R = json.load(open(os.path.join(CK, 'c1.json')))
    by = collections.defaultdict(list)
    for r in R:
        by[r['kind']].append(r)
    gf = {}
    for k, rs in by.items():
        for r in rs:
            fr, ng = C.good_forced(r)
            gf[(k, r['k'])] = fr
            line = '%s %d good chains %d viol min %d med %d forced %d %s' % (k, r['k'], ng, min(r['viol']), np.median(r['viol']), len(fr),
                                                                           dict(collections.Counter(fr.values())))
            if k in ('PC', 'PCNULL', 'UR', 'URNULL'):
                lab = C.PC_LAB if k.startswith('PC') else C.UR_LAB
                st, fl = C.score_forced(fr, lab)
                line += ' | labelled forced %d strict %.2f %s' % (len(fl), st, fl[:25])
            if k == 'PLANT':
                line += ' | plants %s' % {w: fr.get(w) for w in ('PLANT-T', 'PLANT-H', 'PLANT-V', 'PLANT-E')}
            print(line)
    pe = gf[('PE', 0)]
    print('PE good-forced by role:')
    for ro in ROLES:
        ws = [w for w, r in pe.items() if r == ro]
        nn = [sum(1 for k in ('NULLE', 'NULLA') for (kk, i), f in gf.items() if kk == k and f.get(w) == ro) for w in ws]
        print(' ', ro, len(ws), [(w, n) for w, n in zip(ws, nn)][:60])


def c2():
    R = json.load(open(os.path.join(CK, 'c2.json')))
    A = R['a']
    tr = np.array([x[1] for x in A]); nl = np.array(A[0][2] and [x[2] for x in A])
    print('a: n', len(A), 'true top5', np.sort(tr)[-5:].round(3), 'median', np.median(tr).round(3))
    print('   null per-perm max: mean %.3f range %.3f-%.3f' % (nl.max(0).mean(), nl.max(0).min(), nl.max(0).max()))
    print('   rank of true max among perm maxima:', (nl.max(0) >= tr.max()).sum(), 'of', nl.shape[1])
    for tag in ('top', 'null', 'med'):
        b = [x for x in R['b'] if x['tag'] == tag]
        print('b', tag, 'PCB %.3f (%.3f-%.3f)' % (np.mean([x['PCB'] for x in b]), min(x['PCB'] for x in b), max(x['PCB'] for x in b)),
              'UR %.3f (%.3f-%.3f)' % (np.mean([x['UR'] for x in b]), min(x['UR'] for x in b), max(x['UR'] for x in b)))
    for c in R['c']:
        print('c', c['src'], '->', c['dst'], 'lam', c['lam'], 'argmax %.3f gibbs %.3f null %.3f (max %.3f)' %
              (c['argmax'], c['gibbs'], np.mean(c['null_argmax']), max(c['null_argmax'])))
    if 'd' in R:
        print('d', json.dumps(R['d'])[:3000])


def c3():
    R = json.load(open(os.path.join(CK, 'c3.json')))
    print(json.dumps(R.get('summary', {}), indent=1)[:6000])


if __name__ == '__main__':
    {'c1': c1, 'c1g': c1g, 'c2': c2, 'c3': c3}[sys.argv[1]]()
