"""la63 cycle 2 report: Linear A loss classes, seed-half stability, within-document null, targets.
usage: python3 la63_report.py
"""
import collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import la63_sum as S
import la63_lib as L

TARGETS = {
    'la57 commodity candidates': ['DI-DE-RU', 'DA-SI-*118', '*28B-NU-MA-RE', 'U-*325-ZA', 'TE-TU'],
    'between-line single signs': ['RA', 'PA', 'PA₃', 'TU', 'ME', '*318'],
    'totals and transaction words': ['KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'SA-RA₂', 'A-DU', 'KA-PA', 'KU-PA'],
    'la60 commodity single signs': ['NI', 'KI', 'DI', 'RE', 'TI', 'MA', '*304', '*308', '*306', 'E', 'SU', '*307'],
    'la45/la49 heading signs': ['*301', 'KA', 'KU', 'SI', 'RO', 'ZE', 'TE', 'A', 'I', 'TA', 'O', 'JA', 'DA'],
    'la60 entry words': ['KU-PA₃-NU', 'MA-DI', 'SA-RU', 'SA-RO', 'DA-RE', 'TA-I'],
}


def frac_words(C):
    st = collections.defaultdict(lambda: [0, 0])
    for d in C:
        for e in d['ents']:
            if e['kind'] == 'num':
                for w in e['toks']:
                    st[w][0] += 1; st[w][1] += 1 if e['frac'] else 0
    return st


def main():
    R = S.analyse('la')
    n = len(R['models'])
    A = S.analyse('la', models=set(m for m in R['models'] if m % 2 == 0))
    B = S.analyse('la', models=set(m for m in R['models'] if m % 2 == 1))
    N = S.analyse('la_nw')
    print('\n'.join(S.report(R, None, top=40)))
    stab = {w: (A['C'].get(w) == R['C'][w] and B['C'].get(w) == R['C'][w]) for w in R['C']}
    cls = [w for w in R['C'] if R['C'][w] != 'SILENT']
    print('classed %d; stable in both seed halves %d: %s' % (len(cls), sum(stab[w] for w in cls),
          ' '.join('%s=%s' % (w, R['C'][w]) for w in cls if stab[w])))
    print('half agreement on classed types: A %s | B %s' % (
        ' '.join('%s=%s' % (w, A['C'].get(w)) for w in cls), ' '.join('%s=%s' % (w, B['C'].get(w)) for w in cls)))
    if N:
        print('within-doc null la_nw: models %d, classes %s, alpha %.4f' % (len(N['models']), dict(collections.Counter(N['C'].values())), N['alpha']))
        print('  classed in null: %s' % ' '.join('%s=%s' % (w, c) for w, c in N['C'].items() if c != 'SILENT'))
        print('  real classed types in null: %s' % ' '.join('%s=%s' % (w, N['C'].get(w, '-')) for w in cls))
    C = L.build_la()
    fw = frac_words(C)
    for name, ws in TARGETS.items():
        print('\n' + name)
        for w in ws:
            s = fw.get(w, [0, 0])
            print('  ' + S.profile(R, w) + '   [num entries %d, with fraction %d]%s' % (
                s[0], s[1], ('  null: %s' % N['C'].get(w, '-')) if N else ''))
    # words before fractions: types with >= 50% fraction share and >= 3 numeral entries
    print('\nwords before fractions (>= 3 numeral entries, >= 50% with a fraction)')
    for w, (a, b) in sorted(fw.items(), key=lambda x: -x[1][1]):
        if a >= 3 and b / a >= 0.5:
            print('  ' + S.profile(R, w) + '   [%d/%d]' % (b, a))


if __name__ == '__main__':
    main()
