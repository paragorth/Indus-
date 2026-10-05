#!/usr/bin/env python3
"""pe56 cycle 1 report: residue counts per corpus / family / threshold, planted recall, top residues."""
import sys, json, collections, os
from pe56_common import CK, FAMS

UR_KEY = {'total': {'szu-nigin2', 'szu-nigin', 'sza3-bi-ta', 'la2-ia3', 'zi-ga', 'nig2-ka9-ak', 'sag-nig2-gur11-ra-kam'},
          'commodity': {'udu', 'masz2', 'gu4', 'sze', 'kasz', 'ninda', 'i3', 'siki', 'u8', 'sila4', 'ab2', 'ansze',
                        'ud5', 'kir11', 'dabin', 'zi3', 'gu4-niga', 'udu-niga', 'tug2', 'gesz', 'sa', 'ku6',
                        'munu4', 'zu2-lum', 'amar', 'ga', 'dug', 'gi', 'niga', 'kusz', 'u2', 'gukkal', 'masz2-gal',
                        'durah', 'szah2', 'u8-niga', 'sze-ba', 'sila3', 'gur', 'ma-na', 'gin2', 'sar', 'iku', 'gun2'},
          'worker': {'gurusz', 'geme2', 'dumu', 'erin2', 'a2', 'u4', 'ugula', 'lu2'},
          'date': {'iti', 'mu', 'u4', 'ki', 'giri3', 'kiszib3', 'mu-kux(DU)', 'ba-zi', 'i3-dab5', 'szu', 'ba-ti',
                   'ba-hul', 'ba-hun', 'us2-sa', 'lugal'}}
PC_KEY = {'commodity': {'P_' + x for x in ['SZE', 'GAR', 'KASZ', 'KU6', 'UDU', 'U8', 'SZAH2', 'GU4', 'AB2', 'DUG',
                                           'TUG2', 'SILA4', 'MASZ', 'KISZ', 'NINDA2', 'ZIZ2', 'SZE3', 'SZE~a', 'GAR~a',
                                           'KU6~a', 'BA', 'ZATU659', 'SZAH2~a']},
          'title': {'P_' + x for x in ['EN', 'SANGA', 'NAM2', 'GAL', 'SUKKAL', 'UMBISAG', 'NUN', 'SAL', 'KUR', 'ERIM',
                                       'SAG', 'GURUSZ']}}


def label(name, a):
    K = UR_KEY if name.startswith('UR') else PC_KEY if name.startswith('PC') else {}
    for k, v in K.items():
        if a in v:
            return k
    return ''


def load(name):
    fn = os.path.join(CK, 'c1_%s.json' % name)
    return json.load(open(fn)) if os.path.exists(fn) else None


def main(names):
    TH = [1e-3, 1e-5, 1e-7]
    for name in names:
        d = load(name)
        if d is None:
            print(name, 'missing'); continue
        res = d['res']
        n = len(res)
        bonf = 0.05 / n
        by = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        for k, R, E, M, pm, px in res:
            f = k[0]
            by[f][0] += 1
            for i, t in enumerate(TH):
                if pm < t:
                    by[f][i + 1] += 1
            if pm < bonf:
                by[f][4] += 1
        tot = [sum(v[i] for v in by.values()) for i in range(5)]
        print('== %s docs %d tok %d tested %d | p<1e-3 %d, 1e-5 %d, 1e-7 %d, bonf(%.1e) %d | max-group rule bonf %d' % (
            name, d['n_docs'], d['n_tok'], n, tot[1], tot[2], tot[3], bonf, tot[4],
            sum(1 for x in res if x[5] < bonf)))
        print('   ', {f: by[f] for f in FAMS if f in by})
        if d['truth']:
            T = set(map(tuple, d['truth']))
            hit = [x for x in res if tuple(x[0]) in T]
            fp = [x for x in res if x[4] < bonf and tuple(x[0]) not in T]
            print('    planted: %d/%d tested; found(bonf) %d; found(1e-3) %d; false passes(bonf) %d' % (
                len(hit), len(T), sum(x[4] < bonf for x in hit), sum(x[4] < 1e-3 for x in hit), len(fp)))
            for x in hit:
                print('      truth', x[0], 'R %d E %.2f p %.1e' % (x[1], x[2], x[4]))
            for x in sorted(fp, key=lambda x: x[4])[:8]:
                print('      FP', x[0], 'R %d E %.2f p %.1e' % (x[1], x[2], x[4]))
        else:
            top = sorted([x for x in res if x[4] < bonf], key=lambda x: x[4])
            for x in top[:int(os.environ.get('TOPN', '25'))]:
                k = x[0]
                print('      %-4s %-18s %-18s R %3d E %6.2f Emax %6.2f p %.1e %s %s' % (
                    k[0], k[1], k[2], x[1], x[2], x[3], x[4], label(name, k[1]), label(name, k[2])))


if __name__ == '__main__':
    main(sys.argv[1:] or ['PE', 'W1', 'W2', 'SH1', 'SH2', 'PL1', 'PL2', 'PC1', 'PC2', 'UR1', 'UR2'])
