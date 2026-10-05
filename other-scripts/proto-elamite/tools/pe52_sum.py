"""pe52 summary of job files: python3 pe52_sum.py <glob-tag>"""
import sys, os, json, glob, re
import numpy as np
import pe52_lib as L

HEADS = {'udu', 'sila4', 'gu4', 'masz2', 'masz2-gal', 'u8', 'ab2', 'ud5', 'amar', 'kir11', 'gukkal', 'masz-da3',
         'udu-nita2', 'dara4', 'munus', 'udu-a-lum'}
HERD_DESC = HEADS | {'niga', 'u2', 'ga', 'us2', 'saga', 'sag', 'kam', 'mu', 'lum', 'gun3', 'babbar', 'ge6', 'nita2',
                     'sig5', 'gaba', 'bar', 'gal2', 'ki', 'sal'}
PREREG = {'niga': -1}   # fattened animals counted in small numbers (stated before the run)


def herd_truth(s):
    return s in HERD_DESC or bool(re.match(r'^\d', s))


def describe(f):
    R = json.load(open(f))
    T = R['table']; sc = R['scores']; ex = R['explained']
    D = sorted([s for s, v in T.items() if v['cls'] == 'D'], key=lambda s: -abs(T[s]['med']))
    I = [s for s, v in T.items() if v['cls'] == 'I']
    line = (f"{R['corpus']:8s} {R['mode']} {R['null']:7s} n={R['n']} bestC={sc['best_gC']:+.3f} "
            f"(pos {sc['best_gC_pos']:.1f}) novel={sc['best_gCn']:+.3f} rel={sc['rel_gCn']:.3f} "
            f"nsurv={sc['nsurv']:.0f} D={len(D)} I={len(I)} full_str={ex['full_strings']:.2f} "
            f"full_ent={ex['full_entries']:.2f} share={ex['mean_desc_share']:.2f}")
    extra = []
    if R['corpus'] in ('HERD', 'HERD2'):
        tp = sum(herd_truth(s) for s in D)
        It = sum(herd_truth(s) for s in I)
        extra.append(f"   HERD: D in descriptor vocab {tp}/{len(D)}; I in descriptor vocab {It}/{len(I)}")
        for s, sg in PREREG.items():
            v = T.get(s)
            if v:
                extra.append(f"   prereg {s}: med {v['med']:+.2f} cls {v['cls']} (expected sign {sg:+d})")
    if R['corpus'] in ('PERS', 'HERD2'):
        nf = R['namefrac']
        dn = [s for s in D if nf.get(s, 0) > 0.5]
        In = [s for s in I if nf.get(s, 0) > 0.5]
        freqn = [s for s, v in T.items() if v['occ'] >= 10 and v['ntab'] >= 3 and nf.get(s, 0) > 0.5]
        freqo = [s for s, v in T.items() if v['occ'] >= 10 and v['ntab'] >= 3 and nf.get(s, 0) <= 0.5]
        do = [s for s in D if nf.get(s, 0) <= 0.5]
        extra.append(f"   PERS: frequent name tokens {len(freqn)}: D {len(dn)} ({len(dn)/max(1,len(freqn)):.2f}) I {len(In)}; "
                     f"frequent other tokens {len(freqo)}: D {len(do)} ({len(do)/max(1,len(freqo)):.2f}); "
                     f"name-D: {dn[:8]}")
    if R['planted']:
        P = R['planted']
        hit = [s for s in D if s in P]
        right = sum(np.sign(T[s]['med']) == np.sign(P[s]) for s in hit)
        extra.append(f"   PLANT: recall {len(hit)}/{len(P)} precision {len(hit)}/{len(D)} right sign {right}/{len(hit)}")
    top = ', '.join(f"{s}({T[s]['med']:+.2f},{T[s]['occ']})" for s in D[:14])
    extra.append(f"   top D: {top}")
    return line + '\n' + '\n'.join(extra)


if __name__ == '__main__':
    for f in sorted(glob.glob(os.path.join(L.CK, sys.argv[1] + '*.json'))):
        print(describe(f))
