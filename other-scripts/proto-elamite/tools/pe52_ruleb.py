"""pe52 rule B (set in cycle 2 after the Ur III herd calibration): a DESCRIPTOR is a sign with >= 10 tokens on
>= 3 tablets, present in >= 75% of splits' survivors, same sign of effect in >= 90% of splits and |median effect| >= EFF.
(The novel-string ablation of rule A is reported as a second flag 'G' = also generalises to never-seen strings.)
Real Ur III attribute words (udu, niga, saga ...) recur in the same few strings, so novel strings carrying them
are rare and atypical; rule A missed them although their weights were stable.
usage: python3 pe52_ruleb.py <glob-tag> [EFF]"""
import sys, os, json, glob
import numpy as np
import pe52_lib as L
from pe52_sum import herd_truth, HERD_DESC


def ruleb(T, S, eff):
    out = {}
    for s, v in T.items():
        d = (v['occ'] >= 10 and v['ntab'] >= 3 and v['splits'] >= 0.75 * S and v['consist'] >= 0.9
             and abs(v['med']) >= eff)
        g = d and v['nabl'] >= 3 and v['ablpos'] >= 0.75
        out[s] = 'G' if g else 'D' if d else ('I' if v['occ'] >= 10 and v['ntab'] >= 3 else '-')
    return out


def main(tag, eff=0.1):
    for f in sorted(glob.glob(os.path.join(L.CK, tag + '*.json'))):
        R = json.load(open(f))
        if 'table' not in R:
            continue
        T = R['table']; cl = ruleb(T, R['S'], eff)
        D = sorted([s for s in cl if cl[s] in 'DG'], key=lambda s: -abs(T[s]['med']))
        nI = sum(c == 'I' for c in cl.values())
        msg = f"{R['corpus']:7s} {R['mode']} {R['null']:7s} D={len(D)} (G={sum(cl[s]=='G' for s in D)}) of {len(D)+nI} frequent"
        if R['corpus'] in ('HERD', 'HERD2'):
            msg += f" | truth-vocab D {sum(herd_truth(s) for s in D)}/{len(D)}; vocab signs frequent {sum(herd_truth(s) for s in cl if cl[s] != '-')}"
            for s in ('niga', 'udu', 'sila4', 'saga', 'u2', 'ga', 'gu4'):
                if s in T:
                    msg += f" {s}{T[s]['med']:+.2f}{cl[s]}"
        if R.get('namefrac'):
            nf = R['namefrac']
            fn = [s for s in cl if cl[s] != '-' and nf.get(s, 0) > 0.5]
            fo = [s for s in cl if cl[s] != '-' and nf.get(s, 0) <= 0.5]
            msg += (f" | name tokens D {sum(cl[s] in 'DG' for s in fn)}/{len(fn)}; other D "
                    f"{sum(cl[s] in 'DG' for s in fo)}/{len(fo)}")
        if R.get('planted'):
            P = R['planted']
            hit = [s for s in D if s in P]
            msg += f" | plant recall {len(hit)}/{len(P)} precision {len(hit)}/{len(D)}"
        msg += ' | ' + ', '.join(f"{s}{T[s]['med']:+.2f}{cl[s]}" for s in D[:12])
        print(msg)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.1)
