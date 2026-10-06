"""v78 cycle 4: IS THE DOUBLED UNIT LONGER THAN ONE TOKEN? ABAB and ABA in the Voynich vs sub-word streams.

Cycle 1: the Voynich surplus is larger at lag 2 (1.23) than at lag 1 (1.15). Cycle 3: sub-word units of reduplicating
languages (Malay and Tagalog syllables and letters) show a big lag-2 surplus, because a reduplicated two-unit word
is written A B A B. So if the Voynich lag-2 surplus is reduplication of two-token groups, lag-2 matches come in
consecutive pairs: P(T[i+1]=T[i+3] | T[i]=T[i+2]) far above the lag-2 base rate (block ratio), and block repeats
of two tokens (ABAB) exceed a within-line shuffle. An alternation (ABA, as in chant) gives isolated lag-2 matches.
Statistics per stream (E1c for surface texts; plain units for the cycle-3 unit streams):
  blk   = P(lag-2 match at i+1 | lag-2 match at i) / P(lag-2 match)       (ABAB clustering)
  abab  = count of ABAB (A != B) / mean count in 20 within-line shuffles
  aba   = count of ABA with A != B / shuffle mean (isolated alternation)
Kill: Voynich blk and abab inside the generators' band means the lag-2 surplus is not a two-token reduplication.
"""
import os, sys, json, pickle, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L, v78_c3 as C3


def lines_of(pages, key):
    return [[key(w) for w in l['w']] for p in pages for l in p['lines']]


def stats(Ls, nsh=20, seed=0):
    def cnt(Ls):
        n2 = m2 = c = cc = abab = aba = 0
        for T in Ls:
            n = len(T)
            for i in range(n - 2):
                n2 += 1
                if T[i] == T[i + 2]:
                    m2 += 1
                    if T[i] != T[i + 1]:
                        aba += 1
                    if i + 3 < n:
                        c += 1
                        if T[i + 1] == T[i + 3]:
                            cc += 1
                            if T[i] != T[i + 1]: abab += 1
        return n2, m2, c, cc, abab, aba
    n2, m2, c, cc, abab, aba = cnt(Ls)
    rng = random.Random(seed); sa = []; sb = []
    for _ in range(nsh):
        S = [rng.sample(T, len(T)) for T in Ls]; r = cnt(S); sa.append(r[4]); sb.append(r[5])
    base = m2 / max(1, n2)
    return dict(lag2=base, blk=(cc / max(1, c)) / max(1e-9, base), n_abab=abab, abab=abab / max(0.5, np.mean(sa)),
                aba=aba / max(0.5, np.mean(sb)), n_aba=aba)


def main():
    C = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))
    res = {}
    for name in ('VOY_ZL', 'VOY_IT', 'GEN_SELFCIT', 'GEN_SC10', 'GEN_MK2', 'GEN_JUNC', 'GEN_STACK', 'MS_1001', 'TL_MED', 'LA_ISID'):
        r = stats(lines_of(C[name]['pages'], L.e1c)); res[name] = r
        print(name, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
        if name.startswith('VOY'):
            for h in (0, 1):
                r = stats(lines_of(L.half(C[name]['pages'], h), L.e1c)); res['%s_h%d' % (name, h)] = r
                print(' half', h, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    S = C3.sources()
    for src in ('MS', 'TL', 'LA', 'IT'):
        for lev in ('syll', 'letter'):
            pages = L.plain_pages(C3.units(S[src], lev), src.lower(), cap=36000, line_w=9)
            r = stats(lines_of(pages, lambda w: w)); res['%s_%s' % (src, lev)] = r
            print(src, lev, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    r = stats(lines_of(C3.chant_pages(), lambda w: w)); res['CHANT'] = r
    print('CHANT', json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c4.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
