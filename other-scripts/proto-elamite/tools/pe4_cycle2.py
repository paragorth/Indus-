"""pe4 cycle 2: the two predictions of a factorial item code, tested against
shuffles and calibrated on the same positive / negative corpora as cycle 1.

(a) exclusivity + order, held out: strict factorial with fixed k (3) fitted on
    half A of the strings; share of half-B strings (all tokens seen in A) that
    are valid (<= 1 value per slot, slots in order).  Null: slot labels permuted
    among sign types, slot sizes kept (200x).  Statistic: lift = valid / null.
(b) forbidden pairs: among the 25 most frequent tokens, pairs expected >= 2
    times together (from presence rates) that never co-occur; share vs a global
    token shuffle keeping string lengths (50x).  Same-slot values of a factorial
    code must never co-occur.
(c) combinatorial filling: fit k = 3 on all strings; the two most-present
    slots; values seen >= 3 times; distinct value pairs observed / expected
    under independence (slot-B values shuffled among the strings holding both
    slots, 200x).  A factorial code fills its cross-table (ratio ~1); fixed
    collocations (names) under-fill it (ratio < 1).
Run: python3 pe4_cycle2.py -> data/pe4_cycle2.json
"""
import json, os, random, statistics as st, sys
from collections import Counter
from pe4_common import *

N = int(os.environ.get('PE4_N', 110))
R = int(os.environ.get('PE4_R', 8))
K = int(os.environ.get('PE4_K', 3))
OUT = os.path.join(PEDATA, 'pe4_cycle2_k%d_N%d.json' % (K, N))


def heldout_valid(smp, rng):
    idx = list(range(len(smp)))
    rng.shuffle(idx)
    A = [smp[i] for i in idx[:len(idx) // 2]]
    B = [smp[i] for i in idx[len(idx) // 2:]]
    F = Factorial(A, K, rng=rng)
    F.fit()
    seen = set(F.g)
    Bs = [w for w in B if all(s in seen for s in w)]
    if len(Bs) < 5:
        return None
    def vshare(g):
        ok = 0
        for w in Bs:
            last = -1
            good = True
            for s in w:
                if g[s] <= last:
                    good = False
                    break
                last = g[s]
            ok += good
        return ok / len(Bs)
    obs = vshare(F.g)
    keys = list(F.g)
    vals = [F.g[s] for s in keys]
    null = []
    for _ in range(200):
        rng.shuffle(vals)
        null.append(vshare(dict(zip(keys, vals))))
    m = st.mean(null)
    return {'valid': obs, 'null': m, 'lift': obs / m if m else float('nan'),
            'p': (1 + sum(x >= obs for x in null)) / 201, 'nB': len(Bs)}


def forbidden(smp, rng, top=25):
    def share(C):
        n = len(C)
        pres = Counter(s for w in C for s in set(w))
        tops = [s for s, _ in pres.most_common(top)]
        sets = [set(w) for w in C]
        cand = zero = 0
        for i in range(len(tops)):
            for j in range(i + 1, len(tops)):
                a, b = tops[i], tops[j]
                e = pres[a] * pres[b] / n
                if e >= 2:
                    cand += 1
                    if not any(a in s and b in s for s in sets):
                        zero += 1
        return zero / cand if cand else float('nan'), cand
    obs, cand = share(smp)
    toks = [s for w in smp for s in w]
    null = []
    for _ in range(50):
        rng.shuffle(toks)
        C, i = [], 0
        for w in smp:
            C.append(tuple(toks[i:i + len(w)]))
            i += len(w)
        null.append(share(C)[0])
    null = [x for x in null if x == x]
    m = st.mean(null) if null else float('nan')
    return {'forb': obs, 'null': m, 'excess': obs - m, 'cand': cand}


def filling(smp, rng):
    F = Factorial(smp, K, rng=rng)
    F.fit()
    pres = sorted(range(K), key=lambda s: -F.pres[s])[:2]
    a, b = sorted(pres)
    rows = []
    for w, ok in zip(F.C, F.ok):
        if not ok:
            continue
        d = {F.g[s]: s for s in w}
        if a in d and b in d:
            rows.append((d[a], d[b]))
    ca = Counter(x for x, _ in rows)
    cb = Counter(y for _, y in rows)
    rows = [(x, y) for x, y in rows if ca[x] >= 2 and cb[y] >= 2]
    if len(rows) < 10:
        return None
    obs = len(set(rows))
    xs = [x for x, _ in rows]
    ys = [y for _, y in rows]
    null = []
    for _ in range(200):
        rng.shuffle(ys)
        null.append(len(set(zip(xs, ys))))
    m = st.mean(null)
    return {'fill': obs / m, 'cells': obs, 'null': m, 'rows': len(rows),
            'p_low': (1 + sum(x <= obs for x in null)) / 201}


def main():
    corp = all_corpora()
    prof = profile(corp['PE_all'][0])
    res = {}
    for name, (types, role) in corp.items():
        H, Fb, Fl = [], [], []
        for r in range(R):
            rng = random.Random(1000 * r + 7)
            smp = matched(types, prof, N, rng)
            h = heldout_valid(smp, random.Random(r))
            if h: H.append(h)
            Fb.append(forbidden(smp, random.Random(r)))
            f = filling(smp, random.Random(r))
            if f: Fl.append(f)
        mean = lambda L, k: st.mean(x[k] for x in L) if L else float('nan')
        summ = {'valid': mean(H, 'valid'), 'valid_null': mean(H, 'null'), 'lift': mean(H, 'lift'),
                'p_valid_med': st.median(x['p'] for x in H) if H else float('nan'),
                'forb': mean(Fb, 'forb'), 'forb_null': mean(Fb, 'null'), 'forb_excess': mean(Fb, 'excess'),
                'fill': mean(Fl, 'fill'), 'fill_n': len(Fl), 'p_fill_low_med': st.median(x['p_low'] for x in Fl) if Fl else float('nan')}
        res[name] = {'role': role, 'summary': summ, 'heldout': H, 'forbidden': Fb, 'filling': Fl}
        print('%-10s %-3s heldout valid %.2f vs null %.2f lift %.2f (med p %.3f) | forbidden %.2f vs %.2f (+%.2f) | fill %.2f (n=%d, med p_low %.3f)'
              % (name, role, summ['valid'], summ['valid_null'], summ['lift'], summ['p_valid_med'], summ['forb'],
                 summ['forb_null'], summ['forb_excess'], summ['fill'], summ['fill_n'], summ['p_fill_low_med']), flush=True)
    json.dump({'N': N, 'R': R, 'K': K, 'res': res}, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
