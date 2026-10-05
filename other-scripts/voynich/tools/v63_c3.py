"""v63 cycle 3: SLIP OR PAIRING RULE?  Tests on the adjacency-specific ordered alternations of cycle 2
(S* = operators whose orientation beats the column shuffle by >= 1.5 sd in BOTH ZL3b and IT2a).

For pairs (A, B) adjacent with A = op(B), op in S*:
 (a) frequency direction: share with f(A) < f(B) and mean log f(A) - log f(B)  (slip: first twin rarer)
 (b) line breaks: share of pairs straddling a line break (A line-final, B next line-initial) vs all adjacent pairs
 (c) deletion test: dDel = [gain from deleting A] - [gain from deleting B] under a held-out (2-fold by folio)
     interpolated bigram + junction model.  Slip (A extraneous): dDel > 0.
 Reference distribution: the same quantities for all other adjacent near pairs (edit 1-2).
Calibration: ZL + 2% planted 6-op slips (pairs = the planted ones, and the S*-style search output); real Plaoul
struck words (a, b); copy-and-modify generator (a, c).
Also the 'corrected text': S* first twins deleted; what is left, and does adjacency orientation drop to the shuffle.
"""
import json, os, sys, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L, v63_plant as PL
from v63_c1 import BiLM
from v63_c2 import script, inv_full, plant_ops, counts, agg, zset, column_shuffle_folio

SSTAR = ['Sl>o@I', 'Ss>l@I', 'Sr>s@F', 'Sy>l@F', 'ST>C@I', 'Ia@M']


def models(streams, seed=0):
    fol = sorted(set(s['folio'] for s in streams)); random.Random(seed).shuffle(fol)
    h = set(fol[:len(fol) // 2])
    m0 = BiLM([s for s in streams if s['folio'] in h]); m1 = BiLM([s for s in streams if s['folio'] not in h])
    return lambda f: (m1 if f in h else m0)


def pair_stats(streams, select, label=''):
    """select(A,B,sc,s,i) -> bool"""
    f = Counter(w for s in streams for w in s['words'])
    M = models(streams)
    sel = defaultdict(list); n_all = 0; n_lb = 0
    for s in streams:
        ws = s['words']; lm = M(s['folio']); line = s.get('line')
        for i in range(len(ws) - 1):
            A, B = ws[i], ws[i + 1]
            lb = bool(line) and line[i][0] != line[i + 1][0]
            n_all += 1; n_lb += lb
            if A == B: continue
            sc = script(B, A)
            if sc is None: continue
            grp = 'S' if select(A, B, sc, s, i) else 'R'
            dd = None
            if 0 < i < len(ws) - 2:
                prev, nxt = ws[i - 1], ws[i + 2]
                gA = lm.lp(B, prev) - lm.lp(A, prev) - lm.lp(B, A)        # delete A
                gB = lm.lp(nxt, A) - lm.lp(B, A) - lm.lp(nxt, B)          # delete B
                dd = gA - gB
            sel[grp].append((math.log(f[A]) - math.log(f[B]), lb, dd))
    out = {}
    for g, v in sel.items():
        a = np.array([x[0] for x in v]); b = np.array([x[1] for x in v]); d = np.array([x[2] for x in v if x[2] is not None])
        out[g] = dict(n=len(v), rarer_first=float(np.mean(a < 0)), dlogf=float(a.mean()), dlogf_se=float(a.std() / math.sqrt(len(a))),
                      lb=float(b.mean()), dDel=float(d.mean()) if len(d) else 0, dDel_se=float(d.std() / math.sqrt(max(1, len(d)))))
    out['lb_all'] = n_lb / max(1, n_all)
    s, r = out.get('S'), out.get('R')
    if s and r:
        print(f"{label:34s} S* n {s['n']:4d}: rarer-first {s['rarer_first']:.2f} dlogf {s['dlogf']:+.2f}+-{s['dlogf_se']:.2f}  "
              f"line-break {s['lb']:.3f}  dDel {s['dDel']:+.2f}+-{s['dDel_se']:.2f} | rest n {r['n']}: rarer-first {r['rarer_first']:.2f} "
              f"dlogf {r['dlogf']:+.2f}  lb {r['lb']:.3f}  dDel {r['dDel']:+.2f}+-{r['dDel_se']:.2f} | all-pairs lb {out['lb_all']:.3f}", flush=True)
    return out


def in_sstar(A, B, sc, s, i):
    return any(o in SSTAR for o in sc)


def corrected(streams):
    """delete the first twin of every S* pair; returns streams + list of (deleted, kept) examples."""
    out, ex = [], Counter()
    for s in streams:
        ws, ln, skip = [], [], set()
        w0 = s['words']
        for i in range(len(w0) - 1):
            sc = script(w0[i + 1], w0[i]) if w0[i] != w0[i + 1] else None
            if sc and any(o in SSTAR for o in sc) and i not in skip:
                skip.add(i); ex[(''.join(w0[i]), ''.join(w0[i + 1]))] += 1
        for i, w in enumerate(w0):
            if i in skip: continue
            ws.append(w); ln.append(s['line'][i])
        out.append(dict(s, words=ws, line=ln))
    return out, ex


def main():
    res = {}
    zl = L.voynich_paras('ZL3b'); it = L.voynich_paras('IT2a')
    res['ZL'] = pair_stats(zl, in_sstar, 'V-ZL3b')
    res['IT'] = pair_stats(it, in_sstar, 'V-IT2a')
    # planted 6-op slips in ZL: S = the planted pairs (known labels)
    S, tab = plant_ops(zl, 0.02, seed=5)
    tops = ['S%s>%s@%s' % (x, y, p) for x, y in tab.items() for p in 'IMF']
    res['plant'] = pair_stats(S, lambda A, B, sc, s, i: any(o in tops for o in sc), 'V-ZL + 2% planted 6-op slips')
    cg = PL.copygen(zl, seed=0)
    for s2, s in zip(cg, zl): s2['line'] = s['line']
    res['copygen'] = pair_stats(cg, in_sstar, 'copy-and-modify (S* ops)')
    for k in range(3):
        cs = column_shuffle_folio(zl, k)
        res[f'colshuf{k}'] = pair_stats(cs, in_sstar, f'ZL folio column-shuffle #{k}')
    # real Latin slips: struck A vs next kept B, frequencies from corrected text; straddling line break
    P = L.plaoul_witnesses()
    fr = Counter(t['after'] for p in P for t in p['toks'] if t['st'] != 'd' and t['after'])
    ev = json.load(open(os.path.join(L.CK, 'slipcat.json')))['events']
    d = [math.log(fr[e['A']] + 1) - math.log(fr[e['B']] + 1) for e in ev if e['c'] in ('nearmiss', 'falsestart', 'sameend')]
    d_all = [math.log(fr[e['A']] + 1) - math.log(fr[e['B']] + 1) for e in ev]
    lb = np.mean([e['lb'] for e in ev])
    print(f"Plaoul real struck words: near-type n {len(d)} rarer-first {np.mean(np.array(d) < 0):.2f} dlogf {np.mean(d):+.2f}; "
          f"all n {len(d_all)} rarer-first {np.mean(np.array(d_all) < 0):.2f} dlogf {np.mean(d_all):+.2f}; line break after A {lb:.3f} (all words 0.143)")
    res['plaoul'] = dict(near_rarer=float(np.mean(np.array(d) < 0)), near_dlogf=float(np.mean(d)), all_rarer=float(np.mean(np.array(d_all) < 0)), lb=float(lb))
    # corrected text
    for nm, T in (('ZL', zl), ('IT', it)):
        C, ex = corrected(T)
        F = set(s['folio'] for s in T)
        before = [zset(agg(counts(T, F, 1), F), [o]) for o in SSTAR]
        after = [zset(agg(counts(C, F, 1), F), [o]) for o in SSTAR]
        print(f'{nm} corrected text: {sum(ex.values())} first twins deleted ({100*sum(ex.values())/sum(len(s["words"]) for s in T):.2f}% of words); '
              f'S* orientation z before {np.round(before,2).tolist()} after {np.round(after,2).tolist()}')
        print('   most common deleted/kept pairs:', ex.most_common(25))
        res[f'corr_{nm}'] = dict(n=sum(ex.values()), ex=[(a, b, c) for (a, b), c in ex.most_common(60)])
    json.dump(res, open(os.path.join(L.CK, 'c3_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
