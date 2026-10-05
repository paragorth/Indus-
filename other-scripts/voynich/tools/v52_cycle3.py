"""v52 cycle 3: WHERE DOES THE INFORMATION LIVE?  A field code keeps each attribute in its own slot:
the slot alone predicts the attribute as well as the whole word does, and adding the other slots
(additively) changes little. A text whose words carry attributes as whole-word identity (vocabulary,
topic) or as conditioned phonotactics spreads it through the word, so the whole word beats the slot.

For the best climbed schema of each corpus (cycle 2, else cycle 1) and each target variable v:
naive-Bayes-style predictors trained on half A, scored on held-out half B, in bits/token gained over
the prior P(v):  SLOT (the slot that tracks v), NB (all slots additively), WORD (full word type,
backoff to SLOT when unseen). Ratios  r_slot = SLOT/WORD, r_nb = NB/WORD.
Also the SWAP test: on B, does the attribute slot keep its meaning across records? For words whose
free part (all non-target slots) also occurs in A with a DIFFERENT target-slot value, predict v from
the target slot through the A mapping (records re-filed under another value)."""
import sys, os, pickle, json
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v52_lib as L
from v52_cycle1 import build


def table(x, y, ny, alpha=0.5):
    nx = int(x.max()) + 1
    T = np.full((nx, ny), alpha)
    np.add.at(T, (x, y), 1)
    return np.log2(T / T.sum(1, keepdims=True))


def gain(C, sch, v, slot):
    codes, _ = L.slot_codes(C, sch)
    A = C['A']; B = ~A
    y = C['vars'][v]; ny = int(y.max()) + 1
    prior = np.bincount(y[A], minlength=ny) + 0.5; lp = np.log2(prior / prior.sum())
    base = -lp[y[B]].mean()
    # slot alone
    Ts = table(codes[slot][A], y[A], ny)
    xs = codes[slot][B]; ok = xs < Ts.shape[0]
    ls = np.where(ok, Ts[np.minimum(xs, Ts.shape[0] - 1), y[B]], lp[y[B]])
    g_slot = base + ls.mean()
    # NB additive over all slots, temperature calibrated on A
    def nbmat(M):
        acc = np.tile(lp, (M.sum(), 1))
        for c in codes:
            T = table(c[A], y[A], ny); xb = c[M]; ok = xb < T.shape[0]
            acc += np.where(ok[:, None], T[np.minimum(xb, T.shape[0] - 1)] - lp[None, :], 0)
        return acc
    accA = nbmat(A); best_tau, best_ll = 1.0, -1e9
    for tau in (1, 0.8, 0.6, 0.45, 0.33, 0.25, 0.15):
        a = accA * tau; a -= np.log2(np.exp2(a).sum(1, keepdims=True))
        ll = a[np.arange(A.sum()), y[A]].mean()
        if ll > best_ll:
            best_ll, best_tau = ll, tau
    acc = nbmat(B) * best_tau
    acc -= np.log2(np.exp2(acc).sum(1, keepdims=True))
    nb = acc[np.arange(B.sum()), y[B]]
    g_nb = base + nb.mean()
    Pslot = np.exp2(np.where(ok[:, None], Ts[np.minimum(xs, Ts.shape[0] - 1)], lp[None, :]))
    # whole word (unbounded type), backoff to SLOT when unseen in A
    tt = C['tok_type']
    Tw = defaultdict(lambda: np.zeros(ny))
    for t, yy in zip(tt[A], y[A]):
        Tw[t][yy] += 1
    lw = []
    for i, (t, yy) in enumerate(zip(tt[B], y[B])):
        if t in Tw and Tw[t].sum() >= 1:
            row = Tw[t] + 2.0 * Pslot[i]
            lw.append(np.log2(row[yy] / row.sum()))
        else:
            lw.append(ls[i])
    g_word = base + float(np.mean(lw))
    # swap test
    others = [k for k in range(len(codes)) if k != slot]
    free = np.zeros(C['n'], dtype=np.int64)
    for k in others:
        free = free * (int(codes[k].max()) + 1) + codes[k]
    fa = defaultdict(set)
    for f, s in zip(free[A], codes[slot][A]):
        fa[f].add(s)
    sw = np.array([(f in fa) and (s not in fa[f]) for f, s in zip(free[B], codes[slot][B])])
    g_swap = float(-lp[y[B]][sw].mean() + ls[sw].mean()) if sw.sum() > 30 else float('nan')
    return dict(base=base, slot=g_slot, nb=g_nb, word=g_word, r_slot=g_slot / max(1e-9, g_word),
                r_nb=g_nb / max(1e-9, g_word), swap=g_swap, nswap=int(sw.sum()), nB=int(B.sum()))


if __name__ == '__main__':
    out = {}
    S2 = pickle.load(open(os.path.join(L.CK, 'c2_summary.pkl'), 'rb')) if os.path.exists(os.path.join(L.CK, 'c2_summary.pkl')) else {}
    S1 = pickle.load(open(os.path.join(L.CK, 'c1_summary.pkl'), 'rb'))
    TV = dict(V=['SEC', 'LANG', 'POS', 'NBR', 'NXT'], VI=['SEC', 'LANG', 'POS', 'NBR', 'NXT'],
              VMARK=['SEC', 'LANG', 'POS', 'NBR', 'NXT'], PL6=['SEC', 'POS', 'NBR'], PL3=['SEC', 'POS', 'NBR'],
              GORILA=['SITE', 'SUPPORT', 'PERIOD'], UNICODE=['BLOCK', 'CASE', 'DECOMP'])
    for n, vs in TV.items():
        src = S2.get(n) or S1.get(n)
        if not src:
            continue
        best = max(src['top'], key=lambda t: t['A']['FS'])
        C = build(n); res = {}
        for v in vs:
            j = best['A']['varlist'].index(v)
            slot = int(np.argmax(np.array(best['A']['E'])[:, j]))
            g = gain(C, best['sch'], v, slot)
            g['slot_idx'] = slot
            res[v] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in g.items()}
            print(n, v, res[v], flush=True)
        out[n] = dict(sch=best['sch'], res=res)
    pickle.dump(out, open(os.path.join(L.CK, 'c3_summary.pkl'), 'wb'))
    print('all done')
