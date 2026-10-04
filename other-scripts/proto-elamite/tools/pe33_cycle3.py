"""pe33 cycle 3: massive random guessing with held-out seals.
A hypothesis = (picture feature m, set A of 1-3 content features): 'tablets sealed with m write A'.
Draw H random hypotheses, score each on the tablets of a random half of the SEALS (log-odds that any of A
is present given m, 0.5-corrected), keep the top 1%, and re-score the survivors on the other half of the
seals. Repeat over splits. Compare with the same pipeline after motif vectors are shuffled among seals
(within volume), and with a planted 2-sign bundle on 50% of BOVID tablets."""
import json, sys, collections
import numpy as np
from pe33_common import load, matrices, perm_seal, CK

H = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
NSPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NSHUF = int(sys.argv[3]) if len(sys.argv) > 3 else 20


def lor_any(X, y, A, idx):
    a = X[np.ix_(idx, A)].max(1)
    yy = y[idx]
    n11 = (a & yy).sum(); n10 = (yy & ~a).sum(); n01 = (a & ~yy).sum(); n00 = (~a & ~yy).sum()
    l = np.log((n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5)))
    se = np.sqrt(1 / (n11 + .5) + 1 / (n10 + .5) + 1 / (n01 + .5) + 1 / (n00 + .5))
    return l / se


def pipeline(X, Y, seal, hyps, rng):
    useals = np.unique(seal)
    out = []
    for s in range(NSPLIT):
        tr_s = set(rng.choice(useals, len(useals) // 2, replace=False))
        tr = np.array([x in tr_s for x in seal]); te = ~tr
        tri, tei = np.where(tr)[0], np.where(te)[0]
        sc = np.array([lor_any(X, Y[:, m].astype(bool), A, tri) for m, A in hyps])
        keep = np.argsort(-sc)[:max(1, len(hyps) // 100)]
        ho = np.array([lor_any(X, Y[:, hyps[k][0]].astype(bool), hyps[k][1], tei) for k in keep])
        out.append((float(np.mean(ho)), float(np.mean(ho > 2)), [(int(hyps[k][0]), [int(a) for a in hyps[k][1]], round(float(sc[k]), 2), round(float(h), 2)) for k, h in zip(keep[:5], ho[:5])]))
    return out


if __name__ == '__main__':
    rows, _ = load(True)
    X, F, Y, mot = matrices(rows, min_n=3)
    X = X.astype(bool)
    seal = np.array([r['seal'] for r in rows])
    rng = np.random.default_rng(3)
    hyps = [(int(rng.integers(Y.shape[1])), list(rng.choice(X.shape[1], int(rng.integers(1, 4)), replace=False))) for _ in range(H)]
    print(X.shape, mot, flush=True)
    real = pipeline(X, Y, seal, hyps, np.random.default_rng(10))
    rm = np.mean([r[0] for r in real]); rf = np.mean([r[1] for r in real])
    print('REAL held-out mean z', rm, 'share z>2', rf, flush=True)
    shuf = []
    for k in range(NSHUF):
        r2 = np.random.default_rng(200 + k)
        Ys = perm_seal(Y, rows, r2)
        o = pipeline(X, Ys, seal, hyps, np.random.default_rng(10))
        shuf.append((np.mean([r[0] for r in o]), np.mean([r[1] for r in o])))
        print('SHUF', k, shuf[-1], flush=True)
    shuf = np.array(shuf)
    # planted
    bi = mot.index('BOVID')
    pl = []
    for k in range(5):
        r2 = np.random.default_rng(700 + k)
        Xp = X.copy()
        cand = [j for j in range(X.shape[1]) if F[j].startswith('S:') and 3 <= X[:, j].sum() <= 10]
        J = r2.choice(cand, 2, replace=False)
        for i in np.where(Y[:, bi])[0]:
            if r2.random() < 0.5:
                Xp[i, r2.choice(J)] = True
        hp = hyps + [(bi, [int(J[0])]), (bi, [int(J[1])]), (bi, [int(J[0]), int(J[1])])]
        o = pipeline(Xp, Y, seal, hp, np.random.default_rng(10))
        hits = np.mean([any(t[0] == bi and set(t[1]) & set(map(int, J)) for t in r[2]) for r in o])
        pl.append(dict(signs=[F[j] for j in J], heldout_mean=float(np.mean([r[0] for r in o])), share_z2=float(np.mean([r[1] for r in o])), top5_hit=float(hits)))
        print('PLANT', pl[-1], flush=True)
    # tally which hypotheses recur among real survivors' top lists
    rec = collections.Counter()
    for r in real:
        for m, A, s, h in r[2]:
            rec[(mot[m], tuple(F[a] for a in A))] += 1
    out = dict(H=H, nsplit=NSPLIT, real_mean=float(rm), real_share=float(rf), shuf_mean=shuf[:, 0].tolist(), shuf_share=shuf[:, 1].tolist(),
               p_mean=float((shuf[:, 0] >= rm).mean()), p_share=float((shuf[:, 1] >= rf).mean()), planted=pl,
               recurrent=[(k[0], list(k[1]), v) for k, v in rec.most_common(15)])
    print(json.dumps({k: v for k, v in out.items() if k not in ('shuf_mean', 'shuf_share')}, indent=0))
    json.dump(out, open(f'{CK}/cycle3.json', 'w'), indent=1)
