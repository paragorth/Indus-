"""la73 cycle 3: frozen predictions for the next 50 Hagia Triada tablets, after checking the
prediction machinery where the answer is known: (a) HT random splits (fit on 108, predict 50);
(b) Ur III windows (fit on 158, predict 50 more from the same window). Control: a prior-only
prediction and a frequency-only baseline (Good-Turing new-type rate) on the same splits."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la73_abc as A, la73_common as C, la73_prep as P
import numpy as np
C.set_version(2)

def predict(o, th, L, rng, extra=50, n=400, Lnew=None):
    idx = o['_idx']; w = o['_w'] / o['_w'].sum(); out = []
    for i in rng.choice(idx, n, p=w):
        if Lnew is not None:
            Lall = np.concatenate([L, Lnew]); d, _ = C.simulate(th[i], Lall, rng)
        else:
            d, _ = C.simulate(th[i], L, rng, extra=extra)
        out.append(measure(d[:len(L)], d[len(L):]))
    return np.array(out)

def measure(old, new):
    seen = {x for d in old for x in d}
    tok = [x for d in new for x in d]; types = set(tok)
    newt = [x for x in types if x not in seen]
    rep_tok = sum(1 for x in tok if x in seen)
    return [len(newt), len(types) - len(newt), rep_tok / max(1, len(tok)), len(tok)]

def good_turing(old, ntok_new):
    from collections import Counter
    c = Counter(x for d in old for x in d); n = sum(c.values()); f1 = sum(1 for v in c.values() if v == 1)
    return f1 / n  # probability that a new token is an unseen type

if __name__ == '__main__':
    rng = np.random.default_rng(7303); res = {'val': []}
    D = P.la_docs('Haghia Triada'); L = np.array([len(x['w']) for x in D])
    # (a) HT splits
    for sp in range(3):
        perm = rng.permutation(len(D)); tr = [D[i] for i in perm[50:]]; te = [D[i] for i in perm[:50]]
        Ltr = np.array([len(x['w']) for x in tr]); Lte = np.array([len(x['w']) for x in te])
        tth, ts, tdd = A.table('v2_split%d' % sp, 15000, Ltr, seed=30 + sp)
        ids = {}; old = [[ids.setdefault(w, len(ids)) for w in x['w']] for x in tr]
        new = [[ids.setdefault(w, len(ids)) for w in x['w']] for x in te]
        truth = measure(old, new)
        o = A.abc(C.stats(old), tth, ts, tdd)
        pr = predict(o, tth, Ltr, rng, Lnew=Lte, n=300)
        prior = predict(dict(_idx=np.arange(len(tth)), _w=np.ones(len(tth))), tth, Ltr, rng, Lnew=Lte, n=300)
        gt = good_turing(old, truth[3])
        row = dict(kind='HT split %d' % sp, truth=truth,
                   post=[np.percentile(pr[:, j], [10, 50, 90]).round(3).tolist() for j in range(3)],
                   prior=[np.percentile(prior[:, j], [10, 50, 90]).round(3).tolist() for j in range(3)],
                   gt_rep_rate=round(1 - gt, 3))
        res['val'].append(row); print(row, flush=True)
    # (b) Ur III
    uth, us, ud = A.table('v2_ur3', 40000, None)
    U = P.ur3()
    for site, win in (('Umma', 12), ('Puzriš-Dagan', 36), ('Umma', 480)):
        X = [x for x in U if x['site'] == site]; t = np.array([x['t'] for x in X])
        starts = [s for s in range(t.min(), t.max() - win + 2) if np.sum((t >= s) & (t < s + win)) >= 208]
        s0 = starts[rng.integers(len(starts))]; pool = np.flatnonzero((t >= s0) & (t < s0 + win))
        pick = rng.choice(pool, 208, replace=False)
        ids = {}; docs = [[ids.setdefault(w, len(ids)) for w in X[i]['w']] for i in pick]
        old, new = docs[:158], docs[158:]
        truth = measure(old, new); Lo = np.array([len(x) for x in old]); Ln = np.array([len(x) for x in new])
        o = A.abc(C.stats(old), uth, us, ud)
        pr = predict(o, uth, Lo, rng, Lnew=Ln, n=300)
        prior = predict(dict(_idx=np.arange(len(uth)), _w=np.ones(len(uth))), uth, Lo, rng, Lnew=Ln, n=300)
        row = dict(kind='Ur III %s %dmo' % (site, win), truth=truth,
                   post=[np.percentile(pr[:, j], [10, 50, 90]).round(3).tolist() for j in range(3)],
                   prior=[np.percentile(prior[:, j], [10, 50, 90]).round(3).tolist() for j in range(3)],
                   gt_rep_rate=round(1 - good_turing(old, 0), 3))
        res['val'].append(row); print(row, flush=True)
    # frozen prediction for the next 50 HT tablets (lengths resampled from HT)
    th, s, d = A.table('v2_ht', 60000, L, seed=21)
    so = C.stats(C.to_int_docs([x['w'] for x in D]))
    o = A.abc(so, th, s, d)
    pr = predict(o, th, L, rng, extra=50, n=1000)
    pred = dict(created='2026-10-07', loop='la73', corpus='corpus_ra.json rd, HT tablets with >=1 word of >=2 signs (158 tablets, 615 tokens)',
                target='the next 50 Hagia Triada tablets with words (e.g. RILA-S1 / new finds), word tokens of >=2 signs, read+damaged',
                new_types=np.percentile(pr[:, 0], [10, 50, 90]).round(1).tolist(),
                old_types_reappearing=np.percentile(pr[:, 1], [10, 50, 90]).round(1).tolist(),
                share_of_tokens_already_seen=np.percentile(pr[:, 2], [10, 50, 90]).round(3).tolist(),
                tokens=np.percentile(pr[:, 3], [10, 50, 90]).round(1).tolist(),
                kill='share_of_tokens_already_seen outside [q10, q90] on >= 50 new HT tablets',
                note='lengths of the new tablets resampled from HT; per-token share is the length-free test')
    txt = json.dumps(pred, indent=1, sort_keys=True)
    h = hashlib.sha256(txt.encode()).hexdigest()
    open(os.path.join(A.CK, '..', 'la73_predictions.json'), 'w').write(txt)
    res['prediction'] = pred; res['sha256'] = h
    print(txt, '\nsha256', h, flush=True)
    json.dump(res, open(os.path.join(A.CK, 'c3.json'), 'w'), indent=1, default=float)
