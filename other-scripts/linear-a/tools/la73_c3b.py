"""la73 cycle 3b: model-free baseline for the next-50-tablets prediction (Good-Turing), calibrated
on 1,000 random HT 108/50 splits (and Ur III windows); the frozen file carries both the calibrated
baseline and the ABC prediction that failed its own validation (kept as a test of the simulator)."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, la73_prep as P
from collections import Counter
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la73_ckpt')

def gt(old):
    c = Counter(w for d in old for w in d); n = sum(c.values()); return 1 - sum(1 for v in c.values() if v == 1) / n

def split_stats(D, ntest, reps, rng):
    E = []
    for _ in range(reps):
        p = rng.permutation(len(D)); te = [D[i] for i in p[:ntest]]; tr = [D[i] for i in p[ntest:]]
        seen = {w for d in tr for w in d}; tok = [w for d in te for w in d]
        truth = np.mean([w in seen for w in tok]); E.append((truth, gt(tr), len({w for w in tok if w not in seen}), len({w for w in tok if w in seen})))
    return np.array(E)

if __name__ == '__main__':
    rng = np.random.default_rng(7304); res = {}
    D = [x['w'] for x in P.la_docs('Haghia Triada')]
    E = split_stats(D, 50, 1000, rng)
    err = E[:, 0] - E[:, 1]
    res['ht_splits'] = dict(truth_mean=round(E[:, 0].mean(), 3), gt_mean=round(E[:, 1].mean(), 3), bias=round(err.mean(), 3),
                            err_q10_q90=np.quantile(err, [0.1, 0.9]).round(3).tolist(),
                            cover_raw_gt_pm_0p1=round(float(np.mean(np.abs(err) <= 0.1)), 3))
    # ABC validation from c3 for comparison
    c3 = json.load(open(os.path.join(CK, 'c3.json')))
    res['abc_val'] = [dict(kind=v['kind'], truth=round(v['truth'][2], 3), abc=v['post'][2], gt=v['gt_rep_rate'],
                           abc_in80=v['post'][2][0] <= v['truth'][2] <= v['post'][2][2]) for v in c3['val']]
    # Ur III: same split test for the baseline
    import la73_c1 as c1
    ue = []
    for g in c1.ur3_draws(np.random.default_rng(75)):
        Eu = split_stats(g['docs'], 50, 20, rng); ue.append((Eu[:, 0] - Eu[:, 1]).mean())
    res['ur3_gt_bias'] = dict(mean=round(float(np.mean(ue)), 3), q10_q90=np.quantile(ue, [0.1, 0.9]).round(3).tolist())
    # frozen: full HT (158) -> next 50; GT at 158 plus split-calibrated error
    g158 = gt(D)
    share = (g158 + np.quantile(err, [0.1, 0.5, 0.9])).round(3).tolist()
    pred = dict(created='2026-10-07', loop='la73',
                corpus='corpus_ra.json rd; Hagia Triada tablets with >= 1 word of >= 2 signs (158 tablets, 615 tokens; bridged and erased words out)',
                target='the next >= 50 Hagia Triada tablets with words (RILA-S1 or new finds), word tokens of >= 2 signs, read + damaged',
                calibrated_baseline=dict(method='Good-Turing seen-token share at n = 158 (%.3f) + error distribution of 1,000 random HT 108/50 splits' % g158,
                                         share_of_tokens_already_seen_q10_q50_q90=share,
                                         kill='observed share outside [q10, q90] on >= 50 new HT tablets',
                                         assumption='new tablets come from the same archive population; a new deposit or genre (la15 habitat novelty) is expected to push the share down'),
                abc_simulator=dict(share_of_tokens_already_seen_q10_q50_q90=c3['prediction']['share_of_tokens_already_seen'],
                                   new_types_q10_q50_q90=c3['prediction']['new_types'],
                                   old_types_reappearing_q10_q50_q90=c3['prediction']['old_types_reappearing'],
                                   status='FAILED its own validation (truth inside the 80 % interval in 0 of 3 HT splits (3 of 3 Ur III); misses in both directions); kept only as a test of the writers-in-time simulator: expected to be killed'))
    txt = json.dumps(pred, indent=1, sort_keys=True); h = hashlib.sha256(txt.encode()).hexdigest()
    open(os.path.join(CK, '..', 'la73_predictions.json'), 'w').write(txt)
    res['prediction'] = pred; res['sha256'] = h
    print(json.dumps(res, indent=1, default=float)); print('sha256', h)
    json.dump(res, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1, default=float)
