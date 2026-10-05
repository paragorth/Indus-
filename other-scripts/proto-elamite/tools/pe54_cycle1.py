"""pe54 cycle 1: calibrate flock linking on Ur III and planted archives, first PE run.
usage: pe54_cycle1.py ur3|plant|pe  (each writes data/pe54_ckpt/c1_<part>.json)"""
import sys, time
import numpy as np
from pe54_lib import *

import pe54_lib as L
part = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else 'zc'
L.ZERO_MISSING = MODE == 'zm'
rng = np.random.default_rng(54)
MID = dict(b=0.85, sy=0.65, sa=0.85, of=0.07, om=0.35, qf=0.5, k=6.0)
res = {}


def evaluate(C, r, th, strings, years):
    s, M, S = hyp_score(C, r, th)
    f = sym_forbid(C)
    out = dict(score=s, links=link_eval(M, strings, years), auc=pair_auc(S, strings, f),
               retr=retrieval(S, strings, f))
    if years is not None:
        out['order'] = order_eval(S, strings, years, f)
    return out, M, S


def blind(C, strings, years, n_hyp, climb=5, truth=None):
    t = time.time()
    top, alls = search(C, n_hyp, rng, keep=50, climb=climb)
    s, r, th = top[0]
    ev, M, S = evaluate(C, r, th, strings, years)
    ev.update(roles=r, th=th, n_hyp=n_hyp, secs=time.time() - t, all_q=np.percentile(alls, [50, 90, 99, 100]))
    if truth is not None:
        ev['role_match'] = int(sum((r[c] > 0) == (truth[c] > 0) and ((r[c] - 1) % 3 == (truth[c] - 1) % 3 if r[c] > 0 else True)
                                   for c in range(len(r))))
    # ensemble AUC over top 20 (mean S)
    Sm = np.mean([pair_scores(C, rr, tt) for _, rr, tt in top[:20]], 0)
    f = sym_forbid(C)
    ev['ens_auc'] = pair_auc(Sm, strings, f)
    ev['ens_retr'] = retrieval(Sm, strings, f)
    if years is not None:
        ev['ens_order'] = order_eval(Sm, strings, years, f)
    return ev, top, M


if part == 'ur3':
    d, X, tabs, st, yr = ur3_corpus()
    C = Corpus(X, tabs, st, yr)
    f = sym_forbid(C)
    truth = np.array([1, 2, 3, 3, 4, 5, 6, 6])
    res['n'] = C.n
    res['size_only'] = dict(auc=pair_auc(size_scores(C), st, f), retr=retrieval(size_scores(C), st, f))
    res['truth'] = {}
    for k in KAPPAS:
        th = dict(MID); th['k'] = k
        res['truth'][str(k)] = evaluate(C, truth, th, st, yr)[0]
    ev, top, M = blind(C, st, yr, 3000, truth=truth)
    res['blind'] = ev
    res['blind_links'] = [(st[i], yr[i], st[j], yr[j], s) for i, j, s in M if st[i] == st[j]]
    # null: same blind search on within-class shuffled compositions (strings kept)
    res['null_shuffle'] = []
    for rep in range(5):
        C0 = Corpus(shuffle_within_class(X, rng), tabs, st, yr)
        e0, _, _ = blind(C0, st, yr, 600, climb=2)
        res['null_shuffle'].append(dict(score=e0['score'], auc=e0['auc'], ens_auc=e0['ens_auc'], retr=e0['retr']))
    res['blind_600'] = blind(C, st, yr, 600, climb=2)[0]
elif part == 'plant':
    res['plants'] = []
    for surv, nf, yrs in ((1.0, 30, 6), (0.3, 40, 8), (0.1, 100, 8), (0.04, 150, 8)):
        for rep in range(3):
            X, tabs, own, years, truth, th0 = plant_archive(rng, n_flocks=nf, years=yrs, survive=surv, het=1.0)
            years = [int(y) for y in years]
            C = Corpus(X, tabs, own, years)
            f = sym_forbid(C)
            row = dict(survive=surv, n=C.n, n_true_pairs=int(sum(own[i] == own[j] for i in range(C.n) for j in range(i + 1, C.n))))
            row['size_only'] = dict(auc=pair_auc(size_scores(C), own, f), retr=retrieval(size_scores(C), own, f))
            row['truth'] = evaluate(C, truth, th0, own, years)[0]
            ev, top, M = blind(C, own, years, 800 if C.n > 100 else 2000, climb=3, truth=truth)
            row['blind'] = ev
            res['plants'].append(row)
            print(surv, rep, C.n, 'TRUTH', row['truth']['links']['prec'], row['truth']['links']['chance'], row['truth']['auc'], row['truth']['retr'], row['truth']['order'], 'BLIND', ev['links']['prec'], ev['auc'], ev['retr'], ev['order'], ev['role_match'], 'SIZE', row['size_only']['auc'], flush=True)
elif part == 'pe':
    for copy in (True, False):
        R, X, tabs, st = pe_corpus(drop_copy=not copy)
        C = Corpus(X, tabs, st)
        f = sym_forbid(C)
        key = 'with_copy' if copy else 'main'
        ev, top, M = blind(C, st, None, 20000, climb=8)
        lab = ['%s %s' % (r[0], r[1]) for r in R]
        ev['links_list'] = [(lab[i], st[i], lab[j], st[j], round(s, 2)) for i, j, s in sorted(M, key=lambda x: -x[2])]
        ev['top10'] = [(s, r, th) for s, r, th in top[:10]]
        res[key] = ev
        print(key, ev['score'], ev['roles'], ev['links']['same'], ev['links']['n_links'], ev['auc'], flush=True)
    # nulls at equal budget (2,000 hyps + climb 3) for main
    R, X, tabs, st = pe_corpus()
    C = Corpus(X, tabs, st)
    res['main_2000'] = blind(C, st, None, 2000, climb=3)[0]['score']
    res['null_shuffle'], res['null_synth'] = [], []
    for rep in range(12):
        C0 = Corpus(shuffle_within_class(X, rng), tabs, st)
        res['null_shuffle'].append(blind(C0, st, None, 2000, climb=3)[0]['score'])
        C1 = Corpus(synthetic_unrelated(X, rng), tabs, st)
        res['null_synth'].append(blind(C1, st, None, 2000, climb=3)[0]['score'])
        print(rep, res['null_shuffle'][-1], res['null_synth'][-1], flush=True)
dump(res, os.path.join(CK, 'c1_%s_%s.json' % (part, MODE)))
print('done', part)
