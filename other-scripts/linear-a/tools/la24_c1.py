#!/usr/bin/env python3
"""LA-24 cycle 1: does any Linear A list compress as an apportionment beyond the nulls?

Per list: best MDL gain (baseline bits - apportionment bits) over 143 share alphabets x 13
rule/grid combinations. Nulls: N3b (per list; each distinct value -> random corpus amount within
+-25 % in value, same letter status, repeats kept; R = 200), N1 (amounts shuffled across lists,
list sizes kept), N2 (amounts drawn from the corpus distribution). Corpus statistics: number of
lists with gain > 0, summed positive gain. Positive controls: Ur III ration lists (CDLI, sila3),
Linear B single-commodity line runs (DAMOS), and 40 planted apportionments embedded in the
Linear A corpus.
"""
import json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import *

R = int(os.environ.get('R', 200))
G = {}


def setup(name, lists, numval, grids):
    B = Baseline([l['amts'] for l in lists], numval)
    G[name] = dict(lists=lists, B=B, nv=numval, grids=grids, pool=band_pool(B, numval))


def one_list_nulls(args):
    name, i, R, seed = args
    g = G[name]; rng = random.Random(seed)
    toks = g['lists'][i]['amts']
    out = []
    for r in range(R):
        t2 = null_band(toks, g['pool'], g['nv'], rng)
        x = np.array([g['nv'](t) for t in t2]); bc = g['B'].costs(t2)
        mb, _ = model_best(x, bc, g['grids'])
        out.append(bc.sum() - mb)
    return i, out


def corpus_null(args):
    name, kind, seed = args
    g = G[name]; rng = random.Random(seed); nrng = np.random.default_rng(seed)
    L = [l['amts'] for l in g['lists']]
    L2 = null_shuffle(L, rng) if kind == 'N1' else null_marg(L, g['B'], nrng)
    res = score_lists(L2, g['B'], g['nv'], g['grids'])
    gains = np.array([r[0] for r in res])
    return kind, int((gains > 0).sum()), float(gains[gains > 0].sum())


def run_dataset(name, pool, Rn, corpus_nulls=True):
    g = G[name]
    t0 = time.time()
    real = score_lists(g['lists'], g['B'], g['nv'], g['grids'])
    gains = np.array([r[0] for r in real])
    jobs = [(name, i, Rn, 1000 * i + 7) for i in range(len(g['lists']))]
    nulls = np.zeros((len(g['lists']), Rn))
    for i, out in pool.imap_unordered(one_list_nulls, jobs, chunksize=4):
        nulls[i] = out
    p = ((nulls >= gains[:, None]).sum(1) + 1) / (Rn + 1)
    q = bh(p)
    # corpus level from N3b replicates
    n3_cnt = (nulls > 0).sum(0); n3_sum = np.where(nulls > 0, nulls, 0).sum(0)
    obs_cnt = int((gains > 0).sum()); obs_sum = float(gains[gains > 0].sum())
    summ = {'name': name, 'n_lists': len(gains), 'obs_cnt': obs_cnt, 'obs_sum': obs_sum,
            'N3b_cnt_mean': float(n3_cnt.mean()), 'N3b_cnt_P': float(((n3_cnt >= obs_cnt).sum() + 1) / (Rn + 1)),
            'N3b_sum_mean': float(n3_sum.mean()), 'N3b_sum_P': float(((n3_sum >= obs_sum).sum() + 1) / (Rn + 1)),
            'n_p05': int((p < 0.05).sum()), 'n_q10': int((q < 0.1).sum())}
    if corpus_nulls:
        cj = [(name, k, s) for k in ('N1', 'N2') for s in range(Rn // 2)]
        acc = {'N1': [], 'N2': []}
        for k, c, s in pool.imap_unordered(corpus_null, cj, chunksize=4):
            acc[k].append((c, s))
        for k in acc:
            a = np.array(acc[k])
            summ[k + '_cnt_mean'] = float(a[:, 0].mean()); summ[k + '_cnt_P'] = float(((a[:, 0] >= obs_cnt).sum() + 1) / (len(a) + 1))
            summ[k + '_sum_mean'] = float(a[:, 1].mean()); summ[k + '_sum_P'] = float(((a[:, 1] >= obs_sum).sum() + 1) / (len(a) + 1))
    per = []
    for i, (gn, info, bc) in enumerate(real):
        l = g['lists'][i]
        per.append({'id': l['id'], 'key': str(l['key']), 'vals': [g['nv'](a) for a in l['amts']],
                    'amts': [[float(a[0]), list(a[1])] for a in l['amts']],
                    'total': None if l.get('total') is None else g['nv'](l['total']),
                    'gain': float(gn), 'base_bits': bc, 'p': float(p[i]), 'q': float(q[i]),
                    'null_mean': float(nulls[i].mean()), 'info': info})
    summ['secs'] = time.time() - t0
    return summ, per


def main():
    out = {}
    la = la_lists()
    setup('LA', la, la_numval(CONV), (1.0, 0.5, 0.25))
    # planted: 40 LA lists replaced by apportionments of the same length (n capped at 10)
    rng = random.Random(24)
    idx = rng.sample(range(len(la)), 40)
    pl = [dict(l) for l in la]; truth = {}
    for i in idx:
        n = min(max(len(la[i]['amts']), 3), 10)
        pp = None
        while pp is None: pp = plant_list(rng, n)
        pl[i] = {'id': 'PLANT%d' % i, 'site': 'PL', 'key': 'PLANT', 'amts': pp['amts'], 'total': None}
        truth[i] = pp
    setup('PLANT', pl, la_numval(CONV), (1.0, 0.5, 0.25))
    os.environ.get('UR_ATF')
    ur = ur_lists(max_lists=150, seed=1)
    if ur: setup('UR', ur, lambda a: float(a[0]), (1.0, 5.0, 10.0))
    lb = lb_lists()
    setup('LB', lb, lambda a: float(a[0]), (1.0, 1 / 60, 1 / 72))
    with Pool(2) as pool:
        for name, Rn, cn in (('LA', R, True), ('PLANT', 100, False), ('UR', 60, False), ('LB', 100, False)):
            if name not in G: continue
            summ, per = run_dataset(name, pool, Rn, cn)
            if name == 'PLANT':
                rec = []
                for i, tr in truth.items():
                    x = per[i]
                    xs = np.array(x['vals'])
                    r = RULES.index(tr['rule'])
                    consistent = any((fit_all(xs, r, gg)[0] == len(xs)).any() for gg in (1.0,)) if r else True
                    rec.append({'i': i, 'true_rule': tr['rule'], 'true_S': tr['S'], 'w': tr['w'], 'found': x['info'],
                                'p': x['p'], 'q': x['q'], 'gain': x['gain'], 'rule_ok': x['info'] is not None and x['info']['rule'] == tr['rule'],
                                'true_rule_reproduces': bool(consistent)})
                pq = bh([r_['p'] for r_ in rec])
                summ['planted_n'] = len(rec)
                summ['planted_p05'] = int(sum(r_['p'] < 0.05 for r_ in rec))
                summ['planted_q10_within'] = int((pq < 0.1).sum())
                summ['planted_gain_pos'] = int(sum(r_['gain'] > 0 for r_ in rec))
                summ['planted_rule_ok'] = int(sum(r_['rule_ok'] for r_ in rec))
                real_idx = [i for i in range(len(per)) if i not in truth]
                summ['unplanted_p05'] = int(sum(per[i]['p'] < 0.05 for i in real_idx))
                out['PLANT_rec'] = rec
            out[name] = {'summary': summ, 'lists': per}
            print(json.dumps(summ), flush=True)
            json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'))


if __name__ == '__main__':
    main()
