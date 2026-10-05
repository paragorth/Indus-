#!/usr/bin/env python3
"""LA-48 cycle 3: does conservation PREDICT held-out quantities?

5-fold over documents. The flow labelling is fitted (annealing, combined objective) on the training
documents only. For every quantity on a held-out document whose node (site, word, commodity) has >= 2
training occurrences, the physics prediction is the training residual of that node (the amount that
would make it balance, either direction): hit if the held-out amount equals it exactly.
Baselines: COPY (the held-out amount equals any training amount of the same node, up to k candidates);
RANDLAB (physics prediction under 100 random labellings of the training documents = label-shuffle null);
QSHUF (the whole pipeline on quantity-shuffled corpora). 'Beyond copy' = physics hits that no copy
predicts (the residual is a genuine sum or difference).
"""
import sys, os, json, time, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la48_common import *

STEPS = 120000
RESTARTS = 3
NF = 5


def residuals(B, d):
    return residuals_nb(d, len(B['names']), B['nptr'], B['o_doc'], B['o_slot'], B['o_val'])


def evaluate(docs, rs, sd, nrand=100):
    idx = list(range(len(docs))); rs.shuffle(idx)
    folds = [idx[f::NF] for f in range(NF)]
    tot = {'elig': 0, 'phys': 0, 'copy': 0, 'beyond': 0, 'rand_phys': 0.0, 'rand_beyond': 0.0}
    examples = []
    for f in range(NF):
        test = set(folds[f])
        tr = [docs[i] for i in range(len(docs)) if i not in test]
        B = build(tr)
        key = {nm: n for n, nm in enumerate(B['names'])}
        best, d = search(B, 1.0, 1.0, 3.0, restarts=RESTARTS, steps=STEPS, sd=sd + f)
        r, k = residuals(B, d)
        tv = collections.defaultdict(set)
        for n in range(len(B['names'])):
            for j in range(B['nptr'][n], B['nptr'][n + 1]):
                tv[n].add(int(B['o_val'][j]))
        rnd = []
        nr = np.random.default_rng(sd + 99 * f)
        for _ in range(nrand):
            dr = nr.integers(-1, 2, B['nd']).astype(np.int64)
            rnd.append(residuals(B, dr)[0])
        for i in test:
            T = build([docs[i]])
            for j in range(len(T['o_node'])):
                nm = T['names'][T['o_node'][j]]
                if nm not in key:
                    continue
                n = key[nm]
                if k[n] < 2:
                    continue
                h = int(T['o_val'][j])
                tot['elig'] += 1
                ph = (h == int(r[n]) or h == -int(r[n]))
                cp = h in tv[n]
                tot['phys'] += ph; tot['copy'] += cp; tot['beyond'] += (ph and not cp)
                if ph and not cp and len(examples) < 30:
                    examples.append([docs[i]['id'], list(map(str, nm)), T['occ_q'][j]])
                rp = 0; rb = 0
                for rr in rnd:
                    x = (h == int(rr[n]) or h == -int(rr[n]))
                    rp += x; rb += (x and not cp)
                tot['rand_phys'] += rp / nrand; tot['rand_beyond'] += rb / nrand
    tot['examples'] = examples
    return tot


def run_one(args):
    name, docs, kind, rep = args
    rs = random.Random(seed('c3-%s-%s-%d' % (name, kind, rep)))
    if kind == 'QSHUF':
        docs = shuffle_quantities(docs, rs)
    t = evaluate(docs, random.Random(seed('fold' + name)), seed(name + kind) % 10000 + 31 * rep)
    t.update({'name': name, 'kind': kind, 'rep': rep})
    return t


if __name__ == '__main__':
    L = la_docs()
    qpool = [e[2] for d in L for e in d['entries']]
    LB = lb_docs()
    DS = [('LA', L), ('LB_KN', [d for d in LB if d['site'] == 'KN']), ('LB_PY', [d for d in LB if d['site'] == 'PY']),
          ('UR_n393_0', ur3_docs(393, random.Random(seed('urdraw0')))),
          ('UR_n2000', ur3_docs(2000, random.Random(seed('urdraw2000'))))]
    for sv in (1.0, 0.5, 0.2):
        DS.append(('PL_s%.1f' % sv, planted_docs(393, random.Random(seed('pl%.1f' % sv)), qpool, survival=sv)))
    for sv in (1.0, 0.3):
        DS.append(('PLAGG_s%.1f' % sv, planted_docs(393, random.Random(seed('plagg%.1f' % sv)), qpool, survival=sv, agg=3)))
    jobs = []
    for name, docs in DS:
        jobs.append((name, docs, 'REAL', 0))
        for r in range(10 if name == 'LA' else (2 if name.startswith('UR') else 4)):
            jobs.append((name, docs, 'QSHUF', r))
    fn = os.path.join(CK, 'c3_results.jsonl')
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            x = json.loads(l); done.add((x['name'], x['kind'], x['rep']))
    jobs = [j for j in jobs if (j[0], j[2], j[3]) not in done]
    print('jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(2) as p:
        for out in p.imap_unordered(run_one, jobs):
            with open(fn, 'a') as f:
                f.write(json.dumps(out) + '\n')
            print('%.0fs' % (time.time() - t0), json.dumps({k: v for k, v in out.items() if k != 'examples'}), flush=True)
