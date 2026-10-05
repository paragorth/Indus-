#!/usr/bin/env python3
"""LA-48 cycle 2: read the network. Does the best flow labelling recover known directions and hubs?

Objective: wG * (pure source/sink nodes) + K2 + 3 * K3 (combined), and separately graph-only (wG=1, K=0)
and Kirchhoff-only (wG=0). The labelling is symmetric under d -> -d (except STOCK terms), so direction
is scored up to a global sign.
Truth (scoring only):
  planted: document IN / OUT / STOCK; words HUB / PASS / LEAF
  Ur III Puzrish-Dagan: document OUT (ba-zi, zi-ga) vs IN (mu-kux) vs TRANSFER (i3-dab5); tokens SOURCE
       (ki X-ta), RECEIVER (X i3-dab5), DELIVERER (mu-kux X), DATE
  Linear B: series IN (Ma, Mc, Na, ...), OUT (Fr, Fn, Un, Fp, Es, ...), STOCK (D-, Cn, Sc, ...)
Metrics: direction accuracy on truth IN/OUT documents (max over global sign) with a label-permutation P;
STOCK recall; node-role separation AUC (net flow sign of SOURCE vs RECEIVER, of HUB vs LEAF);
hub precision (top-5 throughput nodes that are HUB / non-DATE). Each run is repeated on quantity-shuffled
data (QSHUF) so the quantity (physics) contribution is separated from the graph contribution.
"""
import sys, os, json, time, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la48_common import *

STEPS = 150000
RESTARTS = 4
OBJ = {'COMB': (1.0, 1.0, 3.0), 'GRAPH': (1.0, 0.0, 0.0), 'KIRCH': (0.0, 1.0, 3.0)}


def node_net(B, d):
    nn = len(B['names'])
    net_sign = np.zeros(nn); thr = np.zeros(nn); nflow = np.zeros(nn, int)
    for n in range(nn):
        for j in range(B['nptr'][n], B['nptr'][n + 1]):
            dd = d[B['o_doc'][j]]
            q = B['occ_q'][j]
            mag = sum(abs(v) for v in q.values())
            thr[n] += mag
            if dd != 0:
                net_sign[n] += dd * B['o_slot'][j] * mag
                nflow[n] += 1
    return net_sign, thr, nflow


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def dir_metrics(docs, d, rs, nperm=2000):
    lab = [(i, doc['group']) for i, doc in enumerate(docs)]
    io = [(i, 1 if g == 'IN' else -1) for i, g in lab if g in ('IN', 'OUT') and d[i] != 0]
    out = {}
    if len(io) >= 6:
        def acc(pairs):
            a = np.mean([1.0 if d[i] == t else 0.0 for i, t in pairs])
            return max(a, 1 - a)
        a0 = acc(io)
        ts = [t for _, t in io]
        ge = 0
        for _ in range(nperm):
            rs.shuffle(ts)
            if acc([(i, t) for (i, _), t in zip(io, ts)]) >= a0 - 1e-12:
                ge += 1
        out['dir_acc'] = round(float(a0), 3); out['dir_P'] = (ge + 1) / (nperm + 1); out['dir_n'] = len(io)
    st = [i for i, g in lab if g == 'STOCK']
    if st:
        out['stock_recall'] = round(float(np.mean([d[i] == 0 for i in st])), 3)
        out['zero_rate'] = round(float(np.mean(d == 0)), 3)
    return out


def role_metrics(name, B, d, docs):
    net, thr, nflow = node_net(B, d)
    names = B['names']
    out = {}
    if name.startswith('PL'):  # PL and PLAGG
        hub = [net[n] for n in range(len(names)) if nflow[n] >= 2 and ROLE_TRUE(names[n][1]) == 'HUB']
        leaf = [net[n] for n in range(len(names)) if nflow[n] >= 2 and ROLE_TRUE(names[n][1]) == 'LEAF']
        a = auc(hub, leaf); out['auc_hub_leaf'] = round(max(a, 1 - a), 3) if a == a else None
        top = np.argsort(-thr)[:5]
        out['top5_hub'] = sum(ROLE_TRUE(names[n][1]) == 'HUB' for n in top)
        comp = components(B, d)
        bal = [n for n in range(len(names)) if comp[n, 1] or comp[n, 2]]
        out['balanced_pass_frac'] = round(float(np.mean([ROLE_TRUE(names[n][1]) == 'PASS' for n in bal])), 3) if bal else None
        out['n_bal'] = len(bal)
    if name.startswith('UR'):
        T = ur3_node_truth(docs)
        src = [net[n] for n in range(len(names)) if nflow[n] >= 2 and T.get(names[n][1]) == 'SOURCE']
        rcv = [net[n] for n in range(len(names)) if nflow[n] >= 2 and T.get(names[n][1]) == 'RECEIVER']
        a = auc(src, rcv); out['auc_src_rcv'] = round(max(a, 1 - a), 3) if a == a else None
        out['n_src'] = len(src); out['n_rcv'] = len(rcv)
        top = np.argsort(-thr)[:10]
        out['top10'] = [names[n][1] + '/' + names[n][2] + ':' + T.get(names[n][1], '-') for n in top]
        out['top10_nondate'] = sum(T.get(names[n][1]) != 'DATE' for n in top)
    if name.startswith('LA') or name.startswith('LB'):
        top = np.argsort(-thr)[:12]
        out['top12'] = ['%s/%s/%s:%+.0f' % (names[n][0][:2], names[n][1], names[n][2], net[n]) for n in top]
    return out


def run_one(args):
    name, docs, obj, kind, rep = args
    rs = random.Random(seed('c2-%s-%s-%s-%d' % (name, obj, kind, rep)))
    if kind == 'QSHUF':
        docs = shuffle_quantities(docs, rs)
    B = build(docs)
    wG, w2, w3 = OBJ[obj]
    best, bd = search(B, wG, w2, w3, restarts=RESTARTS, steps=STEPS, sd=seed(name + obj + kind) % 10000 + rep)
    out = {'name': name, 'obj': obj, 'kind': kind, 'rep': rep, 'score': best}
    out.update(dir_metrics(docs, bd, rs))
    out.update(role_metrics(name, B, bd, docs))
    if name == 'LA' and kind == 'REAL':
        out['d'] = bd.tolist(); out['ids'] = [x['id'] for x in docs]
    return out


if __name__ == '__main__':
    L = la_docs()
    qpool = [e[2] for d in L for e in d['entries']]
    LB = lb_docs()
    DS = [('LA', L), ('LB_KN', [d for d in LB if d['site'] == 'KN']), ('LB_PY', [d for d in LB if d['site'] == 'PY'])]
    for r in range(2):
        DS.append(('UR_n393_%d' % r, ur3_docs(393, random.Random(seed('urdraw%d' % r)))))
    for sv in (1.0, 0.5, 0.2):
        DS.append(('PL_s%.1f' % sv, planted_docs(393, random.Random(seed('pl%.1f' % sv)), qpool, survival=sv)))
    for sv in (1.0, 0.3):
        DS.append(('PLAGG_s%.1f' % sv, planted_docs(393, random.Random(seed('plagg%.1f' % sv)), qpool, survival=sv, agg=3)))
    jobs = []
    for name, docs in DS:
        for obj in OBJ:
            jobs.append((name, docs, obj, 'REAL', 0))
            for r in range(3 if name in ('LA', 'PL_s1.0', 'PLAGG_s0.3') else 1):
                jobs.append((name, docs, obj, 'QSHUF', r))
    fn = os.path.join(CK, 'c2_results.jsonl')
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            x = json.loads(l); done.add((x['name'], x['obj'], x['kind'], x['rep']))
    jobs = [j for j in jobs if (j[0], j[2], j[3], j[4]) not in done]
    print('jobs', len(jobs), flush=True)
    t0 = time.time()
    with Pool(2) as p:
        for out in p.imap_unordered(run_one, jobs):
            with open(fn, 'a') as f:
                f.write(json.dumps(out) + '\n')
            print('%.0fs' % (time.time() - t0), json.dumps({k: v for k, v in out.items() if k not in ('d', 'ids')}),
                  flush=True)
