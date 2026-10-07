#!/usr/bin/env python3
"""la83 cycle 3: does any established result rest on the records the audit calls contested?
 (a) contested-free corpus: remove every token of a record with a SOURCE or EDITORIAL finding (audit T3, T4, T5, T7)
     and every token SigLA reads differently or not at all (T6 flags variant / nodraw); run the battery (full nulls).
     Control: 200 corpora with the same number of tokens of each kind removed at random (seeded), battery fast.
 (b) jackknife: drop each record with words or numbers once; record the change in B1 exact closures, B4 -ME z,
     B2a reversals and B5 agreement; list the records whose removal alone moves a result across its pass line,
     and how many of them the audit flags (vs the flagged share of all records).
usage: la83_c3.py -> data/la83_ckpt/c3.json"""
import json, os, sys, random, copy
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la83_battery as B
import la79_claims as K
import la71_parse as P

DATA = os.path.join(HERE, '..', 'data'); CK = os.path.join(DATA, 'la83_ckpt')
C = json.load(open(os.path.join(DATA, 'corpus_ra_v2.json')))
A = json.load(open(os.path.join(DATA, 'la83_audit.json')))
flag_docs = {r['doc'] for r in A['rows'] if r['test'] in ('T3', 'T4', 'T5', 'T7')}
PH = {'t': 'unk', 'v': '#R', 'st': 'illegible', 'fl': ['la83cut']}


def cut(C, pred):
    out = []; n = Counter()
    for d in C:
        e = dict(d); toks = []
        for t in d['tokens']:
            if t['t'] not in ('nl', 'div') and pred(d, t):
                toks.append(dict(PH)); n[t['t']] += 1
            else:
                toks.append(t)
        e['tokens'] = toks; out.append(e)
    return out, n


def main():
    res = {}
    Cf, n = cut(C, lambda d, t: d['id'] in flag_docs or bool({'variant', 'nodraw'} & set(t.get('fl', []))))
    res['contested_removed'] = dict(n)
    res['contested_free'] = B.battery(Cf)
    print('removed', dict(n)); print(json.dumps({k: {a: b for a, b in v.items()} for k, v in res['contested_free'].items()}, default=float))
    pools = {}
    for di, d in enumerate(C):
        for ti, t in enumerate(d['tokens']):
            if t['t'] in n: pools.setdefault(t['t'], []).append((di, ti))
    ctrl = []
    for s in range(200):
        rng = random.Random(8300 + s); drop = set()
        for k, m in n.items(): drop |= set(rng.sample(pools[k], m))
        Cr = []
        for di, d in enumerate(C):
            e = dict(d); e['tokens'] = [dict(PH) if (di, ti) in drop else t for ti, t in enumerate(d['tokens'])]; Cr.append(e)
        ctrl.append(B.battery(Cr, fast=True, seed=s))
    res['random_cut'] = ctrl
    # (b) jackknife
    base = B.battery(C, fast=True)
    Crd_all = P.version(C, 'rd')
    D_all = B.docs79(C)
    jk = []
    ids = [d['id'] for d in C if any(t['t'] in ('word', 'num') for t in d['tokens'])]
    idx = {d['id']: i for i, d in enumerate(C)}
    for k in ids:
        Crd = [x for x in Crd_all if x['id'] != k]
        D = [x for x in D_all if x['id'] != k]
        ku = B.b1_kuro(Crd, nnull=0 or 1)
        zs = K.affix_z(D, 'suf').get('ME')
        fr = B.b2a_fixed(Crd, nnull=1)
        jk.append(dict(id=k, kuro=ku['exact'], kuro_testable=ku['testable'], me=zs, rev=fr['reversals'], pairs=fr['pairs'],
                       flagged=k in flag_docs or any({'variant', 'nodraw'} & set(t.get('fl', [])) for t in C[idx[k]]['tokens'])))
    res['jackknife'] = jk
    res['base_fast'] = base
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), default=float)


if __name__ == '__main__':
    main()
