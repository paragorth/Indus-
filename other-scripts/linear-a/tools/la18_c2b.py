#!/usr/bin/env python3
"""LA-18 cycle 2b: is the context sharing of LA alternation classes more than shared site / document type?
Context bags WITHOUT site and support features (prev/next token class + logograms on the doc only), and a
stricter random baseline: random pairs matched on document frequency AND on the (site, support) of each
member's first attestation. 5,000 resamples per class. Also LB: do truth classes have lower context P
(AUC) in the c2 run?
"""
import json, random, collections, numpy as np
from la18_common import *
from la18_c2 import klass


def ctx_vectors_lite(docs):
    V = collections.defaultdict(collections.Counter)
    for d in docs:
        toks = d['toks']; logos = {t[1] for t in toks if t[0] == 'F' and t[1] != 'NUM'}
        for i, t in enumerate(toks):
            if t[0] != 'W' or len(t[1]) < 2: continue
            c = V[t[1]]
            p = toks[i - 1] if i else None; n = toks[i + 1] if i + 1 < len(toks) else None
            f = lambda x: 'END' if x is None else ('W' if x[0] == 'W' else (x[1] if x[0] == 'F' else x[0]))
            c['p:' + f(p)] += 1; c['n:' + f(n)] += 1
            for L in logos: c['L:' + L] += 0.5
    return V


def pair_ctx(by, a, b):
    da = [d for d in by[a] if b not in d['words']]; db = [d for d in by[b] if a not in d['words']]
    Va = ctx_vectors_lite(da).get(a); Vb = ctx_vectors_lite(db).get(b)
    return cos(Va, Vb) if Va and Vb else np.nan


def main():
    docs = la_corpus()
    by = collections.defaultdict(list)
    for d in docs:
        for w in set(d['words']): by[w].append(d)
    df = {w: len(v) for w, v in by.items()}
    strat = {w: (v[0]['site'], v[0]['support']) for w, v in by.items()}
    P = form_pairs(df)
    cls = collections.defaultdict(list)
    for (a, b), f in P.items(): cls[klass(a, b, f)].append((a, b))
    bins = collections.defaultdict(list)
    bk = lambda w: (min(df[w], 4), strat[w])
    for w in df: bins[bk(w)].append(w)
    rnd = random.Random(7)
    cache = {}
    def rc(a, b):
        k = (a, b)
        if k not in cache: cache[k] = pair_ctx(by, a, b)
        return cache[k]
    out = []
    for c, pairs in sorted(cls.items()):
        if len(pairs) < 3: continue
        m = np.nanmean([rc(a, b) for a, b in pairs])
        null = []
        for _ in range(400):
            vals = []
            for a, b in pairs:
                for _t in range(30):
                    x = rnd.choice(bins[bk(a)]); y = rnd.choice(bins[bk(b)])
                    if x != y and x[:1] != y[:1] and x[-1:] != y[-1:]: break
                vals.append(rc(x, y))
            null.append(np.nanmean(vals))
        null = np.array(null)
        out.append(dict(cls=c, n=len(pairs), ctx=round(float(m), 3), null=round(float(np.nanmean(null)), 3),
                        p=round(float(((null >= m).sum() + 1) / (len(null) + 1)), 4)))
    p = np.array([r['p'] for r in out]); q = bh(p)
    for r, qq in zip(out, q): r['q'] = round(float(qq), 3)
    out.sort(key=lambda r: r['p'])
    # LB truth AUC from c2 (context P of truth classes vs others), full LB
    lb = json.load(open(os.path.join(OUT, 'c2_lb.json')))['LB_full']['classes']
    t = [r['ctx_p'] for r in lb if r.get('truth')]; o = [r['ctx_p'] for r in lb if not r.get('truth') and r['cls'] != 'RND']
    auc = float(np.mean([(x < y) + 0.5 * (x == y) for x in t for y in o])) if t and o else None
    res = dict(LA=out, LB_truth_ctx_auc=auc, LB_n_truth=len(t), LB_n_other=len(o))
    dump(res, 'c2b.json')
    for r in out[:12]: print(r)
    print('LB truth-vs-other AUC (lower ctx P)', auc, len(t), len(o))


if __name__ == '__main__':
    main()
