#!/usr/bin/env python3
"""LA-18 cycle 2: which ENDING ALTERNATIONS avoid (or seek) each other?
Pool all form pairs by their alternation class (+X suffix added; X+ prefix added; X~Y final swap; ^X~Y initial
swap). Class statistic = sum of tablet co-occurrences vs the curveball null (site strata; frequencies and
tablet sizes fixed), as z; two-sided max-|z| over classes for FWER. Context similarity of the pairs elsewhere
(cosine of context bags, docs where they co-occur removed) vs frequency-matched random pairs.
usage: la18_c2.py lb | la
"""
import sys, json, random, collections, time
import numpy as np
from la18_common import *
from la18_c1 import lb_truth


def klass(a, b, fam):
    if fam == 'SUF1': return '+' + b[-1]
    if fam == 'PRE1': return b[0] + '+'
    if fam == 'FIN': return '~'.join(sorted((a[-1], b[-1])))
    if fam == 'INI': return '^' + '~'.join(sorted((a[0], b[0])))
    return fam


def ctx_sim(docs, pairs, drop_cooc=True):
    """cosine of context bags; bags built from docs where the OTHER member is absent."""
    out = []
    by = collections.defaultdict(list)
    for d in docs:
        for w in set(d['words']): by[w].append(d)
    for a, b in pairs:
        da = [d for d in by[a] if not (drop_cooc and b in d['words'])]
        db = [d for d in by[b] if not (drop_cooc and a in d['words'])]
        Va = context_vectors(da).get(a, {}); Vb = context_vectors(db).get(b, {})
        out.append(cos(Va, Vb) if Va and Vb else np.nan)
    return np.array(out)


def run(docs, S, site, nsamp, seed, label, minpairs, truthf=None, extra=None):
    df = collections.Counter(w for s in S for w in s)
    alltypes = collections.Counter(w for d in docs for w in set(d['words']))
    P = form_pairs(alltypes)              # pairs from all docs (context), counted on multiword units
    if extra: P.update(extra)
    R = random_pairs(sorted(alltypes), alltypes, P, 1, seed + 3)
    allp = dict(P); allp.update(R); keys = list(allp)
    obs, N = null_matrix(S, site, keys, nsamp, seed)
    cl = np.array([klass(k[0], k[1], allp[k]) for k in keys])
    fam = np.array([allp[k] for k in keys])
    cnt = collections.Counter(cl)
    classes = [c for c, n in cnt.items() if n >= minpairs]
    Z, Zn, info = [], [], []
    for c in classes:
        idx = np.where(cl == c)[0]
        o = obs[idx].sum(); ns = N[:, idx].sum(1).astype(float)
        mu, sd = ns.mean(), ns.std() + 0.5  # +0.5 guards discrete near-zero variance
        Z.append((o - mu) / sd); Zn.append((ns - mu) / sd)
        info.append(dict(cls=c, fam=fam[idx[0]], npairs=int(len(idx)), O=int(o), E=round(float(mu), 2)))
    Z = np.array(Z); Zn = np.array(Zn).T if Zn else np.zeros((nsamp, 0))
    mx = np.abs(Zn).max(1) if Zn.size else np.zeros(nsamp)
    # context similarity: per class mean, vs RND
    t0 = time.time()
    cs = ctx_sim(docs, keys)
    rnd_cs = cs[fam == 'RND']; rnd_mu = np.nanmean(rnd_cs)
    rows = []
    for i, c in enumerate(classes):
        idx = np.where(cl == c)[0]
        m = np.nanmean(cs[idx]) if np.isfinite(cs[idx]).any() else np.nan
        # permutation P for context: mean of len(idx) random RND pairs >= m
        rr = np.random.default_rng(seed + i)
        valid = rnd_cs[np.isfinite(rnd_cs)]
        k = int(np.isfinite(cs[idx]).sum())
        pc = float(((np.array([rr.choice(valid, k).mean() for _ in range(2000)]) >= m).sum() + 1) / 2001) if k and len(valid) else np.nan
        d = dict(info[i]); d.update(z=round(float(Z[i]), 2), fwer=round(float(((mx >= abs(Z[i])).sum() + 1) / (nsamp + 1)), 4),
                                   ctx=round(float(m), 3), ctx_p=round(pc, 4))
        if truthf: d['truth'] = bool(truthf(('X', 'Y', 'Q'), ('X', 'Y', 'Q'), 'NA')) if False else None
        rows.append(d)
    rows.sort(key=lambda r: -abs(r['z']))
    fams = {f: family_excess(obs, N, np.where(fam == f)[0]) for f in ['SUF1', 'PRE1', 'FIN', 'INI', 'RND', 'PLANTA', 'PLANTV']
            if (fam == f).any()}
    for f in fams: fams[f]['ctx'] = round(float(np.nanmean(cs[fam == f])), 3)
    return dict(label=label, nunits=len(S), nclasses=len(classes), rnd_ctx=round(float(rnd_mu), 3), fam=fams,
                n_fwer05=int(sum(r['fwer'] < .05 for r in rows)), classes=rows[:40]), (keys, allp, cl)


def lb_class_truth(c):
    if c.startswith('+'): return lb_truth(('X', 'O'), ('X', 'O', c[1:]), 'SUF1')
    if '~' in c and not c.startswith('^'):
        x, y = c.split('~'); return lb_truth(('X', 'Y', x), ('X', 'Y', y), 'FIN')
    return False


def plant(docs, mode, nw, seed, sign):
    """mode 'A' (attract): in each doc holding w, with P 0.5 also write w+sign in the same doc (another token).
       mode 'V' (avoid): in each doc holding w, with P 0.5 replace every w by w+sign."""
    r = random.Random(seed)
    df = collections.Counter(w for d in docs for w in set(d['words']))
    multi = collections.Counter(w for d in docs if len(set(d['words'])) >= 2 for w in set(d['words']))
    cand = [w for w, c in multi.items() if c >= 2]
    ch = set(r.sample(cand, min(nw, len(cand))))
    out = []
    for d in docs:
        d = dict(d); toks = list(d['toks']); ws = list(d['words'])
        hit = [w for w in ch if w in ws]
        for w in hit:
            if r.random() < 0.5:
                if mode == 'V':
                    toks = [('W', w + (sign,)) if t[0] == 'W' and t[1] == w else t for t in toks]
                    ws = [w + (sign,) if x == w else x for x in ws]
                else:
                    toks = toks + [('NL',), ('W', w + (sign,)), ('F', 'NUM')]
                    ws = ws + [w + (sign,)]
        d['toks'] = toks; d['words'] = ws; out.append(d)
    return out, sorted(ch)


def main():
    job = sys.argv[1]; out = {}
    if job == 'lb':
        docs = lb_corpus(); S, site, _ = units(docs)
        res, _ = run(docs, S, site, 1000, 11, 'LB_full', 5)
        for r in res['classes']: r['truth'] = lb_class_truth(r['cls'])
        out['LB_full'] = res; dump(out, 'c2_lb.json')
        print(json.dumps(res['fam']), res['n_fwer05'], flush=True)
        for r in res['classes'][:20]: print(r, flush=True)
        # LA-sized: 245 units sampled, classes with >= 3 pairs
        subs = []
        for rep in range(8):
            rr = random.Random(900 + rep)
            keep = set(rr.sample(range(len(S)), 245))
            ids = set(); k = 0
            mdocs = [d for d in docs if len(set(d['words'])) >= 2]
            sd = [mdocs[i] for i in sorted(keep)]
            S2, site2, _ = units(sd)
            r2, _ = run(sd, S2, site2, 500, 950 + rep, 'LB245_%d' % rep, 3)
            for r in r2['classes']: r['truth'] = lb_class_truth(r['cls'])
            subs.append(r2); print('sub', rep, r2['n_fwer05'], [(r['cls'], r['z'], r['fwer'], r['truth']) for r in r2['classes'][:5]], flush=True)
        out['LB245'] = subs; dump(out, 'c2_lb.json')
    else:
        docs = la_corpus(); S, site, _ = units(docs)
        res, _ = run(docs, S, site, 2000, 21, 'LA', 3)
        out['LA'] = res; dump(out, 'c2_la.json')
        print(json.dumps(res['fam']), res['n_fwer05'], flush=True)
        for r in res['classes'][:20]: print(r, flush=True)
        neg = []
        for rep in range(5):
            S2 = doc_shuffle(S, site, 1300 + rep)
            # rebuild docs consistent with shuffled units: contexts keep real docs (context test is not the target)
            r2, _ = run(docs, S2, site, 1000, 1400 + rep, 'LAshuf_%d' % rep, 3)
            neg.append({k: v for k, v in r2.items() if k != 'classes'} | dict(top=r2['classes'][:5]))
            print('neg', rep, r2['n_fwer05'], [(r['cls'], r['z'], r['fwer']) for r in r2['classes'][:3]], flush=True)
        out['LA_shuffled'] = neg; dump(out, 'c2_la.json')
        pl = []
        for rep in range(8):
            mode = 'A' if rep % 2 == 0 else 'V'
            pd, ch = plant(docs, mode, 8, 1500 + rep, '#' + mode)
            S2, site2, _ = units(pd)
            r2, _ = run(pd, S2, site2, 1000, 1600 + rep, 'LAplant%s_%d' % (mode, rep), 3)
            hit = [r for r in r2['classes'] if r['cls'] == '+#' + mode]
            rank = [i for i, r in enumerate(r2['classes']) if r['cls'] == '+#' + mode]
            pl.append(dict(rep=rep, mode=mode, words=['-'.join(w) for w in ch], planted=hit[0] if hit else None,
                           rank=rank[0] if rank else None, n_fwer05=r2['n_fwer05']))
            print('plant', mode, rep, hit, rank, flush=True)
        out['LA_planted'] = pl; dump(out, 'c2_la.json')


if __name__ == '__main__':
    main()
