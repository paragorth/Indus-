"""pe12 search engine: build the hypothesis bank for one (corpus, slot), run the
A/B/C held-out search on real labels and on shuffled-label nulls, checkpoint each run.

usage (library): run_corpus(tag, E, slots, reps, nhash, npairs, workers)
"""
import os, sys, json, time, math
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe12_common import (slot_entries, base_vars, add_running, single_features, random_hashes,  # noqa
                         baseline_probs, score_batch, split_tabs, shuffle_within, tablet_types,
                         _code, CKPT)

FAMS = ['ARITH', 'SIZE', 'CTX', 'SEQ', 'PAIR']
NULLS = ['STRICT', 'LOOSE', 'SIZEM']
_G = {}


def slot_sign_all(E, slot):
    out = []
    for e in E:
        s = e['signs']
        if slot in ('FIRST', 'AFTERHDR'):
            out.append(s[0])
        elif slot == 'LAST':
            out.append(s[-1])
        elif slot == 'FIRSTm':
            out.append(s[0] if len(s) > 1 else 'NONE')
        else:
            out.append(s[-1] if len(s) > 1 else 'NONE')
    return out


def size_cell(e):
    """Fine size cell: half-octave of the value x fraction flag x highest numeral code."""
    order = ['N48', 'N34', 'N45', 'N14', 'N01', 'N39B', 'N24', 'N30C', 'N30D', 'N39C']
    top = next((c for c in order if e['dig'].get(c)), 'frac')
    return '%d/%d/%s' % (int(math.floor(2 * math.log2(max(e['v'], 0.01)))), e['frac'], top)


def build(E, slot, nhash=20000, npairs=6000, seed=12, ky=30, fine=False):
    rng = np.random.default_rng(seed)
    ents, ys, oth = slot_entries(E, slot)
    V = add_running(base_vars(ents), ents, E)
    F = single_features(ents, oth, V, E, slot_sign_all(E, slot))
    y, Ky = _code(ys, ky + 1)
    sysv = np.array([0 if e['sys'] == 'S' else 1 for e in ents])
    ty = tablet_types(E)
    tabs = [e['tab'] for e in ents]
    lv = np.floor(np.log2(np.maximum([e['v'] for e in ents], 0.01))).astype(int)
    g_strict = [(t, s) for t, s in zip(tabs, sysv)]
    g_loose = [(ty[t], s) for t, s in zip(tabs, sysv)]
    fr = [e['frac'] for e in ents]
    g_sizem = [(ty[t], s, b, f) for t, s, b, f in zip(tabs, sysv, lv, fr)]
    if fine:   # baseline already knows the fine size cell; nulls shuffle within it
        cells = [size_cell(e) for e in ents]
        strat, _ = _code([str(s) + c for s, c in zip(sysv, cells)], 100000)
        g_sizem = [(ty[t], s, c) for t, s, c in zip(tabs, sysv, cells)]
        g_strict = [(t, s, c) for t, s, c in zip(tabs, sysv, cells)]
    else:
        strat, _ = _code([ty[t] + str(s) for t, s in zip(tabs, sysv)], 1000)
    bank = []  # (name, family, codes)
    for n, (c, k, fam) in F.items():
        bank.append((n, fam, c))
    H, desc = random_hashes(V, nhash, rng)
    for h, d in zip(H, desc):
        bank.append(('hash:' + d, 'ARITH', h))
    singles = [n for n, (c, k, f) in F.items() if f in ('ARITH', 'SIZE', 'CTX')]
    for _ in range(npairs):
        a, b = rng.choice(len(singles), 2, replace=False)
        ca, ka, fa = F[singles[a]]
        cb, kb, fb = F[singles[b]]
        c, k = _code(ca * 64 + cb, 60)
        bank.append(('pair:%s&%s' % (singles[a], singles[b]), 'PAIR:%s' % '+'.join(sorted([fa, fb])), c))
    return dict(ents=ents, y=y, Ky=Ky, ylab=ys, sysv=sysv, tabs=np.array(tabs), strat=strat,
                groups={'STRICT': g_strict, 'LOOSE': g_loose, 'SIZEM': g_sizem}, bank=bank)


def fam_of(f):
    return 'PAIR' if f.startswith('PAIR') else f


def score_all(G, y, fit, ev):
    P0 = baseline_probs(y, G['Ky'], G['strat'], fit)
    bank = G['bank']
    out = np.zeros(len(bank))
    # batch by cardinality
    order = sorted(range(len(bank)), key=lambda i: bank[i][2].max())
    B = 400
    for s in range(0, len(order), B):
        ii = order[s:s + B]
        X = np.stack([bank[i][2] for i in ii])
        Kx = int(X.max()) + 1
        out[ii] = score_batch(X, Kx, y, G['Ky'], G['sysv'], P0, G['strat'], fit, ev)
    return out


def one_run(G, y, nsplit=2, seed=0, top=10):
    """Family statistics averaged over nsplit random A/B/C tablet splits."""
    rng = np.random.default_rng(1000 + seed)
    tabs = sorted(set(G['tabs']))
    res = defaultdict(list)
    best = defaultdict(list)
    for sp in range(nsplit):
        A, Bs, C = split_tabs(tabs, np.random.default_rng(77 + sp))   # same splits real & null
        ia = np.where(np.isin(G['tabs'], list(A)))[0]
        ib = np.where(np.isin(G['tabs'], list(Bs)))[0]
        ic = np.where(np.isin(G['tabs'], list(C)))[0]
        sB = score_all(G, y, ia, ib)
        fams = np.array([fam_of(b[1]) for b in G['bank']])
        sub = {}
        for f in FAMS + ['ALL']:
            m = np.where(fams == f)[0] if f != 'ALL' else np.arange(len(fams))
            if len(m) == 0:
                continue
            t = m[np.argsort(-sB[m])[:top]]
            sub[f] = t
        allt = sorted(set(np.concatenate(list(sub.values())).tolist()))
        Gs = dict(G); Gs['bank'] = [G['bank'][i] for i in allt]
        iab = np.concatenate([ia, ib])
        sC = dict(zip(allt, score_all(Gs, y, iab, ic)))
        for f, t in sub.items():
            res[f + ':top1'].append(sC[t[0]])
            res[f + ':top10'].append(float(np.mean([sC[i] for i in t])))
            best[f].append([(G['bank'][i][0], round(float(sB[i]), 4), round(float(sC[i]), 4)) for i in t[:5]])
    return {k: float(np.mean(v)) for k, v in res.items()}, dict(best)


def _task(args):
    tag, slot, mode, rep, nsplit = args
    path = os.path.join(CKPT, '%s_%s_%s_%03d.json' % (tag, slot, mode, rep))
    if os.path.exists(path):
        return json.load(open(path))
    G = _G[(tag, slot)]
    y = G['y']
    if mode != 'REAL':
        y = shuffle_within(y, G['groups'][mode], np.random.default_rng(rep * 7919 + NULLS.index(mode)))
    t0 = time.time()
    st, best = one_run(G, y, nsplit=nsplit, seed=rep)
    r = dict(tag=tag, slot=slot, mode=mode, rep=rep, stats=st, best=best if mode == 'REAL' else None,
             sec=round(time.time() - t0, 1))
    json.dump(r, open(path, 'w'))
    return r


def run_corpus(tag, E, slots, reps=20, nhash=20000, npairs=6000, workers=2, nsplit=2,
               nulls=NULLS, log=print, fine=False):
    out = {}
    for slot in slots:
        G = build(E, slot, nhash=nhash, npairs=npairs, fine=fine)
        if len(set(G['tabs'])) < 40:
            log('%s %s: too few tablets' % (tag, slot)); continue
        _G[(tag, slot)] = G
        log('%s %s: %d entries, %d tablets, %d hypotheses, Ky %d' %
            (tag, slot, len(G['y']), len(set(G['tabs'])), len(G['bank']), G['Ky']))
        tasks = [(tag, slot, 'REAL', 0, nsplit)] + [(tag, slot, m, r, nsplit) for m in nulls for r in range(reps)]
        with Pool(workers) as p:
            R = p.map(_task, tasks, chunksize=1)
        real = R[0]
        summ = {'n': len(G['y']), 'ntab': len(set(G['tabs'])), 'H': len(G['bank']), 'real': real['stats'],
                'best': real['best'], 'null': {}}
        for m in nulls:
            nr = [r['stats'] for r in R if r['mode'] == m]
            d = {}
            for k, v in real['stats'].items():
                arr = np.array([x.get(k, np.nan) for x in nr])
                d[k] = dict(mean=float(np.nanmean(arr)), sd=float(np.nanstd(arr)),
                            p=float((1 + np.sum(arr >= v)) / (1 + len(arr))),
                            z=float((v - np.nanmean(arr)) / (np.nanstd(arr) + 1e-9)))
            summ['null'][m] = d
        out[slot] = summ
        for f in FAMS:
            k = f + ':top1'
            if k in real['stats']:
                log('  %-6s real %+.4f | ' % (f, real['stats'][k]) +
                    ' '.join('%s %+.4f z%+.1f p%.3f' % (m, summ['null'][m][k]['mean'], summ['null'][m][k]['z'], summ['null'][m][k]['p']) for m in nulls)
                    + ' | ' + real['best'][f][0][0][0])
        del _G[(tag, slot)]
    return out
