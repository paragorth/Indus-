"""v54 cycle 1: find near-repeat families, collate them, compare with nulls; planted-recovery check."""
import sys, os, json, collections, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V

def null_posshuf(pages, seed=1, key='sec'):
    """Shuffle words within (section, in-line slot: first / last / middle): keeps line-position word classes."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    def slot(k, n): return 'F' if k == 0 else ('L' if k == n - 1 else 'M')
    for p in pages:
        for l in p['lines']:
            for k, w in enumerate(l): by[(p['vars'][key], slot(k, len(l)))].append(w)
    for v in by.values(): rng.shuffle(v)
    it = {k: iter(v) for k, v in by.items()}
    return [dict(p, lines=[[next(it[(p['vars'][key], slot(k, len(l)))]) for k in range(len(l))] for l in p['lines']]) for p in pages]

def null_pageshuf(pages, seed=1):
    """Words shuffled within each page (keeps page vocabulary and local drift, destroys phrase order)."""
    rng = random.Random(seed); out = []
    for p in pages:
        ws = [w for l in p['lines'] for w in l]; rng.shuffle(ws); it = iter(ws)
        out.append(dict(p, lines=[[next(it) for _ in l] for l in p['lines']]))
    return out

def summarize(name, pages, inv=None):
    toks, pairs, msg = V.families(pages)
    col = V.collate(toks, pairs)
    loc = collections.Counter()
    for i, j, *_ in pairs:
        pi, pj = toks[i][1], toks[j][1]
        loc['samepage' if pi == pj else ('samesec' if pages[pi]['vars']['sec'] == pages[pj]['vars']['sec'] else 'cross')] += 1
    ex = sum(1 for x in pairs if sum(x[2:]) == 0)
    sc = V.op_scores(col)
    show = lambda k: tuple((inv.get(c, c) if inv else c) for c in k)
    occpos = col['occpos']; oppos = col['oppos']
    posrate = {p: round(oppos[p] / max(occpos[p], 1), 4) for p in 'IMF'}
    urate = {}
    for c, n in col['occ'].items():
        if n >= 100:
            inv_n = sum(v for k, v in col['ops'].items() if c in k[1:])
            urate[show((c,))[0]] = round(inv_n / n, 4)
    return dict(name=name, msg=msg, npairs=len(pairs), exact=ex, loc=dict(loc), posrate=posrate,
                top=[(round(s, 4), n, show(k)) for s, n, k in sc[:15]], unit_var=sorted(urate.items(), key=lambda x: -x[1]),
                ops_per_pair=round(sum(col['ops'].values()) / max(len(pairs), 1), 3)), col

if __name__ == '__main__':
    R = []
    bru, M = V.brumati(); inv = {v: k for k, v in M.items()}
    bru0, M0 = V.brumati(plant=False); inv0 = {v: k for k, v in M0.items()}
    corp = [('ZL', V.voynich('ZL3b'), None), ('IT', V.voynich('IT2a'), None), ('BRU_planted', bru, inv),
            ('BRU_clean', bru0, inv0), ('GER_real', V.german(), None)]
    for nm, P, iv in corp:
        tests = [('real', P)] + [('lineshuf', V.null_lineshuf(P, 1)), ('posshuf', null_posshuf(P, 1)), ('pageshuf', null_pageshuf(P, 1)),
                                 ('markov', V.null_markov(P, 1)), ('selfcit', V.null_selfcit(P, 1))]
        if nm in ('IT', 'BRU_clean'): tests = tests[:3]
        for tn, Q in tests:
            s, col = summarize(nm + ':' + tn, Q, iv)
            R.append(s); print(json.dumps(s, ensure_ascii=False)[:900], flush=True)
    json.dump(R, open(os.path.join(V.CK, 'c1.json'), 'w'), ensure_ascii=False)
