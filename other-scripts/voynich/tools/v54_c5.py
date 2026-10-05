"""v54 cycle 5: more power for the 'witness word' test.  Three directions pooled: (a,b)->c, (b,c)->a, (a,c)->b.
For each, the target word is shuffled among occurrences sharing the adjacent context word (within section).
Statistic: exact and one-edit-variant target agreements among recurring 2-word contexts.  Also on the full ZL
transliteration including label and circular lines (all line types) for extra text."""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
from v54_c4c import lev1
from v54_c4b import plain_pages

def trip_dirs(pages):
    T = []
    for pi, p in enumerate(pages):
        s = p['vars']['sec']
        for l in p['lines']:
            for a, b, c in zip(l, l[1:], l[2:]):
                T.append((('F', s), a, b, c, b))   # context (a,b), target c, shuffle key b
                T.append((('B', s), c, b, a, b))   # context (c,b), target a, shuffle key b
                T.append((('M', s), a, c, b, a))   # context (a,c), target b, shuffle key a
    return T

def counts(T, tgt):
    g = collections.defaultdict(list)
    for (t, x1, x2, _, _), y in zip(T, tgt): g[(t, x1, x2)].append(y)
    ex = var = 0
    for v in g.values():
        if len(v) < 2: continue
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i] == v[j]: ex += 1
                elif lev1(v[i], v[j]): var += 1
    return ex, var

def run(pages, nperm=100, seed=5):
    T = trip_dirs(pages); rng = random.Random(seed)
    tgt = [t[3] for t in T]
    obs = counts(T, tgt)
    grp = collections.defaultdict(list)
    for k, t in enumerate(T): grp[(t[0], t[4])].append(k)
    N = []
    for _ in range(nperm):
        tt = list(tgt)
        for idx in grp.values():
            vals = [tgt[k] for k in idx]; rng.shuffle(vals)
            for k, v in zip(idx, vals): tt[k] = v
        N.append(counts(T, tt))
    out = {}
    for k, nm in [(0, 'exact'), (1, 'variant')]:
        xs = [n[k] for n in N]; m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5
        out[nm] = obs[k]; out[nm + '_null'] = round(m, 1); out[nm + '_ratio'] = round(obs[k] / m, 3) if m else None
        out[nm + '_z'] = round((obs[k] - m) / (sd or 1), 1); out[nm + '_sd'] = round(sd, 1)
    ee = obs[0] - out['exact_null']; ve = obs[1] - out['variant_null']
    out['var_per_exact_excess'] = round(ve / ee, 3) if ee > 0 else None
    out['var_per_exact_excess_se'] = round(out['variant_sd'] / ee, 3) if ee > 0 else None
    return out

def voy_all(name):
    import vlib
    L = vlib.load_voynich(name, ltypes=None)
    pages = collections.OrderedDict()
    for r in L:
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if '?' not in w and w]
        if len(ws) < 3: continue
        p = pages.setdefault(r['folio'], dict(id=r['folio'], vars=dict(sec=r['illus']), lines=[]))
        p['lines'].append(ws)
    return list(pages.values())

if __name__ == '__main__':
    C = json.load(open(os.path.join(V.ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora']
    def cat_pages(key):
        P = C[key]
        return [dict(id=str(i), vars=dict(sec='s%d' % (6 * i // len(P))), lines=[ws[j:j + 9] for j in range(0, len(ws), 9)]) for i, ws in enumerate(P)]
    zl = V.voynich('ZL3b')
    corp = [('ZL', zl), ('IT', V.voynich('IT2a')), ('ZL_alllines', voy_all('ZL3b')), ('IT_alllines', voy_all('IT2a')),
            ('ZL_selfcit', V.null_selfcit(zl, 3)), ('ZL_markov', V.null_markov(zl, 3)),
            ('BRU_clean', V.brumati(plant=False)[0]), ('BRU_planted', V.brumati()[0]), ('GER_real', V.german()),
            ('LAT_catmus', cat_pages('I_Lat')), ('ITA_catmus', cat_pages('I_Ita')),
            ('plain_la', plain_pages(os.path.join(V.ROOT, 'data', 'plain', 'la.txt')))]
    out = {}
    for nm, P in corp:
        r = run(P); out[nm] = r; print(nm, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c5.json'), 'w'))
