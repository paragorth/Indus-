"""v54 cycle 4b: passage repetition index (PRI).  For every in-line word triple (a, b, c): O = number of pairs of
occurrences sharing a, b AND c; E = expected number if the successors of each word b were shuffled among the
occurrences of b in the same section (keeps every word pair a-b and b-c count, removes any link between a and c).
PRI = O / E.  Real languages: PRI > 1 (a phrase repeats as a whole).  PRI < 1: when two words recur together, the
third word is steered AWAY from what was written last time.  Permutation z from 200 successor shuffles.
Also split by whether the two occurrences lie on the same page or on different pages."""
import sys, os, json, collections, random, math, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V

def triples(pages, key='sec'):
    T = []   # (section, page, a, b, c)
    for pi, p in enumerate(pages):
        for l in p['lines']:
            for a, b, c in zip(l, l[1:], l[2:]): T.append((p['vars'][key], pi, a, b, c))
    return T

def observed(T):
    g = collections.defaultdict(list)
    for s, pi, a, b, c in T: g[(s, a, b)].append((pi, c))
    O = Osame = 0
    for v in g.values():
        if len(v) < 2: continue
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i][1] == v[j][1]:
                    O += 1; Osame += v[i][0] == v[j][0]
    return O, Osame

def shuffle_succ(T, rng):
    g = collections.defaultdict(list)
    for k, t in enumerate(T): g[(t[0], t[3])].append(k)
    cs = [t[4] for t in T]
    for idx in g.values():
        vals = [cs[k] for k in idx]; rng.shuffle(vals)
        for k, v in zip(idx, vals): cs[k] = v
    return [(t[0], t[1], t[2], t[3], c) for t, c in zip(T, cs)]

def pri(pages, nperm=200, seed=1, key='sec'):
    T = triples(pages, key); rng = random.Random(seed)
    O, Os = observed(T)
    null = [observed(shuffle_succ(T, rng)) for _ in range(nperm)]
    m = sum(x[0] for x in null) / nperm; sd = (sum((x[0] - m) ** 2 for x in null) / nperm) ** .5
    ms = sum(x[1] for x in null) / nperm
    return dict(O=O, E=round(m, 1), PRI=round(O / m, 3) if m else None, z=round((O - m) / (sd or 1), 1),
                O_samepage=Os, E_samepage=round(ms, 1), O_cross=O - Os, E_cross=round(m - ms, 1),
                PRI_cross=round((O - Os) / (m - ms), 3) if m - ms else None, ntrip=len(T))

def plain_pages(path, n_tokens=40000, line_w=9, page_w=180, nsec=6):
    t = open(path, encoding='utf-8', errors='replace').read().lower()
    ws = re.findall(r'[^\W\d_]+', t)[:n_tokens]
    pages = []
    for i in range(0, len(ws), page_w):
        chunk = ws[i:i + page_w]
        pages.append(dict(id='p%d' % i, vars=dict(sec='s%d' % (nsec * i // len(ws))), lines=[chunk[j:j + line_w] for j in range(0, len(chunk), line_w)]))
    return pages

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v54_c4 import null_ngram
    C = json.load(open(os.path.join(V.ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora']
    def cat_pages(key):
        P = C[key]; out = []
        for i, ws in enumerate(P):
            out.append(dict(id=str(i), vars=dict(sec='s%d' % (6 * i // len(P))), lines=[ws[j:j + 9] for j in range(0, len(ws), 9)]))
        return out
    zl = V.voynich('ZL3b')
    corp = [('ZL', zl), ('IT', V.voynich('IT2a')),
            ('ZL_A', [p for p in zl if p['vars']['lang'] == 'A']), ('ZL_B', [p for p in zl if p['vars']['lang'] == 'B']),
            ('ZL_selfcit', V.null_selfcit(zl, 3)), ('ZL_markov', V.null_markov(zl, 3)), ('ZL_bigramchain', null_ngram(zl, 3, order=2)),
            ('BRU_clean', V.brumati(plant=False)[0]), ('BRU_planted', V.brumati()[0]), ('GER_real', V.german()),
            ('LAT_catmus', cat_pages('I_Lat')), ('ITA_catmus', cat_pages('I_Ita'))]
    for k in ['la', 'it', 'de', 'he']:
        f = os.path.join(V.ROOT, 'data', 'plain', k + '.txt')
        if os.path.exists(f): corp.append(('plain_' + k, plain_pages(f)))
    out = {}
    for nm, P in corp:
        r = pri(P); out[nm] = r; print(nm, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c4b.json'), 'w'))
