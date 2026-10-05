"""v54 cycle 4: are there passages at all beyond word pairs?  Near-repeat distance spectrum of the real text vs a
word-bigram resynthesis (per section; line-initial words from line-initial words; line lengths kept) and a
word-trigram resynthesis.  If the bigram chain reproduces the Voynich near-repeats but not a real text's, the
Voynich 'witnesses' are pairwise junction effects, not copied passages."""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V

def null_ngram(pages, seed=1, key='sec', order=2):
    rng = random.Random(seed)
    tab = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for p in pages:
        s = p['vars'][key]
        for l in p['lines']:
            ctx = ('<s>',) * (order - 1)
            for w in l:
                tab[s][ctx][w] += 1; ctx = (ctx + (w,))[1:] if order > 1 else ()
    T = {s: {c: (list(v.keys()), list(v.values())) for c, v in d.items()} for s, d in tab.items()}
    out = []
    for p in pages:
        t = T[p['vars'][key]]; nl = []
        for l in p['lines']:
            ctx = ('<s>',) * (order - 1); q = []
            for _ in l:
                kv = t.get(ctx)
                if kv is None:   # back off: restart from a line start context
                    kv = t[('<s>',) * (order - 1)]
                w = rng.choices(*kv)[0]; q.append(w); ctx = (ctx + (w,))[1:] if order > 1 else ()
            nl.append(q)
        out.append(dict(p, lines=nl))
    return out

if __name__ == '__main__':
    out = {}
    for nm, P in [('ZL', V.voynich('ZL3b')), ('IT', V.voynich('IT2a')), ('BRU_planted', V.brumati()[0]), ('BRU_clean', V.brumati(plant=False)[0]), ('GER_real', V.german())]:
        for tn, order in [('real', 0), ('unigram', 1), ('bigram', 2), ('trigram', 3)]:
            hs = []
            for s in ([0] if order == 0 else [1, 2, 3]):
                Q = P if order == 0 else null_ngram(P, s, order=order)
                toks, pairs, _ = V.families(Q)
                h = collections.Counter(sum(x[2:]) for x in pairs); hs.append([h[d] for d in range(4)])
            m = [sum(x[d] for x in hs) / len(hs) for d in range(4)]
            out[nm + ':' + tn] = m; print(nm, tn, [round(x) for x in m], flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c4.json'), 'w'))
