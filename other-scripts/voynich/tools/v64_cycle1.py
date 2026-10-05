"""v64 cycle 1: massive random concept-alphabet search.
For every corpus: N random alphabets (6-20 glyph n-grams, n<=4, from the 200 commonest
n-grams of the selection-train quarter) scored on the selection split (train folio%4==0,
test ==1); the top 20 by Gm1 re-scored on the held split (train ==2, test ==3).
Gm1 = held-out bits/word of a glyph Markov-2 minus bits/word of the concept model
(concept Markov-1 + padding gaps); Gfree = the same with the free-combination
(Lullian wheel/table) term instead of the concept Markov-1."""
import sys, json, random, os, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V

CORP = ['ars', 'med', 'lat', 'voy', 'voyit', 'voy_gshuf', 'voy_mk2', 'voy_sc', 'ars_mk2', 'med_mk2']
N = {'lat': 8000}


def run(k):
    out = os.path.join(V.CK, 'c1_%s.json' % k)
    if os.path.exists(out):
        return k
    D = V.all_corpora(); C = D['corpora'][k]
    ps, ph = os.path.join(V.CK, 'c_%s_sel.txt' % k), os.path.join(V.CK, 'c_%s_held.txt' % k)
    tr, _ = V.write_corpus(C, ps, 'sel'); V.write_corpus(C, ph, 'held')
    pool = V.ngram_pool(tr)
    rng = random.Random(hash(k) % 1000)
    n = N.get(k, 20000)
    al = [V.sample_alphabet(pool, rng) for _ in range(n)]
    r = V.score(ps, al)
    top = sorted(range(n), key=lambda i: -r[i]['Gm1'])[:20]
    topf = sorted(range(n), key=lambda i: -r[i]['Gfree'])[:20]
    h = V.score(ph, [al[i] for i in top]); hf = V.score(ph, [al[i] for i in topf])
    res = {'sel': r, 'al': al, 'top': top, 'held': h, 'topf': topf, 'heldf': hf}
    base = {}
    for name, codes in D['codes'].items():
        if k.startswith(name):
            base['truth'] = sorted(codes.values())
            base['truth_sel'] = V.score(ps, [base['truth']])[0]
            base['truth_held'] = V.score(ph, [base['truth']])[0]
    # single-glyph alphabet (every glyph a concept) as reference
    glyphs = sorted({c for w in tr for c in w})
    base['glyph_sel'] = V.score(ps, [glyphs[:40]])[0]
    base['glyph_held'] = V.score(ph, [glyphs[:40]])[0]
    res['base'] = base
    json.dump(res, open(out, 'w'))
    return k


if __name__ == '__main__':
    V.all_corpora(); V.build_scorer()
    with Pool(2) as P:
        for k in P.imap_unordered(run, CORP):
            print('done', k, time.strftime('%X'), flush=True)
