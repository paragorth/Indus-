"""v60 cycle 1: lattice-shaped item search. Hypothesis = items (a, b) for quality 1 and (c, d) for quality 2, with
a/b followed within 3 tokens (or in the same word, prefix->suffix) by c/d; per entry the single cell that occurs.
Gates: coverage >= 0.35, exactly-one cell in >= 80% of covered entries, all four cells used, KL(T||Q) <= 0.15.
Rank by log Bayes factor of the external complexion table Q against any 2x2 table. Train/test split by page."""
import sys, random, json, collections, time
import numpy as np
import v60_lib as L

ext = L.jload('external.json')
pool = collections.Counter()
for s in ('CI', 'BAN', 'MAC'):
    pool.update(ext[s]['cells'])
Q = L.qvec(pool)
QA1 = Q[[0, 1, 3, 2]]


def split(ents, seed):
    pages = sorted(set(e['page'] for e in ents))
    rng = random.Random(seed); rng.shuffle(pages)
    tr = set(pages[: len(pages) // 2])
    return [i for i, e in enumerate(ents) if e['page'] in tr], [i for i, e in enumerate(ents) if e['page'] not in tr]


def is_true(C, h):
    d = [L.decode_item(C, i, 1) for i in h['items']]
    d = [x[0] if x else '' for x in d]
    qual = lambda x: bool(L.HOT.match(x) or L.COLD.match(x))
    dm = lambda x: bool(L.DRY.match(x) or L.MOIST.match(x))
    return (qual(d[0]) and qual(d[1]) and dm(d[2]) and dm(d[3])) or (dm(d[0]) and dm(d[1]) and qual(d[2]) and qual(d[3])), d


if __name__ == '__main__':
    vh = L.voynich_entries('ZL3b', ('H', 'P'))
    runs = [('V_herb', vh, Q), ('V_herb_QA1', vh, QA1), ('N_markov', L.markov_null(vh, 1), Q),
            ('V_stars', L.voynich_entries('ZL3b', ('S',)), Q),
            ('NEG_CURY', L.encode_entries(L.plain_entries('CURY'), seed=63, pad=0.35, max_len=128), Q),
            ('V_herb_IT2a', L.voynich_entries('IT2a', ('H', 'P')), Q)]
    out = {}
    for name, ents, q in runs:
        t = time.time()
        tr, te = split(ents, 1)
        R = L.run_lattice(ents, q, tr, te, top=50)
        top = R['top']
        sm = dict(n_hyp=R['n_hyp'], n_pass=R['n_pass'], n_train=len(tr), n_test=len(te))
        if top:
            sm.update(best_lbf=round(top[0]['lbf'], 2), med_lbf=round(float(np.median([h['lbf'] for h in top])), 2),
                      med_test_lbf=round(float(np.median([h['test']['lbf'] for h in top])), 2),
                      frac_test_pass=round(float(np.mean([h['test']['cov'] >= .35 and h['test']['ex'] >= .8 and h['test']['kl'] <= .15
                                                          for h in top])), 2),
                      best_items=top[0]['items'], best_T=top[0]['T'], best_test_T=top[0]['test']['T'])
        out[name] = dict(summary=sm, top=[{k: v for k, v in h.items()} for h in top[:20]])
        print(name, round(time.time() - t), json.dumps(sm), flush=True)
    L.jsave('cycle1.json', out)
