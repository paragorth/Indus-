"""v60 cycle 1: order-free search for a closed 2x2 complexion lattice (two mutually exclusive item pairs whose joint
table fits the external hot/cold x dry/moist distribution), on train pages, re-scored on held-out pages."""
import sys, random, json, collections
import numpy as np
import v60_lib as L

ext = L.jload('external.json')
pool = collections.Counter()
for s in ('CI', 'BAN', 'MAC'):
    pool.update(ext[s]['cells'])
Q = L.qvec(pool)
# permuted targets: same cell values, dominant cell's diagonal partner changed (not a relabeling of Q)
QA1 = Q[[0, 1, 3, 2]]; QA2 = Q[[0, 3, 2, 1]]
print('Q', Q.round(3), 'QA1', QA1.round(3), 'QA2', QA2.round(3))


def split(ents, seed):
    pages = sorted(set(e['page'] for e in ents))
    rng = random.Random(seed); rng.shuffle(pages)
    tr = set(pages[: len(pages) // 2])
    a = [i for i, e in enumerate(ents) if e['page'] in tr]
    b = [i for i, e in enumerate(ents) if e['page'] not in tr]
    return a, b


def corpora():
    C = {}
    vh = L.voynich_entries('ZL3b', ('H', 'P'))
    C['V_herb'] = vh
    C['V_herb_IT2a'] = L.voynich_entries('IT2a', ('H', 'P'))
    C['N_markov'] = L.markov_null(vh, 1)
    C['N_wordshuf'] = L.wordshuf_null(vh, 1)
    C['V_stars'] = L.voynich_entries('ZL3b', ('S',))
    C['V_bio'] = L.voynich_entries('ZL3b', ('B',))
    for s in ('CI', 'BAN', 'MAC'):
        E = L.herbal_entries(s)
        C['P_' + s] = L.encode_entries(E, seed=60 + len(s), pad=0.35)
    C['P_CI_trunc60'] = L.encode_entries(L.herbal_entries('CI'), seed=61, pad=0.35, max_len=40)
    C['P_GER'] = L.encode_entries(L.herbal_entries('GER')[:300], seed=62, pad=0.35)
    C['NEG_CURY'] = L.encode_entries(L.plain_entries('CURY'), seed=63, pad=0.35)
    C['NEG_AST'] = L.encode_entries(L.plain_entries('AST'), seed=64, pad=0.35)
    return C


if __name__ == '__main__':
    C = corpora()
    for k, v in C.items():
        print(k, len(v), 'entries, mean len', round(np.mean([len(e['toks']) for e in v]), 1), flush=True)
    out = {}
    for k, ents in C.items():
        for tgt, q in (('Q', Q), ('QA1', QA1), ('QA2', QA2)):
            if tgt != 'Q' and k not in ('V_herb', 'P_CI', 'P_BAN', 'N_wordshuf'):
                continue
            for seed in (1, 2, 3):
                tr, te = split(ents, seed)
                r = L.lattice_search(ents, q, tr, te, K=3000, top=50)
                sm = L.summarize(r)
                out['%s|%s|%d' % (k, tgt, seed)] = dict(summary=sm, top=r['top'][:10])
                print(k, tgt, seed, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in sm.items()}),
                      (r['top'][0]['items'] if r['top'] else None), flush=True)
    L.jsave('cycle1.json', out)
