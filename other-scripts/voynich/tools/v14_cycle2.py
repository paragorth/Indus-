"""v14 cycle 2: strata and the junction.
(a) device fit + within-word slot independence per Currier language and per section
    (ZL3b; >= 1,500 tokens), plus IT2a as a transcription check;
(b) pooled independence with the permutation done within section / within language
    (a mixture of per-section dice would otherwise fake dependence);
(c) between-word independence: slot i of word t vs slot j of word t+1 in the same
    line; null = words shuffled within their line (dice rolled per word give 0)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
from v14_cycle1 import analyse, get_tokens, v7model
from multiprocessing import Pool

def junction(toks, K, nperm=200, seed=0):
    rng = np.random.default_rng(seed)
    C = code_cols(toks, K)
    line = np.array([t['line'] for t in toks])
    n = len(toks)
    a = np.arange(n - 1); same = line[a] == line[a + 1]
    A = a[same]; B = A + 1
    res = {}
    def mis(order):
        Cp = C[:, order]
        return {(i, j): mi(Cp[i][A], Cp[j][B]) for i in range(K) for j in range(K)}
    real = mis(np.arange(n))
    nulls = []
    for _ in range(nperm):
        order = perm_within(np.arange(n), line, rng)
        nulls.append(mis(order))
    for key in real:
        nl = np.array([x[key] for x in nulls])
        res[key] = {'mi': real[key], 'null_mean': float(nl.mean()), 'z': float((real[key] - nl.mean()) / (nl.std() + 1e-12)),
                    'p': float((1 + np.sum(nl >= real[key])) / (nperm + 1))}
    return res

def job(spec):
    name, kind, val = spec
    if load('c2_' + name) is not None: return name
    t = time.time()
    if kind == 'it2a':
        toks = parse(voy_lines('IT2a'), v7model('voynich_IT2a_K4'))
        r = analyse(toks, 4, name, nnull=20)
    elif kind in ('lang', 'illus'):
        toks = parse(voy_lines('ZL3b'), v7model('voynich_ZL3b_K4'))
        toks = [x for x in toks if x[kind] == val]
        r = analyse(toks, 4, name, nnull=20)
    elif kind == 'pooled':
        toks = parse(voy_lines('ZL3b'), v7model('voynich_ZL3b_K4'))
        if val == 'lang': toks = [x for x in toks if x['lang'] in ('A', 'B')]
        r = {'label': name, 'n': len(toks), 'slots': [],
             'indep': independence(toks, 4, 200, strata=[x[val] for x in toks])}
    elif kind == 'junction':
        toks, m, truth = get_tokens(val)
        r = {'label': name, 'n': len(toks), 'slots': [], 'junction': junction(toks, 4)}
    save('c2_' + name, r)
    print(name, 'done', round(time.time() - t), flush=True)
    return name

SPECS = ([('lang_A', 'lang', 'A'), ('lang_B', 'lang', 'B'), ('IT2a', 'it2a', None),
          ('pooled_by_illus', 'pooled', 'illus'), ('pooled_by_lang', 'pooled', 'lang')] +
         [('illus_' + s, 'illus', s) for s in 'HSBP'] +
         [('junc_' + c, 'junction', c) for c in ('voynich_ZL3b', 'planted_mixed', 'latin_verbose', 'italian_verbose', 'voynich_shuffled')])

if __name__ == '__main__':
    with Pool(2) as p:
        list(p.imap_unordered(job, SPECS))
