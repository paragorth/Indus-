"""v52 cycle 1: massive random field-schema search on the Voynich and on controls.
Discovery on half A of the pages (documents), the top schemas re-tested on held-out half B.
Corpora: V (ZL3b, all tokens incl. labels), VSHUF (external variables permuted: meta-null),
VMARK (unit-trigram resynthesis conditioned on SEC x POS: conditioned phonotactics, no fields),
PL6 / PL3 (planted field catalogue, signal 0.6 / 0.3), GORILA (real Linear A catalogue ids, opaque),
UNICODE (real Unicode-name abbreviation codes, opaque)."""
import sys, os, json, pickle, random, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v52_lib as L

NS = int(os.environ.get('V52_NS', 2000))
TOP = 30
_C = {}


def build(name):
    if name in _C:
        return _C[name]
    pk = os.path.join(L.CK, f'corpus_{name}.pkl')
    if os.path.exists(pk):
        _C[name] = pickle.load(open(pk, 'rb')); return _C[name]
    rng = np.random.default_rng(7)
    true = None
    if name in ('V', 'VSHUF', 'VMARK', 'PL6', 'PL3'):
        V = L.voynich_corpus('ZL3b')
        if name == 'V':
            C = V
        elif name == 'VSHUF':
            C = L.shuffled_corpus(V, rng)
        elif name == 'VMARK':
            C = L.replace_words(V, L.markov_cell_words(V, rng), 'VMARK')
        else:
            W, true = L.planted_words(V, seed=3, p=0.6 if name == 'PL6' else 0.3)
            C = L.replace_words(V, W, name)
    elif name == 'VI':
        C = L.voynich_corpus('IT2a')
    elif name == 'GORILA':
        C, true = L.gorila_corpus()
    elif name == 'UNICODE':
        C, true = L.unicode_corpus()
    C['true'] = true
    C['nulls'] = L.null_vars(C, np.random.default_rng(11), 3)
    C['A'] = L.page_split(C, 5)
    pickle.dump(C, open(pk, 'wb'))
    _C[name] = C
    return C


def task(args):
    name, i0, i1 = args
    C = build(name)
    A = C['A']
    nullsA = [{k: v for k, v in nd.items()} for nd in C['nulls']]
    out = []
    for i in range(i0, i1):
        r = random.Random(1000003 * i + 17)
        sch = L.random_schema(r, C['top_units'])
        res = L.score(C, sch, nullsA, mask=A, dep=False)
        out.append((i, sch, res['FS'], res['TOT']))
    return name, out


def retest(args):
    name, sch = args
    C = build(name)
    rA = L.score(C, sch, C['nulls'], mask=C['A'], dep=True)
    rB = L.score(C, sch, C['nulls'], mask=~C['A'], dep=True)
    return name, sch, rA, rB


if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['PL6', 'PL3', 'GORILA', 'UNICODE', 'V', 'VSHUF', 'VMARK']
    tag = os.environ.get('V52_TAG', 'c1')
    for n in names:
        build(n)
    chunk = 100
    jobs = [(n, i, min(NS, i + chunk)) for n in names for i in range(0, NS, chunk)]
    t = time.time()
    res = {n: [] for n in names}
    with Pool(2) as P:
        for name, out in P.imap_unordered(task, jobs):
            res[name] += out
        print('search done', round(time.time() - t), flush=True)
        rj = []
        for n in names:
            res[n].sort(key=lambda x: -x[2])
            rj += [(n, x[1]) for x in res[n][:TOP]]
        rt = list(P.imap(retest, rj))
    summ = {n: dict(search=[(x[0], x[2], x[3]) for x in res[n]], top=[]) for n in names}
    for name, sch, rA, rB in rt:
        summ[name]['top'].append(dict(sch=sch, A=rA, B=rB))
    pickle.dump(summ, open(os.path.join(L.CK, f'{tag}_summary.pkl'), 'wb'))
    print('all done', round(time.time() - t), flush=True)
