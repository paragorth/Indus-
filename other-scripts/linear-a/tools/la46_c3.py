#!/usr/bin/env python3
"""LA-46 cycle 3: the cast test with sides merged into tablets and topic forgers.
Changes vs cycle 1: (1) sides a/b of one tablet are merged into one document (kills the side-duplication
artefact); (2) 300 random architectures with tablet clusters up to K = 32 (topic forgers that nearly memorise
tablet vocabularies); (3) residues scored against the ensemble mean (calibrated by the null corpora) AND against
the strongest group; (4) CO residues are joined into casts (connected components) and every cast is re-tested
leave-one-tablet-out on held-out tablets vs forgeries.
usage: la46_c3.py NAME [NAME ...]  (LA, W1, W2, SH1, PL1, PL2, LB1, LB2)"""
import sys, json, time, random
import la46_common as C
from la46_common import *

N_ARCH = int(os.environ.get('LA46_NARCH', '300'))


def merged_la():
    L = la_docs(1)
    by = collections.OrderedDict()
    for d in L:
        base = re.sub(r'[ab]$', '', d['id'])
        if base in by:
            by[base]['toks'] = by[base]['toks'] + [('NL',)] + d['toks']
        else:
            by[base] = {'id': base, 'site': d['site'], 'toks': list(d['toks'])}
    return [d for d in by.values() if sum(x[0] != 'NL' for x in d['toks']) >= 4]


def build(name):
    L = merged_la()
    if name == 'LA':
        return L, None
    if name == 'W1':
        return world(L, random.Random(seed('c3w1'))), None
    if name == 'W2':
        a = {'k': 3, 'fmin': 1, 'nmode': 'bin', 'lmode': 'exact', 'alpha': 1.0, 'lineidx': False,
             'site': False, 'K': 16, 'template': False, 'nl_reset': False}
        return world(L, random.Random(seed('c3w2')), a), None
    if name.startswith('SH'):
        return shuffle_corpus(L, random.Random(seed('c3' + name))), None
    if name.startswith('PL'):
        W = world(L, random.Random(seed('c3w' + name)))
        P, truth = plant(W, random.Random(seed('c3plant' + name)))
        return P, sorted(truth)
    if name.startswith('LB'):
        return sample_like(lb_docs_all(), ntok(L), random.Random(seed('c3' + name))), None
    raise ValueError(name)


def arch3(rng):
    a = random_arch(rng)
    if rng.random() < 0.5:
        a['K'] = rng.choice([8, 12, 16, 24, 32])
    return a


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, 'c3_%s.json' % name)
        if os.path.exists(fn):
            continue
        t0 = time.time()
        docs, truth = build(name)
        rng = random.Random(seed('la46-c3-archs'))
        archs = [arch3(rng) for _ in range(N_ARCH)]
        logf = open(os.path.join(CK, 'c3_%s.log' % name), 'w')

        def log(s):
            logf.write(s + '\n'); logf.flush()
        log('%s docs %d tokens %d' % (name, len(docs), ntok(docs)))
        real, gE, n, _ = contest(docs, N_ARCH, random.Random(seed('c3contest' + name)), reps=2,
                                 arch_list=archs, log=log)
        gn = C.contest.last_gn
        rmax = residues(real, gE)
        rmean = residues_mean(real, gE, gn)
        rmin = [(k, R, M, m, pois_sf(R, m)) for k, R, M, m, p in rmax]
        out = {'name': name, 'n_docs': n, 'n_tok': ntok(docs), 'truth': truth,
               'res': [[list(k), R, M, m, pmax, pm[2], pm[4], pmn[4]]
                       for (k, R, M, m, pmax), pm, pmn in zip(rmax, rmean, rmin)],
               'secs': time.time() - t0}
        json.dump(out, open(fn, 'w'))
        json.dump(docs, open(os.path.join(CK, 'c3_%s_docs.json' % name), 'w'))
        log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
