#!/usr/bin/env python3
"""LA-46 cycle 1: run the inside-out contest on one corpus (real or control).
usage: la46_c1.py NAME [NAME ...]   (names: LA, W1, W2, SH1, SH2, PL1, PL2, LB1, LB2, UR1, UR2)
Writes data/la46_ckpt/c1_<NAME>.json (residue table + family summary)."""
import sys, json, time, random, gzip
from la46_common import *

N_ARCH = int(os.environ.get('LA46_NARCH', '200'))
TAG = os.environ.get('LA46_TAG', 'c1')

W2_ARCH = {'k': 3, 'fmin': 1, 'nmode': 'bin', 'lmode': 'exact', 'alpha': 1.0, 'lineidx': False,
           'site': False, 'K': 0, 'template': False, 'nl_reset': False}


def build(name):
    L = la_docs()
    if name == 'LA':
        return L, None
    if name == 'W1':
        return world(L, random.Random(seed('w1'))), None
    if name == 'W2':
        return world(L, random.Random(seed('w2')), W2_ARCH), None
    if name.startswith('SH'):
        return shuffle_corpus(L, random.Random(seed(name))), None
    if name.startswith('PL'):
        W = world(L, random.Random(seed('w' + name)))
        P, truth = plant(W, random.Random(seed('plant' + name)))
        return P, sorted(truth)
    if name.startswith('LB'):
        return sample_like(lb_docs_all(), ntok(L), random.Random(seed(name))), None
    if name.startswith('UR'):
        return sample_like(ur3_docs_all(), ntok(L), random.Random(seed(name))), None
    raise ValueError(name)


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, '%s_%s.json' % (TAG, name))
        if os.path.exists(fn):
            continue
        t0 = time.time()
        docs, truth = build(name)
        rng = random.Random(seed('la46-archs'))
        archs = [random_arch(rng) for _ in range(N_ARCH)]
        logf = open(os.path.join(CK, '%s_%s.log' % (TAG, name)), 'w')

        def log(s):
            logf.write(s + '\n'); logf.flush()
        log('%s docs %d tokens %d' % (name, len(docs), ntok(docs)))
        real, gE, n, _ = contest(docs, N_ARCH, random.Random(seed('contest' + name)), arch_list=archs, log=log)
        res = residues(real, gE)
        out = {'name': name, 'n_docs': n, 'n_tok': ntok(docs), 'truth': truth,
               'summary': summarize(res), 'groups': sorted(gE),
               'res': [[list(k), R, M, m, p] for k, R, M, m, p in res],
               'secs': time.time() - t0}
        json.dump(out, open(fn, 'w'))
        if name in ('LA',) or name.startswith('LB') or name.startswith('UR'):
            json.dump(docs, open(os.path.join(CK, '%s_%s_docs.json' % (TAG, name)), 'w'))
        log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
