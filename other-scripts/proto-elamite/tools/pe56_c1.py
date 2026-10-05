#!/usr/bin/env python3
"""pe56 cycle 1: the inside-out forger contest on PE and its controls.
usage: pe56_c1.py NAME [NAME ...]   names: PE W1 W2 SH1 SH2 PL1 PL2 PC1 PC2 UR1 UR2
Writes data/pe56_ckpt/c1_NAME.json: every tested relation with R, ensemble-mean E, max-group E, p_mean, p_max."""
import sys, json, time, random
from pe56_common import *

N_ARCH = int(os.environ.get('PE56_NARCH', '150'))
REPS = int(os.environ.get('PE56_REPS', '2'))


def build(name):
    P = pe_docs()
    if name == 'PE':
        return P, None
    if name in W_ARCHS:
        return world(P, random.Random(seed('pe56' + name)), W_ARCHS[name]), None
    if name.startswith('SH'):
        return shuffle_corpus(P, random.Random(seed('pe56' + name))), None
    if name.startswith('PL'):
        w = 'W1' if name == 'PL1' else 'W2'
        W = world(P, random.Random(seed('pe56pl' + name)), W_ARCHS[w])
        return plant(W, random.Random(seed('pe56plant' + name)))
    if name.startswith('PC'):
        return sample_like(pc_docs(), ntok(P), random.Random(seed('pe56' + name))), None
    if name.startswith('UR'):
        return sample_like(ur3_docs(), ntok(P), random.Random(seed('pe56' + name))), None
    raise ValueError(name)


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, 'c1_%s.json' % name)
        if os.path.exists(fn):
            continue
        t0 = time.time()
        docs, truth = build(name)
        freq = freq_set(docs)
        rng = random.Random(seed('pe56-archs'))
        archs = [random_arch(rng) for _ in range(N_ARCH)]
        logf = open(os.path.join(CK, 'c1_%s.log' % name), 'w')

        def log(s):
            logf.write(s + '\n'); logf.flush()
        log('%s docs %d tokens %d freq %d' % (name, len(docs), ntok(docs), len(freq)))
        real, gE, gn = contest(docs, archs, random.Random(seed('pe56c1' + name)), freq, reps=REPS, log=log)
        res = residues(real, gE, gn)
        out = {'name': name, 'n_docs': len(docs), 'n_tok': ntok(docs), 'truth': truth, 'gn': gn,
               'res': [[list(k), R, E, M, pm, px] for k, R, E, M, pm, px in res], 'secs': time.time() - t0}
        json.dump(out, open(fn, 'w'))
        json.dump(docs, open(os.path.join(CK, 'c1_%s_docs.json' % name), 'w'))
        log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
