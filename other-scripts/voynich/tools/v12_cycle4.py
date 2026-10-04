"""v12 cycle 4: power repair for the latent weave model + 3-way weaves.
Cycle 2 showed EM from random starts fails to find strict alternation when both streams look alike
(LI-alt, VV-alt). Here every model is fitted from 3 starts -- random, parity (odd/even), first-glyph hash --
and the best TRAINING likelihood is kept (no peeking at the test fold); W2 and H2 then get the same starts.
k = 3 (W3 vs H3) is added for Voynich ZL, the LF-hmm weave and verbose Latin.
Held-out bits/token, fold = folio parity."""
import sys, os, json, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V, v12_hmm as H

ITERS = 10
STARTS = [('random', 1), ('parity', 1), ('firstglyph', 1)]


def fit_best(tr, k, kind):
    best = None; tag = None
    for init, sd in STARTS:
        m = H.StreamModel(k, kind, seed=sd); m.fit(tr, iters=ITERS, init=init)
        if best is None or m.train_bits > best.train_bits: best, tag = m, init
    return best, tag


def job(arg):
    name, fold, k = arg
    ck = os.path.join(V.CK, f'c4_{name}_{fold}_k{k}.json')
    if os.path.exists(ck): return json.load(open(ck))
    t0 = time.time()
    C = V.corpus(name); tr, te = V.split(C, fold)
    out = {'name': name, 'fold': fold, 'k': k}
    for kind in ('H', 'W'):
        m, tag = fit_best(tr, k, kind)
        out[kind] = m.score(te); out[kind + '_start'] = tag; out[kind + '_A'] = m.A; out[kind + '_train'] = m.train_bits
    out['sec'] = time.time() - t0
    json.dump(out, open(ck, 'w'), indent=1)
    print(name, fold, f'k{k}', 'H', round(out['H'], 3), out['H_start'], 'W', round(out['W'], 3), out['W_start'],
          'W-H', round(out['W'] - out['H'], 3), 'Wstay', [round(out['W_A'][s][s], 2) for s in range(k)], round(out['sec']), flush=True)
    return out


if __name__ == '__main__':
    jobs = [(n, f, 2) for n in ['Weave-LI-alt', 'Weave-VV-alt', 'Voynich-ZL', 'Voynich-IT', 'vLatin', 'vItalian', 'Weave-LF-hmm']
            for f in (0, 1)]
    jobs += [(n, f, 3) for n in ['Voynich-ZL', 'Weave-LF-hmm', 'vLatin'] for f in (0, 1)]
    with Pool(2) as p:
        R = p.map(job, jobs, chunksize=1)
    json.dump(R, open(os.path.join(V.CK, 'cycle4.json'), 'w'), indent=1)
    print('done all')
