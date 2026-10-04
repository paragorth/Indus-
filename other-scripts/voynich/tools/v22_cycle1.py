"""v22 cycle 1: one hidden program for every paragraph? Held-out likelihood of an S-stage left-to-right program
(elastic alignment learned by EM from random segmentations) against rigid position models, on ZL3b paragraphs,
with line-shuffled, head-kept line-shuffled and lp-matched word-shuffled nulls, Markov resynthesis (v21 F3),
planted elastic programs, and Latin / Italian herbal paragraphs as positive controls."""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import v22_lib as L
import v21_lib as V21

SL = [2, 3, 4, 6, 9]
R = 8


def corpus(kind):
    if kind in ('ZL3b', 'IT2a'): return L.voynich_pages(kind)
    return L.herbal_pages(kind)


def task(spec):
    name, kind, level, mode, seed = spec
    out_f = f'c1_{name}.json'
    got = L.load(out_f)
    if got: return got
    t0 = time.time()
    C = corpus(kind)
    U = L.units(C, level)
    coder = L.Coder(U)
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    extra = {}
    if mode == 'real': U2 = U
    elif mode == 'lineshuf': U2 = L.shuffle_lines(U, rng)
    elif mode == 'lineshuf_head': U2 = L.shuffle_lines(U, rng, keep_head=True)
    elif mode == 'wordshuf': U2 = L.shuffle_words_lp(U, rng)
    elif mode == 'markov':
        random.seed(seed)
        F = V21.Forger(C, scope='sec', pos=True, name='F3')
        Cf = F.forge(C, rng)
        U2 = L.units(Cf, level)
    elif mode.startswith('plant') or mode.startswith('sham'):
        rho = float(mode[5:] if mode.startswith('plant') else mode[4:])
        U2, truth = L.plant_program(U, nrng, S=4, rho=rho, tilt=1.0 if mode.startswith('plant') else 0.0)
    codes = L.code_all(U2, coder)
    fold = L.folds_by_unit(U2, 5, 1000 + seed)
    res = L.evaluate(codes, coder.sizes, fold, SL, R, seed)
    if mode.startswith('plant'):
        seq = L.Seq(codes)
        m = L.fit_hmm(seq, coder.sizes, 4, 3 * R, np.random.default_rng(seed + 7))
        paths = L.viterbi_paths(m, seq, 4)
        res['ari4'] = L.ari(np.concatenate(truth), np.concatenate(paths))
    res.update(name=name, kind=kind, level=level, mode=mode, seed=seed, nunits=len(U2), secs=time.time() - t0)
    L.save(out_f, res)
    print(name, f"{res['delta']:+.1f}", res['best_hmm_k'], res['best_rigid_k'], f"{res['secs']:.0f}s", flush=True)
    return res


def specs():
    S = [('ZL_real', 'ZL3b', 'para', 'real', 0), ('LA_real', 'LA', 'para', 'real', 0), ('IT_real', 'IT', 'para', 'real', 0)]
    for i in range(3):
        S.append((f'ZL_lineshuf_{i}', 'ZL3b', 'para', 'lineshuf', 100 + i))
        S.append((f'ZL_wordshuf_{i}', 'ZL3b', 'para', 'wordshuf', 200 + i))
    for rho, sd in (('0.25', 50), ('0.5', 70)):
        S.append((f'ZL_plant{rho}_0', 'ZL3b', 'para', f'plant{rho}', sd))
        S.append((f'ZL_sham{rho}_0', 'ZL3b', 'para', f'sham{rho}', sd))
    for k in ('LA', 'IT'):
        for i in range(2):
            S.append((f'{k}_lineshuf_{i}', k, 'para', 'lineshuf', 500 + i))
            S.append((f'{k}_wordshuf_{i}', k, 'para', 'wordshuf', 700 + i))
    for i in range(2):
        S.append((f'ZL_lineshufhead_{i}', 'ZL3b', 'para', 'lineshuf_head', 300 + i))
        S.append((f'ZL_markov_{i}', 'ZL3b', 'para', 'markov', 400 + i))
    return S


if __name__ == '__main__':
    with Pool(2) as P:
        res = P.map(task, specs(), chunksize=1)
    L.save('c1_all.json', res)
