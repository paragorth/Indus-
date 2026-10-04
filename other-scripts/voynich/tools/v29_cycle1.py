"""v29 cycle 1: blind feature-spread search on Voynich (ZL, IT), harmony controls and planted harmony.
usage: python3 v29_cycle1.py NAME [NAME ...]   -> data/v29_ckpt/c1_NAME.pkl"""
import sys, os, pickle, time
import numpy as np
import v29_lib as L, v25_shapes as S


def corpus(name):
    if name in ('ZL', 'IT'):
        return L.voynich_lines('ZL3b' if name == 'ZL' else 'IT2a'), S.VOYNICH
    if name in ('ZL_A', 'ZL_B'):
        return L.voynich_lines('ZL3b', lang=name[-1]), S.VOYNICH
    if name.startswith('plant'):
        base = L.voynich_lines('ZL3b')
        p = dict(plant7=0.7, plant3=0.3, plantJ=0.0)[name]
        j = 0.5 if name == 'plantJ' else 0.0
        return L.plant(base, {'k', 't', 'f', 'p'}, {'t', 'p'}, p=p, seed=7, junction=j), S.VOYNICH
    if name.startswith('copy'):
        import v29_copygen
        rate, ed = dict(copy5=(0.5, 1.0), copy8=(0.8, 1.5))[name]
        return v29_copygen.gen(L.voynich_lines('ZL3b'), rate=rate, edits=ed, seed=3), S.VOYNICH
    if name == 'ko':
        return L.hangul_lines(), S.HANGUL
    if name in ('tr', 'hu', 'fi'):
        lines = L.leipzig(dict(tr='tur', hu='hun', fi='fin')[name], name)
    else:
        lines = L.plain_lines(name)
    letters = {c for l in lines for w in l for c in w}
    return lines, L.phon_shapes(name, letters)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        t = time.time()
        lines, sh = corpus(name)
        voy = sh is S.VOYNICH
        res = L.run(lines, sh, R=10, n_random=1500 if voy else 800, n_class=400 if voy else 200, seed=1)
        res['feats'] = [dict(m=f['m'], v=f['v'], name=f['name'], kind=f['kind']) for f in res['feats']]
        res['maxnull'] = L.maxnull(res)
        pickle.dump(res, open(os.path.join(L.CK, f'c1_{name}.pkl'), 'wb'))
        print(name, len(res['feats']), 'feats', res['ntok'], 'tok', f'{time.time()-t:.0f}s', flush=True)
