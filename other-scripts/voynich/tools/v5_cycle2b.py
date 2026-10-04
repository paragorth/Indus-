"""v5 cycle 2b -- is the Voynich lag-6/7 symbol-repeat excess (M1) a metre or neighbour-word similarity?
Split symbol pairs at lag k into: same word, ADJACENT words, words >= 2 apart. Excess over within-line word
shuffle (40) for each class separately. A metre is blind to word boundaries; neighbour similarity lives in the
adjacent class only. Also: aligned pairs (same within-word position in adjacent words) vs misaligned."""
import sys, os, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib

LAGS = range(1, 17)

def rates(lines):
    num = np.zeros((4, 17)); den = np.zeros((4, 17))
    for L in lines:
        s, wi, pos = [], [], []
        for j, w in enumerate(L):
            for p, x in enumerate(w):
                s.append(x); wi.append(j); pos.append(p)
        n = len(s)
        for k in LAGS:
            for i in range(n - k):
                d = wi[i + k] - wi[i]
                c = 0 if d == 0 else (1 if d == 1 else 2)
                eq = s[i] == s[i + k]
                num[c, k] += eq; den[c, k] += 1
                if d == 1 and pos[i] == pos[i + k]:
                    num[3, k] += eq; den[3, k] += 1
    return num / np.maximum(den, 1), den

def run(units, R=30, seed=0):
    rng = random.Random(seed)
    lines = [L for u in units for L in u['lines']]
    obs, den = rates(lines)
    nn = np.array([rates([rng.sample(L, len(L)) for L in lines])[0] for _ in range(R)])
    return obs - nn.mean(0), (obs - nn.mean(0)) / (nn.std(0) + 1e-12), den

res = {}
C = {'V-ZL': vc.voynich('ZL3b'), 'V-IT': vc.voynich('IT2a'), 'chant': vc.chant(max_units=600),
     'vs-Italian-Manzoni': vc.verbose_lang('Italian-Manzoni', max_words=15000)}
for name, units in C.items():
    ex, z, den = run(units)
    res[name] = {'excess': ex.tolist(), 'z': z.tolist()}
    print('==', name)
    for c, lab in enumerate(['same word', 'adjacent words', 'words>=2 apart', 'adjacent, same in-word position']):
        print(f"  {lab:32s} " + ' '.join(f"{k}:{ex[c,k]*1000:+.0f}({z[c,k]:+.0f})" for k in range(3, 13)))
vlib.save('v5_cycle2b', res)
