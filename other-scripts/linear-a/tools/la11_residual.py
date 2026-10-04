#!/usr/bin/env python3
"""LA-11 residual test: is LA's target profile more than absorbency?
Regress LA's z profile over targets on the mean Markov-control profile (absorbency of random structure) and the mean
hidden-LB profile (absorbency of a foreign-spelled real syllabary); report residual tops and the Afro-Asiatic
residual mean against 5,000 random target subsets of the same size. usage: la11_residual.py TAG [INPUT ...]"""
import sys, statistics as st
import numpy as np
from scipy.stats import spearmanr
import la11_report as R

def run(tag, names=('LA',)):
    B, _ = R.load(tag); targets = sorted({t for X in B for t in B[X]})
    G = {}
    for X in B:
        for T in targets:
            d = B[X].get(T, {}); sh = [v for k, v in d.items() if k > 0]
            if 0 in d and sh: G[X, T] = st.mean(sh) - d[0]
    langX = [X for X in B if X.startswith('X:')]
    def z(X, T):
        ref = [G[Y, T] for Y in langX if Y != X and R.famgroup(R.fam(Y[2:])) != R.famgroup(R.fam(T))]
        return (G[X, T] - st.mean(ref)) / st.pstdev(ref)
    mk = np.array([np.mean([z(X, T) for X in B if X.startswith('MK')]) for T in targets])
    lb = np.array([np.mean([z(f'LB{i}', T) for i in range(3)]) for T in targets])
    aa = [i for i, T in enumerate(targets) if R.famgroup(R.fam(T)) == 'AfroAsiatic']
    for name in names:
        if name not in B: continue
        la = np.array([z(name, T) for T in targets])
        A = np.c_[np.ones(len(targets)), mk, lb]; coef, *_ = np.linalg.lstsq(A, la, rcond=None); res = la - A @ coef
        o = np.argsort(-res); rng = np.random.default_rng(1); r_aa = res[aa].mean()
        p = (1 + sum(res[rng.permutation(len(res))[:len(aa)]].mean() >= r_aa for _ in range(5000))) / 5001
        print(f'{tag} {name}: rho(MK)={spearmanr(la, mk)[0]:.2f} rho(LB)={spearmanr(la, lb)[0]:.2f}; top z ' +
              ', '.join(f'{targets[i]} {la[i]:+.2f}' for i in np.argsort(-la)[:4]) + '; residual top ' +
              ', '.join(f'{targets[i]} {res[i]:+.2f}' for i in o[:5]) + '; bottom ' + ', '.join(f'{targets[i]} {res[i]:+.2f}' for i in o[-3:]) +
              f'; AA residual {r_aa:+.2f} p={p:.3f} (AA: ' + ', '.join(f'{targets[i]} {res[i]:+.2f}' for i in aa) + ')')

if __name__ == '__main__':
    run(sys.argv[1], tuple(sys.argv[2:]) or ('LA',))
