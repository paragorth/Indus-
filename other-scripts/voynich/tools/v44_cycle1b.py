"""v44 cycle 1b: structure-keeping planted control. The 1.1-1.6 plants re-ordered EVERY word, which wipes the
real sequential structure and shrinks the null (sd 0.2%); the real texts have a null sd of 2-5%. Here a fraction
`mix` of word tokens is replaced by a near-minimum-cost ordering (beta 4) and the rest keep their real order.
Detection = saving above the 97.5th percentile of the randomised-cost-model null. Second control: the same plants
scored with 200 randomised models on average must look like the unplanted text (the effect lives in the true model).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v44_lib as L

FN = 'v44_cycle1.txt'


def main():
    out = {}
    i = 30
    for nm, w, sig in (('Voynich ZL', L.words_of(L.voynich('ZL3b')), L.VOYNICH_HAND),
                       ('ms Latin', L.words_of(L.v31_corpus('L_msI_Lat')), L.CURSIVA),
                       ('CS raw', L.words_of(L.cs_lines()), L.CS_HAND)):
        alph = L.alphabet(w, sig); S = L.sig_array(sig, alph); c = L.cost_matrix(S)
        B0, Q0, _ = L.pair_mats(w, alph)
        rnd = np.random.default_rng(5); Rm = [L.cost_matrix(L.random_sigs(len(alph), rnd)) for _ in range(200)]
        base_r = np.mean([L.saving(B0, Q0, r) for r in Rm])
        res = []
        for mix in (0.1, 0.25, 0.5, 1.0):
            pw = L.plant(w, alph, c, 4.0, mix=mix, seed=1)
            r = L.summarize(pw, sig, 1000)
            B, Q, _ = L.pair_mats(pw, alph)
            mr = np.mean([L.saving(B, Q, x) for x in Rm])
            res.append(f"mix {mix}: S_shuf {100*r['shuf']['S']:+.1f}% pct {r['shuf']['pct']:.3f}, S_edge pct {r['edge']['pct']:.3f}; mean S under 200 random models {100*mr:+.1f}% (unplanted {100*base_r:+.1f}%)")
            out[f'{nm}_{mix}'] = dict(r=r, rand_mean=mr, rand_base=base_r)
            print(nm, res[-1], flush=True)
        i += 1
        L.row(FN, f'V-44.1.{i}', f'Control: structure-keeping plant in {nm}: a fraction mix of word tokens set to a near-minimum-cost ordering (beta 4, 24 candidates), rest real; true-model saving vs 1000 permuted models; also mean saving under 200 random signature models',
              '; '.join(res), 'see cycle verdict')
    L.save('cycle1b.json', out)


if __name__ == '__main__':
    main()
