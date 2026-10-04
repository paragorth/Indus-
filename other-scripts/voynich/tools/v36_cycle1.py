"""v36 cycle 1: do the v30 rewrite rules target feature-defined glyph classes?
Existing v30 guided-search rule sets (g3_*: 30 rules, 2 seeds) for Voynich pairs and the real
calibration pairs, scored with COH / CTX / SUB against frequency-stratified random glyph classes.
Usage: python3 v36_cycle1.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
from v30_ladder import PAIRS
import v25_shapes as S

VOY = ['V_AB', 'S_ABherb', 'V_BA', 'V_B2B3', 'N_AA', 'N_BB']
REAL = ['L_BavAlem', 'S_BavAlem', 'R_AlemBav', 'L_BavBav', 'L_ComCom', 'L_LatIta', 'S_LatIta', 'R_ItaLat',
        'L_CzReform', 'R_CzReform', 'U_GerIta', 'K_Swap3', 'K_Sub']

_F = {}


def feat_for(name, kind):
    key = (name, kind)
    if key in _F:
        return _F[key]
    C = L.corpora30()
    xc, yc = PAIRS[name][:2]
    script = 'voynich' if kind.startswith('v') else 'latin'
    tok = L.tok_for(script)
    types = L.counts_of(C[xc]) + L.counts_of(C[yc])
    fr = L.glyph_freq(types, tok)
    if script == 'voynich':
        alph = [g for g in S.VOYNICH if fr[g] >= 20]
    else:
        alph = [g for g, c in fr.most_common() if c >= 20]
    F = L.Feat(kind, alph, fr)
    _F[key] = F
    return F


def analyse(name, seed, kind, k):
    f = os.path.join(L.CK30, f'g3_{name}_s{seed}.json')
    if not os.path.exists(f):
        return None
    rules, d = L.load_rules(f, k)
    C = L.corpora30()
    xc = PAIRS[name][0]
    script = 'voynich' if kind.startswith('v') else 'latin'
    types = L.counts_of(C[xc])
    eff = [L.rule_effect(r, types, L.tok_for(script)) for r in rules]
    eff = [e for e in eff if e['mass'] > 0]
    F = feat_for(name, kind)
    st = L.stats(eff, F, nperm=2000, seed=seed)
    return st


def main():
    out = {}
    for name in VOY + REAL:
        kinds = ['vhand', 'vimg'] if name in VOY else ['phon']
        for kind in kinds:
            for seed in (0, 1):
                for k in (10, 30):
                    st = analyse(name, seed, kind, k)
                    if st is None:
                        continue
                    out[f'{name}|{kind}|s{seed}|k{k}'] = st
                    print(name, kind, seed, k, ' '.join(f"{m} {st[m]['obs']:.3f} z{st[m]['z']:+.2f}" for m in ('COH', 'CTX', 'SUB')),
                          f"edge {st['edge']:.2f} nT {st['n_targets']}", flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c1.json'), 'w'), default=float, indent=0)


if __name__ == '__main__':
    main()
