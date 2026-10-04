"""v36 cycle 2: score the rules the v30 search learned on PLANTED changes (v36_plant.py).
Also: plant recovery (share of planted source glyphs among the targets of the top-10 rules).
Usage: python3 v36_cycle2.py
"""
import os, sys, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v36_plant as P
from v36_fitnull import feat


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(L.CK, 'plant_*.json'))):
        d = json.load(open(f)); name = d['name']
        sc = name[0]
        script = {'P': 'voynich', 'H': 'hangul', 'G': 'latin'}[sc]
        X, Y, mp = P.make(name)
        xw = [w for p in X for w in p]
        types = collections.Counter(xw)
        tok = L.tok_for('voynich' if script == 'voynich' else 'latin')
        kinds = {'voynich': ['vhand', 'vimg'], 'hangul': ['hangul'], 'latin': ['phon']}[script]
        planted = set(mp) if mp else ({'b', 'd', 'g', 'p', 't', 'k'} if name == 'G_FEAT' else set())
        for k in (10, 30):
            rules = [tuple(r) for r in d['path'][-1]['rules']][:k]
            eff = [e for e in (L.rule_effect(r, types, tok) for r in rules) if e['mass'] > 0]
            tg = collections.Counter()
            for e in eff:
                s = sum(e['targets'].values())
                for g, c in e['targets'].items():
                    tg[g] += c / s
            rec = sum(c for g, c in tg.items() if g in planted) / max(sum(tg.values()), 1e-9)
            for kind in kinds:
                F = feat(script, xw, kind)
                st = L.stats(eff, F, nperm=2000)
                st['recovery'] = rec
                out[f'{name}|{kind}|k{k}'] = st
                print(f'{name:7s} {kind:6s} k{k} rec {rec:.2f} ' + ' '.join(f"{m} {st[m]['obs']:.3f} z{st[m]['z']:+.2f}" for m in ('COH', 'CTX', 'SUB')), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c2.json'), 'w'), default=float, indent=0)


if __name__ == '__main__':
    main()
