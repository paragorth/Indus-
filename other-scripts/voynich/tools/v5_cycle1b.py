"""v5 cycle 1b -- is the paragraph's CLOSING glyph special, or does any line-final glyph track
paragraph make-up?  T4 rerun with label = final glyph of the unit's last line vs of its second line
vs of its penultimate line; strata = section x Currier language (Voynich), mode (chant)."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, v5_cycle1 as c1, vlib

def relabel(units, k):
    out = []
    for u in units:
        if len(u['lines']) < 3:
            continue
        L = u['lines']
        tgt = {'close': L[-1], 'second': L[1], 'penult': L[-2]}[k]
        # put the target line last so t4('close') reads its final; keep all lines for make-up
        rest = [x for x in L if x is not tgt]
        out.append(dict(u, lines=rest + [tgt]))
    return out

res = {}
C = {'V-ZL': vc.voynich('ZL3b'), 'V-IT': vc.voynich('IT2a'), 'chant': vc.chant(max_units=1500)}
for name, units in C.items():
    if name.startswith('V'):
        units = [dict(u, stratum=str(u['stratum']) + str(u['lang'])) for u in units]
    for k in ['close', 'second', 'penult']:
        r = c1.t4(relabel(units, k), 'close', stratified=True)
        res[f'{name}:{k}'] = r
        print(name, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
vlib.save('v5_cycle1b', res)
