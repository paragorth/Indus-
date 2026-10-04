"""v28 cycle 3b: planted free allographs + twin-pair drop test (run after v28_run on the planted corpora)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_lib as X, v28_pos as P, v28_run as R
NAMES = ['latinprint+free6', 'latinprint+free6i', 'voy:ZL3b:v25', 'voy:IT2a:v25', 'voy:ZL3b:split', 'catmus:allo', 'cast:Val_S-Hand_C:minim', 'am']
if __name__ == '__main__':
    out = X.load('c3_twins.pkl') or {}
    for n in NAMES:
        if n not in out:
            R.run(n)
            out[n] = P.twins(n)
            X.save('c3_twins.pkl', out)
        for m, s, top in out[n]:
            print('TWIN', n, m, '|', s, '| top twins', top, flush=True)
