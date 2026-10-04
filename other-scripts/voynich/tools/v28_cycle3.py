"""v28 cycle 3: (a) SIZE-MATCHED Mantel (every rival restricted to its 23 most frequent units, as many as the Voynich
v25 inventory; behaviour from the full set; image combo averaged over fonts); (b) positional-variant / allograph test
(v28_pos.test) on Voynich and on the medieval sets with known allographs.  Results cached in v28_ckpt/c3_*.pkl."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_lib as X, v28_pos as P

SM = ['voy:ZL3b:v25', 'voy:IT2a:v25', 'voy:ZL3b:split', 'voy:ZL3b:istr', 'latinprint', 'latinprint+bpe25', 'catmus:allo', 'catmus:plain',
      'cast:pool:allo', 'cast:Sev_Z:allo', 'cast:Sev_Z:graph', 'cast:Sev_Z:minim', 'cast:Mad_A-Hand_A:allo', 'cast:Mad_A-Hand_A:graph', 'cast:Mad_A-Hand_A:minim',
      'cast:Val_S-Hand_C:allo', 'cast:Val_S-Hand_C:graph', 'cast:Val_S-Hand_C:minim', 'cast:Val_S-Hand_D:allo', 'cast:Val_S-Hand_D:graph', 'cast:Val_S-Hand_D:minim',
      'enhg:allo', 'am', 'cu', 'glag', 'ru', 'hy']
POS = ['voy:ZL3b:v25', 'voy:IT2a:v25', 'voy:ZL3b:split', 'cast:pool:allo', 'cast:Sev_Z:allo', 'cast:Mad_A-Hand_A:allo', 'cast:Val_S-Hand_C:allo',
       'cast:Val_S-Hand_D:allo', 'enhg:allo', 'catmus:allo']
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('sm', 'both'):
        out = X.load('c3_sm.pkl') or {}
        for n in SM:
            if n in out or X.load(f'beh_{n}.pkl') is None:
                continue
            out[n] = P.size_matched(n)
            X.save('c3_sm.pkl', out)
            print('SM', n, {m: (round(r, 3), round(p, 4)) for m, (r, p) in out[n][0].items()}, flush=True)
    if which in ('pos', 'both'):
        out = X.load('c3_pos.pkl') or {}
        for n in POS:
            if n in out:
                continue
            out[n] = P.test(n)
            X.save('c3_pos.pkl', out)
            o, top, sk = out[n]
            for m, s in o:
                print('POS', n, sk, m, '|', s, flush=True)
            print('POS', n, 'allograph-like:', top, flush=True)
