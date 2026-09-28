"""Pairwise glyph similarity of all used Wells signs (symmetric dilated overlap)."""
import sys,json,collections,numpy as np
sys.path.insert(0,'tools')
from sign_glyphs import render
from scipy.ndimage import binary_dilation
def load():
    m=json.load(open('data/derived/merged-corpus-reading-order.json'))
    return m if isinstance(m,list) else list(m.values())[0]
def glyph_sim(signs):
    G=[];D=[]
    for w in signs:
        g=render(w); G.append(g.ravel()); D.append(binary_dilation(g,iterations=2).ravel())
    G=np.array(G,dtype=np.float32); D=np.array(D,dtype=np.float32)
    inter=G@D.T                      # |A ∩ dil(B)|
    n=G.sum(1)
    S=(inter+inter.T)/(n[:,None]+n[None,:])
    return S
if __name__=='__main__':
    m=load(); freq=collections.Counter(x for r in m for x in (r.get('seq') or []) if x!=999)
    signs=sorted(freq); S=glyph_sim(signs)
    np.save('data/derived/glyph_sim.npy',S); json.dump(signs,open('data/derived/glyph_sim_signs.json','w'))
    iu=np.triu_indices(len(signs),1); v=S[iu]
    print('pairs',len(v),'quantiles',np.quantile(v,[.5,.9,.99,.999]).round(3))
    top=np.argsort(-v)[:60]
    for k in top: i,j=iu[0][k],iu[1][k]; print(f'W{signs[i]}~W{signs[j]} {v[k]:.3f} f={freq[signs[i]]},{freq[signs[j]]}')
