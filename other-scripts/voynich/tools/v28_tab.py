import sys, numpy as np
sys.path.insert(0,'.')
import v28_lib as X, glob, os
def tab(names):
    out={}
    for n in names:
        r=X.load(f'res_{n}.pkl')
        if r is None: continue
        d={}
        for m in ('ppmi','svd','potts'):
            v=[(rr,p) for (k,mm),(rr,p,p2,pr) in r['summ'].items() if k.startswith('img:') and mm==m]
            vp=[pr for (k,mm),(rr,p,p2,pr) in r['summ'].items() if k.startswith('img:') and mm==m]
            d[m]=(np.mean([a for a,_ in v]), max(b for _,b in v), np.mean(vp))
        h=[(rr,p) for (k,mm),(rr,p,p2,pr) in r['summ'].items() if k=='hand']
        out[n]=(r['n'],d,h)
    return out
if __name__=='__main__':
    names=sys.argv[1:] or sorted(os.path.basename(f)[4:-4] for f in glob.glob(os.path.join(X.CK,'res_*.pkl')))
    for n,(k,d,h) in tab(names).items():
        print(f'{n:28s} n={k:3d} '+'  '.join(f'{m} {d[m][0]:+.2f} (pmax {d[m][1]:.3f}, |pos {d[m][2]:+.2f})' for m in d)+('  hand '+' '.join(f'{a:+.2f}/{b:.4f}' for a,b in h) if h else ''))
