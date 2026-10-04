"""Summarise cycle-1 checkpoints: per corpus x level, under-dispersion hits after family-wise correction."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L

def summ(name):
    d = np.load(os.path.join(L.CK, 'c1_%s.npz' % name), allow_pickle=True)
    names = list(d['names']); out = []
    for lev in L.LEVELS:
        g = lambda k: d[lev + '__' + k]
        R = g('RD'); z = g('zD')
        lowD = np.where(g('fwD_low') < 0.05)[0]; highD = np.where(g('fwD_high') < 0.05)[0]
        lowM = np.where(g('fwMax_low') < 0.05)[0]; lowZ = np.where(g('fwZero_low') < 0.05)[0]
        o = np.argsort(z)[:3]
        top = ', '.join('%s R%.2f z%.1f fw%.2f' % (names[i], R[i], z[i], g('fwD_low')[i]) for i in o)
        cm = np.argsort(g('zMax'))[:2]
        topm = ', '.join('%s z%.1f fw%.2f' % (names[i], g('zMax')[i], g('fwMax_low')[i]) for i in cm)
        out.append(dict(lev=lev, F=len(names), medR=float(np.median(R)), shareRlt1=float((R < 1).mean()),
                        nlowD=len(lowD), nhighD=len(highD), nlowMax=len(lowM), nlowZero=len(lowZ), top=top, topmax=topm,
                        lowD=[names[i] for i in lowD][:12], lowM=[names[i] for i in lowM][:12], lowZ=[names[i] for i in lowZ][:12],
                        minz=float(z.min()), nullmin=float(np.median(g('nullminz')))))
    return out

if __name__ == '__main__':
    for f in sorted(glob.glob(os.path.join(L.CK, 'c1_*.npz'))):
        name = os.path.basename(f)[3:-4]
        for r in summ(name):
            print(name, r['lev'], 'F=%d medR=%.2f R<1=%.2f lowD=%d highD=%d lowMax=%d lowZero=%d minz=%.1f (null min med %.1f)' % (
                r['F'], r['medR'], r['shareRlt1'], r['nlowD'], r['nhighD'], r['nlowMax'], r['nlowZero'], r['minz'], r['nullmin']))
            print('    top:', r['top']); print('    ceil:', r['topmax'])
            if r['lowD'] or r['lowM'] or r['lowZ']: print('    lowD', r['lowD'], 'lowM', r['lowM'], 'lowZ', r['lowZ'])
