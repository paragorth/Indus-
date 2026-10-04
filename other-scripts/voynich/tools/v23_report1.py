"""v23 cycle-1 table: z per signature per corpus, reversible-surrogate calibration, FDR."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L

D = {}
for f in glob.glob(os.path.join(L.CK, 'c1_*.json')):
    d = json.load(open(f)); D[d['name']] = d
cols = ['ZL', 'IT', 'ZL19', 'LA', 'ITA', 'DE', 'CS', 'HE', 'HEvis', 'PL_REV', 'PL_WREV', 'MkG_1', 'MkW_1']
cols = [c for c in cols if c in D]
rv = sorted(k for k in D if k.startswith('Rv'))
sig = list(D['LA']['signed'].keys()); uns = list(D['LA']['unsigned'].keys())
print('SIGNED (z, sign-flip over pages; + = text as written is the forward reading)')
print('%-15s' % 'sig' + ''.join('%8s' % c for c in cols) + '   RvG|z|max RvW|z|max')
for s in sig:
    zr = [D[k]['signed'][s]['z'] for k in rv]
    print('%-15s' % s + ''.join('%8.1f' % D[c]['signed'][s]['z'] for c in cols) +
          '   %5.1f %5.1f' % (max(abs(D[k]['signed'][s]['z']) for k in rv if 'RvG' in k),
                            max(abs(D[k]['signed'][s]['z']) for k in rv if 'RvW' in k)))
print('\nSIGNED effects (eff)')
for s in sig:
    print('%-15s' % s + ''.join('%9.4g' % D[c]['signed'][s]['eff'] for c in cols))
print('\nUNSIGNED (z of obs vs random-page-direction null; obs-nullmean)')
for s in uns:
    print('%-13s' % s + ''.join('%8.1f' % D[c]['unsigned'][s]['z'] for c in cols) +
          '   Rv max z %5.1f' % max(D[k]['unsigned'][s]['z'] for k in rv))
for s in uns:
    print('%-13s' % s + ''.join('%9.4f' % D[c]['unsigned'][s]['excess'] for c in cols) +
          '   Rv max %7.4f' % max(D[k]['unsigned'][s]['excess'] for k in rv))
# calibration: all surrogate signed z
allz = np.array([D[k]['signed'][s]['z'] for k in rv for s in sig])
print('\nReversible surrogates: %d signed z, |z|>2: %d, |z|>3: %d, sd %.2f' % (len(allz), (abs(allz) > 2).sum(), (abs(allz) > 3).sum(), allz.std()))
for c in cols:
    ps = [D[c]['signed'][s]['p'] for s in sig]
    keep = L.bh(ps)
    print(c, 'BH-significant signed:', [s for s, k in zip(sig, keep) if k])
