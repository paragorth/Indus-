#!/usr/bin/env python3
"""LA-19 cycle 1 report: K selection table + calibration of LA against single languages and planted mixtures."""
import glob
from la19_common import *

rows = {}
for f in sorted(glob.glob(os.path.join(CK, 'c1_*.json'))):
    r = json.load(open(f)); rows[r['name']] = r


def line(r):
    K = r['K_sel']; st = r['stab']
    g2 = r['gain_per_word']['2']; gk = r['gain_per_word'][str(K)]; gb = r['gain_per_word'][str(r['K_best'])]
    s2 = st.get('2', {}); sk = st.get(str(K), {})
    return dict(name=r['name'], n=r['n'], V=r['V'], len=r['mean_len'], K_sel=K, K_best=r['K_best'],
                gain_sel=gk, gain2=g2, gain_best=gb, ari2=s2.get('ari_mean'), ari_sel=sk.get('ari_mean'),
                ari_truth2=s2.get('ari_truth'), tri_z=r['tri_vs_mix_z'], beta1=r['beta']['1'])


L = [line(r) for r in rows.values()]
out = dict(rows=L)
L1 = [x for x in L if x['name'].startswith('L1_')]
la = [x for x in L if x['name'] == 'LA'][0]
if L1:
    gb = np.array([x['gain_best'] for x in L1]); ks = np.array([x['K_sel'] for x in L1]); a2 = np.array([x['ari2'] for x in L1])
    out['single_lang'] = dict(n=len(L1), K_sel_hist=collections.Counter(ks.tolist()), frac_K_ge2=round(float((ks >= 2).mean()), 3),
                              gain_best_median=round(float(np.median(gb)), 4), gain_best_q90=round(float(np.quantile(gb, 0.9)), 4),
                              LA_gain_best_rank=int((gb >= la['gain_best']).sum()), ari2_median=round(float(np.median(a2)), 3),
                              LA_ari2_rank=int((a2 >= la['ari2']).sum()))
dump(os.path.join(OUT, 'c1_report.json'), out)
print('%-18s %5s %4s %5s %5s %6s %7s %7s %7s %6s %6s %6s %6s' % ('name', 'n', 'V', 'len', 'Ksel', 'Kbest', 'g_sel', 'g2', 'g_best', 'ari2', 'ariK', 'ariT2', 'triz'))
for x in L:
    print('%-18s %5d %4d %5.2f %5d %6d %7.4f %7.4f %7.4f %6s %6s %6s %6.2f' % (x['name'], x['n'], x['V'], x['len'], x['K_sel'], x['K_best'], x['gain_sel'], x['gain2'], x['gain_best'], x['ari2'], x['ari_sel'], x['ari_truth2'], x['tri_z']))
print(json.dumps(out.get('single_lang'), default=str))
