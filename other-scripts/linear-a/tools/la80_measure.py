"""Measure outlines for a list of doc ids (photo at two thresholds + facsimile), consensus QC.
usage: python3 la80_measure.py ids.json out.json"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L
import numpy as np
ids = json.load(open(sys.argv[1])); out = {}
KEYS = None
for d in ids:
    fp = os.path.join(L.IMG, '%s-Inscription.jpg' % d); fx = os.path.join(L.IMG, '%s-Facsimile.jpg' % d)
    cands = []
    if os.path.exists(fp):
        g, dpi = L.load_gray(fp)
        for fr in (0.4, 0.6):
            o = L.segment(g, frac=fr)
            if o is not None and o.sum() > 400:
                f = L.shape_features(o, g, dpi); f['src'] = 'photo%.1f' % fr; f['dpi'] = dpi; cands.append(f)
    if os.path.exists(fx):
        f, o = L.measure_fx(fx)
        if f is not None:
            f['src'] = 'fx'; cands.append(f)
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if abs(a['aspect'] - b['aspect']) < 0.07 and abs(a['rectness'] - b['rectness']) < 0.07:
                if best is None: best = (a, b)
    rec = {'n_cand': len(cands), 'ok': best is not None}
    if best:
        a, b = best
        for k in a:
            if isinstance(a[k], float) and isinstance(b.get(k), float):
                rec[k] = (a[k] + b[k]) / 2
        rec['pair'] = a['src'] + '+' + b['src']
        ph = [c for c in cands if c['src'].startswith('photo')]
        if ph: rec['long_cm'] = ph[0]['long_cm']; rec['short_cm'] = ph[0]['short_cm']; rec['area_cm2'] = ph[0]['area_cm2']
        if ph: rec['tone'] = ph[0]['tone']; rec['tone_sd'] = ph[0]['tone_sd']
    out[d] = rec
json.dump(out, open(sys.argv[2], 'w'), indent=0)
print(len(out), sum(r['ok'] for r in out.values()))
