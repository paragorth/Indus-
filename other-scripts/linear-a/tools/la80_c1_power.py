"""Power for the c1 outside test: plant each frozen survivor's link into the test shapes at a given strength."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L, la80_text as TX
from la80_c1 import *
fr = json.load(open(os.path.join(L.DATA, 'la80_frozen_c1.json')))
ids = json.load(open(os.path.join(L.CK, 'test_ids.json')))
B = build(ids, os.path.join(L.CK, 'shape_test.json'))
keep = [k for k in B['tabs'] if k in B['TS']]; keep_idx = np.array([B['tabs'].index(k) for k in keep])
M = shape_matrix(B, keep); sites = [B['site'][i] for i in keep_idx]; us = sorted(set(sites))
X = np.c_[B['cov'][keep_idx], np.array([[s == u for u in us[1:]] for s in sites], float)]
rng = np.random.default_rng(3)
for beta in (0.15, 0.25, 0.35):
    shares = []
    for x in fr['survivors'][::4]:
        h, j, sg = hyp_from_json(x)
        t = TX.eval_text_feature(h, B['F'], B['S'], B['G'], B['first'])[keep_idx]
        col = M[:, j].copy(); ok = ~np.isnan(col); tt = t[ok]
        if len(set(tt)) < 2: continue
        z = (tt - tt.mean()) / tt.std()
        col[ok] = col[ok] + sg * beta * np.nanstd(col) * z
        shares.append(float(resid(tt, X[ok]) @ resid(col[ok], X[ok])) * sg > 0)
    print('beta', beta, 'share', round(np.mean(shares), 3), len(shares))
