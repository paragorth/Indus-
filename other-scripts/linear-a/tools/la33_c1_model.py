#!/usr/bin/env python3
"""LA-33: the cycle-1 archive regressor (w = scheduled share), retrained from the stored simulations."""
import json, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from la33_common import CK, feats

_d = json.load(open(os.path.join(CK, 'sims_c1.json')))['rows']
_X = np.array([r[3:] for r in _d]); _w = np.array([r[2] for r in _d])
REG = RandomForestRegressor(400, min_samples_leaf=3, n_jobs=1, random_state=0).fit(_X, _w)


def score_w(docs, rng):
    return float(REG.predict(feats(docs, rng)[None])[0])
