#!/usr/bin/env python3
"""la71 cycle 1: hand check of the restoration-aware parser against SigLA drawings.
(1) sign-box wear: 96 boxes (32 per cover stratum), labelled blind from crops (W = stipple touches the
    strokes, C = clean); data/la71_ckpt/hand_wear.json.  (2) numbers on 8 KU-RO tablets labelled from the
    whole SigLA drawing (C = complete and clear, W = in stipple or at a break).  Prints confusion tables."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la71_parse as P
CK = P.CK
HAND_NUM = {'HT85a': 'CCCCCCCC', 'HT88': 'CCCCCCCCCC', 'HT89': 'CCCCCCCCW', 'HT104': 'CWWC', 'HT122b': 'CCCCCCW',
            'HT118': 'CCCCCCCCW', 'HT11b': 'CCCCCC', 'HT2': 'WCWWCC'}


def main():
    H = json.load(open(os.path.join(CK, 'hand_wear.json')))
    lab = np.array([c == 'W' for c in H['labels']]); cov = np.array([k[4] for k in H['key']])
    N = H['strata']; st = np.where(cov < 0.1, 'lo', np.where(cov < 0.3, 'mid', 'hi'))
    w = np.array([N[s] / 32 for s in st])
    out = {'wear': {}}
    for t in (0.05, 0.1, 0.2):
        p = cov >= t
        out['wear'][t] = dict(acc=float((w * (p == lab)).sum() / w.sum()), prec=float((w * (p & lab)).sum() / (w * p).sum()),
                              rec=float((w * (p & lab)).sum() / (w * lab).sum()))
    out['wear']['hand_share_any_stipple'] = float((w * lab).sum() / w.sum())
    C = {d['id']: d for d in json.load(open(os.path.join(P.DATA, 'corpus_ra.json')))}
    tp = fp = fn = tn = 0; rows = []
    for k, s in HAND_NUM.items():
        nums = [t for t in C[k]['tokens'] if t['t'] == 'num']
        assert len(nums) == len(s), (k, len(nums), len(s))
        for t, h in zip(nums, s):
            pdmg = t['st'] != 'read'; hd = h == 'W'
            tp += pdmg and hd; fp += pdmg and not hd; fn += hd and not pdmg; tn += not pdmg and not hd
            if pdmg != hd: rows.append((k, t['v'], h, t['st'], t['fl']))
    out['numbers'] = dict(n=tp + fp + fn + tn, tp=tp, fp=fp, fn=fn, tn=tn, prec=tp / max(1, tp + fp), rec=tp / max(1, tp + fn),
                          read_clean=tn / max(1, tn + fn), disagreements=rows)
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(os.path.join(CK, 'c1_check.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
