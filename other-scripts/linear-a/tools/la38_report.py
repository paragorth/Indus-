#!/usr/bin/env python3
"""LA-38: print compact summaries of the checkpoint JSONs."""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L

KEYS = ['ocpC', 'ocpP', 'sameV', 'vinit', 'son', 'freqC', 'freqV', 'comp', 'la21', 'ALL7', 'SEL']


def line(res):
    out = []
    for k in KEYS:
        r = res.get(k)
        if not r:
            continue
        out.append(f"{k} inv" if r.get('inv') else f"{k} z{r['z']:+.1f} P{r['p']:.2g}")
    return '; '.join(out)


def tiers(fn):
    for r in json.load(open(os.path.join(L.CK, fn))):
        if r['fake'] is None:
            print(r['tag'], r['tier'], r['N'], '|', line(r['res']))


def pooled(fn, prefix, fake):
    """for LB draws / shuffles / fake truths: median z and share P<0.05 per tier and key."""
    rs = [r for r in json.load(open(os.path.join(L.CK, fn))) if r['tag'].startswith(prefix) and ((r['fake'] is not None) == fake)]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        for k in KEYS:
            x = r['res'].get(k)
            if x and not x.get('inv'):
                by[r['tier']][k].append((x['z'], x['p']))
    for t in sorted(by):
        s = []
        for k in KEYS:
            v = by[t].get(k)
            if v:
                z = np.array(v)
                s.append(f"{k} medz{np.median(z[:, 0]):+.1f} P<.05 {np.mean(z[:, 1] < .05):.2f}")
        print(prefix, 'fake' if fake else '', t, f'n={len(by[t]["SEL"]) if "SEL" in by[t] else 0}', '|', '; '.join(s))


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'tiers':
        tiers(sys.argv[2])
    elif what == 'pooled':
        pooled(sys.argv[2], sys.argv[3], sys.argv[4] == '1')
