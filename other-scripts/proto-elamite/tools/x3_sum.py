#!/usr/bin/env python3
"""X-3 summary: usage x3_sum.py TAG [TAG...]. Prints, per ordered pair and mapping mode, gains in
bits per token on held-out Y (positive = helps), paired by seed:
  G_real  = scratch - real          total transfer
  G_body  = scratch - body          structure only (pretrained MLP, fresh embeddings)
  G_shuf  = scratch - shuf ; G_mark = scratch - markov
  ID      = relab - real            identity component (lost when X's signs are relabelled)
  IDb     = band - real             identity beyond frequency band (relabel within rank bands of 8)
and the transfer matrices (rank mode) of G_real, G_body and ID."""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCR = os.environ.get('X3_SCR', '/tmp/claude-0/x3')


def load(tags):
    R = []
    for t in tags:
        for L in open(os.path.join(SCR, f'res_{t}.jsonl')):
            R.append(json.loads(L))
    return R


def table(R, metric='test'):
    sc = defaultdict(dict)            # (y, slice) -> seed -> loss
    v = defaultdict(lambda: defaultdict(dict))   # (x,y,slice,mode) -> cond -> seed -> loss
    for r in R:
        if r['cond'] == 'scratch':
            sc[(r['y'], r['slice'])][r['seed']] = r[metric]
        else:
            mode = r['mode']
            cond = r['cond'] if mode != 'body' else 'body'
            mm = 'rank' if mode == 'body' else mode
            v[(r['x'], r['y'], r['slice'], mm)][cond][r['seed']] = r[metric]
            if mode == 'body':
                v[(r['x'], r['y'], r['slice'], 'label')]['body'][r['seed']] = r[metric]
                v[(r['x'], r['y'], r['slice'], 'syl')]['body'][r['seed']] = r[metric]
    out = {}
    for k, d in v.items():
        x, y, sl, mode = k
        if 'real' not in d:
            continue
        s = sc.get((y, sl), {})
        seeds = sorted(set(d['real']) & set(s))
        if not seeds:
            continue

        def diff(a, b):
            if a not in d and a != 'scratch':
                return None
            A = s if a == 'scratch' else d[a]
            B = s if b == 'scratch' else d.get(b, {})
            ss = [q for q in seeds if q in A and q in B]
            if not ss:
                return None
            arr = np.array([A[q] - B[q] for q in ss])
            return (float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                    float(arr.min()), float(arr.max()), len(arr))
        out[k] = {'G_real': diff('scratch', 'real'), 'G_body': diff('scratch', 'body'),
                  'G_shuf': diff('scratch', 'shuf'), 'G_mark': diff('scratch', 'markov'),
                  'G_relab': diff('scratch', 'relab'),
                  'ID': diff('relab', 'real'), 'IDnull': diff('relab2', 'relab'),
                  'IDc': diff('cperm', 'real'), 'IDv': diff('vperm', 'real'), 'IDb': diff('band', 'real'),
                  'scratch': float(np.mean([s[q] for q in seeds]))}
    return out


def fmt(t):
    return '   .  ' if t is None else f'{t[0]:+.3f}'


def main():
    tags = sys.argv[1:]
    R = load(tags)
    T = table(R)
    names = ['LA', 'PE', 'VOY', 'LB', 'PC', 'UR3', 'AKK', 'ELX', 'LAT', 'GRC']
    for sl in sorted({k[2] for k in T}):
        for met in ('G_real', 'G_body', 'G_relab', 'ID', 'IDb'):
            for mode in ('rank', 'label'):
                rows = [k for k in T if k[2] == sl and k[3] == mode]
                if not rows:
                    continue
                print(f'\n== {met}  mode {mode}  slice {sl}  (rows X = pretrain, cols Y = target; bits/token)')
                print('X\\Y   ' + ' '.join(f'{n:>6}' for n in names))
                for x in names:
                    cells = []
                    for y in names:
                        t = T.get((x, y, sl, mode))
                        cells.append(fmt(t[met]) if t else '   .  ')
                    if any(c.strip() != '.' for c in cells):
                        print(f'{x:5} ' + ' '.join(cells))
        print(f'\n== controls / all pairs, slice {sl}: mean +- se (min..max over seeds)')
        for k in sorted(T):
            if k[2] != sl:
                continue
            t = T[k]
            line = f'{k[0]:>4}>{k[1]:<4} {k[3]:5} scr {t["scratch"]:.3f} '
            for met in ('G_real', 'G_body', 'G_relab', 'G_shuf', 'G_mark', 'ID', 'IDb', 'IDnull', 'IDc', 'IDv'):
                v = t[met]
                line += f' {met} ' + ('.' if v is None else f'{v[0]:+.3f}+-{v[1]:.3f}')
            print(line)


if __name__ == '__main__':
    main()
