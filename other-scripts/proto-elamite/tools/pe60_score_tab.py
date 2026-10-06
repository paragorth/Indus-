#!/usr/bin/env python3
"""pe60 cycle 2b: score tablet-level photo readings (made blind) against CDLI ATF.

Usage: pe60_score_tab.py MY.tsv TRUTH.atf [--json out.json]
MY.tsv columns: P H157 TAG CAP TOT NE  (1/0/? ; NE integer = numeric entries on the obverse)
Truth from ATF (same definitions as pe59 roles):
  H157 first obverse line has signs, no numerals, and starts with M157 (any variant)
  TAG  a top/bottom/left/right/edge line with numerals and no signs
  CAP  any capacity code (pe59 CAPSET) anywhere
  TOT  any numeric line on the reverse
  NE   numeric lines on the obverse
Controls: majority-class guesser per feature; my calls permuted across tablets (2,000 permutations).
"""
import sys, os, re, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe60_score_read import blocks, parse, CAP  # noqa: E402

FEATS = ['H157', 'TAG', 'CAP', 'TOT']


def truth_feats(b):
    L = parse(b)
    obv = [l for l in L if l['surf'] == 'obverse']
    first = obv[0] if obv else None
    h = bool(first and first['signs'] and not first['nums'] and first['signs'][0] == 'M157')
    tag = any(l['surf'] in ('top', 'bottom', 'left', 'right', 'edge') and l['nums'] and not l['signs'] for l in L)
    cap = any(c in CAP for l in L for _, c in l['nums'])
    tot = any(l['surf'] == 'reverse' and l['nums'] for l in L)
    ne = sum(1 for l in obv if l['nums'])
    return {'H157': int(h), 'TAG': int(tag), 'CAP': int(cap), 'TOT': int(tot), 'NE': ne}


def read_tsv(path):
    rows = {}
    for i, line in enumerate(open(path)):
        f = line.rstrip('\n').split('\t')
        if i == 0 or len(f) < 6:
            continue
        rows[f[0]] = {k: (None if v == '?' else int(v)) for k, v in zip(FEATS + ['NE'], f[1:6])}
    return rows


def evaluate(mine, truth, ids):
    out = {}
    for k in FEATS:
        pairs = [(mine[p][k], truth[p][k]) for p in ids if mine[p][k] is not None]
        n = len(pairs)
        acc = sum(a == b for a, b in pairs) / n if n else None
        tp = sum(a == 1 and b == 1 for a, b in pairs); fp = sum(a == 1 and b == 0 for a, b in pairs)
        fn = sum(a == 0 and b == 1 for a, b in pairs); tn = sum(a == 0 and b == 0 for a, b in pairs)
        maj = max(sum(b for _, b in pairs), n - sum(b for _, b in pairs)) / n if n else None
        out[k] = {'decided': n, 'abstained': len(ids) - n, 'acc': round(acc, 3) if acc is not None else None,
                  'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                  'sens': round(tp / (tp + fn), 3) if tp + fn else None, 'spec': round(tn / (tn + fp), 3) if tn + fp else None,
                  'majority_baseline': round(maj, 3) if maj is not None else None}
    ne = [(mine[p]['NE'], truth[p]['NE']) for p in ids if mine[p]['NE'] is not None]
    out['NE'] = {'mean_abs_err': round(sum(abs(a - b) for a, b in ne) / len(ne), 2),
                 'baseline_median_guess_err': round(sum(abs(sorted(b for _, b in ne)[len(ne) // 2] - b) for _, b in ne) / len(ne), 2)}
    return out


def main():
    mine = read_tsv(sys.argv[1])
    tb = blocks(sys.argv[2])
    ids = [p for p in mine if p in tb]
    truth = {p: truth_feats(tb[p]) for p in ids}
    res = evaluate(mine, truth, ids)
    rng = random.Random(60602)
    null = {k: [] for k in FEATS}
    for _ in range(2000):
        perm = ids[:]
        rng.shuffle(perm)
        sh = {p: mine[q] for p, q in zip(ids, perm)}
        r = evaluate(sh, truth, ids)
        for k in FEATS:
            null[k].append(r[k]['acc'] if r[k]['acc'] is not None else 0)
    for k in FEATS:
        res[k]['perm_mean_acc'] = round(sum(null[k]) / len(null[k]), 3)
        res[k]['perm_p'] = round(sum(x >= res[k]['acc'] for x in null[k]) / len(null[k]), 4)
    res['per'] = {p: {'mine': mine[p], 'true': truth[p]} for p in ids}
    if '--json' in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    for k in FEATS + ['NE']:
        print(k, res[k])
    print('true base rates:', {k: sum(truth[p][k] for p in ids) for k in FEATS}, 'n', len(ids))


if __name__ == '__main__':
    main()
