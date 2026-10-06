#!/usr/bin/env python3
"""la71 helper: print documents of corpus_ra.json with token status marks (for hand checks).
mark: plain = read, <x>L / x>R edge, {x} restored, ~x~ other damage, [[x]] erased."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
C = {d['id']: d for d in json.load(open(os.path.join(HERE, '..', 'data', 'corpus_ra.json')))}


def show(t):
    if t['t'] == 'nl': return '/'
    if t['t'] == 'div': return '.'
    v = '-'.join(t['s']) if t['t'] == 'word' else (str(t['v']) + ('+' + ''.join(t['frac']) if t.get('frac') else '') if t['t'] == 'num' else str(t['v']))
    fl = set(t['fl'])
    if t['st'] == 'erased': return '[[%s]]' % v
    if t['st'] == 'restored': return '{%s:%s}' % (v, ','.join(sorted(fl & {'bridge', 'nodraw'})))
    if t['st'] == 'damaged':
        return '~%s:%s~' % (v, ','.join(sorted(fl)))
    return v


if __name__ == '__main__':
    for k in sys.argv[1:]:
        d = C[k]
        print(k, d['site'], 'sigla' if d['sigla'] else '', ' '.join(show(t) for t in d['tokens']))
