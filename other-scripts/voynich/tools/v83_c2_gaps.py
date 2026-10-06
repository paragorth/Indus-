#!/usr/bin/env python3
"""v83 cycle 2, v74 short-word gap habit re-run by uncertainty class of the junction.
The v74 measurement: share of physically measured word gaps narrower than the line median (r < 1), by glyph length of
the first word (Voynich 1-glyph 0.72 ... 6-glyph 0.39; Latin flat). Gaps: data/v74_ckpt/gaps_V.json (ZL words a, b at
locus folio / line n / junction k). Here every junction is classed with the v83 ZL flags:
  certain  '.' separator and both words unflagged
  usp      the transcriber marked this space ',' (uncertain)
  flagged  any other flag on either word
  agree    both words read identically by ZL3b, IT2a and GC2a, separator '.'
Control: the gradient (1-glyph minus 6+-glyph share) in each class with 2,000 bootstrap resamples of lines."""
import os, sys, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v83_parse as P
from vlib import glyphs

G = json.load(open(os.path.join(P.DATA, 'v74_ckpt', 'gaps_V.json')))
Z = {(r['folio'], r['n']): r for r in P.records('ZL3b')}
rows = []; miss = 0
for g in G:
    r = Z.get((g['folio'], g['n']))
    k = g['k']
    if r is None or k + 1 >= len(r['words']) or r['words'][k] != g['a'] or r['words'][k + 1] != g['b']:
        miss += 1; continue
    sep = r['seps'][k]
    fa, fb = set(r['flags'][k]), set(r['flags'][k + 1])
    cls = 'usp' if sep == ',' else ('certain' if not ((fa | fb) - {'usp'}) else 'flagged')
    # 'usp' on a word can come from its other side; for the junction only this separator matters
    agree = r['agree'][k] and r['agree'][k + 1] and sep != ','
    L = min(len(glyphs(g['a'])), 6)
    rows.append(dict(line=(g['folio'], g['n']), L=L, narrow=g['r'] < 1, cls=cls, agree=agree,
                     sepdot=sep in '.-~'))
print('gaps', len(G), 'matched', len(rows), 'unmatched', miss)


def table(rs):
    out = {}
    for L in range(1, 7):
        x = [r['narrow'] for r in rs if r['L'] == L]
        out[L] = (float(np.mean(x)) if x else float('nan'), len(x))
    return out


def grad(rs, B=2000, seed=74):
    lines = sorted({r['line'] for r in rs}); by = collections.defaultdict(list)
    for r in rs: by[r['line']].append(r)
    def g(rr):
        a = [r['narrow'] for r in rr if r['L'] == 1]; b = [r['narrow'] for r in rr if r['L'] >= 6]
        return (np.mean(a) - np.mean(b)) if a and b else np.nan
    obs = g(rs); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = rng.choice(len(lines), len(lines))
        bs.append(g([r for i in pick for r in by[lines[i]]]))
    bs = np.array(bs); bs = bs[~np.isnan(bs)]
    return float(obs), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


res = {}
sets = {'all': rows, 'certain': [r for r in rows if r['cls'] == 'certain'], 'usp': [r for r in rows if r['cls'] == 'usp'],
        'flagged': [r for r in rows if r['cls'] == 'flagged'], 'agree': [r for r in rows if r['agree']],
        'certain_agree': [r for r in rows if r['cls'] == 'certain' and r['agree']]}
for k, rs in sets.items():
    t = table(rs); gr = grad(rs)
    res[k] = dict(n=len(rs), by_len=t, grad=gr, narrow_all=float(np.mean([r['narrow'] for r in rs])))
    print('%-14s n=%5d narrow %.3f | ' % (k, len(rs), res[k]['narrow_all']) +
          ' '.join('%d:%.2f(%d)' % (L, v[0], v[1]) for L, v in t.items()) + ' | grad 1-vs-6 %.3f [%.3f, %.3f]' % gr)
# how much of the 1-glyph narrowing do the transcribers' commas carry?
one = [r for r in rows if r['L'] == 1]
print('1-glyph first words: share of junctions marked "," %.3f; narrow among "," %.3f, among "." %.3f' % (
    np.mean([r['cls'] == 'usp' for r in one]), np.mean([r['narrow'] for r in one if r['cls'] == 'usp']),
    np.mean([r['narrow'] for r in one if r['cls'] != 'usp'])))
res['_note'] = 'r < 1: gap narrower than the line median (v74 definition)'
json.dump(res, open(os.path.join(P.CK, 'c2_gaps.json'), 'w'), indent=1, default=str)
