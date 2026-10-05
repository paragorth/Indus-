"""Summarise cycle 2 (climbed alphabets, held test, section topic test)."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V
for k in ['ars', 'med', 'lat', 'voy', 'voyit', 'voy_mk2', 'voy_gshuf', 'voy_sc', 'ars_mk2']:
    p = os.path.join(V.CK, 'c2_%s.json' % k)
    if not os.path.exists(p):
        continue
    r = json.load(open(p)); b = r['best']
    s = r['finals'][b][1]; h = r['held'][b]
    rand = sorted(r['rand']); perm = sorted(r['perm'])
    pr = sum(x >= r['Ic'] for x in rand) / len(rand)
    pp = sum(x >= r['Ic'] for x in perm) / len(perm)
    print('%-9s sel Gfree %.2f Gm1 %.2f | held Gfree %.2f Gm1 %.2f gap %.2f cov %.2f canon %.2f m %.2f | k %d %s' % (
        k, s['Gfree'], s['Gm1'], h['Gfree'], h['Gm1'], h['Gfree'] - h['Gm1'], h['cov'], h['canon'], h['m'], len(r['alpha']), ' '.join(r['alpha'])))
    print('          held all starts Gfree', [round(x['Gfree'], 2) for x in r['held']],
          '| I(concept;sec) %.4f I(glyph;sec) %.4f ratio %.2f | random-alpha median %.4f p %.3f | label-perm median %.4f p %.3f' % (
              r['Ic'], r['Ig'], r['Ic'] / max(1e-9, r['Ig']), rand[len(rand) // 2], pr, perm[len(perm) // 2], pp),
          ('| J truth %.2f (starts %s)' % (r['jacc'], [round(x, 2) for x in r['jacc_start']]) if 'jacc' in r else ''))
    print('          rand sel quantiles', [round(x, 2) for x in r['rand_sel_q']], 'rand held best Gfree %.2f' % max(x['Gfree'] for x in r['rand_held_top20']),
          ('| truth held Gfree %.2f Gm1 %.2f | J finals %s' % (r['truth_held']['Gfree'], r['truth_held']['Gm1'], [round(x, 2) for x in r['jacc_finals']]) if 'truth_held' in r else ''))
    print('          glyph-alphabet held Gm1 %.2f Gfree %.2f | DELTA (climbed Gfree - glyph Gm1) %.2f' % (r['glyph_held']['Gm1'], r['glyph_held']['Gfree'], h['Gfree'] - r['glyph_held']['Gm1']))
    print('          profiles', json.dumps(r['prof'])[:600])
