"""Print cycle-1 tables from results/v15/cycle1/*.json."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V

for p in sorted(glob.glob(os.path.join(V.RES, 'cycle1', '*.json'))):
    r = json.load(open(p))
    re_ = r['real']
    print(f"== {r['text']} [{r['unit']}] n={r['n_words']}  mi_in {re_['mi_in']:.4f} mi_x {re_['mi_x']:.4f} "
          f"h2 {re_['h2']:.3f} rep10 {re_['rep10']} maxrep {re_['maxrep']}")
    for lev, d in r['nulls'].items():
        s = '  '.join(f"{k} z{d[k]['z']:+.1f}(null {d[k]['mean']:.4g})" for k in ('mi_in', 'mi_x', 'h2', 'rep10', 'maxrep'))
        sp = d['spec']
        print(f"   {lev:8s} {s}  spec max {sp['real_max_ratio']:.2f} @P{sp['period_at_max']:.1f} "
              f"(null max mean {sp['null_max_mean']:.2f}, max {sp['null_max_max']:.2f}, p {sp['p']:.3f})")
    pr = r['prank']
    print(f"   prank: length excess {pr['len_excess']:.4f} vs random partitions {pr['rand_mean']:.4f}+-{pr['rand_sd']:.4f}"
          f" (max {pr['rand_max']:.4f}) rank {pr['rank']}/{pr['n_part'] + 1}; first-glyph hash {pr['first_glyph_excess']:.4f}, last-glyph hash {pr['last_glyph_excess']:.4f}"
          f"  ratio len/random {pr['len_excess'] / max(pr['rand_mean'], 1e-9):.1f}")
    if 'gaps' in r:
        g = r['gaps']; print(f"   gaps: comma rate {g['comma_rate']:.3f} MI {g['mi']:.5f} null {g['null_mean']:.5f} z {g['z']:+.1f}")
    w = r['wpl']; print(f"   words/line: MI {w['mi']:.4f} null {w['null_mean']:.4f} z {w['z']:+.1f}")
