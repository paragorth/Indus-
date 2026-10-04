"""Summarise v16 cycle 1 (rule-grammar search) into rows."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v16_lib as L
from v16_cycle1 import NAMES

D = os.path.join(L.RESDIR, 'cycle1')
for n in NAMES:
    p = os.path.join(D, n + '.json')
    if not os.path.exists(p): print(n, 'missing'); continue
    o = json.load(open(p))
    w = o['written_test']; b = o['best'][0]['test']
    tbest = min(o['best'], key=lambda x: x['test']['bpg'])
    print(f"{n:24s} R={o['R']} written {w['bpg']:.3f} best-train-rule on test {b['bpg']:.3f} "
          f"(d {b['bpg']-w['bpg']:+.3f}) [{o['best'][0]['rule']}] beat-frac {o['frac_beat_written_train']:.4f} "
          f"| units {b['ntok']} vs {w['ntok']}, types {b['ntype']} vs {w['ntype']}, len {b['mean_len']:.2f}+-{b['sd_len']:.2f} "
          f"vs {w['mean_len']:.2f}+-{w['sd_len']:.2f}; zipf {b['zipf_slope']:.2f}/{b['zipf_r2']:.3f} vs {w['zipf_slope']:.2f}/{w['zipf_r2']:.3f}; "
          f"junc {b['junc_ex']:.3f} vs {w['junc_ex']:.3f}; bfW {b['bf_written']:.2f}"
          + (f"; bfTruth best {b['bf_truth']:.2f} planted {w['bf_truth']:.2f}; truth bpg {o['truth_test']['bpg']:.3f}" if 'bf_truth' in b else ''))
    print('    kinds best bpg:', {k: round(v, 3) for k, v in sorted(o['kind_best'].items(), key=lambda x: x[1])})
    print('    kinds beat frac:', {k: round(v, 4) for k, v in o['kind_frac_beat'].items() if v > 0})
    print('    top5 test:', [(x['rule'], round(x['test']['bpg'], 3)) for x in o['best'][:5]])
