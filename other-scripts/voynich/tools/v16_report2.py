"""Summarise v16 cycle 2 (annealed free segmentation)."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v16_lib as L

D = os.path.join(L.RESDIR, 'cycle2')
for p in sorted(glob.glob(os.path.join(D, '*.json'))):
    o = json.load(open(p)); f = o['found']; w = o['written']
    line = (f"{o['name']:24s} {o['start']:8s} MDL2 written {w['bpg']:.3f} found {f['bpg']:.3f} (d {f['bpg']-w['bpg']:+.3f}) "
            f"units {f['ntok']} vs {w['ntok']} types {f['ntype']} vs {w['ntype']} len {f['mean_len']:.2f}+-{f['sd_len']:.2f} vs {w['mean_len']:.2f}+-{w['sd_len']:.2f} "
            f"zipf {f['zipf_slope']:.2f}/{f['zipf_r2']:.3f} vs {w['zipf_slope']:.2f}/{w['zipf_r2']:.3f} junc {f['junc_ex']:.3f} vs {w['junc_ex']:.3f} bfW {f['bf_written']:.3f}")
    if 'truth' in o:
        line += f" | truth bpg {o['truth']['bpg']:.3f} bfTruth found {f['bf_truth']:.3f} planted {w['bf_truth']:.3f}"
    print(line)
