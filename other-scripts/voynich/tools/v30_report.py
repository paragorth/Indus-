"""Summarise v30 ladder runs: python3 v30_report.py <tag>"""
import sys, os, json, glob, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, 'data', 'v30_ckpt')


def load(tag):
    R = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CK, f'{tag}_*_s*.json'))):
        r = json.load(open(f)); R[r['name']].append(r)
    return R


def metrics(r, kcut=None):
    p = r['path']
    F = r['floor'][0]; Fp = r['floor'][1]
    D0 = p[0]['held']; P0 = p[0]['held_parts']
    q = p[-1] if kcut is None else [x for x in p if x['k'] <= kcut][-1]
    Dk = q['held']; Pk = q['held_parts']
    ex0 = D0 - F; exk = Dk - F
    c0 = P0[0] + P0[1] - Fp[0] - Fp[1]; ck = Pk[0] + Pk[1] - Fp[0] - Fp[1]
    w0 = P0[2] - Fp[2]; wk = Pk[2] - Fp[2]
    # marginal held gains along the path (first occurrence of each k)
    gains = [p[i - 1]['held'] - p[i]['held'] for i in range(1, len(p))]
    tot = sum(g for g in gains if g > 0) or 1e-9
    top3 = sum(sorted([g for g in gains if g > 0], reverse=True)[:3]) / tot
    fitgain = p[0]['fit'] - p[-1]['fit']
    return dict(F=F, D0=D0, ex0=ex0, exk=exk, gain=D0 - Dk, clos=(D0 - Dk) / ex0 if ex0 > 0.02 else float('nan'),
                cex0=c0, cexk=ck, cclos=(c0 - ck) / c0 if c0 > 0.01 else float('nan'),
                wex0=w0, wexk=wk, k=q['k'], top3=top3, fitgain=fitgain, overfit=fitgain - (D0 - Dk))


def table(tag, kcut=None):
    R = load(tag)
    from v30_ladder import PAIRS
    rows = []
    for name in PAIRS:
        if name not in R:
            continue
        ms = [metrics(r, kcut) for r in R[name]]
        avg = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        sd = {k: float(np.nanstd([m[k] for m in ms])) for k in ms[0]}
        rows.append((name, PAIRS[name][4], len(ms), avg, sd))
    return rows, R


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tag = sys.argv[1]
    kcut = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rows, R = table(tag, kcut)
    print(f'{"pair":11s} {"rung":26s} n  {"F":>6s} {"D0":>6s} {"ex0":>6s} {"exK":>6s} {"clos":>6s} {"c_ex0":>6s} {"c_clos":>6s} {"w_ex0":>6s} {"w_exK":>6s} {"K":>4s} {"top3":>5s} {"ovf":>6s}')
    for name, rung, n, a, s in sorted(rows, key=lambda r: r[3]['ex0']):
        print(f'{name:11s} {rung[:26]:26s} {n}  {a["F"]:6.3f} {a["D0"]:6.3f} {a["ex0"]:6.3f} {a["exk"]:6.3f} {a["clos"]:6.2f} {a["cex0"]:6.3f} {a["cclos"]:6.2f} {a["wex0"]:6.3f} {a["wexk"]:6.3f} {a["k"]:4.0f} {a["top3"]:5.2f} {a["overfit"]:6.3f}')
