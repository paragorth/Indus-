"""X1 planted size control.

For every feature that separates the undeciphered four from the deciphered set in a given regime
(|AUC-0.5|*2 >= THR), ask: if each deciphered corpus is cut down to the size of an undeciphered corpus
(regimes d8k / d18k / d27k = Linear A / Indus / Proto-Elamite token counts), does it acquire the property,
i.e. cross the midpoint between the deciphered and undeciphered medians toward the U side?
A property that deciphered corpora acquire by subsampling is a size effect, not a signature.

usage: python3 x1_planted.py REGIME [THR] [sizes regimes ...]
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'data', 'results')


def med(data, n, f):
    return float(np.nanmedian([r.get(f, np.nan) for r in data[n]['reps']]))


def main():
    regime = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    sizes = sys.argv[3:] or ['d8k', 'd18k', 'd27k']
    an = json.load(open(os.path.join(RES, f'x1_analysis_{regime}.json')))
    base = json.load(open(os.path.join(RES, f'x1_features_{regime}.json')))
    sz = {s: json.load(open(os.path.join(RES, f'x1_features_{s}.json'))) for s in sizes}
    U = an['U']; names = an['names']; D = [n for n in names if n not in U]
    out = []
    feats = [r for r in an['features'] if abs(r['auc'] - 0.5) * 2 >= thr]
    print(f'{regime}: {len(feats)} features with |AUC-.5|*2 >= {thr}')
    for r in feats:
        f = r['feat']
        uv = [med(base, n, f) for n in U]; dv = [med(base, n, f) for n in D]
        mid = (np.median(uv) + np.median(dv)) / 2
        up = np.median(uv) > np.median(dv)
        row = {'feat': f, 'auc': r['auc'], 'u_med': float(np.median(uv)), 'd_med': float(np.median(dv)), 'mid': float(mid)}
        print(f"  {f:30s} AUC {r['auc']:.3f} U med {np.median(uv):.3f} D med {np.median(dv):.3f}")
        for s in sizes:
            vals = {n: med(sz[s], n, f) for n in D if n in sz[s]}
            cross = [n for n, v in vals.items() if (v > mid if up else v < mid)]
            base_cross = [n for n in vals if (med(base, n, f) > mid if up else med(base, n, f) < mid)]
            shift = np.median([vals[n] - med(base, n, f) for n in vals])
            row[s] = {'n': len(vals), 'cross': cross, 'base_cross': base_cross, 'median_shift': float(shift),
                      'd_med_sub': float(np.median(list(vals.values())))}
            print(f"     {s}: D median {np.median(list(vals.values())):.3f} (shift {shift:+.3f} toward U side: "
                  f"{(shift > 0) == up and shift != 0}); D on U side {len(cross)}/{len(vals)} (at full size {len(base_cross)})")
        out.append(row)
    json.dump(out, open(os.path.join(RES, f'x1_planted_{regime}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
