"""v29 cycle-2 summary."""
import sys, os, pickle
import numpy as np
import v29_lib as L

for name in sys.argv[1:]:
    c = pickle.load(open(os.path.join(L.CK, f'c2_{name}.pkl'), 'rb'))
    print('==', name)
    for s, js in c['sel'].items():
        i = L.STATS.index(s)
        print(f'  [{s}] selected {len(js)}')
        for j in js:
            f = c['feats'][j]
            fz = ' '.join(f"f{k}:{f.get(f'z_fold{k}', [np.nan]*8)[i]:+.1f}" for k in (0, 1))
            print(f"    ex {f['e'][i]:+.3f} z {f['z'][i]:+.1f} twin_pct {f['twin_pct'][i]:.2f} twin_mean {f['twin_mean'][i]:+.3f} {fz} | {f['name']}")
            if s in ('tier1', 'tier2') and f['blk']:
                rows = []
                for p, (pw, pn, n) in sorted(f['blk'].items(), key=lambda t: -t[1][2]):
                    nw = np.nanmean([b[p][0] - b[p][1] for b in f['blkn'] if p in b]) if any(p in b for b in f['blkn']) else np.nan
                    rows.append(f"{p}:{pw:+.2f}/{pn:+.2f}(n{n},nullD{nw:+.2f})")
                print('       blocking with/without X:', ' '.join(rows[:8]))
