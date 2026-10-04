#!/usr/bin/env python3
"""LA-38 cycle 3 summaries: row / column P values and sign support; planted-error detection AUC."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L


def auc(pos, neg):
    pos = np.asarray([x for x in pos if x == x]); neg = np.asarray([x for x in neg if x == x])
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main(fn):
    res = json.load(open(os.path.join(L.CK, fn)))
    for r in res:
        rows = {k: v for k, v in r['rows'].items() if v}; cols = {k: v for k, v in r['cols'].items() if v}
        sg = r['signs']
        print(f"== {r['tag']} plant={r['plant']}")
        print(' rows  ', ' '.join(f"{k}:{v['k']}/{v['p']:.3f}" for k, v in sorted(rows.items(), key=lambda x: x[1]['p'])))
        print(' cols  ', ' '.join(f"{k}:{v['k']}/{v['p']:.3f}" for k, v in sorted(cols.items(), key=lambda x: x[1]['p'])))
        print(f" rows P<=.05 {np.mean([v['p'] <= .05 for v in rows.values()]):.2f} (multi-sign rows "
              f"{np.mean([v['p'] <= .05 for v in rows.values() if v['k'] > 1]):.2f}); cols P<=.05 {np.mean([v['p'] <= .05 for v in cols.values()]):.2f}")
        if r['plant']:
            w = [s for s, d in sg.items() if d['wrong']]; ok = [s for s, d in sg.items() if not d['wrong']]
            a1 = auc([sg[s]['supC'] for s in ok], [sg[s]['supC'] for s in w])
            a2 = auc([sg[s]['supV'] for s in ok], [sg[s]['supV'] for s in w])
            a3 = auc([np.nansum([sg[s]['supC'], sg[s]['supV']]) for s in ok], [np.nansum([sg[s]['supC'], sg[s]['supV']]) for s in w])
            print(f" planted-error AUC (right > wrong): C-support {a1:.2f}, V-support {a2:.2f}, sum {a3:.2f}; wrong: "
                  + ' '.join(f"{s}({sg[s]['val']}) {sg[s]['supC']:.2f}/{sg[s]['supV']:.2f}" for s in w))
        else:
            ss = sorted(sg.items(), key=lambda x: np.nansum([x[1]['supC'], x[1]['supV']]))
            print(' signs (C/V support, low first): ' + ' '.join(f"{s}={d['val']}[{d['n']}] {d['supC']:.2f}/{d['supV']:.2f}" for s, d in ss))


if __name__ == '__main__':
    main(sys.argv[1])
