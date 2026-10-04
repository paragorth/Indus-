"""v21 cycle-1 report: ladder AUCs, controls, surviving features."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_lib import *

R = [json.load(open(f)) for f in glob.glob(os.path.join(CK, 'c1_*_*_F*_*.json'))]
by = defaultdict(list)
for r in R: by[(r['tag'], r['corpus'], r['forger'])].append(r)


def summ(rs):
    lr = np.array([r['lr'] for r in rs]); gb = np.array([r['gbm'] for r in rs])
    sub = np.array([a for r in rs for a, _ in r['sub']])
    return lr, gb, sub


def feat_imp(rs, keys):
    tot = defaultdict(list)
    for r in rs:
        for a, cols in r['sub']:
            for c in cols: tot[keys[c]].append(a)
    allm = np.mean([a for r in rs for a, _ in r['sub']])
    return sorted(((np.mean(v) - allm, k) for k, v in tot.items()), reverse=True)


def main(show=True):
    lines = []
    nullsub = np.concatenate([summ(v)[2] for k, v in by.items() if k[0] == 'null'] or [np.array([0.5])])
    q99 = float(np.quantile(nullsub, 0.99)) if len(nullsub) > 10 else 0.6
    lines.append(f'null subset AUC 99th pct {q99:.3f} (n {len(nullsub)})')
    tab = {}
    for k in sorted(by):
        rs = by[k]; lr, gb, sub = summ(rs)
        tab[k] = (lr.mean(), gb.mean(), np.median(sub), sub.max(), (sub > q99).mean())
        lines.append(f'{k[0]:5s} {k[1]:2s} {k[2]} n{len(rs)} ridge {lr.mean():.3f}+-{lr.std():.3f} gbm {gb.mean():.3f}+-{gb.std():.3f} '
                     f'sub median {np.median(sub):.3f} max {sub.max():.3f} frac>null99 {(sub > q99).mean():.3f}')
    # surviving features
    for cn in ('V', 'LA', 'IT'):
        mains = {k[2]: v for k, v in by.items() if k[0] == 'main' and k[1] == cn}
        if not mains: continue
        best = min(mains, key=lambda f: np.mean([r['lr'] for r in mains[f]]))
        rs = mains[best]; keys = rs[0]['keys']
        Z = np.array([r['z'] for r in rs])
        neg = by.get(('neg', cn, best)) or by.get(('neg', cn, 'F7')) or []
        Zn = np.array([r['z'] for r in neg]) if neg else np.zeros((1, len(keys)))
        nul = by.get(('null', cn, best)) or by.get(('null', cn, 'F7')) or []
        Zu = np.array([r['z'] for r in nul]) if nul else np.zeros((1, len(keys)))
        surv = []
        for j, kk in enumerate(keys):
            z = Z[:, j]
            if (np.abs(z) >= 4).all() and len(set(np.sign(z))) == 1 and np.abs(Zn[:, j]).max() < 3.5 and np.abs(Zu[:, j]).max() < 3.5:
                surv.append((float(np.mean(z)), kk, float(np.abs(Zn[:, j]).max()), float(np.abs(Zu[:, j]).max())))
        surv.sort(key=lambda t: -abs(t[0]))
        lines.append(f'{cn}: best forger {best}; features surviving all {len(rs)} seeds at |z|>=4 with neg/null |z|<3.5: {len(surv)}')
        for z, kk, zn, zu in surv:
            lines.append(f'   {kk:20s} z {z:+.1f}  (neg max |z| {zn:.1f}, null max |z| {zu:.1f})')
        imp = feat_imp(rs, keys)
        lines.append(f'   subset importance top: ' + ', '.join(f'{k} {d:+.3f}' for d, k in imp[:10]))
        if neg:
            bad = [(keys[j], float(Zn[:, j].mean())) for j in range(len(keys)) if np.abs(Zn[:, j]).min() >= 4]
            lines.append(f'   negative-control features at |z|>=4 in every seed: {bad}')
    txt = '\n'.join(lines)
    if show: print(txt)
    return tab, txt


if __name__ == '__main__':
    main()
