"""Loop 55 cycle 3: THE STRONGEST SINGLE DISCRIMINATOR. Over every n=3000 Indus-shaped sample from cycles 1 and 2,
rank the 55 statistics by how well they separate known language writing (L) from designed codes (D): AUC on the raw
value, AUC on the Markov-2 z, and a leave-one-corpus-out check (the AUC when each L or D corpus is dropped, so a single
corpus cannot carry the result). Report where Indus falls on the top statistics at all three merge levels with a CI over
resamples, and where the accounting corpora fall. Output: loop55_c3.txt / .json."""
import os, sys, json, math, collections, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop55_common as C
from dark_loop55_c2 import auc

SRC = [os.path.join(C.DARK, 'loop55_c1'), os.path.join(C.DARK, 'loop55_c2')]


def load():
    R = collections.defaultdict(list)   # corpus -> list of dicts at n=3000, natural dup
    for d_ in SRC:
        if not os.path.isdir(d_): continue
        for f in os.listdir(d_):
            if not f.endswith('.json'): continue
            d = json.load(open(os.path.join(d_, f)))
            if d.get('dup', 'natural') != 'natural': continue
            if d.get('n', 3000) != 3000 and d['meta']['n'] < 2500: continue
            R[d['corpus']].append(d)
    return R


def val(d, k, kind):
    if kind == 'raw': x = d['obs'][k]
    else: x = d['nulls']['M2'][k]['z'] if 'M2' in d['nulls'] else float('nan')
    return None if (isinstance(x, float) and math.isnan(x)) else x


def ci(v):
    v = sorted(v)
    if not v: return 'nan'
    return f'{st.mean(v):.3f} [{v[0]:.3f}-{v[-1]:.3f}]' if abs(v[0]) < 100 else f'{st.mean(v):.0f} [{v[0]:.0f}-{v[-1]:.0f}]'


def main(out=None):
    R = load()
    L = [c for c in R if C.TYPE[c] == 'L']; D = [c for c in R if C.TYPE[c] == 'D']; A = [c for c in R if C.TYPE[c] == 'A']
    I = ['indus_seq_raw', 'indus_seq_strong', 'indus_seq_all']
    lines = []; P = lines.append
    P(f'# Loop 55 cycle 3: strongest L-vs-D discriminator at n=3000, Indus shape. Samples: ' + ', '.join(f'{c} {len(R[c])}' for c in sorted(R)))
    table = []
    for kind in ('raw', 'z'):
        for k in C.STATS:
            lv = [val(d, k, kind) for c in L for d in R[c]]; lv = [x for x in lv if x is not None]
            dv = [val(d, k, kind) for c in D for d in R[c]]; dv = [x for x in dv if x is not None]
            if len(lv) < 5 or len(dv) < 5: continue
            a = auc(lv, dv); sep = max(a, 1 - a); hi = a >= 0.5
            # leave-one-corpus-out: minimum separation when any single L or D corpus is removed
            loco = []
            for drop in L + D:
                l2 = [val(d, k, kind) for c in L if c != drop for d in R[c]]; l2 = [x for x in l2 if x is not None]
                d2 = [val(d, k, kind) for c in D if c != drop for d in R[c]]; d2 = [x for x in d2 if x is not None]
                if l2 and d2:
                    a2 = auc(l2, d2); loco.append(a2 if hi else 1 - a2)
            # per-corpus medians
            med = {c: st.median([x for x in (val(d, k, kind) for d in R[c]) if x is not None] or [float('nan')]) for c in L + D + A + I if c in R}
            # Indus position: share of language-side relative to the D-class 5/95 pct threshold; and percentile within L and D pools
            ds = sorted(dv); q = ds[int(0.95 * (len(ds) - 1))] if hi else ds[int(0.05 * (len(ds) - 1))]
            ind = {}
            for c in I:
                iv = [x for x in (val(d, k, kind) for d in R.get(c, [])) if x is not None]
                if not iv: ind[c] = None; continue
                side = (sum(1 for x in iv if x > q) if hi else sum(1 for x in iv if x < q)) / len(iv)
                pL = st.mean(sum(1 for y in lv if y < x) / len(lv) for x in iv); pD = st.mean(sum(1 for y in dv if y < x) / len(dv) for x in iv)
                ind[c] = {'ci': ci(iv), 'language_side_share': side, 'pct_in_L': pL, 'pct_in_D': pD}
            av = [x for c in A for x in (val(d, k, kind) for d in R[c]) if x is not None]
            aside = (sum(1 for x in av if x > q) if hi else sum(1 for x in av if x < q)) / len(av) if av else float('nan')
            table.append({'stat': k, 'kind': kind, 'auc': a, 'sep': sep, 'direction': 'L>D' if hi else 'L<D', 'loco_min': min(loco) if loco else float('nan'),
                          'L_ci': ci(lv), 'D_ci': ci(dv), 'A_language_side': aside, 'indus': ind, 'medians': med})
    table.sort(key=lambda r: (-min(r['sep'], r['loco_min'] if not math.isnan(r['loco_min']) else 0), -r['sep']))
    P('')
    P('## Ranking by separation (AUC folded to >= 0.5), with leave-one-corpus-out minimum. kind raw = the statistic itself; z = its z against the Markov-2 chain of the sample.')
    P('| rank | statistic | kind | AUC | LOCO min | direction | L mean [range] | D mean [range] | accounting on L side | Indus raw / strong / all: mean [range]; share on L side; percentile in L pool / in D pool |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(table[:25], 1):
        ind = '; '.join(f"{c[6:]}: {v['ci']}, L-side {v['language_side_share']:.2f}, pct L {v['pct_in_L']:.2f} / D {v['pct_in_D']:.2f}" if v else f'{c[6:]}: n/a' for c, v in r['indus'].items())
        P(f"| {i} | {r['stat']} | {r['kind']} | {r['auc']:.3f} | {C.fmt(r['loco_min'])} | {r['direction']} | {r['L_ci']} | {r['D_ci']} | {C.fmt(r['A_language_side'])} | {ind} |")
    P('')
    P('## Per-corpus medians on the top 8 statistics')
    top = table[:8]
    cs = L + D + A + I
    P('| statistic (kind) | ' + ' | '.join(cs) + ' |')
    P('|---|' + '---|' * len(cs))
    for r in top:
        P(f"| {r['stat']} ({r['kind']}) | " + ' | '.join(C.fmt(r['medians'].get(c, float('nan'))) for c in cs) + ' |')
    txt = '\n'.join(lines)
    out = out or os.path.join(C.DARK, 'loop55_c3.txt')
    open(out, 'w').write(txt + '\n'); json.dump(table, open(out.replace('.txt', '.json'), 'w'), default=str)
    print(txt)


if __name__ == '__main__':
    main()
