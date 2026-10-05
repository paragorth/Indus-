"""v57 cycle 4: kill controls for the gap darkenings and the ink sawtooth.
(a) WORD-SHUFFLE surrogate: each line's ink trace is cut into word spans (word + its
    trailing gap) which are permuted within the line. Word-internal shape (a dark start, a
    light tail) survives; pen-time order does not. Real gap asymmetry minus shuffled = the
    part that needs the writing sequence (re-dips). Same for the context effects
    (darkening before q- words, after -y words).
(b) Word-level sawtooth skew (v57_c1 'arrow') after a word-TYPE fixed-effect
    residualisation (every type with >= 4 tokens gets its own mean), and per page half.
(c) Event-triggered elevation at lags 2-5 (outside the detection window) after the same
    type residualisation."""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v57_lib import Corpus, CK, vglyphs, lglyphs
from v57_c2 import extract, asym
from v57_c1 import page_lines, arrow
from v57_c3 import gap_events, triggered, LAGS


def shuffle_row(row, rng):
    d = row['dark']; ws = row['words']
    lead = d[:ws[0][1]]
    spans = []
    for i, w in enumerate(ws):
        end = ws[i + 1][1] if i + 1 < len(ws) else len(d)
        spans.append((w, d[w[1]:end], w[2] - w[1]))
    perm = rng.permutation(len(spans))
    nd = list(lead); nw = []; npun = []
    for j in perm:
        w, seg, width = spans[j]
        a = len(nd)
        nw.append([w[0], a, a + width, w[3]]); npun.append(row['punct'][j])
        nd += seg
    r = dict(row); r['dark'] = nd; r['words'] = nw; r['punct'] = npun
    return r


def type_resid(C):
    r = C.r.copy()
    by = collections.defaultdict(list)
    for i, w in enumerate(C.word):
        by[w].append(i)
    for w, ii in by.items():
        if len(ii) >= 4:
            r[ii] = r[ii] - r[ii].mean()
    for p in np.unique(C.page):
        m = C.page == p
        r[m] = (r[m] - r[m].mean()) / (r[m].std() + 1e-9)
    return r


def ctx_ratio(ev, key):
    d, l = collections.Counter(), collections.Counter()
    for e in ev:
        (d if e['sg'] > 0 else l)[key(e)] += 1
    return d, l


if __name__ == '__main__':
    rng = np.random.default_rng(5704)
    out = {}
    for which in ['L', 'V']:
        gl = vglyphs if which == 'V' else lglyphs
        rows = extract(which)
        res = {'asym': []}
        for wg, q, tolg in [(3, 0.9, 1.0), (3, 0.97, 1.0), (6, 0.9, 1.0), (1.5, 0.97, 1.0), (3, 0.9, 0.5), (6, 0.97, 0.5)]:
            real, _ = asym(rows, wg, q, tolg, rng, nnull=300)
            sh = []
            for s in range(12):
                rr = [shuffle_row(r, rng) for r in rows]
                a, _ = asym(rr, wg, q, tolg, rng, nnull=2)
                sh.append((a['A_gap'], a['A_in'], a['gap_share_events']))
            sh = np.array(sh)
            row = dict(wg=wg, q=q, tolg=tolg, A_gap=real['A_gap'], A_in=real['A_in'], gap_share=real['gap_share_events'],
                       shuf_A_gap=(float(sh[:, 0].mean()), float(sh[:, 0].std())),
                       shuf_A_in=(float(sh[:, 1].mean()), float(sh[:, 1].std())),
                       shuf_gap_share=(float(sh[:, 2].mean()), float(sh[:, 2].std())),
                       z_gap_vs_shuf=float((real['A_gap'] - sh[:, 0].mean()) / (sh[:, 0].std() + 1e-9)))
            res['asym'].append(row)
            print(which, 'ASYM', json.dumps(row), flush=True)
        # context under shuffle, primary setting
        wg, q, tolg = 3, 0.9, 1.0
        ev = gap_events(rows, wg, q, tolg)
        keys = {'right_q': lambda e: gl(e['right'])[0] == 'q' if which == 'V' else gl(e['right'])[0] in 'pdf',
                'left_y': lambda e: gl(e['left'])[-1] == 'y' if which == 'V' else gl(e['left'])[-1] in 'eu',
                'punct': lambda e: e['punct']}
        ctx = {}
        for name, kf in keys.items():
            d, l = ctx_ratio(ev, kf)
            rr_ = (d[True] / max(1, l[True])) / (d[False] / max(1, l[False]))
            shr = []
            for s in range(12):
                evs = gap_events([shuffle_row(r, rng) for r in rows], wg, q, tolg)
                d2, l2 = ctx_ratio(evs, kf)
                shr.append((d2[True] / max(1, l2[True])) / (d2[False] / max(1, l2[False])))
            ctx[name] = dict(n_true=(d[True], l[True]), odds_real=round(rr_, 3), odds_shuf=(round(float(np.mean(shr)), 3), round(float(np.std(shr)), 3)))
        res['ctx'] = ctx
        print(which, 'CTX', json.dumps(ctx), flush=True)
        # (b) skew with type residual; (c) triggered with type residual
        C = Corpus(which, 'top')
        pl = page_lines(C)
        rt = type_resid(C)
        res['arrow_base'] = arrow(C, C.r, pl, 300)
        res['arrow_type'] = arrow(C, rt, pl, 300)
        for half in (0, 1):
            pls = {p: v for p, v in pl.items() if p % 2 == half}
            res[f'arrow_type_half{half}'] = arrow(C, rt, pls, 300)
        C.r = rt
        D, Lt = triggered(C, ev)
        prof = (np.nanmean(D, 0) - np.nanmean(Lt, 0))
        se = np.sqrt(np.nanvar(D, 0) / len(D) + np.nanvar(Lt, 0) / len(Lt))
        res['trig_type'] = {int(l): (round(float(p), 3), round(float(p / s), 2)) for l, p, s in zip(LAGS, prof, se)}
        print(which, 'ARROW', json.dumps({k: v for k, v in res.items() if k.startswith('arrow') or k == 'trig_type'}), flush=True)
        out[which] = res
    json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
