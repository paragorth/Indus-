"""v57 cycle 3: are the gap darkenings re-dips, and what do they separate?
(a) Event-triggered ink: word-level residual darkness (v57_lib) at reading-order lags
    -6..+12 around darkening gaps vs lightening gaps (the mirror null). A re-dip should be
    followed by a sustained, slowly fading elevation (a pen load), not one dark word.
(b) Context of re-dip gaps: does the left word's last glyph / right word's first glyph /
    word identity differ between darkening and lightening gaps? Null: per-line mirror
    (dark<->light labels swapped for random lines). Latin control: punctuation.
(c) Phrase units: word strings between consecutive darkening gaps on a page; do they
    recur more than strings cut at lightening gaps, or at randomly shifted gaps?"""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v57_lib import Corpus, CK, vglyphs, lglyphs
from v57_c2 import extract, events, classify

LAGS = np.arange(-6, 13)


def gap_events(rows, wg, q, tolg):
    ev = []
    for row in rows:
        for x, sg, a in events(row, wg, q):
            cl = classify(row, x, tolg * row['unit'])
            if cl[0] == 'gap':
                k = cl[1]
                ev.append(dict(folio=row['folio'], li=row['li'], kl=row['words'][k][3],
                               kr=row['words'][k + 1][3], left=row['words'][k][0], right=row['words'][k + 1][0],
                               sg=sg, a=a, punct=row['punct'][k], nwl=len(row['words']), pos=k))
    return ev


def triggered(C, ev):
    idx = {}
    for i, w in enumerate(C.W):
        idx[(C.pages[w['pi']], w['li'], w['k'])] = i
    # reading-order position within page
    pos = np.zeros(C.N, int); pagestart = {}
    for p in np.unique(C.page):
        ii = np.where(C.page == p)[0]
        pos[ii] = np.arange(len(ii)); pagestart[p] = ii
    T = {1: [], -1: []}
    for e in ev:
        i = idx.get((e['folio'], e['li'], e['kr']))
        if i is None:
            continue
        ii = pagestart[C.page[i]]; j = pos[i]
        v = np.full(len(LAGS), np.nan)
        for t, l in enumerate(LAGS):
            if 0 <= j + l < len(ii):
                v[t] = C.r[ii[j + l]]
        T[e['sg']].append(v)
    D, Lt = np.array(T[1]), np.array(T[-1])
    return D, Lt


def sustained(D, Lt, rng, nb=2000):
    win = (LAGS >= 2) & (LAGS <= 8)
    a = np.nanmean(D[:, win], 1); b = np.nanmean(Lt[:, win], 1)
    obs = np.nanmean(a) - np.nanmean(b)
    allv = np.r_[a, b]; n = len(a)
    nul = []
    for _ in range(nb):
        pm = rng.permutation(len(allv))
        nul.append(np.nanmean(allv[pm[:n]]) - np.nanmean(allv[pm[n:]]))
    nul = np.array(nul)
    return float(obs), float(obs / nul.std())


def context_test(ev, key, rng, nb=2000, mincount=15):
    """Chi-square-like divergence between dark and light gap distributions over key(e);
    null: random relabelling of whole lines (mirror)."""
    lines = collections.defaultdict(list)
    for e in ev:
        lines[(e['folio'], e['li'])].append(e)
    L = list(lines.values())
    cats = collections.Counter(key(e) for e in ev)
    keep = {c for c, n in cats.items() if n >= mincount}
    def stat(flip):
        d, l = collections.Counter(), collections.Counter()
        for f, es in zip(flip, L):
            for e in es:
                c = key(e) if key(e) in keep else '_other'
                s = e['sg'] * (-1 if f else 1)
                (d if s > 0 else l)[c] += 1
        nd, nl = sum(d.values()), sum(l.values())
        chi = 0.0; cells = {}
        for c in set(d) | set(l):
            ed = (d[c] + l[c]) * nd / (nd + nl); el = (d[c] + l[c]) * nl / (nd + nl)
            chi += (d[c] - ed) ** 2 / max(ed, 1e-9) + (l[c] - el) ** 2 / max(el, 1e-9)
            cells[c] = (d[c], l[c], (d[c] - ed) / np.sqrt(max(ed, 1e-9)))
        return chi, cells
    obs, cells = stat(np.zeros(len(L), bool))
    nul = np.array([stat(rng.random(len(L)) < 0.5)[0] for _ in range(nb)])
    top = sorted(cells.items(), key=lambda kv: -kv[1][2])
    return dict(chi=round(obs, 1), null_mu=round(float(nul.mean()), 1), null_sd=round(float(nul.std()), 1),
                z=round(float((obs - nul.mean()) / nul.std()), 2), p=float((nul >= obs).mean()),
                ncat=len(keep), top_dark=[(c, v[0], v[1], round(v[2], 2)) for c, v in top[:6]],
                top_light=[(c, v[0], v[1], round(v[2], 2)) for c, v in top[-4:]])


def segments(rows, ev, sgn, shift=0, rng=None):
    """Word strings between consecutive gaps with sign sgn, per page (reading order)."""
    cut = collections.defaultdict(set)
    for e in ev:
        if e['sg'] == sgn:
            cut[(e['folio'], e['li'])].add(e['pos'])
    segs = []
    bypage = collections.defaultdict(list)
    for row in rows:
        bypage[row['folio']].append(row)
    for fo, rs in bypage.items():
        cur = []
        for row in sorted(rs, key=lambda r: r['li']):
            n = len(row['words'])
            cs = cut.get((fo, row['li']), set())
            if shift and rng is not None:
                cs = {int(rng.integers(0, n - 1)) for _ in cs}
            for k, w in enumerate(row['words']):
                cur.append(w[0])
                if k in cs:
                    segs.append(tuple(cur)); cur = []
        if cur:
            segs.append(tuple(cur))
    return segs


def recurrence(segs, maxlen=4):
    """Share of word bigrams that straddle no cut and recur; plus share of segment-initial
    words that are segment-initial elsewhere (a 'phrase opener' consistency)."""
    ini = collections.Counter(s[0] for s in segs if s)
    allw = collections.Counter(w for s in segs for w in s)
    # opener consistency: for words seen >=5 times, fraction of their tokens that open a segment
    rate = [ini[w] / allw[w] for w in allw if allw[w] >= 5]
    tot_rate = sum(ini.values()) / sum(allw.values())
    # concentration of openers (variance of opener rate across types beyond binomial)
    disp = np.var(rate) / max(1e-9, tot_rate * (1 - tot_rate) / 5)
    short = [s for s in segs if 1 <= len(s) <= maxlen]
    c = collections.Counter(short)
    rec = sum(n for s, n in c.items() if n > 1) / max(1, len(short))
    return dict(nseg=len(segs), mean_len=float(np.mean([len(s) for s in segs])), opener_disp=float(disp),
                short_recur=float(rec))


if __name__ == '__main__':
    rng = np.random.default_rng(5703)
    out = {}
    for which in ['L', 'V']:
        rows = extract(which)
        C = Corpus(which, 'top')
        gl = vglyphs if which == 'V' else lglyphs
        res = []
        for wg, q, tolg in [(3, 0.9, 1.0), (3, 0.97, 1.0), (6, 0.9, 1.0), (1.5, 0.97, 1.0), (6, 0.97, 0.5)]:
            ev = gap_events(rows, wg, q, tolg)
            D, Lt = triggered(C, ev)
            prof = (np.nanmean(D, 0) - np.nanmean(Lt, 0)).round(3).tolist()
            sus = sustained(D, Lt, rng)
            r = dict(wg=wg, q=q, tolg=tolg, ndark=len(D), nlight=len(Lt), trig_diff=dict(zip(LAGS.tolist(), prof)),
                     sustained_2to8=sus)
            r['ctx_leftlast'] = context_test(ev, lambda e: gl(e['left'])[-1], rng, 1000)
            r['ctx_rightfirst'] = context_test(ev, lambda e: gl(e['right'])[0], rng, 1000)
            r['ctx_rightword'] = context_test(ev, lambda e: e['right'], rng, 1000, mincount=10)
            r['ctx_leftword'] = context_test(ev, lambda e: e['left'], rng, 1000, mincount=10)
            r['ctx_linepos'] = context_test(ev, lambda e: min(e['pos'], 5) if e['pos'] < e['nwl'] - 6 else 10 + (e['nwl'] - 2 - e['pos']), rng, 1000)
            if which == 'L':
                r['ctx_punct'] = context_test(ev, lambda e: e['punct'], rng, 1000)
            sd = recurrence(segments(rows, ev, 1)); sl = recurrence(segments(rows, ev, -1))
            sh = [recurrence(segments(rows, ev, 1, shift=1, rng=rng)) for _ in range(50)]
            r['seg_dark'] = sd; r['seg_light'] = sl
            r['seg_shift'] = {k: (round(float(np.mean([s[k] for s in sh])), 4), round(float(np.std([s[k] for s in sh])), 4)) for k in sd}
            res.append(r)
            print(which, json.dumps(r), flush=True)
        out[which] = res
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
