"""v40 cycle 4: dissect the PEN effect (the previous twin choice in the line predicts the next).
Is it copying (the same word or word-shape repeated), a scribe 'mood' that persists, or alternation?
For each slot with an earlier slot of the same pair in the same line:
  prev = choice of the nearest earlier slot (+1 B / -1 A)
Subsets: (a) earlier slot in a DIFFERENT frame (no copy of the word shape), (b) SAME frame, (c) same word,
         distance bins in glyphs (1-5, 6-15, 16-30, >30), (d) the last slot of the PREVIOUS line (across the
         line break, which a pen state should cross and a line-internal rule should not).
Gain = held-out offset gain (bits/token) over a base model SCRIBE + page + CTX + FRAME, with page-grouped folds;
null = 200 permutations of the choice within (frame x page) strata, prev recomputed each time.
PLANTED: y resampled from the base model, then with prob s copy the previous slot's choice in the line."""
import sys, os, json
import numpy as np
from scipy.sparse import hstack
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v40_lib import *
from v40_cycle1 import vpairs, perm_within


def prev_info(toks):
    """index of nearest earlier same-pair slot in the line (-1 none) and in the previous line's last slot."""
    byline = defaultdict(list)
    for i, t in enumerate(toks):
        byline[(t['page'], t['line'])].append(i)
    prv = np.full(len(toks), -1); pl = np.full(len(toks), -1); first = np.zeros(len(toks), bool)
    for (pg, li), idx in byline.items():
        idx = sorted(idx, key=lambda i: toks[i]['off'])
        before = byline.get((pg, li - 1), [])
        lastp = max(before, key=lambda i: toks[i]['off']) if before else -1
        for k, i in enumerate(idx):
            prv[i] = idx[k - 1] if k else -1
            if k == 0:
                pl[i] = lastp; first[i] = True
    return prv, pl, first


def gains_for(y, p0, f, prv, masks):
    out = {}
    for name, m in masks.items():
        x = np.where(prv >= 0, 2 * y[np.maximum(prv, 0)] - 1, 0).astype(float)
        out[name] = offset_gain(y[m], p0[m], x[m], f[m]) if m.sum() > 100 else np.nan
    return out


def run(name, pages, pname, rng, log, plant=None, nperm=200):
    pairs = {pname: vpairs()[pname]}
    toks = extract(pages, pairs, V_TALL)
    y = np.array([t['y'] for t in toks]); groups = np.array([t['page'] for t in toks])
    fams = [onehot(family_cols(toks, fm)) for fm in ('SCRIBE', 'CTX', 'FRAME')]
    if os.environ.get('V40_BASE_LAYOUT'):
        # stricter base: add the whole LAYOUT family (paragraph-first line, x, offsets, line in paragraph...)
        from v40_cycle1 import build_extra, same_above
        e = build_extra(pages, toks, V_TALL, y); e['same_above'] = same_above(pages, toks, pairs)
        fams.append(onehot(family_cols(toks, 'LAYOUT', e)))
    X = hstack(fams).tocsr()
    p0, f = oof_logloss(X, y, groups)
    prv, pl, first = prev_info(toks)
    if plant is not None:
        y = (rng.random(len(y)) < p0).astype(int)
        for i in sorted(range(len(toks)), key=lambda i: (toks[i]['page'], toks[i]['line'], toks[i]['off'])):
            if prv[i] >= 0 and rng.random() < plant:
                y[i] = y[prv[i]]
        for i, t in enumerate(toks):
            t['y'] = int(y[i])
    fr = np.array([t['frame'] for t in toks]); wd = np.array([t['word'] for t in toks])
    has = prv >= 0
    dist = np.array([toks[i]['off'] - toks[prv[i]]['off'] if prv[i] >= 0 else 0 for i in range(len(toks))])
    pidx = np.maximum(prv, 0)
    masks = {'all': has, 'diff_frame': has & (fr != fr[pidx]), 'same_frame': has & (fr == fr[pidx]),
             'diff_word': has & (wd != wd[pidx]), 'same_word': has & (wd == wd[pidx]),
             'd1-5': has & (dist <= 5), 'd6-15': has & (dist > 5) & (dist <= 15),
             'd16-30': has & (dist > 15) & (dist <= 30), 'd>30': has & (dist > 30),
             'samewordslot': has & (dist <= 5) & (np.array([toks[i]['wi'] for i in range(len(toks))]) == np.array([toks[pidx[i]]['wi'] for i in range(len(toks))]))}
    obs = gains_for(y, p0, f, prv, masks)
    # across the line break: previous line's last slot, first slot of each line only
    mpl = first & (pl >= 0)
    xpl = np.where(pl >= 0, 2 * y[np.maximum(pl, 0)] - 1, 0).astype(float)
    obs['prevline'] = offset_gain(y[mpl], p0[mpl], xpl[mpl], f[mpl])
    # sign / raw agreement
    agree = float((y[has] == y[pidx[has]]).mean())
    nulls = defaultdict(list)
    for _ in range(nperm):
        y2 = perm_within(toks, y, rng, keys=('frame', 'page'))
        g2 = gains_for(y2, p0, f, prv, masks)
        xpl2 = np.where(pl >= 0, 2 * y2[np.maximum(pl, 0)] - 1, 0).astype(float)
        g2['prevline'] = offset_gain(y2[mpl], p0[mpl], xpl2[mpl], f[mpl])
        g2['agree'] = float((y2[has] == y2[pidx[has]]).mean())
        for k, v in g2.items():
            nulls[k].append(v)
    res = {'n': len(y), 'n_has': int(has.sum()), 'agree': agree}
    for k in list(obs) + ['agree']:
        v = np.array(nulls[k]); o = obs[k] if k != 'agree' else agree
        res[k] = dict(obs=float(o), null=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                      p=float((v >= o).mean()), z=float((o - np.nanmean(v)) / (np.nanstd(v) + 1e-12)),
                      n=int(masks[k].sum()) if k in masks else int(mpl.sum()) if k == 'prevline' else int(has.sum()))
    msg = '%s %s%s n=%d with-prev=%d agree %.3f (null %.3f, z %.1f) | ' % (
        name, pname, '' if plant is None else ' PLANT%.2f' % plant, len(y), has.sum(), agree, res['agree']['null'], res['agree']['z']) + \
        ' '.join('%s %.4f(z%.1f,n%d)' % (k, res[k]['obs'], res[k]['z'], res[k]['n']) for k in list(masks) + ['prevline'])
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    rng = np.random.default_rng(404)
    log = open(os.path.join(CKPT, 'cycle4_%s.log' % which), 'a')
    out = {}
    tag = which + ('_baselayout' if os.environ.get('V40_BASE_LAYOUT') else '')
    log = open(os.path.join(CKPT, 'cycle4_%s.log' % tag), 'a')
    if which == 'plant':
        P = load_voynich('ZL3b')
        for s in (0.0, 0.05, 0.1):
            out[s] = run('ZL3b', P, 'KT', rng, log, plant=s, nperm=60)
    else:
        P = load_voynich(which)
        for pn in (('KT', 'CS') if os.environ.get('V40_BASE_LAYOUT') else ('KT', 'CS', 'BENCH', 'PF', 'BKT')):
            out[pn] = run(which, P, pn, rng, log)
    json.dump(out, open(os.path.join(CKPT, 'cycle4_%s.json' % tag), 'w'), indent=1)
