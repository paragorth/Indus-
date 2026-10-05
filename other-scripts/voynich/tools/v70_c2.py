"""v70 cycle 2: do ink alphabets track pages better than EVA, and how do they differ from EVA?

(a) Page tracking on held-out lines. Lines of the 23 Q20 pages (Voynich) are split even/odd
    by line index; a multinomial naive Bayes on unit unigrams+bigrams (inside words) trained on
    even lines predicts the page of odd lines (and vice versa). Compared: every k-means ink
    alphabet (theta x K x m), the matched random-Voronoi null alphabets (same embedding, so the
    same page-level ink drift), EVA units on the same words, and a no-alphabet drift baseline
    (page from the line's mean unit embedding, nearest centroid). Latin: page within manuscript.
(b) EVA anatomy of the ink units: each unit is labelled with the EVA glyphs whose estimated centres
    fall inside its span (glyph widths fitted by least squares on word widths); contingency of
    clusters x EVA strings for the alphabet chosen in cycle 1.
Usage: python3 v70_c2.py V|L|SV
"""
import sys, os, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import *
from v70_c1 import embed, words_from, bpe, nearest, THETAS, KS
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as NMI

Q20 = [f'f{n}{s}' for n in list(range(103, 109)) + list(range(111, 117)) for s in 'rv'] + ['f116r']


def lines_of(D, W, keep_pg):
    L = collections.defaultdict(list)
    for r, w in zip(D['recs'], W):
        if r['pg'] in keep_pg and w:
            L[(r['pg'], r['li'])].append(w)
    return L


def nb_acc(L, alpha=0.5):
    def feats(ws):
        c = collections.Counter()
        for w in ws:
            for u in w:
                c[('u', u)] += 1
            for a, b in zip(w, w[1:]):
                c[('b', a, b)] += 1
        return c
    keys = sorted(L)
    accs = []
    for par in (0, 1):
        tr = [k for k in keys if k[1] % 2 == par]; te = [k for k in keys if k[1] % 2 != par]
        pc = collections.defaultdict(collections.Counter)
        for k in tr:
            pc[k[0]].update(feats(L[k]))
        vocab = set(f for c in pc.values() for f in c)
        V = len(vocab)
        tot = {p: sum(c.values()) for p, c in pc.items()}
        ok = 0
        for k in te:
            f = feats(L[k])
            best = max(pc, key=lambda p: sum(n * np.log((pc[p][x] + alpha) / (tot[p] + alpha * V)) for x, n in f.items()))
            ok += best == k[0]
        accs.append(ok / len(te))
    return float(np.mean(accs))


def drift_acc(E, U, D, keep_pg):
    """page from mean unit embedding per line, nearest centroid on the other parity."""
    by = collections.defaultdict(list)
    for e, u in zip(E, U):
        r = D['recs'][u[0]]
        if r['pg'] in keep_pg:
            by[(r['pg'], r['li'])].append(e)
    M = {k: np.mean(v, 0) for k, v in by.items()}
    accs = []
    for par in (0, 1):
        tr = [k for k in M if k[1] % 2 == par]; te = [k for k in M if k[1] % 2 != par]
        C = {p: np.mean([M[k] for k in tr if k[0] == p], 0) for p in set(k[0] for k in tr)}
        ps = list(C); CM = np.array([C[p] for p in ps])
        accs.append(np.mean([ps[int(np.argmin(((CM - M[k]) ** 2).sum(1)))] == k[0] for k in te]))
    return float(np.mean(accs))


def eva_span_labels(D, U, s_by_pg):
    """label units with EVA glyphs whose centres fall in the unit span (Voynich only)."""
    recs = D['recs']
    # word widths in pitch units are not stored for real pages; use unit widths summed + gaps
    wsum = collections.defaultdict(float); spans = collections.defaultdict(list)
    for i, u in enumerate(U):
        spans[u[0]].append((u[2], u[1], i))
    gl = [vglyphs(r['word'].replace('?', '')) for r in recs]
    inv = sorted(set(g for w in gl for g in w))
    # least-squares glyph widths: total unit width ~ sum glyph widths
    rows, yv = [], []
    for wi, sp in spans.items():
        if gl[wi]:
            c = collections.Counter(gl[wi]); rows.append([c[g] for g in inv]); yv.append(sum(x[1] for x in sp))
    gw = np.linalg.lstsq(np.array(rows, float), np.array(yv), rcond=None)[0]
    gw = np.clip(gw, 0.1, None); gwd = dict(zip(inv, gw))
    lab = [''] * len(U)
    for wi, sp in spans.items():
        g = gl[wi]
        if not g:
            continue
        sp = sorted(sp)
        tot_u = sum(x[1] for x in sp); exp = np.array([gwd[x] for x in g])
        cen = (np.cumsum(exp) - exp / 2) / exp.sum() * tot_u
        x = 0.0
        for pos, wd, i in sp:
            inside = [g[j] for j in range(len(g)) if x <= cen[j] < x + wd]
            lab[i] = '+'.join(inside) if inside else '0'
            x += wd
    return lab, gwd


def main(name):
    out = {}
    D0 = json.load(open(os.path.join(ARR_, name + '.json')))
    if name == 'V':
        keep = {r['pg'] for r in D0['recs'] if r['folio'] in Q20}
        ref_words = [tuple(vglyphs(r['word'].replace('?', ''))) for r in D0['recs']]
    elif name == 'L':
        keep = {r['pg'] for r in D0['recs']}
        ref_words = [tuple(r['word']) for r in D0['recs']]
    else:
        keep = {r['pg'] for r in D0['recs']}
        ref_words = [tuple(r['truth']) for r in D0['recs']]
    out['ref_acc'] = nb_acc(lines_of(D0, ref_words, keep))
    print(name, 'reference transcription page acc', out['ref_acc'], flush=True)
    res = []
    for th in THETAS:
        E, D, U = embed(name, th)
        out[f'drift_t{th}'] = drift_acc(E, U, D, keep)
        for K in KS[::2]:
            for kind in ('km', 'rnd'):
                if kind == 'km':
                    lab = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(E)
                else:
                    rng = np.random.default_rng(1000 + K)
                    lab = nearest(E, E[rng.choice(len(E), K, replace=False)])
                W0 = words_from(lab, U, len(D['recs']))
                for m in (0, 8):
                    W = bpe(W0, m)[0] if m else W0
                    res.append({'theta': th, 'K': K, 'kind': kind, 'm': m, 'acc': nb_acc(lines_of(D, W, keep))})
        print(name, th, out[f'drift_t{th}'], res[-1], flush=True)
    out['rows'] = res
    for kind in ('km', 'rnd'):
        a = [r['acc'] for r in res if r['kind'] == kind]
        out[kind] = {'median': float(np.median(a)), 'max': float(max(a)), 'frac_beat_ref': float(np.mean(np.array(a) > out['ref_acc']))}
    # paired km vs rnd at the same theta, K, m
    pairs = {(r['theta'], r['K'], r['m']): {} for r in res}
    for r in res:
        pairs[(r['theta'], r['K'], r['m'])][r['kind']] = r['acc']
    d = [v['km'] - v['rnd'] for v in pairs.values()]
    out['km_minus_rnd'] = {'mean': float(np.mean(d)), 'frac_pos': float(np.mean(np.array(d) > 0))}
    if name == 'V':
        # EVA anatomy at the cycle-1 best alphabet (theta, K from c1_report)
        rep = json.load(open(os.path.join(CK, 'c1_report.json')))
        b = rep['V']['km']['best']
        E, D, U = embed(name, b['theta'])
        lab = MiniBatchKMeans(b['K'], random_state=b['seed'], n_init=3, batch_size=4096).fit_predict(E)
        ev, gwd = eva_span_labels(D, U, None)
        ok = [i for i, t in enumerate(ev) if t and t != '0']
        out['anatomy'] = {'best': b, 'nmi_vs_eva_spans': float(NMI([ev[i] for i in ok], lab[ok])),
                          'glyph_width_pitch': {k: round(float(v), 3) for k, v in gwd.items()}}
        cl = {}
        for c in range(b['K']):
            idx = [i for i in ok if lab[i] == c]
            cn = collections.Counter(ev[i] for i in idx)
            cl[c] = {'n': len(idx), 'top': cn.most_common(4)}
        out['anatomy']['clusters'] = cl
        # spread of each EVA glyph over clusters (when it is the unit's only glyph)
        sp = {}
        for g in sorted(set(t for t in (ev[i] for i in ok) if '+' not in t)):
            cn = collections.Counter(lab[i] for i in ok if ev[i] == g)
            n = sum(cn.values())
            if n >= 50:
                p = np.array(list(cn.values())) / n
                sp[g] = {'n': n, 'eff_clusters': round(float(2 ** -(p * np.log2(p)).sum()), 2)}
        out['anatomy']['eva_spread'] = sp
        mult = collections.Counter(ev[i].count('+') + 1 for i in ok)
        out['anatomy']['glyphs_per_unit'] = dict(mult)
        # cross-check: same anatomy for SV with exact truth
    json.dump(out, open(os.path.join(CK, f'c2_{name}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows',)}, indent=1)[:5000])


ARR_ = os.path.join(SCR, 'v70', 'arr')
if __name__ == '__main__':
    main(sys.argv[1])
