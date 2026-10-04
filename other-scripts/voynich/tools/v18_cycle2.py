"""v18 cycle 2: known-language control and content-artifact control.

(a) LATIN: the same dip search and reset tests on 14th-15th c. Latin manuscripts with
    line transcriptions (HTR-United CREMMA-Medieval-Lat: Phi 10a135, Laur. Plut. 39.34,
    53.08, 53.09, Montpellier H318, Clm 13027, Arras 861); glyph = letter. Page-swap
    surrogates within the Latin set. If dips 'reset' Latin too, the test reads pauses,
    not generators.
(b) CONTENT-ONLY darkness (Voynich): darkness replaced by its prediction from the word's
    own glyph content plus page-shuffled residual noise; any 'reset' found then is a
    measurement artefact of what the word looks like, not of the pen.
(c) Planted reset in the Latin layout (synthetic Latin-letter text, same generator class)
    to show the Latin pipeline has power too.
Checkpoints: data/results/v18/c2_*.json
"""
import sys, os, json, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
from v18_cycle1 import summarise, CANON
D = os.path.join(HERE, '..', 'data', 'derived')
RES = os.path.join(HERE, '..', 'data', 'results', 'v18')


def lglyphs(w):
    return list(w)


def latin():
    pages = json.load(open(os.path.join(D, 'v18_latin_words.json')))
    pages = [p for p in pages if len(p['words']) >= 60]
    lines = collections.OrderedDict()
    for p in pages:
        for w in sorted(p['words'], key=lambda w: (w['li'], w['k'])):
            lines.setdefault((p['folio'], w['li']), []).append(w['word'])
    C = Corpus(pages, lglyphs, list(lines.values()))
    return C, pages


def run_family(C, tag, out, planted=True):
    t0 = time.time()
    ds = all_dips(C)
    print(tag, 'dips', len(ds), round(time.time() - t0), flush=True)
    can = [i for st, i in ds if st == CANON][0]
    d = np.zeros(C.N, bool); d[can] = True
    out[tag] = {'N': C.N, 'pages': C.npages, 'canon_spacing': spacing_stats(C, d)}
    maps = [page_swap_map(C, k) for k in range(1, C.npages)]
    prim = [x for x in ds if x[0][3] in CLEAN and x[0][1]]
    out[tag]['primary'] = summarise(evaluate(C, prim, maps), C.npages)
    print(tag, 'primary', out[tag]['primary'], flush=True)
    out[tag]['full'] = summarise(evaluate(C, ds, maps), C.npages)
    print(tag, 'full', {k: out[tag]['full'][k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'best')}, flush=True)
    return ds, prim, maps, can


def main():
    out = {}
    fn = os.path.join(RES, 'c2.json')
    # (a) Latin
    C, pages = latin()
    ds, prim, maps, can = run_family(C, 'latin', out)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (c) planted in Latin layout: synthetic text from a letter-junction generator
    canm = np.zeros(C.N, bool); canm[can] = True
    res = {}
    for strength in (1.0, 0.0):
        rr = []
        for trial in range(2):
            rng = np.random.default_rng(300 + trial)
            mask = np.zeros(C.N, bool)
            for t in np.where(canm)[0]:
                if rng.random() < strength:
                    u = t + (rng.choice([-1, 1]) if rng.random() < 0.2 else 0)
                    if 0 <= u < C.N:
                        mask[u] = True
            G = Generator(C, C.lines_of_text(), seed=400 + trial)
            words = G.text(mask)
            Cs, _ = latin(); Cs.word = words; Cs.fit_model(Cs.lines_of_text()); Cs.features()
            s = summarise(evaluate(Cs, prim, maps), C.npages)
            rr.append({k: s[k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'best')})
            print('latin planted', strength, trial, rr[-1], flush=True)
        res[str(strength)] = rr
    out['latin_planted'] = res
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (b) content-only darkness, Voynich
    vp = json.load(open(os.path.join(D, 'v18_words.json')))
    ZL = json.load(open(os.path.join(D, 'ZL3b_lines.json')))
    model = [r['words'] for r in ZL if r['ltype'] == 'P']
    V = Corpus(vp, glyphs, model)
    Cm = V.content_matrix()
    rng = np.random.default_rng(7)
    for key in ('med', 'top', 'p90', 'gray', 'areag', 'rb'):
        y = np.array([w['m'][key] if w['m'] else np.nan for w in V.W])
        ok = ~np.isnan(y)
        X = np.hstack([np.ones((ok.sum(), 1)), Cm[ok]])
        b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        pred = np.full(V.N, np.nan); pred[ok] = X @ b
        res_ = y - pred
        # residual noise shuffled across pages (keeps its distribution, kills pen order)
        perm = np.where(ok)[0].copy(); rng.shuffle(perm)
        noise = np.full(V.N, np.nan); noise[np.where(ok)[0]] = res_[perm]
        fake = pred + noise
        for i, w in enumerate(V.W):
            if w['m']:
                w['m'] = dict(w['m']); w['m'][key] = float(fake[i])
    V._C = None
    run_family(V, 'content_only', out)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    print('done')


if __name__ == '__main__':
    main()
