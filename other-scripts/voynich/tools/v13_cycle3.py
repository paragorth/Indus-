"""v13 cycle 3: (1) real-herbal control (Gerard 1636) through the same pipeline;
(2) robustness of the one surviving Voynich cell (root-zone edges ~ benched gallows) to
transcription (IT2a), paragraph-initial lines, line-count covariates, and segmentation settings.
Usage: python3 v13_cycle3.py VOYNICH_CACHE
"""
import json, os, sys, math, random, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from v13_cycle2 import leaf_test, leaf_pairs, order_key, IF, TF
from v13_cycle1 import build
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

OUTD = os.path.join(DER, 'v13')
log = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)


def gerard_rows():
    g = json.load(open(os.path.join(DER, 'v13_gerard.json')))['pages']
    allw = Counter()
    for v in g.values():
        allw.update(w.lower() for w in v['ocr'])
    rows = []
    rng = random.Random(0)
    for n, v in g.items():
        if 'img' not in v or 'error' in v['img']:
            continue
        ws = [re.sub(r'[^a-z]', '', w.lower()) for w in v['ocr']]
        ws = [w for w in ws if len(w) >= 2]
        t = {'log_words': math.log1p(len(ws)), 'n_words': len(ws),
             'word_len': np.mean([len(w) for w in ws]) if ws else 0.0,
             'cap_share': sum(1 for w in v['ocr'] if w[:1].isupper()) / max(1, len(v['ocr'])),
             'digit_tokens': sum(1 for w in v['ocr'] if re.fullmatch(r'\d{1,2}\.?', w)),
             'types40': float(np.mean([len(set(rng.sample(ws, 40))) for _ in range(20)])) if len(ws) >= 40 else float('nan')}
        rows.append({'folio': 'g%d' % int(n), 'n': int(n), 'meta': {'illus': 'G', 'lang': 'E', 'hand': '1', 'quire': 'g'},
                     'img': v['img'], 'txt': t})
    rows.sort(key=lambda r: r['n'])
    return rows


GTF = ['log_words', 'word_len', 'cap_share', 'digit_tokens', 'types40']


def gerard_pairs(rows, shift=0):
    by = {r['n']: r for r in rows}
    pairs = [(by[n], by[n + 1]) for n in sorted(by) if n % 2 == 0 and n + 1 in by]
    if shift:
        imgs = [(a['img'], b['img']) for a, b in pairs]
        pairs = [(dict(a, img=imgs[(i + shift) % len(pairs)][0]), dict(b, img=imgs[(i + shift) % len(pairs)][1]))
                 for i, (a, b) in enumerate(pairs)]
    return pairs


def leaf_vals(pairs, imgf, txtfn):
    dx, dy = [], []
    for a, b in pairs:
        ya, yb = txtfn(a), txtfn(b)
        if ya is None or yb is None:
            continue
        dx.append(math.log1p(100 * a['img'][imgf]) - math.log1p(100 * b['img'][imgf])); dy.append(ya - yb)
    return np.array(dx), np.array(dy)


def signflip_r(dx, dy, nperm=20000, seed=0, covs=None):
    if covs is not None:  # residualise dy on difference covariates (no intercept: symmetric H0)
        C = np.column_stack(covs)
        b, *_ = np.linalg.lstsq(C, dy, rcond=None); dy = dy - C @ b
    r = (dx @ dy) / math.sqrt((dx @ dx) * (dy @ dy))
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(nperm, len(dx)))
    rn = (S * dx) @ dy / math.sqrt((dx @ dx) * (dy @ dy))
    return r, (np.sum(np.abs(rn) >= abs(r)) + 1) / (nperm + 1), len(dx)


def benched_rate(words):
    gl = [g for w in words for g in glyphs(w)]
    return sum(g in 'TKPF' for g in gl) / len(gl) if gl else None


def feat_variant(args):
    fn, d, z = args
    return features(load_norm(fn), ink_delta=d, zone=z)


if __name__ == '__main__':
    cache = sys.argv[1]
    # ---------------- (1) Gerard real herbal
    g = gerard_rows()
    P('GERARD 1636: %d pages with image+OCR, %d leaf pairs' % (len(g), len(gerard_pairs(g))))
    res = leaf_test(gerard_pairs(g), seed=1, tf=GTF)
    P('  same-leaf contrast: max|r|95=%.3f' % res['max95'])
    R = res['R']
    idx = sorted(((abs(R[i, j]), i, j) for i in range(R.shape[0]) for j in range(R.shape[1])), reverse=True)[:6]
    for _, i, j in idx:
        P('    %s~%s r=%+.2f p=%.4f pFW=%.3f' % (IF[i], GTF[j], R[i, j], res['P'][i, j], res['Pfw'][i, j]))
    for sh in (1, 3):
        rw = leaf_test(gerard_pairs(g, shift=sh), seed=5 + sh, tf=GTF)
        P('  GERARD wrong leaf +%d: min pFW %.3f, cells p<0.01 %d/%d' % (sh, rw['Pfw'].min(), int((rw['P'] < .01).sum()), rw['P'].size))
    json.dump({'R': R.tolist(), 'P': res['P'].tolist(), 'Pfw': res['Pfw'].tolist(), 'tf': GTF}, open(os.path.join(OUTD, 'c3_gerard.json'), 'w'))
    # ---------------- (2) Voynich robustness of bottom_edges ~ benched
    rows, gc = build()
    pairs = leaf_pairs(rows)
    herb = [p for p in pairs if p[0]['meta']['illus'] == 'H']
    P('VOYNICH root-zone edges ~ benched gallows, same-leaf sign-flip test (single pre-named cell now)')
    for nm, pp in (('all leaves', pairs), ('herbal', herb)):
        dx, dy = leaf_vals(pp, 'bottom_edges', lambda r: r['txt']['g_benched'])
        P('  ZL %s: r=%+.3f p=%.4f n=%d' % ((nm,) + signflip_r(dx, dy)))
    # IT2a transcription
    recs = json.load(open(os.path.join(DER, 'IT2a_lines.json')))
    itP = defaultdict(list); itNP = defaultdict(list)
    zlNP = defaultdict(list); nl = Counter(); npar = Counter()
    for r in recs:
        if r['ltype'] != 'L':
            itP[r['folio']] += r['words']
            if not r['para_start']:
                itNP[r['folio']] += r['words']
    for r in json.load(open(os.path.join(DER, 'ZL3b_lines.json'))):
        if r['ltype'] != 'L':
            nl[r['folio']] += 1; npar[r['folio']] += r['para_start']
            if not r['para_start']:
                zlNP[r['folio']] += r['words']
    for nm, src in (('IT2a', itP), ('ZL non-paragraph-initial lines', zlNP), ('IT2a non-paragraph-initial', itNP)):
        for lab, pp in (('all', pairs), ('herbal', herb)):
            dx, dy = leaf_vals(pp, 'bottom_edges', lambda r: benched_rate(src.get(r['folio'], [])))
            P('  %s %s: r=%+.3f p=%.4f n=%d' % ((nm, lab) + signflip_r(dx, dy)))
    # line / paragraph count covariates
    for lab, pp in (('all', pairs), ('herbal', herb)):
        dx, dy = leaf_vals(pp, 'bottom_edges', lambda r: r['txt']['g_benched'])
        okp = [(a, b) for a, b in pp]
        c1 = np.array([nl[a['folio']] - nl[b['folio']] for a, b in okp], float)
        c2 = np.array([npar[a['folio']] - npar[b['folio']] for a, b in okp], float)
        c3 = np.array([math.log1p(100 * a['img']['draw_area']) - math.log1p(100 * b['img']['draw_area']) for a, b in okp])
        P('  ZL %s | d(lines), d(paragraphs), d(draw_area): r=%+.3f p=%.4f n=%d' % ((lab,) + signflip_r(dx, dy, covs=[c1, c2, c3])))
    # segmentation variants (herbal leaves; 2 workers)
    can = json.load(open(os.path.join(DER, 'v13_canvases.json')))['canvases']
    fnm = {'f' + c['label']: os.path.join(cache, c['iiif'].rsplit('/', 1)[1] + '.jpg') for c in can if c['single']}
    fols = sorted({r['folio'] for p in herb for r in p})
    vfile = os.path.join(OUTD, 'c3_variants.json')
    var = json.load(open(vfile)) if os.path.exists(vfile) else {}
    for d, z in ((0.08, 4), (0.13, 4), (0.10, 3), (0.10, 5)):
        key = '%.2f_%d' % (d, z)
        if key not in var:
            with ProcessPoolExecutor(2) as ex:
                fs = list(ex.map(feat_variant, [(fnm[f], d, z) for f in fols]))
            var[key] = dict(zip(fols, fs))
            json.dump(var, open(vfile, 'w'))
        vp = [(dict(a, img=var[key][a['folio']]), dict(b, img=var[key][b['folio']])) for a, b in herb]
        dx, dy = leaf_vals(vp, 'bottom_edges', lambda r: r['txt']['g_benched'])
        P('  segmentation ink_delta=%.2f zone=1/%d herbal: r=%+.3f p=%.4f n=%d' % ((d, z) + signflip_r(dx, dy)))
    open(os.path.join(OUTD, 'c3_log.txt'), 'w').write('\n'.join(log) + '\n')
