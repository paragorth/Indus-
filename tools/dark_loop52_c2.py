"""Loop 52, cycle 2: total inventory at each merge threshold, with estimators calibrated on scripts of known size.

Estimators on a corpus (list of texts): Chao1 (bias-corrected), ACE (Good-Turing coverage, rare cut 10),
Good-Turing unseen share f1/N, and a Zipf-Mandelbrot fit p_r = C (r+q)^-s, r = 1..S, where S is chosen so that the
expected frequency spectrum (observed types, f1..f10) under Poisson sampling at the corpus token count matches the data.
Bootstrap CIs by resampling texts (200x; ZM 60x).
Calibration: reference corpora (loop32 library) subsampled to the Indus size (3,000 distinct texts, 30 draws); truth =
the types of the whole reference corpus (a lower bound on its real inventory; for Linear B syllabograms the known
syllabary is 87 + a few rare signs). Bias = estimate / truth.
Indus: seq_raw, seq_strong, seq_all, graph k>=2 (contextual-merged), graph k>=1 (maximal), on all texts and on distinct texts.
Writes data/derived/dark/loop52_cycle2.txt.
"""
import json, collections, random, sys, math
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
LIB = OUT + 'loop32_corpora/'
rng = np.random.default_rng(522)
random.seed(522)


def spectrum(texts):
    c = collections.Counter(s for t in texts for s in t)
    return np.array(sorted(c.values(), reverse=True))


def chao1(cnt):
    S = len(cnt); f1 = int((cnt == 1).sum()); f2 = int((cnt == 2).sum())
    return S + f1 * (f1 - 1) / (2 * (f2 + 1)) if f2 == 0 else S + f1 * f1 / (2 * f2)


def ace(cnt, k=10):
    rare = cnt[cnt <= k]; abund = (cnt > k).sum()
    n_rare = rare.sum(); f1 = (rare == 1).sum()
    if n_rare == 0 or n_rare == f1:
        return chao1(cnt)
    C = 1 - f1 / n_rare
    fk = np.array([(rare == i).sum() for i in range(1, k + 1)])
    g2 = max(len(rare) / C * np.sum(np.arange(1, k + 1) * np.arange(0, k) * fk) / (n_rare * (n_rare - 1)) - 1, 0)
    return abund + len(rare) / C + f1 / C * g2


def gt_unseen(cnt):
    return (cnt == 1).sum() / cnt.sum()


def zm_fit(cnt, S_grid=None, s_grid=np.linspace(0.6, 2.2, 17), q_grid=(0, 1, 2, 4, 8, 16, 32, 64)):
    """Zipf-Mandelbrot inventory: S minimising the distance between the expected spectrum (observed types, f1..f10)
    and the observed one. Returns (S, s, q, loss)."""
    N = cnt.sum(); Sobs = len(cnt)
    fobs = np.array([(cnt == i).sum() for i in range(1, 11)], float)
    if S_grid is None:
        S_grid = np.unique(np.round(Sobs * np.exp(np.linspace(0, np.log(6), 40))).astype(int))
    best = (None, None, None, np.inf)
    for s in s_grid:
        for q in q_grid:
            r = np.arange(1, S_grid.max() + 1)
            w = (r + q) ** (-s)
            cw = np.cumsum(w)
            for S in S_grid:
                lam = N * w[:S] / cw[S - 1]
                e = np.exp(-lam)
                Sexp = np.sum(1 - e)
                fexp = np.array([np.sum(e * lam ** k / math.factorial(k)) for k in range(1, 11)])
                # chi-square style distance on counts
                loss = (Sobs - Sexp) ** 2 / max(Sexp, 1) + np.sum((fobs - fexp) ** 2 / np.maximum(fexp, 1))
                if loss < best[3]:
                    best = (int(S), float(s), float(q), float(loss))
    return best


def estimates(texts, nboot=200, nboot_zm=60):
    cnt = spectrum(texts)
    res = {'S_obs': len(cnt), 'N': int(cnt.sum()), 'f1': int((cnt == 1).sum()), 'f2': int((cnt == 2).sum()),
           'chao1': chao1(cnt), 'ace': ace(cnt), 'gt_unseen': gt_unseen(cnt)}
    zm = zm_fit(cnt); res['zm'] = zm[0]; res['zm_par'] = zm[1:3]
    n = len(texts)
    boot = {'chao1': [], 'ace': [], 'zm': []}
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        c = spectrum([texts[i] for i in idx])
        boot['chao1'].append(chao1(c)); boot['ace'].append(ace(c))
        if b < nboot_zm:
            boot['zm'].append(zm_fit(c, s_grid=np.linspace(0.6, 2.2, 9), q_grid=(0, 2, 8, 32))[0])
    for k, v in boot.items():
        res[k + '_ci'] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return res


def fmt(r):
    return 'S_obs %d N %d f1 %d f2 %d | Chao1 %.0f [%.0f-%.0f] | ACE %.0f [%.0f-%.0f] | ZM %d [%d-%d] (s %.2f q %g) | GT unseen %.3f' % (
        r['S_obs'], r['N'], r['f1'], r['f2'], r['chao1'], *r['chao1_ci'], r['ace'], *r['ace_ci'], r['zm'], *r['zm_ci'], r['zm_par'][0], r['zm_par'][1], r['gt_unseen'])


# ------------------------------------------------------------------ reference corpora
def load(name):
    return [json.loads(l)['seq'] for l in open(LIB + name + '.jsonl')]


def linb_syllabograms(texts):
    out = []
    for t in texts:
        q = [s for s in t if s.islower() or s.startswith('*')]
        if q: out.append(q)
    return out


def strip_variants(texts, keep_compounds=True):
    out = []
    for t in texts:
        q = []
        for s in t:
            if s.startswith('|') and not keep_compounds:
                continue
            q.append(s.split('~')[0])
        if q: out.append(q)
    return out


refs = {}
lb = load('linb_syll')
refs['LinearB_syllabograms'] = (linb_syllabograms(lb), 'L', 'known syllabary 87 (+ ~10 rare/undeciphered); ideograms and NUM removed')
refs['LinearB_syll+ideograms'] = ([t for t in lb if t], 'L', 'syllabograms + ideograms + NUM as given (mixed system)')
refs['UrIII_syllables'] = (load('ur3_syll'), 'L', 'Ur III seal legends as syllable signs (logosyllabic cuneiform in a formulaic genre)')
refs['UrIII_names_syll'] = (load('ur3_names_syll'), 'L', 'Ur III owner names, syllable signs')
pe = load('proto_elamite')
refs['ProtoElamite_raw'] = (pe, 'A', 'Dahl sign list with variant forms X~a.. and compounds, as Wells raw')
refs['ProtoElamite_basemerged'] = (strip_variants(pe), 'A', 'variants X~a folded into X (compounds kept): as a merged inventory')
pc = load('proto_cuneiform')
refs['ProtoCuneiform_raw'] = (pc, 'A', 'CDLI qpc sign names with ~variants')
refs['ProtoCuneiform_basemerged'] = (strip_variants(pc), 'A', 'variants folded into base sign')
refs['ChessECO_moves'] = (load('chess_eco'), 'G', 'grammar without language (closed move inventory)')
refs['Khipu'] = (load('khipu'), 'A', 'cord clusters colour:magnitude')

out = ['# Loop 52 cycle 2: inventory estimators calibrated on known systems, then applied to the Indus ladder']
NTEXT = 3000
NDRAW = 30
out.append('\n## calibration: reference corpora subsampled to %d distinct texts, %d draws; truth = whole-corpus types (lower bound on the real inventory)' % (NTEXT, NDRAW))
out.append('| corpus | type | whole: texts / tokens / types | sub: tokens / types (coverage) | Chao1/truth | ACE/truth | ZM/truth | GT unseen | note |')
calib = {}
for name, (texts, typ, note) in refs.items():
    distinct = list({tuple(t): t for t in texts}.values())
    truth = len(set(s for t in texts for s in t))
    rat = {'chao1': [], 'ace': [], 'zm': [], 'cov': [], 'tok': [], 'gt': []}
    for d in range(NDRAW):
        if len(distinct) > NTEXT:
            sub = random.sample(distinct, NTEXT)
        else:
            sub = distinct
        cnt = spectrum(sub)
        rat['chao1'].append(chao1(cnt) / truth); rat['ace'].append(ace(cnt) / truth); rat['cov'].append(len(cnt) / truth); rat['tok'].append(cnt.sum()); rat['gt'].append(gt_unseen(cnt))
        if d < 8:
            rat['zm'].append(zm_fit(cnt, s_grid=np.linspace(0.6, 2.2, 9), q_grid=(0, 2, 8, 32))[0] / truth)
    calib[name] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rat.items()}
    c = calib[name]
    out.append('| %s | %s | %d / %d / %d | %.0f / %.0f (%.2f) | %.2f +/- %.2f | %.2f +/- %.2f | %.2f +/- %.2f | %.3f | %s |' % (
        name, typ, len(distinct), sum(len(t) for t in texts), truth, c['tok'][0], c['cov'][0] * truth, c['cov'][0], c['chao1'][0], c['chao1'][1], c['ace'][0], c['ace'][1], c['zm'][0], c['zm'][1], c['gt'][0], note))
    print(out[-1], flush=True)

# ------------------------------------------------------------------ Indus ladder
corpus = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
merges = json.load(open(OUT + 'loop52_merges.json'))
levels = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']


def apply_classes(texts, classes):
    m = {}
    for cl in classes:
        h = min(cl, key=lambda s: -tokall[s])
        for s in cl: m[s] = h
    return [[m.get(s, s) for s in t] for t in texts]


tokall = collections.Counter(s for r in corpus for s in r['seq_raw'])
ladder = [('raw Wells', [r['seq_raw'] for r in corpus], None),
          ('canonical strong', [r['seq_strong'] for r in corpus], None),
          ('canonical all', [r['seq_all'] for r in corpus], None),
          ('contextual-merged (all + graph k>=2)', [r['seq_raw'] for r in corpus], merges['classes']['all_k2']),
          ('maximal-merged (all + graph k>=1)', [r['seq_raw'] for r in corpus], merges['classes']['all_k1']),
          ('strict-merged (all + graph k>=3)', [r['seq_raw'] for r in corpus], merges['classes']['all_k3'])]
out.append('\n## Indus ladder, all %d texts (%d tokens) and distinct texts' % (len(corpus), sum(tokall.values())))
results = {}
for label, texts, classes in ladder:
    if classes is not None:
        texts = apply_classes(texts, classes)
    texts = [t for t in texts if t]
    distinct = list({tuple(t): list(t) for t in texts}.values())
    r_all = estimates(texts); r_dist = estimates(distinct)
    results[label] = {'all': r_all, 'distinct': r_dist}
    out.append('\n### %s' % label)
    out.append('  all texts:      ' + fmt(r_all))
    out.append('  distinct texts: ' + fmt(r_dist))
    print(label, fmt(r_dist), flush=True)

# calibrated corrections: divide each estimator by its mean bias on the reference corpora of each class
out.append('\n## bias-corrected totals (distinct texts): estimate / mean bias ratio of the reference class')
groups = {'language-writing (LinB syll, UrIII syll, UrIII names)': ['LinearB_syllabograms', 'UrIII_syllables', 'UrIII_names_syll'],
          'accounting raw lists (PE raw, PC raw)': ['ProtoElamite_raw', 'ProtoCuneiform_raw'],
          'accounting base-merged (PE, PC)': ['ProtoElamite_basemerged', 'ProtoCuneiform_basemerged'],
          'all references': list(refs)}
out.append('| ladder step | S_obs | Chao1 | ACE | ZM | ' + ' | '.join('corrected (%s): Chao1 / ACE / ZM' % g for g in groups) + ' |')
for label in results:
    r = results[label]['distinct']
    cells = []
    for g, names in groups.items():
        bc = np.mean([calib[n]['chao1'][0] for n in names]); ba = np.mean([calib[n]['ace'][0] for n in names]); bz = np.mean([calib[n]['zm'][0] for n in names])
        cells.append('%.0f / %.0f / %.0f' % (r['chao1'] / bc, r['ace'] / ba, r['zm'] / bz))
    out.append('| %s | %d | %.0f | %.0f | %d | %s |' % (label, r['S_obs'], r['chao1'], r['ace'], r['zm'], ' | '.join(cells)))

# where does the Indus observed spectrum sit relative to references at matched n: unseen share and f1/S
out.append('\n## spectrum shape at n = %d texts: Good-Turing unseen share and singleton share (f1/S_obs)' % NTEXT)
for name, (texts, typ, note) in refs.items():
    distinct = list({tuple(t): t for t in texts}.values())
    sub = random.sample(distinct, NTEXT) if len(distinct) > NTEXT else distinct
    cnt = spectrum(sub)
    out.append('  %-28s GT unseen %.3f  f1/S %.2f  S_obs %d  N %d' % (name, gt_unseen(cnt), (cnt == 1).sum() / len(cnt), len(cnt), cnt.sum()))
for label in results:
    texts = None
for label, texts, classes in ladder:
    if classes is not None:
        texts = apply_classes(texts, classes)
    distinct = list({tuple(t): list(t) for t in texts if t}.values())
    sub = random.sample(distinct, NTEXT) if len(distinct) > NTEXT else distinct
    cnt = spectrum(sub)
    out.append('  Indus %-22s GT unseen %.3f  f1/S %.2f  S_obs %d  N %d' % (label[:22], gt_unseen(cnt), (cnt == 1).sum() / len(cnt), len(cnt), cnt.sum()))

open(OUT + 'loop52_cycle2.txt', 'w').write('\n'.join(out) + '\n')
json.dump({'calibration': calib, 'indus': results}, open(OUT + 'loop52_cycle2.json', 'w'), indent=0, default=float)
print('\n'.join(out))
