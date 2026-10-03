"""S-DARK-32 cycle 2/3: can the typology be learned at all? Nested feature selection inside leave-one-corpus-out, stronger
regularisation, binary 'language writing or not', numeral features dropped (annotation-dependent), reliability of the LOCO
probabilities, and only then Indus. Cycle 3 adds a granularity-matched library (corpora whose vocabulary growth is within
a factor of the Indus value) and feature-family ablations.
Reads data/derived/dark/loop32_features.json (tools/dark_loop32.py features).
Usage: python3 tools/dark_loop32_c2.py [--nonum] [--gran LO HI] [--nperm N] [--out name]
"""
import os, sys, json, math, random, collections, statistics as st
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from dark_loop32 import TYPE, TRAIN3, TRAIN_EXCLUDE, LEN_FEATS, fit_logreg, predict
OUTD = os.path.join(ROOT, 'data/derived/dark')
A = sys.argv[1:]
NONUM = '--nonum' in A
GRAN = (float(A[A.index('--gran') + 1]), float(A[A.index('--gran') + 2])) if '--gran' in A else None
NPERM = int(A[A.index('--nperm') + 1]) if '--nperm' in A else 30
OUTNAME = A[A.index('--out') + 1] if '--out' in A else 'loop32_c2'
DROP = {'n_texts', 'bits_bigram', 'bits_slot'} | LEN_FEATS | ({'num_share', 'num_end_bias', 'num_run_share'} if NONUM else set())

rows = json.load(open(os.path.join(OUTD, 'loop32_features.json')))
feats = sorted(k for k in rows[0]['feats'] if k not in DROP)
corp = sorted({r['corpus'] for r in rows}, key=lambda c: (TYPE[c], c))
X = {c: np.array([[r['feats'][f] for f in feats] for r in rows if r['corpus'] == c]) for c in corp}
MEAN = {c: X[c].mean(axis=0) for c in corp}
gi = feats.index('types_per_2000tok')
indus_g = np.mean([MEAN[c][gi] for c in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all', 'indus_im77')])
train_c = [c for c in corp if TYPE[c] in TRAIN3 and c not in TRAIN_EXCLUDE]
if GRAN:
    train_c = [c for c in train_c if GRAN[0] * indus_g <= MEAN[c][gi] <= GRAN[1] * indus_g]
tests = [c for c in corp if c not in train_c and c not in TRAIN_EXCLUDE] + ['runes_words']
out = []; P = out.append
P(f'S-DARK-32 cycle {"3 (granularity-matched)" if GRAN else "2"}: nested feature selection inside LOCO. numerals {"DROPPED" if NONUM else "kept"}; '
  f'{len(feats)} candidate features; training corpora {len(train_c)}: ' + ', '.join(f'{c}[{TRAIN3[TYPE[c]]}]' for c in train_c))
if GRAN:
    P(f'  granularity window: types per 2,000 tokens within [{GRAN[0]}, {GRAN[1]}] x Indus ({indus_g:.0f}) -> [{GRAN[0]*indus_g:.0f}, {GRAN[1]*indus_g:.0f}]; '
      'dropped: ' + ', '.join(f'{c} ({MEAN[c][gi]:.0f})' for c in corp if TYPE[c] in TRAIN3 and c not in TRAIN_EXCLUDE and c not in train_c))


def fstat(cs, labels, i):
    """one-way ANOVA F over CORPUS MEANS (one point per corpus) for feature i"""
    groups = collections.defaultdict(list)
    for c in cs: groups[labels[c]].append(MEAN[c][i])
    allv = [v for g in groups.values() for v in g]; gm = st.mean(allv); k = len(groups); n = len(allv)
    if k < 2 or n <= k: return 0.0
    ssb = sum(len(g) * (st.mean(g) - gm) ** 2 for g in groups.values())
    ssw = sum((v - st.mean(g)) ** 2 for g in groups.values() for v in g)
    return (ssb / (k - 1)) / (ssw / (n - k) + 1e-12)


def select(cs, labels, k):
    return sorted(range(len(feats)), key=lambda i: -fstat(cs, labels, i))[:k]


def loco(labels, classes, k, lam, cs=None, knn=False):
    cs = cs or train_c; ci = {c: i for i, c in enumerate(classes)}; res = {}; sel_all = collections.Counter()
    for c in cs:
        tr = [d for d in cs if d != c]
        keep = select(tr, labels, k); sel_all.update(feats[i] for i in keep)
        Atr = np.vstack([X[d][:, keep] for d in tr]); mu, sd = Atr.mean(axis=0), Atr.std(axis=0) + 1e-9
        if knn:
            Zt = {d: ((MEAN[d][keep] - mu) / sd) for d in tr}; zc = (MEAN[c][keep] - mu) / sd
            nn = sorted(tr, key=lambda d: np.linalg.norm(zc - Zt[d]))[:3]
            votes = collections.Counter(labels[d] for d in nn); p = np.array([votes[x] / 3 for x in classes]); res[c] = p
        else:
            W = fit_logreg((Atr - mu) / sd, np.array([ci[labels[d]] for d in tr for _ in range(len(X[d]))]), len(classes), lam=lam, iters=2000)
            res[c] = predict(W, (X[c][:, keep] - mu) / sd).mean(axis=0)
    acc = sum(1 for c in cs if classes[int(np.argmax(res[c]))] == labels[c]) / len(cs)
    return acc, res, sel_all


def report(title, labels, classes, grid, knn=False):
    P(f'\n== {title} ==')
    best = None
    for k in grid['k']:
        for lam in grid['lam']:
            acc, res, sel = loco(labels, classes, k, lam, knn=knn)
            P(f'  k={k:2d} lam={lam:4g}: LOCO accuracy {acc:.2f}' + ('' if not knn else ' (3-NN on corpus means)'))
            if best is None or acc > best[0]: best = (acc, k, lam, res, sel)
            if knn: break
    acc, k, lam, res, sel = best
    maj = max(collections.Counter(labels[c] for c in train_c).values()) / len(train_c)
    P(f'  best: k={k} lam={lam} accuracy {acc:.2f} (majority-class chance {maj:.2f}); features chosen across folds: ' + ', '.join(f'{f} x{n}' for f, n in sel.most_common(10)))
    for c in train_c:
        P(f'    {c:16s} true {labels[c]} P=' + ' '.join(f'{classes[i]} {res[c][i]:.2f}' for i in range(len(classes))) + ('' if classes[int(np.argmax(res[c]))] == labels[c] else '  WRONG'))
    # permutation null for the best config, nested selection included
    R = random.Random(7); accs = []
    for _ in range(NPERM):
        vals = [labels[c] for c in train_c]; R.shuffle(vals); lab2 = dict(zip(train_c, vals))
        accs.append(loco(lab2, classes, k, lam, knn=knn)[0])
    pv = (sum(1 for a in accs if a >= acc) + 1) / (len(accs) + 1)
    P(f'  label-permutation null ({NPERM}, same nested selection): mean {st.mean(accs):.2f}, 95th pct {sorted(accs)[int(0.95*len(accs))]:.2f}, max {max(accs):.2f}; P(null >= {acc:.2f}) = {pv:.3f}')
    # reliability of LOCO probabilities
    conf = [(max(res[c]), classes[int(np.argmax(res[c]))] == labels[c]) for c in train_c]
    hi = [ok for p, ok in conf if p >= 0.8]
    P(f'  reliability: of {len(hi)} held-out corpora given P >= 0.80, {sum(hi)} were right; of {len(conf)-len(hi)} given P < 0.80, {sum(ok for p, ok in conf if p < 0.8)} right')
    return best, pv


def classify_tests(best, labels, classes, title):
    acc, k, lam, res, sel = best; ci = {c: i for i, c in enumerate(classes)}
    keep = select(train_c, labels, k)
    Atr = np.vstack([X[d][:, keep] for d in train_c]); mu, sd = Atr.mean(axis=0), Atr.std(axis=0) + 1e-9
    W = fit_logreg((Atr - mu) / sd, np.array([ci[labels[d]] for d in train_c for _ in range(len(X[d]))]), len(classes), lam=lam, iters=2000)
    P(f'\n== {title}: unknowns and controls with the best config (k={k}: {", ".join(feats[i] for i in keep)}; lam={lam}); LOCO accuracy of this config {acc:.2f} ==')
    Zt = {d: (MEAN[d][keep] - mu) / sd for d in train_c}
    for c in tests:
        Pr = predict(W, (X[c][:, keep] - mu) / sd); m = Pr.mean(axis=0); lo = Pr.min(axis=0); hi = Pr.max(axis=0)
        zc = (MEAN[c][keep] - mu) / sd
        nn = sorted(((np.linalg.norm(zc - Zt[d]), d) for d in train_c))[:4]
        P(f'  {c:16s} ' + ' '.join(f'P({classes[i]})={m[i]:.2f}[{lo[i]:.2f}-{hi[i]:.2f}]' for i in range(len(classes))) + f' -> {classes[int(np.argmax(m))]}; nearest: ' + ', '.join(f'{d}[{labels[d]}] {v:.1f}' for v, d in nn))
    # per selected feature: Indus value vs class means
    P('  selected features, Indus (mean of 4 versions) vs class means over corpora:')
    ind = np.mean([MEAN[c][keep] for c in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all', 'indus_im77')], axis=0)
    for j, i in enumerate(keep):
        cm = {cl: st.mean(MEAN[d][i] for d in train_c if labels[d] == cl) for cl in classes}
        rng = {cl: (min(MEAN[d][i] for d in train_c if labels[d] == cl), max(MEAN[d][i] for d in train_c if labels[d] == cl)) for cl in classes}
        P(f'    {feats[i]:24s} Indus {ind[j]:7.3f} | ' + ' | '.join(f'{cl} {cm[cl]:7.3f} [{rng[cl][0]:.2f}-{rng[cl][1]:.2f}]' for cl in classes))


grid = {'k': [3, 5, 8, 12], 'lam': [1, 10, 50]}
lab3 = {c: TRAIN3[TYPE[c]] for c in train_c}
best3, p3 = report('Three classes L / D / A, logistic regression', lab3, ['L', 'D', 'A'], grid)
bestk, pk = report('Three classes, 3-NN on corpus means', lab3, ['L', 'D', 'A'], {'k': [3, 5, 8, 12], 'lam': [1]}, knn=True)
lab2 = {c: ('L' if TRAIN3[TYPE[c]] == 'L' else 'N') for c in train_c}
best2, p2 = report('Binary: language writing (L) vs not (N = designed, grammar-only, accounting)', lab2, ['L', 'N'], grid)
classify_tests(best3, lab3, ['L', 'D', 'A'], 'Three-class')
classify_tests(best2, lab2, ['L', 'N'], 'Binary')
if not GRAN:
    # feature-family ablation on the binary question (which family carries the L / not-L signal?)
    FAM = {'order': ['fixed_pair_share', 'free_pair_share', 'fixed_inst_share', 'free_inst_share', 'anagram_rate'],
           'ends/slots': ['H_first/H_mid', 'H_last/H_mid', 'H_second/H_mid', 'H_penult/H_mid', 'n80_first/n80_all', 'n80_last/n80_all', 'slot_exclusivity', 'top10_last_share', 'top10_first_share', 'frames_per_text', 'top10_frame_share', 'slot_minus_bigram_bits'],
           'lexicon': ['zipf_slope', 'hapax_type_share', 'types_per_2000tok', 'H_uni_norm', 'unique_share', 'uniq_vs_bigram', 'repeat_ratio', 'nest_share'],
           'adjacency': ['bigram_MI/H1', 'h2/H1'],
           'complement/determinative': ['hubs_per_sign', 'hub_free_median', 'hub_mobility', 'det_loy50_max', 'loyal_per_sign', 'smallset_share']}
    P('\n== Binary L vs N by feature family alone (LOCO, k = all in family, lam=10) ==')
    allfe = feats[:]
    for fam, fl in FAM.items():
        fl = [f for f in fl if f in feats]
        feats_bak = feats[:]
        idx = [feats.index(f) for f in fl]
        # temporarily restrict
        def loco_fam():
            ci = {'L': 0, 'N': 1}; res = {}
            for c in train_c:
                tr = [d for d in train_c if d != c]
                Atr = np.vstack([X[d][:, idx] for d in tr]); mu, sd = Atr.mean(axis=0), Atr.std(axis=0) + 1e-9
                W = fit_logreg((Atr - mu) / sd, np.array([ci[lab2[d]] for d in tr for _ in range(len(X[d]))]), 2, lam=10, iters=2000)
                res[c] = predict(W, (X[c][:, idx] - mu) / sd).mean(axis=0)
            return sum(1 for c in train_c if ['L', 'N'][int(np.argmax(res[c]))] == lab2[c]) / len(train_c), res
        a, res = loco_fam()
        Atr = np.vstack([X[d][:, idx] for d in train_c]); mu, sd = Atr.mean(axis=0), Atr.std(axis=0) + 1e-9
        W = fit_logreg((Atr - mu) / sd, np.array([{'L': 0, 'N': 1}[lab2[d]] for d in train_c for _ in range(len(X[d]))]), 2, lam=10, iters=2000)
        pi = {c: predict(W, (X[c][:, idx] - mu) / sd).mean(axis=0)[0] for c in ('indus_seq_raw', 'indus_seq_all', 'indus_im77', 'indus_slotshuf', 'indus_markov2')}
        P(f'  {fam:26s} LOCO acc {a:.2f}; P(L) Indus raw {pi["indus_seq_raw"]:.2f} all {pi["indus_seq_all"]:.2f} IM77 {pi["indus_im77"]:.2f} | slotshuf {pi["indus_slotshuf"]:.2f} markov2 {pi["indus_markov2"]:.2f}')
txt = '\n'.join(out)
open(os.path.join(OUTD, OUTNAME + '.txt'), 'w').write(txt + '\n')
print(txt)
