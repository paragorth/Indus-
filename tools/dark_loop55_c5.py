"""Loop 55 cycle 5: PROFILE CLASSIFIER (language vs designed code) on the whole 55-statistic profile, and its power.

Question: the count of statistics beyond the chain does not separate language (L) from designed codes (D) (cycle 1).
Does the PROFILE (which statistics, in which direction) separate them, with corpus-level leave-one-corpus-out (LOCO)
validation, and how does that power grow with n (cycle 2 samples, n = 250-3000)? Where does Indus fall, at all three
merge levels, and does the classifier pass the chain control (an order-2 chain fitted to a language sample must NOT be
called language if the classifier reads more than chain-level structure)?

Feature sets: raw (the statistic itself), zM2 / zM2E (its z against the sample's own order-2 chain, lengths kept / END
state). Features that are undefined in any L or D sample (multi-site shares: codes have no sites) are dropped.
Classifiers: standardised L2 logistic regression (C=0.1) and nearest centroid. Exact permutation null: all C(11,5) = 462
assignments of the class labels to the 11 L/D corpora.
Usage: python3 tools/dark_loop55_c5.py [controls]   (controls = also run the chain-synthetic batteries; ~15 min)"""
import os, sys, json, math, random, itertools, collections, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dark_loop55_common as C
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D1 = os.path.join(C.DARK, 'loop55_c1'); D2 = os.path.join(C.DARK, 'loop55_c2')
CTRL = os.path.join(C.DARK, 'loop55_c5_controls.json')
I = ['indus_seq_raw', 'indus_seq_strong', 'indus_seq_all']


def load(dirs, cond):
    R = collections.defaultdict(list)
    for d_ in dirs:
        if not os.path.isdir(d_): continue
        for f in sorted(os.listdir(d_)):
            if f.endswith('.json'):
                d = json.load(open(os.path.join(d_, f)))
                if cond(d): R[d['corpus']].append(d)
    return R


def vec(d, kind, feats):
    out = []
    for k in feats:
        x = d['obs'][k] if kind == 'raw' else d['nulls'][kind[1:]][k]['z']
        x = float('nan') if x is None else float(x)
        if kind != 'raw' and not math.isnan(x): x = max(-30.0, min(30.0, x))
        out.append(x)
    return out


def usable(R, kind, corp):
    keep = []
    for k in C.STATS:
        ok = True
        for c in corp:
            for d in R[c]:
                x = d['obs'][k] if kind == 'raw' else d['nulls'].get(kind[1:], {}).get(k, {}).get('z', float('nan'))
                if x is None or (isinstance(x, float) and math.isnan(x)): ok = False
        if ok: keep.append(k)
    return keep


def fit_predict(Xtr, ytr, Xte, model):
    mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    sc = StandardScaler().fit(Xtr); A = sc.transform(Xtr); B = np.clip(sc.transform(Xte), -10, 10)
    if model == 'logit':
        m = LogisticRegression(C=0.1, max_iter=2000).fit(A, ytr); return m.predict_proba(B)[:, 1]
    cL = A[ytr == 1].mean(0); cD = A[ytr == 0].mean(0)
    dL = ((B - cL) ** 2).sum(1); dD = ((B - cD) ** 2).sum(1)
    return 1 / (1 + np.exp(np.clip((dL - dD) / A.shape[1], -50, 50)))   # soft score, > 0.5 = closer to L


def loco(R, corp, lab, kind, feats, model):
    """corpus-level LOCO: returns per-corpus mean P(L) when held out, and corpus accuracy."""
    P = {}
    for held in corp:
        tr = [c for c in corp if c != held]
        Xtr = np.array([vec(d, kind, feats) for c in tr for d in R[c]]); ytr = np.array([lab[c] for c in tr for d in R[c]])
        Xte = np.array([vec(d, kind, feats) for d in R[held]])
        P[held] = float(np.mean(fit_predict(Xtr, ytr, Xte, model)))
    acc = sum(1 for c in corp if (P[c] > 0.5) == (lab[c] == 1)) / len(corp)
    return P, acc


def perm_null(R, corp, kind, feats, model, true_lab):
    L0 = [c for c in corp if true_lab[c] == 1]; accs = []
    for Ls in itertools.combinations(corp, len(L0)):
        lab = {c: (1 if c in Ls else 0) for c in corp}
        accs.append(loco(R, corp, lab, kind, feats, model)[1])
    return accs


def apply_all(R, corp, lab, kind, feats, model, targets):
    Xtr = np.array([vec(d, kind, feats) for c in corp for d in R[c]]); ytr = np.array([lab[c] for c in corp for d in R[c]])
    out = {}
    for name, ds in targets.items():
        if not ds: continue
        p = fit_predict(Xtr, ytr, np.array([vec(d, kind, feats) for d in ds]), model)
        out[name] = (float(np.mean(p)), float(np.min(p)), float(np.max(p)), len(p))
    return out


def controls():
    """Chain-synthetic batteries: for each n=3000 cycle-1 sample of the L and D corpora and of Indus, regenerate the
    sample (same seed) and compute the battery on one order-2 chain (lengths kept) and one order-2 END-state chain fitted on it."""
    if os.path.exists(CTRL): return json.load(open(CTRL))
    from multiprocessing import Pool
    jobs = [(c, r) for r in range(2) for c in ['ur3_words', 'ur3_syll', 'ur3_names_syll', 'linb_syll', 'linb_words', 'latin_edh',
                                               'icd10', 'hts', 'aircraft_reg', 'unicode_names', 'heraldry'] + I]
    with Pool(int(os.environ.get('NPROC', '2'))) as P:
        res = P.map(_ctrl_job, jobs)
    json.dump(res, open(CTRL, 'w')); return res


def _ctrl_job(a):
    name, r = a
    rnd = random.Random(1000 * r + 55)
    hist, copies, _ = C.indus_shape('seq_all')
    if name.startswith('indus_'):
        src = [t for t in C.load_indus(name[6:]) if 2 <= len(t[2]) <= 12]; data = rnd.sample(src, 3000); bt = True
    else:
        data, _ = C.sample_indus_shaped(C.load_ref(name), rnd, hist, copies, dup='natural'); bt = False
    out = {'corpus': name, 'resample': r}
    for nm, end in (('M2', False), ('M2E', True)):
        g = C.markov_corpus(data, random.Random(r + 99), 2, bt, end=end)
        out[nm] = {'obs': C.battery(g, random.Random(r + 7))}
    return out


def main(with_controls=False):
    lines = []; P = lines.append; J = {}
    R1 = load([D1], lambda d: d.get('dup') == 'natural')
    corp = [c for c in R1 if C.TYPE[c] in 'LD']; lab = {c: 1 if C.TYPE[c] == 'L' else 0 for c in corp}
    P('# Loop 55 cycle 5: language-vs-code PROFILE classifier on the 55-statistic battery (n = 3,000 Indus-shaped samples, cycle 1).')
    P(f'L corpora: {", ".join(c for c in corp if lab[c])}; D corpora: {", ".join(c for c in corp if not lab[c])}. Corpus-level LOCO; exact permutation null over all 462 label assignments.')
    P('')
    P('## (a) LOCO accuracy (11 corpora), permutation P, held-out P(L) per corpus, and where Indus / accounting / grammar-without-language fall (model trained on all 11)')
    P('| features | model | n feats | LOCO acc | perm mean / 95th / P | held-out P(L): L corpora | held-out P(L): D corpora | Indus raw / strong / all P(L) mean [min-max over 4 resamples] | proto-cun / proto-elam / khipu | chess / chords |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    J['a'] = []
    for kind in ('raw', 'zM2', 'zM2E'):
        feats = usable(R1, kind, corp)
        for model in ('logit', 'centroid'):
            Pp, acc = loco(R1, corp, lab, kind, feats, model)
            null = perm_null(R1, corp, kind, feats, model, lab)
            pv = sum(1 for x in null if x >= acc) / len(null)
            tg = apply_all(R1, corp, lab, kind, feats, model, {c: R1.get(c, []) for c in I + ['proto_cuneiform', 'proto_elamite', 'khipu', 'chess_eco', 'chords']})
            f = lambda c: f'{tg[c][0]:.2f} [{tg[c][1]:.2f}-{tg[c][2]:.2f}]' if c in tg else 'n/a'
            P(f"| {kind} | {model} | {len(feats)} | {acc:.2f} | {st.mean(null):.2f} / {sorted(null)[int(0.95 * len(null)) - 1]:.2f} / {pv:.3f} | "
              + ', '.join(f'{c} {Pp[c]:.2f}' for c in corp if lab[c]) + ' | ' + ', '.join(f'{c} {Pp[c]:.2f}' for c in corp if not lab[c]) + ' | '
              + ' / '.join(f(c) for c in I) + ' | ' + ' / '.join(f(c) for c in ('proto_cuneiform', 'proto_elamite', 'khipu')) + ' | ' + ' / '.join(f(c) for c in ('chess_eco', 'chords')) + ' |')
            J['a'].append({'kind': kind, 'model': model, 'nfeat': len(feats), 'acc': acc, 'perm_p': pv, 'heldout': Pp, 'targets': tg})
    # (b) power curve of the profile classifier
    R2 = load([D2], lambda d: True)
    P('')
    P('## (b) Power curve: LOCO accuracy of the profile classifier (logit) by sample size, cycle 2 samples (4 resamples x 11 L/D corpora per n); Indus(all) P(L) at each n')
    P('| n | raw acc (perm P) | zM2 acc (perm P) | zM2E acc (perm P) | Indus all P(L) raw / zM2 / zM2E |')
    P('|---|---|---|---|---|')
    J['b'] = {}
    for n in (250, 500, 1000, 2000, 3000):
        Rn = collections.defaultdict(list)
        for c, ds in R2.items():
            for d in ds:
                if d['n'] == n: Rn[c].append(d)
        cn = [c for c in corp if Rn.get(c)]
        if len(cn) < 11: P(f'| {n} | incomplete ({len(cn)} corpora) | | | |'); continue
        cells = []; ind = []
        for kind in ('raw', 'zM2', 'zM2E'):
            feats = usable(Rn, kind, cn)
            _, acc = loco(Rn, cn, lab, kind, feats, 'logit')
            null = perm_null(Rn, cn, kind, feats, 'logit', lab); pv = sum(1 for x in null if x >= acc) / len(null)
            cells.append(f'{acc:.2f} ({pv:.3f})')
            tg = apply_all(Rn, cn, lab, kind, feats, 'logit', {'indus_seq_all': Rn.get('indus_seq_all', [])})
            ind.append(f"{tg['indus_seq_all'][0]:.2f}" if 'indus_seq_all' in tg else 'n/a')
            J['b'][f'{n}_{kind}'] = {'acc': acc, 'perm_p': pv, 'indus_all': tg.get('indus_seq_all')}
        P(f'| {n} | ' + ' | '.join(cells) + ' | ' + ' / '.join(ind) + ' |')
    # (c) chain control on raw features
    if with_controls:
        ctl = controls()
        P('')
        P('## (c) Chain control (raw-feature classifier trained on the 11 real L/D corpora): P(L) of an order-2 chain (lengths kept) and of an order-2 END-state chain fitted on each sample (2 resamples). A classifier that reads more than chain structure must give chain-synthetic language LOW P(L).')
        P('| corpus | class | real P(L) | M2 synthetic P(L) | M2E synthetic P(L) |')
        P('|---|---|---|---|---|')
        J['c'] = {}
        for model in ('logit',):
            feats = usable(R1, 'raw', corp)
            Xtr = np.array([vec(d, 'raw', feats) for c in corp for d in R1[c]]); ytr = np.array([lab[c] for c in corp for d in R1[c]])
            by = collections.defaultdict(list)
            for x in ctl: by[x['corpus']].append(x)
            for c in corp + I:
                real = apply_all(R1, corp, lab, 'raw', feats, model, {c: R1[c]})[c][0] if c in I else \
                    loco(R1, corp, lab, 'raw', feats, model)[0][c]
                ps = {}
                tr = [k for k in corp if k != c]   # LOCO: the synthetic of corpus c is scored by a model that never saw c
                Xt = np.array([vec(d, 'raw', feats) for k in tr for d in R1[k]]); yt = np.array([lab[k] for k in tr for d in R1[k]])
                for nm in ('M2', 'M2E'):
                    ps[nm] = float(np.mean(fit_predict(Xt, yt, np.array([vec(x[nm], 'raw', feats) for x in by[c]]), model)))
                P(f"| {c} | {C.TYPE[c]} | {real:.2f} | {ps['M2']:.2f} | {ps['M2E']:.2f} |")
                J['c'][c] = {'real': real, **ps}
            P('(All rows LOCO: an L/D corpus and its synthetics are scored by a model trained on the other 10; Indus by the model on all 11.)')
    txt = '\n'.join(lines)
    open(os.path.join(C.DARK, 'loop55_c5.txt'), 'w').write(txt + '\n'); json.dump(J, open(os.path.join(C.DARK, 'loop55_c5.json'), 'w'), default=str)
    print(txt)


if __name__ == '__main__':
    main('controls' in sys.argv)
