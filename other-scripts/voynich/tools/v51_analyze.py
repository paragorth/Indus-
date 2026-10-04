"""v51 analysis: paired (same mapping) comparisons of speech-model scores.

usage: python3 v51_analyze.py TAG [ref_for_voynich]
"""
import sys, os, json, glob
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v51_lib import CKPT, METRICS, SIGN

def load(tag):
    rows = []
    for fn in glob.glob(os.path.join(CKPT, f'{tag}_*.jsonl')):
        rows += [json.loads(l) for l in open(fn)]
    return rows

def table(rows, metrics=METRICS):
    """-> corpora list, mappings list, array [corpus, mapping, metric] (clips averaged)."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r['corpus'], r['m'])].append([r[k] for k in metrics])
    cs = sorted(set(k[0] for k in acc)); ms = sorted(set(k[1] for k in acc))
    ms = [m for m in ms if all((c, m) in acc for c in cs)]
    A = np.array([[np.mean(acc[(c, m)], 0) for m in ms] for c in cs])
    return cs, ms, A

def composite(A):
    """z across corpora within each mapping, signed so + = speech-like, averaged over metrics."""
    Z = (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-9)
    sg = np.array([SIGN[k] for k in METRICS])
    return (Z * sg).mean(-1), Z * sg

def paired(a, b):
    d = a - b
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12), (d > 0).mean()

def default_pairs(cs):
    P = []
    for c in cs:
        if '~' in c:
            continue
        if c + '~shuf' in cs:
            P.append((c, c + '~shuf')); P.append((c, c + '~mk1')); P.append((c + '~mk1', c + '~shuf'))
        elif c.startswith('V-') or c.startswith('G-'):
            P.append((c, 'V-ZL~shuf'))
    return P

def classifier(cs, Zs):
    """leave-one-language-out logistic regression on the 6 signed z-metrics:
    languages (+) vs their shuffles/markov and the generators (-)."""
    from sklearn.linear_model import LogisticRegression
    langs = [c for c in cs if '~' not in c and not c.startswith('V-') and not c.startswith('G-')]
    neg = [c for c in cs if '~' in c or c.startswith('G-')]
    ci = {c: i for i, c in enumerate(cs)}
    out = {}
    def fit(pos, negs):
        X = np.concatenate([Zs[ci[c]] for c in pos + negs]); y = np.concatenate([np.full(Zs.shape[1], c in pos) for c in pos + negs])
        return LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(X, y)
    for L in langs:
        pos = [c for c in langs if c != L]; negs = [c for c in neg if not c.startswith(L + '~')]
        clf = fit(pos, negs)
        for c in [L, L + '~shuf', L + '~mk1']:
            if c in ci:
                out[c] = float(clf.predict_proba(Zs[ci[c]])[:, 1].mean())
    clf = fit(langs, [c for c in neg if not c.startswith('V-')])
    for c in cs:
        if c.startswith('V-') or c.startswith('G-'):
            out[c] = float(clf.predict_proba(Zs[ci[c]])[:, 1].mean())
    return out, dict(zip(METRICS, clf.coef_[0].round(2)))

def report(tag):
    rows = load(tag)
    cs, ms, A = table(rows)
    S, Zs = composite(A)
    lines = [f'# {tag}: {len(ms)} mappings x {len(cs)} corpora']
    lines.append('corpus | composite (mean +- se) | ' + ' | '.join(METRICS) + ' | top heard languages')
    tops = defaultdict(Counter)
    for r in rows:
        tops[r['corpus']][r['lid_top']] += 1
    order = np.argsort(-S.mean(1))
    for i in order:
        c = cs[i]
        lines.append(f'{c} | {S[i].mean():+.3f} +- {S[i].std(ddof=1)/np.sqrt(len(ms)):.3f} | ' +
                     ' | '.join(f'{A[i,:,j].mean():.3f}' for j in range(len(METRICS))) + ' | ' +
                     ','.join(f'{k}:{v}' for k, v in tops[c].most_common(3)))
    lines.append('\nPAIRED (same mapping): composite diff, paired z, win rate; per-metric z')
    for a, b in default_pairs(cs):
        ia, ib = cs.index(a), cs.index(b)
        d, z, w = paired(S[ia], S[ib])
        mz = [paired(Zs[ia, :, j], Zs[ib, :, j])[1] for j in range(len(METRICS))]
        lines.append(f'{a} vs {b}: {d:+.3f} z {z:+.1f} win {w:.2f} | ' + ' '.join(f'{k}:{v:+.1f}' for k, v in zip(METRICS, mz)))
    try:
        out, coef = classifier(cs, Zs)
        lines.append('\nCLASSIFIER P(language), leave-one-language-out for languages and their nulls; coef ' + str(coef))
        for c in sorted(out, key=lambda c: -out[c]):
            lines.append(f'{c}: {out[c]:.3f}')
    except Exception as e:
        lines.append('classifier failed: ' + repr(e))
    return '\n'.join(lines), (cs, ms, A, S, Zs)

if __name__ == '__main__':
    txt, _ = report(sys.argv[1])
    print(txt)
