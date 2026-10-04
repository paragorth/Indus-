"""v47 cycle 3: CROSS-SECTION PROBE. The pharmaceutical registers were never used to find the
herbal link, so they serve as out-of-sample probes. For every (pharma unit u, herbal page h) pair:
visual similarity s(u,h) (four networks; cosine of each network's embeddings, z-scored per
network over all u x h, averaged) and text overlap t(u,h). Both u x h matrices are double-
centred (removes 'hub' pages and units: long texts, busy drawings). Statistic: correlation of
the two double-centred matrices.
Text views: (P) the unit's paragraph words vs the page's words (tf-idf cosine); (L) the folio's
LABEL words (each label sits next to one drawn part) - number found on page h, exact or with one
glyph edit, divided by sqrt(page length).
Nulls: herbal pages' visual rows permuted within strata (2000x); a planted control (10% of
herbal tokens replaced by words driven by the visual embedding) to show the bipartite
statistic sees a planted link; a positive control in Gerard and Dodoens: one half of the book
(odd pseudo-quires) as probes, the other half as targets.
Writes data/v47_ckpt/c3.json and loops/v47_cycle3.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_setup import *

NPERM = int(os.environ.get('NPERM', 2000))


def bip_vis(E1, E2, nets=NETS4):
    out = 0
    for f in nets:
        A = np.array([e[f] for e in E1], float); B = np.array([e[f] for e in E2], float)
        mu = np.vstack([A, B]).mean(0)
        A = A - mu; B = B - mu
        A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
        M = A @ B.T
        out = out + (M - M.mean()) / M.std()
    return out / len(nets)


def dc(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def tfidf_bip(W1, W2):
    allw = W1 + W2
    n = len(allw)
    df = Counter(w for ws in allw for w in set(ws))
    vocab = {w: i for i, w in enumerate(sorted(df))}
    X = np.zeros((n, len(vocab)))
    for i, ws in enumerate(allw):
        for w, c in Counter(ws).items():
            X[i, vocab[w]] = np.log1p(c) * np.log(n / df[w])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X[:len(W1)] @ X[len(W1):].T


def ed1(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def label_bip(labels, W2):
    M = np.zeros((len(labels), len(W2)))
    for i, L in enumerate(labels):
        L = [l for l in set(labels[i]) if len(l) >= 4]
        for j, ws in enumerate(W2):
            s = set(ws)
            M[i, j] = sum(any(ed1(l, w) for w in s) for l in L) / np.sqrt(len(ws))
    return M


def bip_test(Vb, Tb, rng, strata=None, nperm=NPERM):
    T = dc(Tb).ravel(); T = (T - T.mean()) / T.std()
    def st(p):
        v = dc(Vb[:, p]).ravel()
        return float(np.mean((v - v.mean()) / v.std() * T))
    obs = st(np.arange(Vb.shape[1]))
    null = np.array([st(perm(Vb.shape[1], rng, strata)) for _ in range(nperm)])
    return dict(r=obs, z=float((obs - null.mean()) / null.std()), p=float((1 + (null >= obs).sum()) / (1 + nperm)))


if __name__ == '__main__':
    rng = np.random.default_rng(4703)
    rows, out = [], {}
    A1 = vsetup('A1'); PH = psetup()
    Eh = [A1['emb'][k] for k in A1['keys']]; Ep = [PH['emb'][k] for k in PH['keys']]
    Vb = bip_vis(Ep, Eh)
    TP = tfidf_bip(PH['words'], A1['words'])
    rP = bip_test(Vb, TP, rng)
    labs_f = {}
    for k in PH['keys']:
        labs_f[PH['emb'][k]['folio']] = PH['emb'][k]['labels']
    TL = label_bip([labs_f[PH['emb'][k]['folio']] for k in PH['keys']], A1['words'])
    rL = bip_test(Vb, TL, rng)
    # folio level for labels (labels belong to the folio, not to one register)
    fol = sorted(set(PH['folio']), key=lambda f: PH['folio'].index(f))
    Vf = np.array([Vb[[i for i, x in enumerate(PH['folio']) if x == f]].mean(0) for f in fol])
    TLf = label_bip([labs_f[f] for f in fol], A1['words'])
    rLf = bip_test(Vf, TLf, rng)
    # CLIP-only and MAE-only versions of the paragraph probe
    rPc = bip_test(bip_vis(Ep, Eh, ['clip']), TP, rng, nperm=1000)
    rPm = bip_test(bip_vis(Ep, Eh, ['mae']), TP, rng, nperm=1000)
    out['voynich'] = dict(P=rP, L=rL, Lf=rLf, P_clip=rPc, P_mae=rPm)
    rows.append(('V-47.3.1', 'CROSS-SECTION PROBE, paragraphs: %d pharma register units x %d A1 herbal pages; double-centred four-net visual similarity vs tf-idf overlap of the unit paragraph with the herbal page; herbal visual rows permuted (%d)' % (len(Ep), len(Eh), NPERM),
                 'four-net r %+.4f z %+.2f p %.4f; CLIP alone z %+.2f; MAE alone z %+.2f' % (rP['r'], rP['z'], rP['p'], rPc['z'], rPm['z']),
                 'probe supports' if rP['p'] < 0.01 else ('weak' if rP['p'] < 0.05 else 'no cross-section link')))
    rows.append(('V-47.3.2', 'CROSS-SECTION PROBE, labels: the folio\'s label words (>= 4 glyphs) found (exact or one glyph edit) in the herbal page, per sqrt(page length); unit level (%d units) and folio level (%d folios)' % (len(Ep), len(fol)),
                 'unit r %+.4f z %+.2f p %.4f; folio r %+.4f z %+.2f p %.4f' % (rL['r'], rL['z'], rL['p'], rLf['r'], rLf['z'], rLf['p']),
                 'labels name what is drawn' if min(rL['p'], rLf['p']) < 0.01 else 'no label-picture link across sections'))
    # planted: herbal text gets 10% tokens from a vocabulary driven by the herbal embedding; pharma paragraphs get the same
    # driven by the pharma embedding with the SAME word map, so a real cross-section vocabulary link is planted
    allw = Counter(w for ws in A1['words'] for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    Z = np.array([e['dino'] for e in Eh + Ep], float)
    U, Sv, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    Zr = U[:, :10] * Sv[:10]
    pw = plant_text(A1['words'] + PH['words'], Zr, 0.10, np.random.default_rng(7), pool)
    rPl = bip_test(Vb, tfidf_bip(pw[len(Eh):], pw[:len(Eh)]), rng, nperm=500)
    out['planted'] = rPl
    rows.append(('V-47.3.3', 'PLANTED control for the bipartite statistic: 10% of herbal and pharma tokens drawn from one vocabulary map driven by the shared DINO embedding', 'z %+.2f p %.4f' % (rPl['z'], rPl['p']),
                 'statistic sees a planted cross-section link' if rPl['z'] > 3 else 'statistic too weak at this size'))
    for src, adjw in [('gerard', 8), ('dodoens', 6)]:
        H = hsetup(src, adjw)
        q = np.array([int(x[1:]) for x in H['quire']])
        a = np.where(q % 2 == 0)[0]; b = np.where(q % 2 == 1)[0]
        a = a[:len(Ep)]
        Ea = [H['emb'][H['keys'][i]] for i in a]; Eb = [H['emb'][H['keys'][i]] for i in b]
        r = bip_test(bip_vis(Ea, Eb), tfidf_bip([H['words'][i] for i in a], [H['words'][i] for i in b]), rng, nperm=1000)
        out[src] = r
        rows.append(('V-47.3.%d' % (4 if src == 'gerard' else 5), 'POSITIVE control, %s: %d probe pages (even pseudo-quires) x %d target pages (odd), identical bipartite statistic' % (src, len(a), len(b)),
                     'r %+.4f z %+.2f p %.4f' % (r['r'], r['z'], r['p']), 'control works' if r['p'] < 0.01 else 'control weak'))
    json.dump(out, open(os.path.join(CK47, 'c3.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v47_cycle3.txt'), 'w') as fh:
        fh.write('# v47 cycle 3 - cross-section probe: pharmaceutical registers against herbal pages\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v47_cycle3.txt')).read())
