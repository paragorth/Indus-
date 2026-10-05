#!/usr/bin/env python3
"""LA-54 cycle 2 report: calibration of confidences, imputed commodities, word -> commodity links.
Links use the BLIND model (no word features), so a link cannot be produced by the word itself.
Null for links: imputed probability vectors permuted among unlabelled documents of the same site (10^4)."""
import os, json, glob, collections, random, math
import numpy as np
import la54_common as C


def load(model):
    R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(C.CK, 'c2_%s_*.json' % model)))]
    return R


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    m = 1.0
    for r, i in reversed(list(enumerate(o))):
        m = min(m, p[i] * n / (r + 1)); q[i] = m
    return q


def calib(R):
    cl = R[0]['classes']
    y = np.array([cl.index(c) for c in R[0]['y']])
    P = np.mean([np.array(r['oof']) for r in R], 0)
    conf = P.max(1); hit = P.argmax(1) == y
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
    tab = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (conf >= a) & (conf < b)
        if m.any(): tab.append((a, b, int(m.sum()), float(hit[m].mean())))
    # per class recall / precision
    pc = {}
    for k, c in enumerate(cl):
        pr = P.argmax(1) == k
        pc[c] = {'n': int((y == k).sum()), 'recall': float((pr[y == k]).mean()) if (y == k).any() else None,
                 'precision': float((y[pr] == k).mean()) if pr.any() else None}
    return float(hit.mean()), tab, pc, P


def main():
    la = {d['id']: d for d in C.la_docs()}
    out = {}
    lines = []
    for model in ('FULL', 'BLIND'):
        R = load(model)
        if not R: continue
        acc, tab, pc, oofP = calib(R)
        cl = R[0]['classes']
        Pu = np.mean([np.array(r['imp']) for r in R], 0)
        sdu = np.std([np.array(r['imp']) for r in R], 0)
        ids = R[0]['unl_ids']
        stab = float(np.mean([np.mean(np.array(r['imp']).argmax(1) == Pu.argmax(1)) for r in R]))
        out[model] = {'oof_acc': acc, 'calib': tab, 'per_class': pc, 'n_seeds': len(R), 'stability': stab,
                      'imp': {i: dict(zip(cl, map(float, p))) for i, p in zip(ids, Pu)}}
        lines.append('%s: seeds %d, out-of-fold acc %.3f, seed stability of argmax %.3f' % (model, len(R), acc, stab))
        lines.append('  calibration (conf bin, n, acc): ' + '; '.join('%.1f-%.1f n=%d acc=%.2f' % t for t in tab))
        lines.append('  per class: ' + '; '.join('%s n=%d R=%.2f P=%s' % (c, v['n'], v['recall'] or 0, ('%.2f' % v['precision']) if v['precision'] is not None else '-') for c, v in pc.items()))
        dist = collections.Counter(cl[k] for k in Pu.argmax(1))
        lines.append('  imputed class counts (166 docs): ' + str(dict(dist)))
        # confident docs
        conf = Pu.max(1)
        order = np.argsort(-conf)
        lines.append('  top imputations: ' + ', '.join('%s=%s(%.2f)' % (ids[i], cl[Pu[i].argmax()], conf[i]) for i in order[:30]))
        out[model]['labelled_prior'] = dict(collections.Counter(R[0]['y']))
        if model == 'BLIND':
            # word -> commodity links on unlabelled docs, permutation within site
            rng = np.random.default_rng(54)
            site = np.array([la[i]['site'] for i in ids])
            W = collections.defaultdict(set)
            for k, i in enumerate(ids):
                for w in set(la[i]['words']): W[w].add(k)
            words = [w for w, s in W.items() if len(s) >= 3]
            M = np.zeros((len(words), len(ids)))
            for a, w in enumerate(words):
                M[a, list(W[w])] = 1
            obs = M @ Pu  # words x classes
            NP = 10000
            ge = np.zeros_like(obs); sm = np.zeros_like(obs); sq = np.zeros_like(obs)
            groups = [np.where(site == s)[0] for s in set(site)]
            for t in range(NP):
                perm = np.arange(len(ids))
                for g in groups:
                    perm[g] = rng.permutation(g)
                e = M @ Pu[perm]
                ge += e >= obs - 1e-12; sm += e; sq += e * e
            p = (ge + 1) / (NP + 1)
            mu = sm / NP; sd = np.sqrt(np.maximum(sq / NP - mu ** 2, 1e-12))
            z = (obs - mu) / sd
            flat = [(words[a], cl[c], int(M[a].sum()), float(obs[a, c]), float(z[a, c]), float(p[a, c]))
                    for a in range(len(words)) for c in range(len(cl))]
            q = bh([f[5] for f in flat])
            # labelled-doc support for each link (independent of the imputation)
            labw = collections.defaultdict(collections.Counter)
            for d in la.values():
                if d['label']:
                    for w in set(d['words']): labw[w][d['label']] += 1
            links = []
            for f, qq in zip(flat, q):
                if f[5] < 0.05:
                    links.append({'word': f[0], 'class': f[1], 'n_unl': f[2], 'mass': f[3], 'z': f[4], 'p': f[5], 'q': float(qq),
                                  'labelled': dict(labw[f[0]])})
            links.sort(key=lambda r: r['p'])
            out['links'] = links
            out['n_tests'] = len(flat)
            lines.append('  link tests %d (words on >=3 unlabelled docs: %d); p<0.05: %d (expected %.1f); q<0.10: %d' % (
                len(flat), len(words), len(links), 0.05 * len(flat), sum(1 for l in links if l['q'] < 0.10)))
            for l in links[:25]:
                lines.append('   %s -> %s  n=%d z=%.2f p=%.4f q=%.2f labelled=%s' % (l['word'], l['class'], l['n_unl'], l['z'], l['p'], l['q'], l['labelled']))
    # agreement FULL vs BLIND
    if 'FULL' in out and 'BLIND' in out:
        ag = np.mean([max(out['FULL']['imp'][i], key=out['FULL']['imp'][i].get) == max(out['BLIND']['imp'][i], key=out['BLIND']['imp'][i].get) for i in out['FULL']['imp']])
        out['full_blind_agree'] = float(ag)
        lines.append('FULL vs BLIND argmax agreement on unlabelled docs: %.3f' % ag)
    json.dump(out, open(os.path.join(C.CK, 'c2_report.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
