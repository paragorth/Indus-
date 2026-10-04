"""v47 cycle 5: LEAVE-ONE-QUIRE-OUT word transfer (more held-out power than the 2-fold of cycle 2).
For each quire q: select the top K = 10 words by word-visual partial Mantel on all OTHER quires
(word on >= 3 discovery pages and >= 2 pages of q); on q's pages, AUC of word presence from the
page's visual similarity to the discovery pages carrying the word (length and area partialled).
Statistic: mean AUC over all quires x frozen words. Null: the whole selection + test rerun with
the discovery visual rows permuted (NPERM per quire). Also: which words are frozen in >= 4 of the
quire folds (stable candidates) and their pooled held-out AUC.
Run on Voynich A1 herbal and on Dodoens and Gerard (pseudo-quires).
Usage: v47_cycle5.py SRC  -> data/v47_ckpt/c5_SRC.json ; v47_cycle5.py report -> loops/v47_cycle5.txt
"""
import os, sys, json
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_setup import *
import v47_cycle2 as c2

NPERM = int(os.environ.get('NPERM', 100))
c2.K = 10

if __name__ == '__main__':
    src = sys.argv[1]
    if src != 'report':
        rng = np.random.default_rng(4705)
        S = vsetup('A1') if src == 'voynich' else hsetup(src, 8 if src == 'gerard' else 6)
        V = c2.vis_comp(S)
        vocab, B = c2.presence(S['words'])
        q = np.array(S['quire'])
        per, allauc, nullmeans = {}, [], np.zeros(NPERM)
        freq, wauc = Counter(), defaultdict(list)
        cnt = 0
        for x in sorted(set(q)):
            t = np.where(q == x)[0]; d = np.where(q != x)[0]
            if len(t) < 6:
                continue
            ws, a = c2.heldout_words(S, V, vocab, B, d, t, rng)
            ok = ~np.isnan(a)
            per[x] = dict(n=len(t), words=ws, mean=float(np.nanmean(a)))
            allauc += list(a[ok])
            for w, v in zip(ws, a):
                freq[w] += 1
                if not np.isnan(v):
                    wauc[w].append(float(v))
            for k in range(NPERM):
                _, an = c2.heldout_words(S, V, vocab, B, d, t, rng, perm_rows=rng.permutation(len(d)))
                nullmeans[k] += np.nansum(an);
            cnt += int(ok.sum())
            print(src, x, per[x], flush=True)
        obs = float(np.mean(allauc))
        null = nullmeans / max(cnt, 1)
        stable = {w: dict(folds=c, auc=float(np.mean(wauc[w])) if wauc[w] else None) for w, c in freq.most_common() if c >= 4}
        json.dump(dict(per=per, obs=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                       z=float((obs - null.mean()) / null.std()), p=float((1 + (null >= obs).sum()) / (1 + NPERM)),
                       stable=stable, nfolds=len(per)), open(os.path.join(CK47, 'c5_%s.json' % src), 'w'))
        sys.exit()
    rows = []
    for i, s in enumerate(['voynich', 'gerard', 'dodoens']):
        fn = os.path.join(CK47, 'c5_%s.json' % s)
        if not os.path.exists(fn):
            continue
        r = json.load(open(fn))
        st = '; '.join('%s (%d folds, AUC %.2f)' % (w, v['folds'], v['auc'] or 0) for w, v in list(r['stable'].items())[:10])
        rows.append(('V-47.5.%d' % (i + 1), 'LEAVE-ONE-QUIRE-OUT word transfer, %s: %d quire folds, top 10 words per fold by word-visual Mantel on the other quires, AUC on the held-out quire; null: discovery visual rows permuted, whole pipeline (%d per fold)' % (s, r['nfolds'], NPERM),
                     'mean AUC %.3f (null %.3f +- %.3f), z %+.2f, p %.3f; per fold %s; words frozen in >= 4 folds: %s' % (
                         r['obs'], r['null_mean'], r['null_sd'], r['z'], r['p'], ', '.join('%s %.2f' % (k, v['mean']) for k, v in r['per'].items()), st or 'none'),
                     'held-out word transfer' if r['p'] < 0.05 else 'no held-out word transfer'))
    with open(os.path.join(LOOPS, 'v47_cycle5.txt'), 'w') as fh:
        fh.write('# v47 cycle 5 - leave-one-quire-out word transfer\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v47_cycle5.txt')).read())
