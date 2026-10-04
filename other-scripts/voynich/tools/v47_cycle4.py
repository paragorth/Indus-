"""v47 cycle 4: power checks and the candidate words.
(a) POWER of the pharmaceutical tests: planted 10% link inside the 41 register units (within-
    section composite, as V-47.5). If the planted link is not found, V-47.5 / V-47.3.1 say nothing.
(b) CANDIDATE WORDS: words selected independently in BOTH held-out folds of cycle 2, plus the top 5
    word-visual Mantel words. For each: word-visual Mantel z in ZL3b and in IT2a (same pages,
    other transliteration), the CLIP trait profile of the pages carrying it (mean trait z vs the
    rest, permutation p, 15 traits so Bonferroni 0.0033), and its folios.
(c) WORD FAMILY: '-ar' family (any word = up to two glyphs + 'ar') and its complement; word-visual
    Mantel for family presence, and held-out quire AUC (fold 1 -> fold 2 and back) with the
    discovery-row permutation null of cycle 2.
Writes data/v47_ckpt/c4.json and loops/v47_cycle4.txt.
"""
import os, sys, json, re
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_setup import *
from v47_cycle1 import comp, sub
from v47_cycle2 import vis_comp, presence, word_mantel, clip_traits, fold_split, resid_on, auc
from v8_lib import voynich_pages

NPERM = int(os.environ.get('NPERM', 1000))


def family_heldout(S, V, fam, rng, nperm):
    f1 = fold_split(S['quire'], 'voynich')
    has = np.array([any(fam(w) for w in ws) for ws in S['words']])
    res = {}
    for name, d, t in [('1->2', np.where(f1)[0], np.where(~f1)[0]), ('2->1', np.where(~f1)[0], np.where(f1)[0])]:
        ln = np.log([len(S['words'][i]) for i in t]); ar = np.log([S['emb'][S['keys'][i]]['area'] + 1e-4 for i in t])
        def score(pr):
            Vtd = V[np.ix_(t, d)][:, pr]
            sc = Vtd[:, has[d]].mean(1) - Vtd[:, ~has[d]].mean(1)
            return auc(resid_on(sc, [ln, ar]), has[t])
        obs = score(np.arange(len(d)))
        null = np.array([score(rng.permutation(len(d))) for _ in range(nperm)])
        res[name] = (float(obs), float((1 + (null >= obs).sum()) / (1 + nperm)), int(has[d].sum()), int(has[t].sum()))
    return res


if __name__ == '__main__':
    rng = np.random.default_rng(4704)
    rows, out = [], {}
    # (a) pharma power
    PH = psetup()
    allw = Counter(w for ws in PH['words'] for w in ws)
    pool = [w for w, c in allw.items() if 3 <= c <= 30]
    zp = []
    for s in range(5):
        Z = np.array([PH['emb'][k]['dino'] for k in PH['keys']], float)
        U, Sv, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        pw = plant_text(PH['words'], U[:, :10] * Sv[:10], 0.10, np.random.default_rng(100 + s), pool)
        zp.append(comp(PH, NETS4, rng, words=pw, nperm=300)['z'])
    out['pharma_power'] = zp
    rows.append(('V-47.4.1', 'POWER of the pharma within-section test: planted link (10% of paragraph tokens from a DINO-driven vocabulary) in the 41 register units, same confounds, 5 plantings x 300 perms',
                 'planted z: %s (mean %+.2f)' % (', '.join('%+.1f' % z for z in zp), np.mean(zp)),
                 'test has power: V-47.5 is a real non-replication' if np.mean(zp) > 3 else 'test underpowered: V-47.5 is uninformative'))
    # (b) candidates
    A1 = vsetup('A1')
    V = vis_comp(A1)
    vocab, B = presence(A1['words'])
    part = MPartial(list(A1['conf'].values()), len(A1['keys']))
    obs, z = word_mantel(V, B, part, rng, A1['strata'], NPERM)
    c2 = json.load(open(os.path.join(CK47, 'c2.json')))['voynich']['heldout']
    both = sorted(set(c2['1->2']['words']) & set(c2['2->1']['words']))
    top5 = [vocab[i] for i in np.argsort(-z)[:5]]
    cands = list(dict.fromkeys(both + top5))
    # IT2a
    P = {p['id']: p for p in voynich_pages(min_tokens=1, name='IT2a')}
    ok = [i for i, k in enumerate(A1['keys']) if k in P]
    Sit = sub(A1, ok)
    Sit['words'] = [[w for l in P[k]['lines'] for w in l] for k in Sit['keys']]
    vit, Bit = presence(Sit['words'])
    pit = MPartial(list(Sit['conf'].values()), len(ok))
    _, zit = word_mantel(V[np.ix_(ok, ok)], Bit, pit, rng, Sit['strata'], NPERM)
    zit_d = dict(zip(vit, zit))
    names, traits = clip_traits(A1)
    cand_out = {}
    for w in cands:
        h = B[:, vocab.index(w)] > 0
        d = traits[h].mean(0) - traits[~h].mean(0)
        null = np.array([(lambda p: traits[p[:h.sum()]].mean(0) - traits[p[h.sum():]].mean(0))(rng.permutation(len(h))) for _ in range(2000)])
        pv = (1 + (np.abs(null) >= np.abs(d)).sum(0)) / 2001
        j = int(np.argmin(pv))
        cand_out[w] = dict(z=float(z[vocab.index(w)]), z_it2a=float(zit_d.get(w, np.nan)), df=int(h.sum()),
                           best_trait=names[j], trait_diff=float(d[j]), trait_p=float(pv[j]),
                           folios=[A1['keys'][i] for i in np.where(h)[0]])
    out['candidates'] = cand_out
    rows.append(('V-47.4.2', 'CANDIDATE WORDS (frozen in BOTH held-out folds of cycle 2: %s; plus top-5 word-visual Mantel: %s). Word-visual Mantel z in ZL3b and IT2a (%d perms); CLIP trait of the pages carrying the word (strongest of 15, 2000 perms, Bonferroni 0.0033)' % (' '.join(both) or 'none', ' '.join(top5), NPERM),
                 '; '.join('%s (df %d): z %+.1f, IT2a %+.1f, %s %+.2f p %.4f' % (w, c['df'], c['z'], c['z_it2a'], c['best_trait'], c['trait_diff'], c['trait_p']) for w, c in cand_out.items()),
                 ''))
    # (c) -ar family
    fam = lambda w: re.fullmatch(r'[a-z]{0,2}ar', w) is not None
    famB = np.array([[any(fam(w) for w in ws)] for ws in A1['words']], float)
    _, zf = word_mantel(V, famB, part, rng, A1['strata'], NPERM)
    fh = family_heldout(A1, V, fam, rng, NPERM)
    out['ar_family'] = dict(z=float(zf[0]), heldout=fh, members=Counter(w for ws in A1['words'] for w in ws if fam(w)).most_common(12))
    rows.append(('V-47.4.3', "'-ar' FAMILY (word = up to two glyphs + ar: %s): word-visual Mantel of family presence (%d perms); held-out AUC fold 1->2 and 2->1 (discovery-row permutation null)" % (', '.join('%s %d' % t for t in out['ar_family']['members'][:8]), NPERM),
                 'z %+.2f; held-out %s' % (zf[0], '; '.join('%s AUC %.3f p %.3f (carriers %d/%d)' % (k, *v) for k, v in fh.items())),
                 ''))
    json.dump(out, open(os.path.join(CK47, 'c4.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v47_cycle4.txt'), 'w') as fh_:
        fh_.write('# v47 cycle 4 - power of the pharma tests; the candidate words\n')
        fh_.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh_.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v47_cycle4.txt')).read())
