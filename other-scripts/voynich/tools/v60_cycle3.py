"""v60 cycle 3: a published complexion-marker guess as a pre-registered hypothesis, plus a blind degree-ladder search.
(1) Families from an online forum post (data only, not its interpretation): HOT = ched/shed/chey/cheey/sheedy words,
COLD = saiin/daiin/chol/shol/chy words, WET = ot- words and shey, DRY = qok- words; degree endings -y / -aiin / -chedy.
Per paragraph: dominant quality on each axis (majority of family tokens). Tests: share of hot, hot-dry association
(odds ratio; external 3.7), once-per-entry (share of paragraphs where exactly one family of each axis appears).
Nulls: Markov resynthesis, word shuffle across paragraphs, star paragraphs; controls: the same rule applied to the
real complexion words of encoded Circa instans (should give the external numbers).
(2) Degree ladder: four mutually exclusive items within 3 tokens after any family-free anchor... replaced by a blind
search: 4-item sets (from the top 150 items) whose per-paragraph first occurrence gives a 4-way split matching the
external degree distribution [0.12, 0.52, 0.25, 0.11] up to relabeling, ranked by log Bayes factor vs Dirichlet;
held-out pages; nulls as above."""
import sys, random, json, collections, itertools, time
import numpy as np
import v60_lib as L
from scipy.special import gammaln

EVA = lambda w: w  # Voynich entries already use glyph units: ch->C, sh->S, cth->T, ckh->K
FAM = dict(H=lambda w: w.startswith(('Ced', 'Sed', 'Cey', 'Ceey', 'Seed', 'Sey')) and not w.startswith('Sey'),
           C=lambda w: w in ('saiin', 'daiin') or w.startswith(('Col', 'Sol')) or w == 'Cy',
           M=lambda w: w.startswith('ot') or w == 'Sey',
           D=lambda w: w.startswith('qok'))


def axis_values(ents):
    rows = []
    for e in ents:
        c = collections.Counter()
        for w in e['toks']:
            for k, f in FAM.items():
                if f(w): c[k] += 1
        q1 = None if c['H'] == c['C'] else ('H' if c['H'] > c['C'] else 'C')
        q2 = None if c['D'] == c['M'] else ('D' if c['D'] > c['M'] else 'M')
        ex1 = (c['H'] > 0) != (c['C'] > 0); ex2 = (c['D'] > 0) != (c['M'] > 0)
        rows.append(dict(id=e['id'], page=e['page'], q1=q1, q2=q2, ex1=ex1, ex2=ex2, c=dict(c)))
    return rows


def stats(rows):
    q = [r for r in rows if r['q1'] and r['q2']]
    T = collections.Counter(r['q1'] + r['q2'] for r in q)
    hd, hm, cd, cm = [T.get(k, 0) + 0.5 for k in ('HD', 'HM', 'CD', 'CM')]
    return dict(n=len(rows), n_both=len(q), hot_share=round(sum(1 for r in rows if r['q1'] == 'H') / max(1, sum(1 for r in rows if r['q1'])), 3),
                dry_share=round(sum(1 for r in rows if r['q2'] == 'D') / max(1, sum(1 for r in rows if r['q2'])), 3),
                OR=round(hd * cm / (hm * cd), 2), excl1=round(np.mean([r['ex1'] for r in rows]), 3),
                excl2=round(np.mean([r['ex2'] for r in rows]), 3), cells=dict(T))


DEG = np.array([0.12, 0.52, 0.25, 0.11])


def degree_search(ents, train, test, n_items=150, max_sets=2_000_000, seed=0):
    tr = [ents[i] for i in train]; te = [ents[i] for i in test]
    keep, _, _ = L.token_items(tr, max_items=n_items, rmin=0.1, rmax=0.7)
    def firsts(E):
        F = np.full((len(E), len(keep)), 10 ** 6, np.int32)
        idx = {it: j for j, it in enumerate(keep)}
        for i, e in enumerate(E):
            for t, tok in enumerate(e['toks']):
                for it in L.items_of(tok):
                    j = idx.get(it)
                    if j is not None and F[i, j] > t: F[i, j] = t
        return F
    Ftr, Fte = firsts(tr), firsts(te)
    rng = np.random.default_rng(seed)
    I = len(keep)
    S = rng.integers(0, I, size=(max_sets, 4))
    S = S[(np.sort(S, 1)[:, 1:] != np.sort(S, 1)[:, :-1]).all(1)]
    nest = np.array([[L.nested(x, y) for y in keep] for x in keep])
    ok = np.ones(len(S), bool)
    for i, j in itertools.combinations(range(4), 2):
        ok &= ~nest[S[:, i], S[:, j]]
    S = S[ok]
    def table(F, S):
        G = F[:, S]                      # n, h, 4
        mn = G.min(-1)
        cov = mn < 10 ** 6
        arg = G.argmin(-1)
        T = np.stack([((arg == k) & cov).sum(0) for k in range(4)], -1).astype(float)  # h, 4
        return T, cov.mean(0)
    def lbf(T):
        Ts = -np.sort(-T, 1)                # sorted counts vs sorted DEG (relabel-free)
        ll = (Ts * np.log(np.sort(DEG)[::-1])).sum(1)
        n = T.sum(1)
        lm = gammaln(4.0) - gammaln(n + 4.0) + gammaln(T + 1).sum(1)
        return ll - lm
    out = []
    for s in range(0, len(S), 200_000):
        sl = S[s:s + 200_000]
        T, cov = table(Ftr, sl)
        b = np.where((cov >= 0.6) & (T.min(1) >= 3), lbf(T), -1e9)
        out.append((b, sl, T))
    b = np.concatenate([o[0] for o in out]); SS = np.concatenate([o[1] for o in out]); TT = np.concatenate([o[2] for o in out])
    order = np.argsort(-b)[:50]
    Tt, covt = table(Fte, SS[order])
    lt = lbf(Tt)
    top = [dict(items=[keep[x] for x in SS[o]], lbf=float(b[o]), T=TT[o].tolist(), test_T=Tt[k].tolist(), test_lbf=float(lt[k]),
                test_cov=float(covt[k])) for k, o in enumerate(order) if b[o] > -1e8]
    return dict(n_hyp=int(len(SS)), n_pass=int((b > -1e8).sum()), top=top,
                med_test_lbf=float(np.median([t['test_lbf'] for t in top])) if top else None,
                best_lbf=float(top[0]['lbf']) if top else None)


def split(ents, seed):
    pages = sorted(set(e['page'] for e in ents))
    rng = random.Random(seed); rng.shuffle(pages)
    tr = set(pages[: len(pages) // 2])
    return [i for i, e in enumerate(ents) if e['page'] in tr], [i for i, e in enumerate(ents) if e['page'] not in tr]


if __name__ == '__main__':
    out = {}
    vh = L.voynich_entries('ZL3b', ('H', 'P'))
    sets = [('V_herb', vh), ('V_herbA', [e for e in vh if e['strat'][1] == 'A']), ('V_herbB', [e for e in vh if e['strat'][1] == 'B']),
            ('N_markov', L.markov_null(vh, 1)), ('N_wordshuf', L.wordshuf_null(vh, 1)),
            ('V_stars', L.voynich_entries('ZL3b', ('S',))), ('V_bio', L.voynich_entries('ZL3b', ('B',))),
            ('V_herb_IT2a', L.voynich_entries('IT2a', ('H', 'P')))]
    for name, ents in sets:
        st = stats(axis_values(ents))
        out['fam|' + name] = st
        print('FAM', name, json.dumps(st), flush=True)
    # control: the same majority rule on the real complexion words of Circa instans entries (plain words)
    ci = L.herbal_entries('CI')
    rows = []
    for e in ci:
        c = collections.Counter()
        for w in e['toks'][:128]:
            if L.HOT.match(w): c['H'] += 1
            if L.COLD.match(w): c['C'] += 1
            if L.DRY.match(w): c['D'] += 1
            if L.MOIST.match(w): c['M'] += 1
        q1 = None if c['H'] == c['C'] else ('H' if c['H'] > c['C'] else 'C')
        q2 = None if c['D'] == c['M'] else ('D' if c['D'] > c['M'] else 'M')
        rows.append(dict(q1=q1, q2=q2, ex1=(c['H'] > 0) != (c['C'] > 0), ex2=(c['D'] > 0) != (c['M'] > 0)))
    st = stats(rows); out['fam|CTRL_CI_truewords'] = st; print('FAM CTRL_CI_truewords', json.dumps(st), flush=True)
    # per-page implied complexion under the family rule (for the record; testable later)
    pg = collections.defaultdict(collections.Counter)
    for r in axis_values(vh):
        for k, v in r['c'].items(): pg[r['page']][k] += v
    implied = {p: (('H' if c['H'] > c['C'] else 'C' if c['C'] > c['H'] else '?') + ('D' if c['D'] > c['M'] else 'M' if c['M'] > c['D'] else '?'))
               for p, c in pg.items()}
    out['implied_pages_family_rule'] = implied
    # degree ladder search
    ci_enc = L.encode_entries(ci, seed=61, pad=0.35, max_len=128)
    for name, ents in [('P_CI', ci_enc), ('V_herb', vh), ('N_markov', L.markov_null(vh, 1)), ('V_stars', L.voynich_entries('ZL3b', ('S',)))]:
        tr, te = split(ents, 1)
        t = time.time()
        R = degree_search(ents, tr, te)
        if name == 'P_CI':
            for h in R['top'][:10]:
                h['decoded'] = [L.decode_item(ents, it, 2) for it in h['items']]
        out['deg|' + name] = R
        print('DEG', name, round(time.time() - t), json.dumps({k: R[k] for k in ('n_hyp', 'n_pass', 'best_lbf', 'med_test_lbf')}),
              R['top'][0]['items'] if R['top'] else None, R['top'][0].get('decoded') if R['top'] else None, flush=True)
    L.jsave('cycle3.json', out)
