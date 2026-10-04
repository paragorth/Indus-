"""LA-15 cycle 3: the out-of-corpus test. Predict, before looking, what the documents
published after GORILA (1985+; lineara.xyz sources other than GORILA vols 1-5) contain:
how many new word types, how many new sign types, and which known words.
Stage 1 sees only the training corpus and the held-out documents' sizes and sites; it writes
c3_predictions.json with a sha256. Stage 2 opens the held-out content and scores.
Models: EXCH = exchangeable Chao community bootstrap (2000 refits);
        ABC  = Zipf-Mandelbrot random-guessing survivors (posterior predictive);
        HAB  = habitat (site-aware) novelty: per-site leave-one-document-out novelty rates,
               beta-binomial ensemble (2000 draws).
Known-word ranking: exchangeable frequency vs habitat mix (lambda = 0.5, fixed in advance).
Controls: (a) LA random held-out GORILA documents of the same token count (exchangeable truth);
(b) Linear B, LA-sized training drawn from KN+PY+others, held out: Thebes (the most recently
published large LB find) and random PY documents; (c) LA with word tokens shuffled across
documents (word-site links broken); (d) the chronological chain G1+G3 -> G2 -> G4 -> G5.
"""
import sys, json, random, collections, time, math, hashlib
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la15_common import *

rng = np.random.default_rng(153)
prng = random.Random(153)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def community(A):
    A = np.array([x for x in A if x > 0], float)
    n = A.sum()
    C = coverage(A)
    f0 = max(int(round(chao1(A) - len(A))), 0)
    p = A / n * C
    if f0 > 0:
        p = np.concatenate([p, np.full(f0, (1 - C) / f0)])
    return p / p.sum(), len(A)


def exch_new(A, m, B):
    p, S = community(A)
    n = int(sum(A))
    out = np.empty(B, int)
    for b in range(B):
        x = rng.multinomial(n, p)
        pb, Sb = community(x[x > 0])
        y = rng.multinomial(m, pb)
        out[b] = int((y[Sb:] > 0).sum())
    return out


def hab_new(train, held_sites_tokens, key, B):
    """Per-site novelty: share of a training document's tokens absent from all other training
    documents. Held-out tokens at site s are new with rate ~ Beta(new_s + a0, old_s + b0),
    prior centred on the pooled non-majority-site rate. Repeated new types within the held-out
    set are not merged (small effect at this size)."""
    tokc = collections.Counter(w for d in train for w in d[key])
    new_s, tot_s = collections.Counter(), collections.Counter()
    for d in train:
        own = collections.Counter(d[key])
        for w, c in own.items():
            new_s[d['site']] += c if tokc[w] == c else 0
            tot_s[d['site']] += c
    big = max(tot_s, key=tot_s.get)
    pn = sum(new_s[s] for s in tot_s if s != big)
    pt = sum(tot_s[s] for s in tot_s if s != big)
    r0 = pn / pt
    k0 = 20.0
    out = np.zeros(B)
    for s, m in held_sites_tokens.items():
        a = new_s.get(s, 0) + k0 * r0
        b = tot_s.get(s, 0) - new_s.get(s, 0) + k0 * (1 - r0)
        r = rng.beta(a, b, size=B)
        out += rng.binomial(m, r)
    return out


def zm(S, a, q):
    i = np.arange(1, S + 1, dtype=float)
    w = (i + q) ** (-a)
    return w / w.sum()


def summ(A):
    A = np.asarray(A)
    A = A[A > 0]
    return np.array([len(A), (A == 1).sum(), (A == 2).sum(), (A == 3).sum(), A.max()], float)


def abc_new(A, m, n_sims=6000, keep=150):
    n = int(sum(A))
    so = summ(A)
    sc = np.array([so[0], so[1], max(so[2], 5), max(so[3], 5), max(so[4], 5)])
    sims = []
    for _ in range(n_sims):
        S = int(np.exp(rng.uniform(np.log(so[0]), np.log(so[0] * 60))))
        a, q = rng.uniform(0.3, 1.6), rng.uniform(0, 30)
        x = rng.multinomial(n, zm(S, a, q))
        sims.append((np.sqrt((((summ(x) - so) / sc) ** 2).sum()), S, a, q))
    sims.sort()
    out = []
    for d, S, a, q in sims[:keep]:
        p = zm(S, a, q)
        x = rng.multinomial(n, p)
        y = rng.multinomial(m, p)
        out.append(int(((y > 0) & (x == 0)).sum()))
    return np.array(out)


def iv(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def predict(train, held_meta, key, do_abc=True, Bn=B):
    A = list(collections.Counter(w for d in train for w in d[key]).values())
    m = sum(h[1] for h in held_meta)
    hs = collections.Counter()
    for s, k in held_meta:
        hs[s] += k
    P = dict(m=m, EXCH=iv(exch_new(A, m, Bn)), HAB=iv(hab_new(train, hs, key, Bn)))
    if do_abc:
        P['ABC'] = iv(abc_new(A, m))
    return P


def known_scores(train, held, key='words', lam=0.5):
    tokc = collections.Counter(w for d in train for w in d[key])
    N = sum(tokc.values())
    site_c = collections.defaultdict(collections.Counter)
    for d in train:
        site_c[d['site']].update(d[key])
    present = set(w for d in held for w in d[key])
    sc_ex, sc_hab = {}, {}
    for w, c in tokc.items():
        pe, ph = 1.0, 1.0
        for d in held:
            md = len(d[key])
            if md == 0:
                continue
            f = c / N
            Ns = sum(site_c[d['site']].values())
            fs = site_c[d['site']][w] / Ns if Ns else f
            pe *= (1 - f) ** md
            ph *= (1 - (lam * fs + (1 - lam) * f)) ** md
        sc_ex[w], sc_hab[w] = 1 - pe, 1 - ph
    return sc_ex, sc_hab, present


def auc(scores, present):
    pos = [s for w, s in scores.items() if w in present]
    neg = [s for w, s in scores.items() if w not in present]
    if not pos or not neg:
        return None
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ranks for ties
    from scipy.stats import rankdata
    ranks = rankdata(allv)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def truth_new(train, held, key):
    seen = set(w for d in train for w in d[key])
    return len(set(w for d in held for w in d[key]) - seen)


def take_tokens(pool, m, key='words'):
    out, t = [], 0
    for d in pool:
        if t >= m:
            break
        if d[key]:
            out.append(d)
            t += len(d[key])
    return out


def scored(P, truth):
    return {k: dict(pred=v, truth=truth, inside=bool(v[1] <= truth <= v[2]), err=v[0] - truth)
            for k, v in P.items() if k != 'm'}


if __name__ == '__main__':
    t0 = time.time()
    LA = load_la()
    LB = load_lb()
    res = {}
    TR = [d for d in LA if d['pub'] != 'post']
    HO = [d for d in LA if d['pub'] == 'post']
    # ---------------- stage 1: predictions from training + held-out sizes/sites only
    pre = {}
    for key in ('words', 'signs'):
        meta = [(d['site'], len(d[key])) for d in HO if d[key]]
        pre[key] = predict(TR, meta, key)
    tokc = collections.Counter(w for d in TR for w in d['words'])
    held_shell = [dict(site=d['site'], words=['?'] * len(d['words'])) for d in HO]
    sc_ex, sc_hab, _ = known_scores(TR, held_shell)
    pre['top_known_exch'] = [w for w, _ in sorted(sc_ex.items(), key=lambda x: -x[1])[:20]]
    pre['top_known_hab'] = [w for w, _ in sorted(sc_hab.items(), key=lambda x: -x[1])[:20]]
    pre['expected_known_types'] = [float(sum(sc_ex.values())), float(sum(sc_hab.values()))]
    blob = json.dumps(pre, sort_keys=True, ensure_ascii=False)
    pre['sha256'] = hashlib.sha256(blob.encode()).hexdigest()
    json.dump(pre, open(os.path.join(OUT, 'c3_predictions.json'), 'w'), indent=1, ensure_ascii=False)
    print('PREDICTIONS', json.dumps(pre, ensure_ascii=False), time.time() - t0, flush=True)
    # ---------------- stage 2: score
    sc = {}
    for key in ('words', 'signs'):
        sc[key] = scored(pre[key], truth_new(TR, HO, key))
    sc_ex, sc_hab, present = known_scores(TR, HO)
    sc['known'] = dict(n_present=len(present & set(tokc)), auc_exch=auc(sc_ex, present), auc_hab=auc(sc_hab, present),
                       hits_top20_exch=[w for w in pre['top_known_exch'] if w in present],
                       hits_top20_hab=[w for w in pre['top_known_hab'] if w in present],
                       new_types=sorted(set(w for d in HO for w in d['words']) - set(tokc))[:60],
                       new_signs=sorted(set(s for d in HO for s in d['signs']) - set(s for d in TR for s in d['signs'])),
                       present_known=sorted(present & set(tokc)))
    res['LA_post'] = sc
    print('SCORED', json.dumps(sc, ensure_ascii=False), flush=True)

    # ---------------- controls: repeated matched-size tests (coverage of 95% intervals)
    mW = pre['words']['m']
    mS = pre['signs']['m']
    nrep = 60

    def rep_test(name, make, nrep=nrep):
        rows = collections.defaultdict(list)
        aucs = collections.defaultdict(list)
        for r in range(nrep):
            tr, ho = make(r)
            for key in ('words',):
                meta = [(d['site'], len(d[key])) for d in ho if d[key]]
                P = predict(tr, meta, key, do_abc=(r < 10), Bn=300)
                s = scored(P, truth_new(tr, ho, key))
                for k, v in s.items():
                    rows[k].append((v['inside'], v['err'], v['truth'], v['pred'][0]))
            e, h, pr = known_scores(tr, ho)
            a1, a2 = auc(e, pr), auc(h, pr)
            if a1 is not None:
                aucs['exch'].append(a1)
                aucs['hab'].append(a2)
        out = {k: dict(cover=float(np.mean([x[0] for x in v])), mean_err=float(np.mean([x[1] for x in v])),
                       mean_truth=float(np.mean([x[2] for x in v])), mean_pred=float(np.mean([x[3] for x in v])), n=len(v))
               for k, v in rows.items()}
        out['auc'] = {k: [float(np.mean(v)), len(v)] for k, v in aucs.items()}
        res[name] = out
        print(name, json.dumps(out), time.time() - t0, flush=True)

    TRw = [d for d in TR if d['words']]

    def la_random(r):
        pool = TRw[:]
        prng.shuffle(pool)
        ho = take_tokens(pool, mW)
        ids = set(d['id'] for d in ho)
        return [d for d in TR if d['id'] not in ids], ho
    rep_test('LA_random_heldout', la_random)

    LAsh = token_shuffle(LA, prng)
    TRsh = [d for d in LAsh if d['pub'] != 'post']
    HOsh = [d for d in LAsh if d['pub'] == 'post']
    Psh = predict(TRsh, [(d['site'], len(d['words'])) for d in HOsh if d['words']], 'words', Bn=500)
    e, h, pr = known_scores(TRsh, HOsh)
    res['LA_shuffled_post'] = dict(score=scored(Psh, truth_new(TRsh, HOsh, 'words')), auc_exch=auc(e, pr), auc_hab=auc(h, pr))
    print('shuf', json.dumps(res['LA_shuffled_post']), flush=True)

    LBw = [d for d in LB if d['words']]
    nonTH = [d for d in LBw if d['site'] != 'TH']
    TH = [d for d in LBw if d['site'] == 'TH']
    PY = [d for d in LBw if d['site'] == 'PY']
    nTR = sum(len(d['words']) for d in TR)

    def lb_th(r):
        a = nonTH[:]
        prng.shuffle(a)
        b = TH[:]
        prng.shuffle(b)
        return take_tokens(a, nTR), take_tokens(b, mW)
    rep_test('LB_TH_heldout', lb_th)

    def lb_py(r):
        a = PY[:]
        prng.shuffle(a)
        ho = take_tokens(a, mW)
        ids = set(d['id'] for d in ho)
        rest = [d for d in LBw if d['id'] not in ids]
        prng.shuffle(rest)
        return take_tokens(rest, nTR), ho
    rep_test('LB_PY_heldout', lb_py)

    def lb_random(r):
        a = LBw[:]
        prng.shuffle(a)
        ho = take_tokens(a, mW)
        ids = set(d['id'] for d in ho)
        return take_tokens([d for d in a if d['id'] not in ids], nTR), ho
    rep_test('LB_random_heldout', lb_random)

    # chronological chain
    chain = []
    order = [['G1', 'G3'], ['G2'], ['G4'], ['G5'], ['blank'], ['post']]
    for i in range(1, len(order)):
        tr = [d for d in LA if d['pub'] in sum(order[:i], [])]
        ho = [d for d in LA if d['pub'] in order[i]]
        for key in ('words', 'signs'):
            meta = [(d['site'], len(d[key])) for d in ho if d[key]]
            if not meta:
                continue
            P = predict(tr, meta, key, Bn=500)
            chain.append(dict(step='+'.join(order[i]), key=key, m=P['m'], score=scored(P, truth_new(tr, ho, key))))
            print(chain[-1], flush=True)
    res['chain'] = chain
    json.dump(res, open(os.path.join(OUT, 'c3.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    print('done', time.time() - t0)
