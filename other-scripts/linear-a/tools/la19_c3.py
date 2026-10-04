#!/usr/bin/env python3
"""LA-19 cycle 3: two tests the mixture never saw.
(A) Port test (transfer). Fit the K-mixture on words of one place, then read the words of the other places.
    If the palace's lexicon is a mix of tongues whose shares differ by place, the component shares of the
    unseen place's words shift away from the training shares beyond what a random split of the same sizes gives
    (20 random splits = null). Held-out gain over K=1 for the unseen words is also reported.
    LA: train = types attested at Hagia Triada, test = types never attested there.
    LB positive control: train = Pylos-only names, test = Knossos-only names (Knossos is known to hold many
    non-Greek names), and the reverse.
(B) Tablet homogeneity. Is a document written 'in one tongue'? Statistic: number of within-document pairs of
    distinct word types in the same consensus component; null = components permuted among types inside
    (length, frequency) strata, 2,000 times. LB control: personnel names co-listed on the same DAMOS document.
    Planted control: LA documents whose words are relabelled so that each document draws from one component.
usage: la19_c3.py A|B
"""
import sys, time
from la19_common import *
from la19_c2 import strata_of
sys.path.insert(0, HERE)


def fit_read(train, test, K, beta, restarts=5, seed=0):
    enc = Enc(train + test); g = enc.gvec(train)
    held, share_tr, share_te = [], [], []
    for r in range(restarts):
        o = gibbs(enc, train, test, K, 1.0, beta, 300, 150, 5, seed * 100 + r, g=g)
        held.append(o['held'])
        # component posterior of test words at the final state: refit-free approximation via zprob of a joint run
    return np.mean(held, 0)


def shares(train, test, K, beta, restarts=5, seed=0):
    """Fit on train+test jointly but with test words' assignment driven by the train-fitted components:
    implemented as a two-stage Gibbs: (1) fit on train, (2) continue with test words added (their counts included).
    Return mean component share (aligned to restart 0 by train assignments) of train and test words."""
    enc = Enc(train + test); g = enc.gvec(train)
    ref = None; S_tr, S_te, H = [], [], []
    for r in range(restarts):
        o1 = gibbs(enc, train, test, K, 1.0, beta, 300, 150, 5, seed * 100 + r, g=g)
        zi = np.concatenate([o1['z'], np.random.RandomState(r).randint(0, K, len(test))]).astype(np.int32)
        # second stage: all words, train words start from their fitted state; short run; test word shares read off
        o2 = gibbs(enc, train + test, [], K, 1.0, beta, 60, 20, 2, seed * 100 + r + 50, g=g, zinit=zi)
        zp = o2['zprob']; m = zp[:len(train)].argmax(1)
        if ref is None: ref = m; perm = np.arange(K)
        else:
            from scipy.optimize import linear_sum_assignment
            M = np.zeros((K, K))
            for a, b in zip(ref, m): M[a, b] += 1
            rr, cc = linear_sum_assignment(-M); perm = np.zeros(K, int); perm[rr] = cc
        S_tr.append(zp[:len(train)][:, perm].mean(0)); S_te.append(zp[len(train):][:, perm].mean(0)); H.append(o1['held'])
    s_tr = np.mean(S_tr, 0); s_te = np.mean(S_te, 0)
    tv = 0.5 * np.abs(s_tr - s_te).sum()
    return dict(share_train=np.round(s_tr, 3).tolist(), share_test=np.round(s_te, 3).tolist(), tv=round(float(tv), 4),
                held=np.mean(H, 0))


def port_test(name, train, test, Ks=(2, 3, 4), betas=None, nnull=20, seed=0):
    allw = train + test
    out = dict(name=name, n_train=len(train), n_test=len(test), K={})
    h1 = fit_read(train, test, 1, betas.get(1, 30.0), restarts=1, seed=seed)
    for K in Ks:
        b = betas.get(K, 30.0)
        real = shares(train, test, K, b, seed=seed)
        nulls = []
        for i in range(nnull):
            rnd = random.Random(1000 + i); idx = list(range(len(allw))); rnd.shuffle(idx)
            tr = [allw[j] for j in idx[:len(train)]]; te = [allw[j] for j in idx[len(train):]]
            nulls.append(shares(tr, te, K, b, restarts=3, seed=seed + i + 1)['tv'])
        nulls = np.array(nulls)
        gain = float((real['held'] - h1).mean())
        out['K'][K] = dict(tv=real['tv'], null_mean=round(float(nulls.mean()), 4), null_max=round(float(nulls.max()), 4),
                           p=round(float((1 + (nulls >= real['tv']).sum()) / (nnull + 1)), 4),
                           share_train=real['share_train'], share_test=real['share_test'],
                           heldout_gain_per_word_vs_K1=round(gain, 4))
        print(name, K, out['K'][K], flush=True)
    return out


def run_A():
    T = la_tokens()
    ht = set(t['w'] for t in T if t['site'] == 'Haghia Triada')
    W = la_types()
    tr = [w for w in W if w in ht]; te = [w for w in W if w not in ht]
    bl = json.load(open(os.path.join(CK, 'c1_LA.json')))['beta']; betas = {int(k): v for k, v in bl.items()}
    R = {'LA_HT_to_nonHT': port_test('LA HT -> non-HT', tr, te, betas=betas)}
    L = lb_names()
    py = [w for w, s in L if s == 'PY']; kn = [w for w, s in L if s == 'KN']
    bl = json.load(open(os.path.join(CK, 'c1_LBpers.json')))['beta']; betas = {int(k): v for k, v in bl.items()}
    R['LB_PY_to_KN'] = port_test('LB PY -> KN', py, kn, betas=betas)
    R['LB_KN_to_PY'] = port_test('LB KN -> PY', kn, py, betas=betas)
    dump(os.path.join(OUT, 'c3A.json'), R)


# ------------------------------------------------------------------ (B) homogeneity
def homog_test(doc_types, comp_of, strata_of_type, nperm=2000, seed=0):
    types = sorted(comp_of); ix = {w: i for i, w in enumerate(types)}
    comp = np.array([comp_of[w] for w in types])
    docs = [np.array([ix[w] for w in set(d) if w in ix]) for d in doc_types]
    docs = [d for d in docs if len(d) >= 2]
    def stat(c):
        s = 0
        for d in docs:
            cc = np.bincount(c[d]); s += int((cc * (cc - 1) // 2).sum())
        return s
    obs = stat(comp)
    groups = collections.defaultdict(list)
    for w in types: groups[strata_of_type[w]].append(ix[w])
    groups = [np.array(v) for v in groups.values()]
    rnd = np.random.RandomState(seed); null = np.zeros(nperm)
    for r in range(nperm):
        z = comp.copy()
        for gidx in groups: z[gidx] = comp[rnd.permutation(gidx)]
        null[r] = stat(z)
    npairs = sum(len(d) * (len(d) - 1) // 2 for d in docs)
    return dict(ndocs=len(docs), npairs=npairs, obs=obs, null_mean=round(float(null.mean()), 1),
                z=round(float((obs - null.mean()) / (null.std() + 1e-9)), 2), p=round(float((1 + (null >= obs).sum()) / (nperm + 1)), 4))


def run_B():
    from la5_common import lb_docs
    R = {}
    T = la_tokens()
    row = json.load(open(os.path.join(CK, 'c1_LA.json')))
    words = [tuple(w.split('-')) for w in row['words']]
    freq = collections.Counter(t['w'] for t in T)
    st = dict(zip(words, strata_of(words, freq)))
    bydoc = collections.defaultdict(list)
    for t in T: bydoc[t['doc']].append(t['w'])
    docs = list(bydoc.values())
    for K, e in row['stab'].items():
        if 'map' not in e: continue
        comp = dict(zip(words, e['map']))
        R['LA_K%s' % K] = homog_test(docs, comp, st, seed=int(K))
        # within Hagia Triada only
        dht = [v for k, v in bydoc.items() if k.startswith('HT')]
        R['LA_HT_K%s' % K] = homog_test(dht, comp, st, seed=int(K) + 5)
        # planted: each document's words re-labelled to the document's majority component
        rnd = random.Random(int(K))
        pc = dict(comp)
        for d in docs:
            if len(set(d)) >= 2 and rnd.random() < 0.3:
                k0 = collections.Counter(comp[w] for w in d).most_common(1)[0][0]
                for w in d: pc[w] = k0
        R['PLANT30_K%s' % K] = homog_test(docs, pc, st, seed=int(K) + 9)
        print({k: v for k, v in R.items() if k.endswith('K%s' % K)}, flush=True)
    # LB: names on DAMOS documents
    for nm in ('LBpers',):
        row = json.load(open(os.path.join(CK, 'c1_%s.json' % nm)))
        words = [tuple(w.split('-')) for w in row['words']]; wset = set(words)
        D = lb_docs()
        docs = [[tok[1] for L in d['lines'] for tok in L if tok[0] == 'W' and tok[1] in wset] for d in D]
        f = collections.Counter(w for d in docs for w in set(d))
        st = dict(zip(words, strata_of(words, f)))
        for K, e in row['stab'].items():
            if 'map' not in e: continue
            comp = dict(zip(words, e['map']))
            R['%s_K%s' % (nm, K)] = homog_test(docs, comp, st, seed=int(K))
            gl = {w: lb_greekness(w) for w in words}
            R['%s_greeklabel' % nm] = homog_test(docs, {w: (0 if g == 'G' else 1 if g == 'N' else 2) for w, g in gl.items()}, st, seed=99)
            print(nm, K, R['%s_K%s' % (nm, K)], R['%s_greeklabel' % nm], flush=True)
    dump(os.path.join(OUT, 'c3B.json'), R)


if __name__ == '__main__':
    {'A': run_A, 'B': run_B}[sys.argv[1]]()
