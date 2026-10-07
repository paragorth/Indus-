"""la75 cycle 3: what travels between tablets. (a) carried copies: the same sign-group with the same exact quantity
on two different documents, vs numbers permuted within stratum (site x commodity x magnitude bin); frozen link graph
(sha256) then compared with findspot/scribe the numbers never saw. (b) habits carried by a sign-group: same-word
cross-document residual covariance for 6,000 random fingerprint features (maxT). Planted copies; LB KN/PY controls."""
import sys, os, json, pickle, time, collections
sys.path.insert(0, os.path.dirname(__file__))
from la75_common import *
from la75_c1 import FastPanel, dedupe

NP = int(os.environ.get('NP', 2000))


def word_ok(w):
    return bool(w) and w not in ('', '-')


def copies(rows, perm=None):
    """count cross-document pairs with same word and identical (v, frac), and list them"""
    by = collections.defaultdict(list)
    for k, r in enumerate(rows):
        if not word_ok(r['word']): continue
        src = rows[perm[k]] if perm is not None else r
        by[(r['word'], src['v'], src['frac'])].append(r['doc'])
    links = []
    for key, ds in by.items():
        ds = sorted(set(ds))
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                links.append((key, ds[i], ds[j]))
    return links


def copy_test(rows, rng, nperm=NP):
    P = Panel(rows)
    real = copies(rows); n = len(real)
    null = [len(copies(rows, P.perm(rng))) for _ in range(nperm)]
    # per word: excess copies
    wc = collections.Counter(k[0][0] for k in real)
    nullw = collections.defaultdict(list)
    for _ in range(300):
        c = collections.Counter(k[0][0] for k in copies(rows, P.perm(rng)))
        for w in set(wc) | set(c): nullw[w].append(c.get(w, 0))
    perw = {}
    for w, c in wc.items():
        arr = np.array(nullw[w] + [0] * (300 - len(nullw[w])))
        perw[w] = dict(real=c, null=float(arr.mean()), p=float((1 + (arr >= c).sum()) / 301))
    return dict(n_links=n, null_mean=float(np.mean(null)), null_sd=float(np.std(null)),
                p=float((1 + sum(x >= n for x in null)) / (nperm + 1)), links=real, perword=perw)


def habit_test(rows, feats, rng, nperm=1000):
    """same-word, different-document residual covariance per feature, maxT over features"""
    F = fmatrix(rows, feats); P = FastPanel(rows)
    keep = np.where((P.resid(F) ** 2).sum(0) > 2)[0]; keep = keep[dedupe(F[:, keep])]
    R = P.resid(F[:, keep]).astype(np.float64); R /= (R.std(0) + 1e-9)
    words = [r['word'] if word_ok(r['word']) else None for r in rows]
    wl = sorted(set(w for w in words if w)); wi = {w: i for i, w in enumerate(wl)}
    import scipy.sparse as sp
    ok = np.array([w is not None for w in words]); idx = np.where(ok)[0]
    Mw = sp.csr_matrix((np.ones(len(idx)), ([wi[words[i]] for i in idx], idx)), shape=(len(wl), len(rows)))
    # word x doc combined key to remove same-doc pairs
    wd = sorted(set((words[i], rows[i]['doc']) for i in idx)); wdi = {k: j for j, k in enumerate(wd)}
    Mwd = sp.csr_matrix((np.ones(len(idx)), ([wdi[(words[i], rows[i]['doc'])] for i in idx], idx)), shape=(len(wd), len(rows)))
    def stat(R):
        a = Mw @ R; b = Mwd @ R
        return (a ** 2).sum(0) - (b ** 2).sum(0)       # cross-document same-word pair sums
    obs = stat(R); nulls = np.array([stat(R[P.perm(rng)]) for _ in range(nperm)])
    mu, sd = nulls.mean(0), nulls.std(0) + 1e-12; z = (obs - mu) / sd
    mx = ((nulls - mu) / sd).max(1); fwer = (1 + (mx[:, None] >= z[None, :]).sum(0)) / (nperm + 1)
    p = (1 + (nulls >= obs).sum(0)) / (nperm + 1)
    o = np.argsort(-z)[:10]
    return dict(n_feat=len(keep), n_p01=int((p <= 0.01).sum()), exp_p01=0.01 * len(keep), n_fwer=int((fwer <= 0.05).sum()),
                top=[(feats[keep[i]], float(z[i]), float(p[i]), float(fwer[i])) for i in o])


def outside_links(links, attr, docs_all, rng, nperm=20000):
    lab = {d: attr.get(d, '') for d in docs_all}
    L = [(a, b) for _, a, b in links if lab.get(a) and lab.get(b)]
    if not L: return None
    obs = np.mean([lab[a] == lab[b] for a, b in L])
    labelled = [d for d in docs_all if lab[d]]; vals = [lab[d] for d in labelled]
    ge = 1
    for _ in range(nperm):
        pv = dict(zip(labelled, rng.permutation(vals)))
        ge += np.mean([pv[a] == pv[b] for a, b in L]) >= obs
    return dict(n_links=len(L), same=float(obs), p=ge / (nperm + 1))


def plant_copies(rows, rng, n=10):
    new = [dict(r) for r in rows]
    docs = sorted(set(r['doc'] for r in rows)); site = {r['doc']: r['site'] for r in rows}
    cand = [k for k, r in enumerate(new) if word_ok(r['word'])]
    planted = []
    for k in rng.choice(cand, min(len(cand), int(n * 1.5)), replace=False):
        if len(planted) >= n: break
        src = new[k]; tg = [d for d in docs if site[d] == src['site'] and d != src['doc']]
        if not tg: continue
        d = rng.choice(tg); tgt = [j for j, r in enumerate(new) if r['doc'] == d]
        j = rng.choice(tgt)
        new[j]['word'] = src['word']; new[j]['v'] = src['v']; new[j]['frac'] = src['frac']
        new[j]['tg'] = src['tg']; new[j]['mb'] = src['mb']; new[j]['stratum'] = src['stratum']
        planted.append((src['doc'], d))
    return new, planted


if __name__ == '__main__':
    t0 = time.time(); rng = np.random.default_rng(75003)
    feats = pickle.load(open(os.path.join(CK, 'c1.pkl'), 'rb'))['feats']
    rows, docs = la_entries(); rows = prepare(rows, min_n=1)
    out = {}
    ct = copy_test(rows, rng)
    frozen = sorted([[k[0], k[1], list(k[2]), a, b] for k, a, b in ct['links']])
    h = sha(frozen); json.dump(dict(sha256=h, links=frozen), open(os.path.join(D, 'la75_frozen_links.json'), 'w'))
    open(os.path.join(D, 'la75_frozen_links.sha256'), 'w').write(h + '\n')
    ct['sha256'] = h
    alld = sorted(set(r['doc'] for r in rows))
    ht = {d: docs[d]['findspot'] for d in alld if docs[d]['site'] == 'Haghia Triada'}
    sc = {d: docs[d]['scribe'] for d in alld if docs[d]['site'] == 'Haghia Triada'}
    ct['out_findspot'] = outside_links(ct['links'], ht, alld, rng)
    ct['out_scribe'] = outside_links(ct['links'], sc, alld, rng)
    out['LA_copies'] = ct; print('copies', time.time() - t0, flush=True)
    rows2, _ = la_entries(read_only=False); rows2 = prepare(rows2, min_n=1)
    out['LA_copies_all'] = copy_test(rows2, rng, 1000)
    for s in range(3):
        pr, pl = plant_copies(rows, np.random.default_rng(7800 + s))
        o = copy_test(pr, rng, 1000); o['planted'] = pl; out[f'plant_copies_{s}'] = o
    out['LA_habit'] = habit_test(rows, feats, rng); print('habit', time.time() - t0, flush=True)
    # the same with sign-groups already graded as commodities (NI, *304, *306, *308, E, SU) moved into the stratum
    cw = {'NI', '*304', '*306', '*308', 'E', 'SU'}; rn = [dict(x) for x in rows]
    for x in rn:
        if x['word'] in cw:
            x['tg'] = 'C:' + x['word']; x['stratum'] = (x['site'], x['tg'], x['mb']); x['word'] = ''
    out['LA_habit_noCW'] = habit_test(rn, feats, rng)
    # held-out style check: HT only and non-HT only for the top fraction feature
    for nm, sub in (('HT', [x for x in rows if x['site'] == 'Haghia Triada']), ('nonHT', [x for x in rows if x['site'] != 'Haghia Triada'])):
        out['LA_habit_' + nm] = habit_test(sub, [('frac', (4, 5, 6)), ('frac', (4, 5)), ('div', 5)], rng, 2000)
    # planted habit: a fake word put on 20 random entries across documents, numbers rounded to multiples of 5
    pr = [dict(r) for r in rows]; cand = [k for k, r in enumerate(pr) if r['v'] >= 6]
    for k in np.random.default_rng(7900).choice(cand, 20, replace=False):
        pr[k]['word'] = 'TEST-PLANT'; pr[k]['v'] = int(5 * round(pr[k]['v'] / 5)); pr[k]['mb'] = magbin(pr[k]['v'])
        pr[k]['stratum'] = (pr[k]['site'], pr[k]['tg'], pr[k]['mb'])
    out['plant_habit'] = habit_test(pr, feats, rng, 400)
    for p in ('KN', 'PY'):
        b = prepare(lb_entries(p), min_n=1)
        o = copy_test(b, rng, 500); hand = {x['doc']: x['scribe'] for x in b if x['scribe'] not in ('', '-')}
        o['out_hand'] = outside_links(o['links'], hand, sorted(set(x['doc'] for x in b)), rng, 2000)
        out['LB_copies_' + p] = o
        out['LB_habit_' + p] = habit_test(b, feats, rng, 400); print(p, time.time() - t0, flush=True)
    pickle.dump(out, open(os.path.join(CK, 'c3.pkl'), 'wb'))
    for k, o in out.items():
        print(k, {x: o[x] for x in o if x not in ('links', 'perword')})
        if 'perword' in o:
            top = sorted(o['perword'].items(), key=lambda kv: kv[1]['p'])[:8]
            print('   perword', top)
            print('   links', [(l[0][0], l[0][1], l[0][2], l[1], l[2]) for l in o['links'][:25]])
