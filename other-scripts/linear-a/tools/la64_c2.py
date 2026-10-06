#!/usr/bin/env python3
"""la64 cycle 2: the per-head instrument (E1: person count next to a commodity quantity) on Linear B
(positive control), Linear A, planted per-head archives; nulls: numbers shuffled between documents,
random units on held-out halves, nonsense rulers; ration-word and fraction-value read-offs."""
import sys, json, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la64_lib import *

PART = sys.argv[1] if len(sys.argv) > 1 else 'all'
OUT = os.path.join(CK, 'c2_%s.json' % PART)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
G = np.linspace(*LOGU, 241)


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


def pairs_only(A, i):
    m = A['k'][A['pe']] == i if len(A['pe']) else np.zeros(0, bool)
    return dict(A, pe=A['pe'][m], pn=A['pn'][m], pk=A['pk'][m])


def e1_profile(A, i, theta, ruler=None):
    B = pairs_only(A, i)
    if len(B['pe']) < 3:
        return None
    K = len(A['keys'])
    L = np.zeros((len(G), K))
    L[:, i] = G
    s = score(B, L, np.tile(theta, (len(G), 1)), ruler, use=('E1',))
    return s


def summarize(s):
    if s is None:
        return None
    ok = G[s >= s.max() - 2]
    # local maxima within 2 units of the best
    pk = [float(np.exp(G[j])) for j in range(1, len(G) - 1) if s[j] >= s[j - 1] and s[j] >= s[j + 1] and s[j] >= s.max() - 2]
    return dict(u=float(np.exp(G[np.argmax(s)])), lo=float(np.exp(ok.min())), hi=float(np.exp(ok.max())),
                contrast=float(s.max() - np.median(s)), peaks=[round(x, 2) for x in pk][:8])


def keys_with_pairs(A):
    return [i for i in range(len(A['keys'])) if (A['k'][A['pe']] == i).sum() >= 3]


def heldout_e1(A, rng, theta, nsplit=8, n_rand=300, rulers=None):
    """fit u per key on one half (E1 profile argmax), score frozen on the other half vs random u
    and vs nonsense rulers (each ruler fitted and tested the same way)."""
    out = []
    for s in range(nsplit):
        a, b = split_docs(len(A['docs']), rng)
        TA, TB = subset(A, a), subset(A, b)
        K = len(A['keys'])

        def fit_test(ruler):
            lu = np.zeros(K)
            for i in keys_with_pairs(TA):
                p = e1_profile(TA, i, theta, ruler)
                lu[i] = G[np.argmax(p)]
            return lu, float(score(TB, lu[None], theta[None], ruler, use=('E1',))[0])
        lu, sB = fit_test(None)
        LR = rng.uniform(*LOGU, (n_rand, K))
        sR = score(TB, LR, np.tile(theta, (n_rand, 1)), use=('E1',))
        row = dict(sB=sB, p_units=float((sR >= sB).mean()), u={A['keys'][i]: float(np.exp(lu[i])) for i in keys_with_pairs(TA)})
        if rulers:
            sN = [fit_test(r)[1] for r in rulers]
            row['p_ruler'] = float(np.mean([x >= sB for x in sN]))
            row['ruler_med'] = float(np.median(sN))
        out.append(row)
    return out


def shuffled_contrast(docs, clsf, keyname, rng, theta_f, n=20):
    out = []
    for s in range(n):
        D = shuffle_numbers(docs, rng)
        A = build(D, clsf)
        if keyname not in A['keys']:
            continue
        i = A['keys'].index(keyname)
        sm = summarize(e1_profile(A, i, theta_f(A)))
        if sm:
            out.append(sm)
    return out


if __name__ == '__main__':
    LB, LA = load_lb(), load_la()
    nLA = sum(not e['tot'] for d in LA for e in d['E'])
    thLB = lambda A: np.array([LB_TRUE_FR.get(f, 0.1) for f in A['frkeys']])
    thLA = lambda A: np.full(len(A['frkeys']), 0.4)
    rng = np.random.default_rng(seed('la64c2' + PART))
    rulers = [nonsense_ruler(np.random.default_rng(1000 + r)) for r in range(60)]
    if PART in ('lb', 'all'):
        A = build(LB, lb_class)
        res['lb_full'] = {A['keys'][i]: summarize(e1_profile(A, i, thLB(A))) for i in keys_with_pairs(A)}
        res['lb_full_rulers'] = [summarize(e1_profile(A, A['keys'].index('GRA'), thLB(A), r)) for r in rulers]
        save()
        res['lb_thin'] = []
        for s in range(10):
            r2 = np.random.default_rng(500 + s)
            B = build(thin(LB, r2, nLA), lb_class)
            row = {B['keys'][i]: summarize(e1_profile(B, i, thLB(B))) for i in keys_with_pairs(B)}
            row['_npairs'] = int(len(B['pe']))
            res['lb_thin'].append(row)
        save()
        res['lb_shuf'] = shuffled_contrast(LB, lb_class, 'GRA', rng, thLB, 20)
        save()
        res['lb_heldout'] = heldout_e1(A, rng, thLB(A), 8, 300, rulers[:30])
        save()
        # fraction preference: dT value profile with u_GRA free (joint grid)
        i = A['keys'].index('GRA')
        B = pairs_only(A, i)
        fj = A['frkeys'].index('dT')
        TH = np.exp(np.linspace(math.log(0.01), math.log(0.95), 40))
        best = []
        for t in TH:
            th = thLB(A).copy(); th[fj] = t
            L = np.zeros((len(G), len(A['keys']))); L[:, i] = G
            best.append(float(score(B, L, np.tile(th, (len(G), 1)), use=('E1',)).max()))
        res['lb_frac_dT'] = dict(theta=TH.tolist(), s=best)
        save()
        print('lb done', flush=True)
    if PART in ('la', 'all'):
        A = build(LA, la_class)
        res['la_pairs'] = dict(n=int(len(A['pe'])), by_key={A['keys'][i]: int((A['k'][A['pe']] == i).sum()) for i in range(len(A['keys']))},
                               by_pkey={k: int((A['pk'] == j).sum()) for j, k in enumerate(A['pkeys'])})
        res['la_full'] = {A['keys'][i]: summarize(e1_profile(A, i, thLA(A))) for i in keys_with_pairs(A)}
        res['la_full_adjonly'] = {}
        m = np.array([A['pkeys'][j] != 'DOC' for j in A['pk']], bool)
        Aa = dict(A, pe=A['pe'][m], pn=A['pn'][m], pk=A['pk'][m])
        res['la_full_adjonly'] = {A['keys'][i]: summarize(e1_profile(Aa, i, thLA(A))) for i in keys_with_pairs(Aa)}
        res['la_rulers'] = {A['keys'][i]: [summarize(e1_profile(A, i, thLA(A), r))['contrast'] for r in rulers] for i in keys_with_pairs(A)}
        save()
        res['la_shuf'] = {k: shuffled_contrast(LA, la_class, k, rng, thLA, 30) for k in ['GRA', 'CYP', 'NI', 'OLE', 'VIN']}
        save()
        res['la_heldout'] = heldout_e1(A, rng, thLA(A), 8, 300, rulers[:30])
        save()
        # random fraction values: does any set change the E1 fit?
        TH = np.exp(rng.uniform(math.log(1 / 100), math.log(0.95), (400, len(A['frkeys']))))
        tot = []
        for t in TH:
            tot.append(sum(e1_profile(A, i, t).max() for i in keys_with_pairs(A)))
        tot = np.array(tot)
        cor = {f: float(np.corrcoef(np.log(TH[:, j]), tot)[0, 1]) for j, f in enumerate(A['frkeys'])}
        res['la_frac'] = dict(range=float(tot.max() - tot.min()), sd=float(tot.std()), corr=cor,
                              top=dict(zip(A['frkeys'], np.exp(np.log(TH[np.argsort(-tot)[:20]]).mean(0)).round(3).tolist())))
        save()
        print('la done', flush=True)
    if PART in ('plant', 'all'):
        A0 = build(LA, la_class)
        res['plant'] = []
        for s in range(12):
            r2 = np.random.default_rng(700 + s)
            docs = json.loads(json.dumps(LA))
            tu = {c: float(np.exp(r2.uniform(math.log(0.05), math.log(200)))) for c in A0['keys']}
            per = int(r2.integers(0, 3))
            # plant per-head rations on every paired entry: q = n_persons * ration(per) / u
            B = build(docs, la_class)
            for j, (e_i, npers) in enumerate(zip(B['pe'], B['pn'])):
                c = B['keys'][B['k'][e_i]]
                mu, sd, _ = head_windows(la_class(c))
                x = math.exp(r2.normal(mu[per], sd[per])) * npers / tu[c]
                n = int(math.floor(x)); rem = x - n
                B['n'][e_i] = n + (0.5 if rem > 0.25 else 0)
                B['F'][e_i] = 0
            th = np.zeros(len(B['frkeys']))
            row = {}
            for i in keys_with_pairs(B):
                sm = summarize(e1_profile(B, i, th))
                row[B['keys'][i]] = dict(ratio=sm['u'] / tu[B['keys'][i]], inside=bool(sm['lo'] <= tu[B['keys'][i]] <= sm['hi']),
                                         width=sm['hi'] / sm['lo'], alias=[round(p / tu[B['keys'][i]], 2) for p in sm['peaks']])
            row['_per'] = per
            res['plant'].append(row)
            save()
        print('plant done', flush=True)
    if PART in ('words', 'all'):
        # ration-sized entries with TRUE units: do entries on person lines look like rations?
        from sklearn.metrics import roc_auc_score
        out = {}
        for nm, docs, clsf, truef, frt in [('LB', LB, lb_class, lb_true_u, LB_TRUE_FR), ('UR', load_ur3(), ur_class, lambda c: 1.0, {})]:
            A = build(docs, clsf)
            th = np.array([frt.get(f, 0.5) for f in A['frkeys']])
            LQ = logq(A, th[None])[0]
            x = LQ + np.log([truef(A['keys'][k]) for k in A['k']])
            pr = np.zeros(len(x))
            for j in range(len(x)):
                mu, sd = windows(A['cls'][A['k'][j]])
                lf = -0.5 * ((x[j] - mu) / sd) ** 2 - np.log(sd)
                p = np.exp(lf - lf.max()); p /= p.sum()
                pr[j] = p[:3].sum()
            truth = np.zeros(len(x), bool)
            for di, d in enumerate(A['docs']):
                pl = {p['line'] for p in d['P'] if p['key'] in LB_PERS | UR_PERS}
                for j in np.where(A['doc'] == di)[0]:
                    pass
            # line of each entry
            lines = []
            for di, d in enumerate(A['docs']):
                pass
            ents = [(di, e) for di, d in enumerate(A['docs']) for e in d['E'] if not e['tot'] and e['c'] in A['keys']]
            for j, (di, e) in enumerate(ents):
                pl = {p['line'] for p in A['docs'][di]['P'] if p['key'] in LB_PERS | UR_PERS}
                truth[j] = any(e['line'] + dl in pl for dl in (-1, 0, 1))
            out[nm] = dict(n_person_entries=int(truth.sum()), n=len(truth),
                           auc=float(roc_auc_score(truth, pr)) if 0 < truth.sum() < len(truth) else None,
                           mean_ration_post_person=float(pr[truth].mean()), mean_ration_post_other=float(pr[~truth].mean()))
        res['words'] = out
        save()
        print('words', out, flush=True)
