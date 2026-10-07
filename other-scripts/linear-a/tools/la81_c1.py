"""la81 cycle 1: THE SCRIBE'S HYPHEN.
Where did Linear A scribes cut a sign-group that ran over the end of a physical line?
Random boundary theories (sparse random directions over corpus-internal boundary statistics built only
from documents published by CUT year) are fitted (one scale each) on cuts in documents published by CUT,
survivors frozen, then scored on cuts in documents published later.  Controls: cut positions shuffled
within word length (whole lottery rerun), planted worlds, random reference set."""
import json, os, sys, math, collections, random, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from la81_common import load_phys, CK, DATA, sha
from la78_common import pub_year, series
from la15_common import _pub_source
from la78_common import _rights

CUT = int(os.environ.get('CUT', 1950))
NH = int(os.environ.get('NH', 20000))
NSH = int(os.environ.get('NSH', 50))
FEATS = ['pre_free', 'suf_free', 'pre_fam', 'suf_fam', 'sv', 'pv', 'tp', 'fin_left', 'ini_right', 'both_free']
BETAS = np.linspace(-3, 3, 25)


def years():
    src = _pub_source(); url = _rights(); Y = {}
    C = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    for d in C:
        Y[d['id']] = pub_year(d['id'], src.get(d['id'], 'blank'), url.get(d['id'], ''))[0]
    return C, Y


def cuts_data():
    C, Y = years()
    st = {}
    for d in C:
        for t in d['tokens']:
            if t['t'] == 'word': st.setdefault(d['id'], []).append(('-'.join(t['s']), t.get('st'), t.get('fl', [])))
    P = load_phys(); out = []
    for did, d in P.items():
        cw = list(st.get(did, []))
        for r in d['recs']:
            if not (r['ok'] and r['cls'] and set(r['cls']) == {'S'} and '-' in r['w'] and '+' not in r['w']): continue
            signs = r['w'].split('-')
            if len(signs) != len(r['lines']): continue
            m = next((x for x in cw if x[0] == r['w']), None)
            status = m[1] if m else None
            if m: cw.remove(m)
            ls = r['lines']
            if len(set(ls)) == 2 and all(ls[i] <= ls[i + 1] for i in range(len(ls) - 1)):
                k = sum(1 for x in ls if x == ls[0])
                out.append(dict(doc=did, site=d['site'], sup=d['support'], w=signs, k=k, st=status,
                                year=Y.get(did, 1990), findspot=''))
    return C, Y, out


def lexicon(C, Y, cut):
    T = collections.Counter()
    for d in C:
        if Y.get(d['id'], 9999) > cut: continue
        for t in d['tokens']:
            if t['t'] == 'word' and t.get('st') in ('read', 'damaged'):
                T[tuple(t['s'])] += 1
    return T


class Stats:
    def __init__(self, T):
        self.T = T; self.types = list(T)
        self.big = collections.Counter(); self.uni = collections.Counter(); self.fin = collections.Counter(); self.ini = collections.Counter()
        for t in self.types:
            for a, b in zip(t, t[1:]): self.big[(a, b)] += 1
            for a in t: self.uni[a] += 1
            self.fin[t[-1]] += 1; self.ini[t[0]] += 1
        self.V = len(self.uni) + 1

    def feats(self, w, k):
        w = tuple(w); n = len(w); pre, suf = w[:k], w[k:]
        T = self.T
        pf = [t for t in self.types if t != w and len(t) > k and t[:k] == pre]
        sf = [t for t in self.types if t != w and len(t) > n - k and t[-(n - k):] == suf]
        a, b = w[k - 1], w[k]
        f = dict(pre_free=math.log1p(T.get(pre, 0)), suf_free=math.log1p(T.get(suf, 0)),
                 pre_fam=math.log1p(len(pf)), suf_fam=math.log1p(len(sf)),
                 sv=math.log1p(len({t[k] for t in pf})), pv=math.log1p(len({t[-(n - k) - 1] for t in sf})),
                 tp=-math.log((self.big[(a, b)] + 0.5) / (self.uni[a] + 0.5 * self.V)),
                 fin_left=math.log((self.fin[a] + 0.5) / (self.uni[a] + 1)),
                 ini_right=math.log((self.ini[b] + 0.5) / (self.uni[b] + 1)),
                 both_free=float(T.get(pre, 0) > 0 and T.get(suf, 0) > 0))
        return [f[x] for x in FEATS]


def design(cuts, S, mu=None, sd=None):
    """per cut: list of candidate positions with base features and boundary features"""
    rows = []
    for c in cuts:
        n = len(c['w'])
        B = np.array([[k == 1, k == n - 1, k / n] for k in range(1, n)], float)
        F = np.array([S.feats(c['w'], k) for k in range(1, n)], float)
        rows.append([B, F, c['k'] - 1])
    allF = np.vstack([r[1] for r in rows])
    if mu is None: mu, sd = allF.mean(0), allF.std(0) + 1e-9
    for r in rows: r[1] = (r[1] - mu) / sd
    return rows, mu, sd


def pack(rows):
    """flatten: position arrays with group ids"""
    B = np.vstack([r[0] for r in rows]); F = np.vstack([r[1] for r in rows])
    g = np.concatenate([[i] * len(r[0]) for i, r in enumerate(rows)])
    y = np.concatenate([[j == r[2] for j in range(len(r[0]))] for r in rows]).astype(float)
    return B, F, g, y


def grouped_ll(logit, g, y, ng):
    """sum over groups of log softmax at the chosen position; logit (R, P)"""
    L = np.atleast_2d(logit); M = L.max(1, keepdims=True); E = np.exp(L - M)
    G = np.zeros((len(g), ng)); G[np.arange(len(g)), g] = 1
    den = E @ G
    return (L[:, y > 0] - M).sum(1) - np.log(den).sum(1)


def fit_base(B, g, y, ng):
    from scipy.optimize import minimize
    f = lambda th: -grouped_ll((B @ th)[None], g, y, ng)[0] + 0.05 * (th ** 2).sum()
    return minimize(f, np.zeros(B.shape[1]), method='L-BFGS-B').x


def random_dirs(nh, seed):
    rng = np.random.default_rng(seed); H = np.zeros((nh, len(FEATS)))
    for i in range(nh):
        m = rng.integers(1, 4); idx = rng.choice(len(FEATS), m, replace=False)
        H[i, idx] = rng.normal(0, 1, m)
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    for j in range(len(FEATS)):  # single features in both directions
        H[2 * j] = 0; H[2 * j, j] = 1; H[2 * j + 1] = 0; H[2 * j + 1, j] = -1
    return H


def score_lottery(H, B, F, g, y, ng, th):
    """fit beta per hypothesis on a grid; return (gain bits per cut, beta)"""
    base = B @ th; proj = F @ H.T  # P x nh
    b0 = grouped_ll(base[None], g, y, ng)[0]
    best = np.full(H.shape[0], -1e9); bb = np.zeros(H.shape[0])
    for beta in BETAS:
        L = base[None] + beta * proj.T
        ll = grouped_ll(L, g, y, ng)
        m = ll > best; best[m] = ll[m]; bb[m] = beta
    return (best - b0) / ng / math.log(2), bb


def eval_fixed(H, beta, B, F, g, y, ng, th):
    base = B @ th; L = base[None] + (beta[:, None] * (F @ H.T).T)
    b0 = grouped_ll(base[None], g, y, ng)[0]
    return (grouped_ll(L, g, y, ng) - b0) / ng / math.log(2)


def lottery(rows_tr, H, halves):
    B, F, g, y = pack(rows_tr); ng = len(rows_tr)
    th = fit_base(B, g, y, ng)
    gain, beta = score_lottery(H, B, F, g, y, ng, th)
    hv = []
    for idx in halves:
        sub = [rows_tr[i] for i in idx]; b, f, gg, yy = pack(sub)
        hv.append(eval_fixed(H, beta, b, f, gg, yy, len(sub), th))
    ok = np.all(np.array(hv) > 0, axis=0)
    thr = np.quantile(gain, 0.99)
    surv = np.where((gain >= thr) & ok)[0]
    return th, gain, beta, surv


def main():
    C, Y, cuts = cuts_data()
    cuts = [c for c in cuts if c['st'] in ('read', 'damaged', None)]
    if os.environ.get('READONLY'): cuts = [c for c in cuts if c['st'] == 'read']
    LEXALL = bool(os.environ.get('LEXALL'))
    T = lexicon(C, Y, 9999 if LEXALL else CUT); S = Stats(T)
    PL = os.environ.get('PLANT')  # 'j,seed': cut at argmax of feature j with prob 0.6
    tag = '_lexall' if LEXALL else ''
    if PL:
        j, sd_ = map(int, PL.split(',')); pr = random.Random(sd_); tag += f'_plant{j}_{sd_}'
        cell = collections.defaultdict(list)  # residualise the planted feature on (length, position)
        for c in cuts:
            for k in range(1, len(c['w'])): cell[(len(c['w']), k)].append(S.feats(c['w'], k)[j])
        cm = {q: (np.mean(v), np.std(v) + 1e-6) for q, v in cell.items()}
        for c in cuts:
            if pr.random() < 0.6:
                n = len(c['w'])
                v = [(S.feats(c['w'], k)[j] - cm[(n, k)][0]) / cm[(n, k)][1] for k in range(1, n)]
                c['k'] = 1 + max(range(len(v)), key=lambda i: (v[i], pr.random()))
    tr = [c for c in cuts if c['year'] <= CUT]; te = [c for c in cuts if c['year'] > CUT]
    print('cuts', len(cuts), 'train', len(tr), 'test', len(te), 'lexicon types', len(T))
    print('test sites', collections.Counter(c['site'] for c in te))
    rows_tr, mu, sd = design(tr, S)
    rows_te, _, _ = design(te, S, mu, sd)
    # HT halves for stability: alternate documents
    docs = sorted({c['doc'] for c in tr}); rng = random.Random(81)
    rng.shuffle(docs); h1 = set(docs[::2])
    halves = [[i for i, c in enumerate(tr) if c['doc'] in h1], [i for i, c in enumerate(tr) if c['doc'] not in h1]]
    H = random_dirs(NH, 81)
    th, gain, beta, surv = lottery(rows_tr, H, halves)
    print('train base theta', th.round(3), 'n survivors', len(surv), 'top gains', np.sort(gain)[-5:].round(4))
    # shuffle runs (cut positions permuted within word length), survivors kept for test scoring
    sh_surv = []
    for s in range(NSH):
        r2 = random.Random(1000 + s); bylen = collections.defaultdict(list)
        for i, r in enumerate(rows_tr): bylen[len(r[0])].append(i)
        rows_s = [list(r) for r in rows_tr]
        for L, idx in bylen.items():
            ks = [rows_tr[i][2] for i in idx]; r2.shuffle(ks)
            for i, k in zip(idx, ks): rows_s[i][2] = k
        ths, gs, bs, sv = lottery(rows_s, H, halves)
        sh_surv.append(dict(th=ths.tolist(), idx=sv.tolist(), beta=bs[sv].tolist(), topgain=float(np.sort(gs)[-1]),
                            nsurv=len(sv)))
    frozen = dict(cut=CUT, feats=FEATS, theta=th.tolist(), mu=mu.tolist(), sd=sd.tolist(), seed=81, nh=NH,
                  survivors=[dict(i=int(i), dir=H[i].round(6).tolist(), beta=float(beta[i]), train_gain=float(gain[i]))
                             for i in surv], shuffle_runs=sh_surv,
                  train_cuts=[(c['doc'], '-'.join(c['w']), c['k']) for c in tr])
    h = sha(frozen)
    if not PL and not os.environ.get('READONLY'):
        json.dump(frozen, open(os.path.join(DATA, f'la81_frozen_c1_{CUT}{tag}.json'), 'w'))
        open(os.path.join(DATA, f'la81_frozen_c1_{CUT}{tag}.sha256'), 'w').write(h + '\n')
    print('FROZEN sha256', h)
    # ---- test ----
    Bt, Ft, gt, yt = pack(rows_te); ngt = len(rows_te)
    real = eval_fixed(H[surv], beta[surv], Bt, Ft, gt, yt, ngt, th) if len(surv) else np.array([0.0])
    ref_i = np.random.default_rng(5).choice(NH, 2000, replace=False)
    ref = eval_fixed(H[ref_i], beta[ref_i], Bt, Ft, gt, yt, ngt, th)
    shm = []
    for s in sh_surv:
        if not s['idx']: shm.append(float('nan')); continue
        v = eval_fixed(H[s['idx']], np.array(s['beta']), Bt, Ft, gt, yt, ngt, np.array(s['th']))
        shm.append(float(v.mean()))
    shm = np.array(shm)
    base_te = grouped_ll((Bt @ th)[None], gt, yt, ngt)[0]
    unif = -sum(math.log(len(r[0])) for r in rows_te)
    res = dict(n_train=len(tr), n_test=len(te), n_surv=int(len(surv)),
               surv_test_mean=float(real.mean()), surv_test_pos=float((real > 0).mean()),
               ref_mean=float(ref.mean()), ref_pos=float((ref > 0).mean()),
               shuffle_surv_mean=float(np.nanmean(shm)), shuffle_p=float((np.nan_to_num(shm, nan=-9) >= real.mean()).mean()),
               shuffle_nsurv=[s['nsurv'] for s in sh_surv],
               base_vs_uniform_bits=float((base_te - unif) / ngt / math.log(2)),
               feature_singles={FEATS[j]: [float(gain[2 * j]), float(gain[2 * j + 1])] for j in range(len(FEATS))},
               surv_dirs_mean=np.mean(H[surv], 0).round(3).tolist() if len(surv) else [])
    print(json.dumps(res, indent=1))
    tag += '_ro' if os.environ.get('READONLY') else ''
    json.dump(res, open(os.path.join(CK, f'c1_real_{CUT}{tag}.json'), 'w'))


if __name__ == '__main__':
    main()
