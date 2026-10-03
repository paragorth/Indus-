"""Brute-force search over STRUCTURED READINGS of the 80 commonest signs, scored only against OUTSIDE facts
on HELD-OUT sites (sites absent from IM77, as in tools/strat_credential_heldout.py / S349).

A reading assigns each of the 80 signs one label from a small ontology:
  COMMODITY-1..6, COUNT, OFFICE-1..4, PLACE, QUALIFIER, PERSON-MARK, CONNECTIVE, UNKNOWN   (16 labels)
Signs outside the top 80 get a fixed label OTHER. A text's reading is the label sequence; its features are the
label multiset, first and last label, label bigrams and a length bucket. A multinomial naive-Bayes classifier per
outside fact (object type, emblem, find-region, seal size class, material, coarse period) is fitted on the training
texts and scored on the held-out texts as the mean log-likelihood gain over the prior (bits per text), summed over
facts. Simulated annealing over the sign -> label map maximises that score.

Two objectives are run: 'heldout' (anneal directly on the held-out score, as specified; the permuted-fact control
then measures how much a search can overfit 485 texts) and 'crosscity' (anneal on Mohenjo-daro <-> Harappa
cross-prediction, then report the held-out score once, which is the honest out-of-sample figure).

Controls: (1) facts permuted across texts; (2) sign order shuffled within texts; (3) raw sign-identity classifier
(the ceiling: a reading is a coarsening of sign identity) and a length-only classifier (the floor).
Labels in a pure outside-fact scorer are exchangeable, so stability is measured after aligning each top solution's
labels to the best solution (maximum-overlap matching); the category names are then descriptive only.

Usage: python3 tools/strat_anneal.py [--restarts 50] [--steps 3000] [--level seq|seq_raw|seq_strong] [--quick]
Writes data/derived/strat_anneal.txt and data/derived/strat_anneal.json.
"""
import json, csv, collections, random, sys, argparse, time, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'          # one BLAS thread per worker: 100x faster than oversubscribed threads here
import numpy as np
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ['COMMODITY-%d' % i for i in range(1, 7)] + ['COUNT'] + ['OFFICE-%d' % i for i in range(1, 5)] + \
         ['PLACE', 'QUALIFIER', 'PERSON-MARK', 'CONNECTIVE', 'UNKNOWN']
L = len(LABELS)           # 16 selectable labels
OTHER = L                 # fixed label for signs outside the top 80
LT = L + 1
NTOP = 80
FACTS = ['type', 'emblem', 'region', 'size', 'material', 'period']
REGIONS = {'Lower Indus', 'Inter-Riverine', 'Gujarat/Kutch', 'Eastern Headwaters'}


# ----------------------------------------------------------------------------------------------- data
def load(level):
    C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    import re
    rows_all = list(csv.DictReader(open(os.path.join(ROOT, 'data/raw/inscriptions.csv'))))
    # join corpus -> inscriptions.csv by (site, text) in either reading direction (many held-out rows have cisi '-'),
    # falling back to a unique cisi code; region comes from a site -> region map
    parse = lambda t: tuple(int(x) for x in re.findall(r'\d+', t))
    bykey = collections.defaultdict(list); bycisi = collections.defaultdict(list); site_region = {}
    for r in rows_all:
        bykey[(r['site'], parse(r['text']))].append(r); bykey[(r['site'], parse(r['text'])[::-1])].append(r)
        bycisi[r['cisi']].append(r); site_region.setdefault(r['site'], r['region'])

    def lookup(r):
        cand = bykey.get((r['site'], tuple(r['seq_raw'])), [])
        if not cand and r['cisi'] != '-' and len(bycisi.get(r['cisi'], [])) == 1: cand = bycisi[r['cisi']]
        for x in cand:
            if x['horizontal(mm)'] not in ('0', ''): return x
        return cand[0] if cand else {}
    im = set(r['site'] for r in csv.DictReader(open(os.path.join(ROOT, 'data/im77/im77_corpus_lines.csv'))))
    imn = {s.lower().replace('-', '').replace(' ', '') for s in im}

    def inim(site):
        return site.lower().replace('-', '').replace(' ', '') in imn or \
               site in ('Mohenjo-daro', 'Harappa', 'Chanhu-daro', 'Lothal', 'Kalibangan')

    def mat(m):
        m = m.lower()
        if m in ('-', ''): return None
        if 'steatite' in m: return 'steatite'
        if m in ('clay', 'terracotta', 'ceramic'): return 'clay'
        if m == 'faience': return 'faience'
        if m == 'copper': return 'copper'
        return 'other'

    def emblem(s, typ):
        if s in ('', '-') or not typ.startswith('SEAL'): return None
        if s.startswith('Bull1'): return 'unicorn'
        if s in ('Gaur', 'Elep', 'Zebu', 'Bult'): return s
        return 'other'

    def period(site, p):
        p = p.strip()
        if site == 'Mohenjo-daro':
            return {'Early': 'early', 'Intermediate': 'middle', 'Late': 'late'}.get(p)
        if site == 'Harappa':
            return {'2': 'early', '3': 'middle', '4/5': 'late'}.get(p)
        if site == 'Dholavira':
            return {'4': 'middle', '5': 'middle', '4/5': 'middle', '6': 'late', '7': 'late'}.get(p)
        return None

    def typ(t):
        t = t.split(':')[0]
        return t if t in ('SEAL', 'TAB', 'POT', 'TAG') else 'other'

    seen = set(); T = []
    for r in C:
        s = r[level]
        if not s or len(s) < 1: continue
        key = (r['site'], tuple(s))
        if key in seen: continue        # duplicate texts (moulded tablets, same text at a site) counted once
        seen.add(key)
        rw = lookup(r)
        h = rw.get('horizontal(mm)', '0')
        size = None
        if r['type'].startswith('SEAL') and h not in ('0', ''):
            h = float(h); size = 'small' if h < 24 else ('medium' if h < 30 else 'large')
        reg = site_region.get(r['site'], 'Other')
        T.append(dict(site=r['site'], seq=list(s), held=not inim(r['site']), big=r['site'] in ('Mohenjo-daro', 'Harappa'),
                      facts=dict(type=typ(r['type']), emblem=emblem(r['symbol'], r['type']),
                                 region=reg if reg in REGIONS else None, size=size,
                                 material=mat(r['material']), period=period(r['site'], r['period']))))
    return T


def build(T, nsel):
    """Index signs, build sparse structures for fast feature computation."""
    cnt = collections.Counter(x for t in T for x in t['seq'])
    top = [s for s, _ in cnt.most_common(nsel)]
    sidx = {s: i for i, s in enumerate(top)}
    N = len(T)
    sign_of = np.array([[sidx.get(x, nsel) for x in t['seq']] for t in T], dtype=object)
    # flat token arrays
    tok_text = np.concatenate([[i] * len(t['seq']) for i, t in enumerate(T)]).astype(np.int64)
    tok_sign = np.concatenate([[sidx.get(x, nsel) for x in t['seq']] for t in T]).astype(np.int64)
    first = np.array([sidx.get(t['seq'][0], nsel) for t in T]); last = np.array([sidx.get(t['seq'][-1], nsel) for t in T])
    big_text = np.concatenate([[i] * (len(t['seq']) - 1) for i, t in enumerate(T) if len(t['seq']) > 1] or [[]]).astype(np.int64)
    big_a = np.concatenate([[sidx.get(x, nsel) for x in t['seq'][:-1]] for t in T if len(t['seq']) > 1]).astype(np.int64)
    big_b = np.concatenate([[sidx.get(x, nsel) for x in t['seq'][1:]] for t in T if len(t['seq']) > 1]).astype(np.int64)
    lens = np.array([len(t['seq']) for t in T]); lenb = np.minimum(lens, 7) - 1  # buckets 1..7+
    lenF = np.zeros((N, 7)); lenF[np.arange(N), lenb] = 1
    return dict(top=top, sidx=sidx, N=N, tok_text=tok_text, tok_sign=tok_sign, first=first, last=last,
                big_text=big_text, big_a=big_a, big_b=big_b, lenF=lenF, cnt=cnt)


def features(S, assign):
    """assign: int array of length NTOP+1 (sign index -> label, last entry OTHER)."""
    N = S['N']; lab = assign[S['tok_sign']]
    F1 = np.bincount(S['tok_text'] * LT + lab, minlength=N * LT).reshape(N, LT).astype(float)
    F2 = np.zeros((N, LT)); F2[np.arange(N), assign[S['first']]] = 1
    F3 = np.zeros((N, LT)); F3[np.arange(N), assign[S['last']]] = 1
    F4 = np.bincount(S['big_text'] * (LT * LT) + assign[S['big_a']] * LT + assign[S['big_b']],
                     minlength=N * LT * LT).reshape(N, LT * LT).astype(float)
    return np.hstack([F1, F2, F3, F4, S['lenF']])


def raw_features(S):
    N = S['N']; n = NTOP + 1
    F1 = np.zeros((N, n)); np.add.at(F1, (S['tok_text'], S['tok_sign']), 1)
    F2 = np.zeros((N, n)); F2[np.arange(N), S['first']] = 1
    F3 = np.zeros((N, n)); F3[np.arange(N), S['last']] = 1
    return np.hstack([F1, F2, F3, S['lenF']])


# ----------------------------------------------------------------------------------------------- scoring
def sub_struct(S, rows):
    """Token-level structure restricted to a subset of text rows (re-indexed 0..len(rows)-1)."""
    rows = np.asarray(rows); pos = -np.ones(S['N'], dtype=np.int64); pos[rows] = np.arange(len(rows))
    m = pos[S['tok_text']] >= 0; mb = pos[S['big_text']] >= 0
    return dict(N=len(rows), tok_text=pos[S['tok_text'][m]], tok_sign=S['tok_sign'][m], first=S['first'][rows],
                last=S['last'][rows], big_text=pos[S['big_text'][mb]], big_a=S['big_a'][mb], big_b=S['big_b'][mb],
                lenF=S['lenF'][rows])


def class_tensors(S, rows, Y):
    """Per class: sign counts, first/last sign counts, sign-bigram matrix, length counts (training side)."""
    n = NTOP + 1; C = Y.shape[1]
    sub = sub_struct(S, rows)
    cls_tok = Y[sub['tok_text']]                      # tokens x C
    U = np.zeros((C, n)); Fi = np.zeros((C, n)); La = np.zeros((C, n)); B = np.zeros((C, n, n))
    for c in range(C):
        np.add.at(U[c], sub['tok_sign'], cls_tok[:, c])
        np.add.at(Fi[c], sub['first'], Y[:, c]); np.add.at(La[c], sub['last'], Y[:, c])
        np.add.at(B[c], (sub['big_a'], sub['big_b']), Y[sub['big_text'], c])
    Le = Y.T @ sub['lenF']
    return dict(U=U, Fi=Fi, La=La, B=B, Le=Le)


def class_features(ct, assign):
    """Class x feature count matrix for a reading, from the precomputed sign-level tensors."""
    P = np.zeros((NTOP + 1, LT)); P[np.arange(NTOP + 1), assign] = 1
    C = ct['U'].shape[0]
    big = np.matmul(P.T, np.matmul(ct['B'], P)).reshape(C, LT * LT)
    return np.hstack([ct['U'] @ P, ct['Fi'] @ P, ct['La'] @ P, big, ct['Le']])


def raw_class_features(ct):
    C = ct['U'].shape[0]
    return np.hstack([ct['U'], ct['Fi'], ct['La'], ct['Le']])


class FactSet:
    """Per fact: class tensors for training rows, token structure for test rows, labels (integer-coded)."""
    def __init__(self, T, S, train_mask, test_mask, perm_seed=None):
        self.facts = {}; self.S = S
        rnd = random.Random(perm_seed)
        for f in FACTS:
            y = [t['facts'][f] for t in T]
            if perm_seed is not None:          # control 1: permute facts across texts, within train and within test
                for mask in (train_mask, test_mask):
                    idx = [i for i in range(len(T)) if mask[i]]
                    vals = [y[i] for i in idx]; rnd.shuffle(vals)
                    for i, v in zip(idx, vals): y[i] = v
            tr = [i for i in range(len(T)) if train_mask[i] and y[i] is not None]
            te = [i for i in range(len(T)) if test_mask[i] and y[i] is not None]
            classes = sorted({y[i] for i in tr})
            ci = {c: k for k, c in enumerate(classes)}
            te = [i for i in te if y[i] in ci]
            if len(classes) < 2 or len(te) < 10: continue
            ytr = np.array([ci[y[i]] for i in tr]); yte = np.array([ci[y[i]] for i in te])
            Y = np.zeros((len(tr), len(classes))); Y[np.arange(len(tr)), ytr] = 1
            prior = np.log((np.bincount(ytr, minlength=len(classes)) + 0.5) / (len(tr) + 0.5 * len(classes)))
            d = dict(n_tr=len(tr), ct=class_tensors(S, tr, Y), te=sub_struct(S, te), yte=yte, prior=prior,
                     classes=classes, majority=classes[int(np.argmax(np.bincount(ytr)))], n_te=len(te))
            d['tau'] = self._calibrate(S, tr, ytr, Y, prior)
            self.facts[f] = d

    @staticmethod
    def _calibrate(S, tr, ytr, Y, prior, alpha=0.5):
        """Naive Bayes is over-confident; pick a temperature per fact by 2-fold cross-validation inside the training
        set using raw sign features (independent of any reading)."""
        rnd = np.random.RandomState(3); perm = rnd.permutation(len(tr)); half = len(tr) // 2
        folds = [(perm[:half], perm[half:]), (perm[half:], perm[:half])]
        grid = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]; loss = np.zeros(len(grid))
        tr = np.asarray(tr)
        for a, b in folds:
            cnt = raw_class_features(class_tensors(S, tr[a], Y[a])) + alpha
            logp = np.log(cnt) - np.log(cnt.sum(1, keepdims=True))
            ll0 = raw_features(sub_struct(S, tr[b])) @ logp.T
            for g, tau in enumerate(grid):
                ll = tau * ll0 + prior; ll -= ll.max(1, keepdims=True); ll -= np.log(np.exp(ll).sum(1, keepdims=True))
                loss[g] -= ll[np.arange(len(b)), ytr[b]].sum()
        return grid[int(np.argmin(loss))]

    def score(self, assign, alpha=0.5, detail=False, raw=False):
        """Sum over facts of mean log-likelihood gain (bits/text) over the prior on test rows."""
        tot = 0.0; det = {}
        for f, d in self.facts.items():
            if raw:
                cnt = raw_class_features(d['ct']) + alpha; Fte = raw_features(d['te'])
            else:
                cnt = class_features(d['ct'], assign) + alpha; Fte = features(d['te'], assign)
            logp = np.log(cnt) - np.log(cnt.sum(1, keepdims=True))
            ll = d['tau'] * (Fte @ logp.T) + d['prior']           # test x classes, tempered
            ll -= ll.max(1, keepdims=True); ll -= np.log(np.exp(ll).sum(1, keepdims=True))
            gain = (ll[np.arange(d['n_te']), d['yte']] - d['prior'][d['yte']]).mean() / np.log(2)
            tot += gain
            if detail:
                acc = float((ll.argmax(1) == d['yte']).mean())
                base = float((d['yte'] == d['classes'].index(d['majority'])).mean())
                det[f] = dict(gain_bits=round(float(gain), 4), acc=round(acc, 3), majority_acc=round(base, 3), n_test=d['n_te'])
        return (tot, det) if detail else tot

    def score_length_only(self, alpha=0.5, detail=False):
        tot = 0.0; det = {}
        for f, d in self.facts.items():
            cnt = d['ct']['Le'] + alpha; logp = np.log(cnt) - np.log(cnt.sum(1, keepdims=True))
            ll = d['tau'] * (d['te']['lenF'] @ logp.T) + d['prior']; ll -= ll.max(1, keepdims=True); ll -= np.log(np.exp(ll).sum(1, keepdims=True))
            gain = (ll[np.arange(d['n_te']), d['yte']] - d['prior'][d['yte']]).mean() / np.log(2); tot += gain
            if detail: det[f] = dict(gain_bits=round(float(gain), 4), acc=round(float((ll.argmax(1) == d['yte']).mean()), 3))
        return (tot, det) if detail else tot


# ----------------------------------------------------------------------------------------------- annealing
G = {}   # globals for worker processes


def init_worker(S, factsets):
    G['S'] = S; G['FS'] = factsets


def objective(assign):
    return sum(fs.score(assign) for fs in G['FS']) / len(G['FS'])


def anneal(args):
    seed, steps, T0, T1 = args
    rnd = np.random.RandomState(seed)
    a = np.append(rnd.randint(0, L, NTOP), OTHER)
    cur = objective(a); best = cur; besta = a.copy()
    for k in range(steps):
        T = T0 * (T1 / T0) ** (k / max(1, steps - 1))
        i = rnd.randint(NTOP); old = a[i]; new = rnd.randint(L)
        if new == old: continue
        a[i] = new; s = objective(a)
        if s >= cur or rnd.rand() < np.exp((s - cur) / T):
            cur = s
            if s > best: best = s; besta = a.copy()
        else:
            a[i] = old
    return best, besta.tolist(), seed


CKPT = {}


def run_search(S, factsets, restarts, steps, seed0, procs, name=None):
    """Each stage's restarts are checkpointed to data/derived/strat_anneal_ckpt*.json so a killed run resumes."""
    if name and name in CKPT and len(CKPT[name]) == restarts:
        return [tuple(r) for r in CKPT[name]]
    with Pool(procs, initializer=init_worker, initargs=(S, factsets)) as p:
        res = p.map(anneal, [(seed0 + i, steps, 0.03, 0.0005) for i in range(restarts)])
    res.sort(key=lambda r: -r[0])
    if name:
        CKPT[name] = [list(r) for r in res]
        json.dump(CKPT, open(CKPT['_path'], 'w'))
    return res


def align(ref, sol):
    """Relabel sol's labels to maximise overlap with ref (greedy maximum matching over 16x16 overlap)."""
    M = np.zeros((L, L), int)
    for r, s in zip(ref[:NTOP], sol[:NTOP]): M[s, r] += 1
    mapping = {}; used = set()
    for _ in range(L):
        best = None
        for s in range(L):
            if s in mapping: continue
            for r in range(L):
                if r in used: continue
                if best is None or M[s, r] > best[0]: best = (M[s, r], s, r)
        if best is None: break
        mapping[best[1]] = best[2]; used.add(best[2])
    return [mapping.get(x, x) if x != OTHER else OTHER for x in sol]


def stability(res, topk=10):
    ref = res[0][1]; tops = [align(ref, r[1]) for r in res[:topk]]
    out = []
    for i in range(NTOP):
        votes = collections.Counter(t[i] for t in tops)
        lab, n = votes.most_common(1)[0]
        out.append((i, lab, n / len(tops)))
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, default=50); ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--level', default='seq'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--restarts_b', type=int, default=None, help='restarts for the cross-city stages B/Bp (default: --restarts)')
    ap.add_argument('--steps_b', type=int, default=None, help='steps for the cross-city stages B/Bp (default: --steps)')
    A = ap.parse_args()
    if A.quick: A.restarts, A.steps = 4, 200
    if A.restarts_b is None: A.restarts_b = A.restarts
    if A.steps_b is None: A.steps_b = A.steps
    t0 = time.time()
    suffix = '' if A.level == 'seq' else '_' + A.level
    if A.quick: suffix += '_quick'
    CKPT['_path'] = os.path.join(ROOT, f'data/derived/strat_anneal_ckpt{suffix}.json')
    if os.path.exists(CKPT['_path']):
        old = json.load(open(CKPT['_path'])); old.pop('_path', None); CKPT.update(old)
    T = load(A.level); N = len(T)
    held = np.array([t['held'] for t in T]); big = np.array([t['big'] for t in T])
    md = np.array([t['site'] == 'Mohenjo-daro' for t in T]); ha = np.array([t['site'] == 'Harappa' for t in T])
    S = build(T, NTOP)
    out = []; J = dict(level=A.level, restarts=A.restarts, steps=A.steps, n_texts=N, n_heldout=int(held.sum()),
                       heldout_sites=sorted(collections.Counter(t['site'] for t in T if t['held']).items(), key=lambda x: -x[1]))
    P = lambda *a: (print(*a), out.append(' '.join(str(x) for x in a)))
    P(f'strat_anneal: level={A.level} texts={N} (deduplicated per site), held-out={held.sum()} texts from '
      f'{len(J["heldout_sites"])} non-IM77 sites; top {NTOP} signs cover '
      f'{sum(S["cnt"][s] for s in S["top"]) / sum(S["cnt"].values()):.1%} of tokens')
    # training masks: facts other than region fit on Mohenjo-daro + Harappa; region fits on all IM77 sites
    train_all = ~held
    FS_held = FactSet(T, S, big, held)
    # region needs regional classes in training, so refit it on all non-held-out sites
    FS_held.facts['region'] = FactSet(T, S, train_all, held).facts.get('region')
    if FS_held.facts['region'] is None: del FS_held.facts['region']
    for f, d in FS_held.facts.items():
        P(f'  fact {f}: classes {d["classes"]}, train n={d["n_tr"]}, held-out n={d["n_te"]}, majority={d["majority"]}, NB temperature {d["tau"]}')
    J['facts'] = {f: dict(classes=d['classes'], n_train=int(d['n_tr']), n_test=int(d['n_te']), tau=d['tau']) for f, d in FS_held.facts.items()}

    # ---- control 3 baselines
    raw_s, raw_d = FS_held.score(None, detail=True, raw=True); len_s, len_d = FS_held.score_length_only(detail=True)
    rnd = np.random.RandomState(0); rand_scores = []
    for _ in range(50):
        a = np.append(rnd.randint(0, L, NTOP), OTHER); rand_scores.append(FS_held.score(a))
    P(f'\nBaselines on held-out (sum over facts of bits/text gained over prior):')
    P(f'  raw sign identity classifier: {raw_s:.3f}   per fact {raw_d}')
    P(f'  length-only classifier:       {len_s:.3f}   per fact {len_d}')
    P(f'  random reading (50 draws):    median {np.median(rand_scores):.3f}, max {max(rand_scores):.3f}')
    J['baselines'] = dict(raw=raw_s, raw_detail=raw_d, length=len_s, length_detail=len_d,
                          random_median=float(np.median(rand_scores)), random_max=float(max(rand_scores)))

    results = {}
    # ---- objective A: anneal on held-out score directly (as specified)
    P(f'\n[A] anneal on held-out score, {A.restarts} restarts x {A.steps} steps')
    resA = run_search(S, [FS_held], A.restarts, A.steps, A.seed, A.procs, 'A')
    bestA = np.array(resA[0][1]); sA, dA = FS_held.score(bestA, detail=True)
    P(f'  best {sA:.3f}  top-10 range {resA[min(9, len(resA) - 1)][0]:.3f}..{resA[0][0]:.3f}  per fact {dA}')
    results['A_heldout'] = dict(best=sA, detail=dA, scores=[r[0] for r in resA])

    # ---- control 1: facts permuted across texts (same search)
    P(f'\n[C1] control: outside facts permuted across texts (within train and within held-out), same search')
    FS_perm = FactSet(T, S, big, held, perm_seed=7)
    FS_perm.facts['region'] = FactSet(T, S, train_all, held, perm_seed=7).facts.get('region')
    resC1 = run_search(S, [FS_perm], A.restarts, A.steps, A.seed + 1000, A.procs, 'C1')
    sC1, dC1 = FS_perm.score(np.array(resC1[0][1]), detail=True)
    P(f'  best {sC1:.3f}  top-10 range {resC1[min(9, len(resC1) - 1)][0]:.3f}..{resC1[0][0]:.3f}  per fact {dC1}')
    P(f'  raw-sign classifier on permuted facts: {FS_perm.score(None, raw=True):.3f}')
    results['C1_permuted'] = dict(best=sC1, detail=dC1, scores=[r[0] for r in resC1], raw=FS_perm.score(None, raw=True))

    # ---- control 2: sign order shuffled within texts
    P(f'\n[C2] control: sign order shuffled within each text, same search')
    T2 = [dict(t, seq=random.Random(100 + i).sample(t['seq'], len(t['seq']))) for i, t in enumerate(T)]
    S2 = build(T2, NTOP)   # same sign set (counts unchanged by shuffling), order destroyed
    FS2 = FactSet(T2, S2, big, held); FS2.facts['region'] = FactSet(T2, S2, train_all, held).facts.get('region')
    resC2 = run_search(S2, [FS2], A.restarts, A.steps, A.seed + 2000, A.procs, 'C2')
    sC2, dC2 = FS2.score(np.array(resC2[0][1]), detail=True)
    P(f'  best {sC2:.3f}  top-10 range {resC2[min(9, len(resC2) - 1)][0]:.3f}..{resC2[0][0]:.3f}  per fact {dC2}')
    results['C2_shuffled_order'] = dict(best=sC2, detail=dC2, scores=[r[0] for r in resC2])

    # ---- objective B: anneal on cross-city score, report held-out once (honest out-of-sample)
    P(f'\n[B] anneal on Mohenjo-daro <-> Harappa cross-prediction (held-out never seen by the search), '
      f'{A.restarts_b} restarts x {A.steps_b} steps')
    J['restarts_b'] = A.restarts_b; J['steps_b'] = A.steps_b
    FS_mh = FactSet(T, S, md, ha); FS_hm = FactSet(T, S, ha, md)
    for fs in (FS_mh, FS_hm): fs.facts.pop('region', None)   # one region per city: undefined
    resB = run_search(S, [FS_mh, FS_hm], A.restarts_b, A.steps_b, A.seed + 3000, A.procs, 'B')
    bestB = np.array(resB[0][1]); sB, dB = FS_held.score(bestB, detail=True)
    P(f'  best cross-city {resB[0][0]:.3f}; its held-out score {sB:.3f}  per fact {dB}')
    hb = [FS_held.score(np.array(r[1])) for r in resB[:10]]
    P(f'  held-out scores of top-10 cross-city solutions: median {np.median(hb):.3f}, range {min(hb):.3f}..{max(hb):.3f}')
    # control 1 for B
    resBp = run_search(S, [FactSet(T, S, md, ha, perm_seed=7), FactSet(T, S, ha, md, perm_seed=7)], A.restarts_b, A.steps_b, A.seed + 4000, A.procs, 'Bp')
    sBp = FS_held.score(np.array(resBp[0][1])); hbp = [FS_held.score(np.array(r[1])) for r in resBp[:10]]
    P(f'  [B-C1] permuted-fact cross-city search: best cross-city {resBp[0][0]:.3f}; its true held-out score {sBp:.3f}; '
      f'held-out scores of its top-10: median {np.median(hbp):.3f}, max {max(hbp):.3f}')
    results['B_crosscity'] = dict(best_cross=resB[0][0], heldout=sB, detail=dB, heldout_top10=hb,
                                  permuted_best_cross=resBp[0][0], permuted_heldout=sBp, permuted_heldout_top10=hbp)

    # ---- stability of the best assignments (after label alignment)
    bridge = json.load(open(os.path.join(ROOT, 'data/derived/bridge_extended.json')))
    # positional profile per sign for description
    pos = collections.defaultdict(lambda: [0, 0, 0])
    for t in T:
        for j, x in enumerate(t['seq']):
            pos[x][0] += 1
            if j == 0 and len(t['seq']) > 1: pos[x][1] += 1
            if j == len(t['seq']) - 1 and len(t['seq']) > 1: pos[x][2] += 1

    def report_stab(res, tag):
        st = stability(res); robust = [(S['top'][i], LABELS[l], f) for i, l, f in st if f > 0.8]
        P(f'\n[{tag}] stability across top-10 solutions (labels aligned to the best; names are cluster ids, see note):')
        rr = [(0, np.append(np.random.RandomState(900 + k).randint(0, L, NTOP), OTHER).tolist(), k) for k in range(10)]
        strand = stability(rr)
        P(f'  signs with the same label in >80% of top-10: {len(robust)} of {NTOP}; '
          f'mean agreement {np.mean([f for _, _, f in st]):.2f}; 10 random assignments aligned the same way: '
          f'mean {np.mean([f for _, _, f in strand]):.2f}, >80%: {sum(f > 0.8 for _, _, f in strand)}')
        rows = []
        for i, l, f in sorted(st, key=lambda x: -x[2]):
            w = S['top'][i]; n, ini, fin = pos[w]
            rows.append(dict(W=w, M=bridge.get(str(w)), label=LABELS[l], stability=round(f, 2), n=n,
                             initial=round(ini / n, 2), final=round(fin / n, 2)))
        for r in rows[:25]:
            P(f'  W{r["W"]:<4} M{str(r["M"]):<12} {r["label"]:<12} stab {r["stability"]:.2f}  n={r["n"]:<4} initial {r["initial"]:.2f} final {r["final"]:.2f}')
        return rows, robust

    rowsA, robustA = report_stab(resA, 'A')
    rowsB, robustB = report_stab(resB, 'B')
    # agreement between A and B best solutions (partition level): adjusted Rand-like pair agreement
    aA = np.array(resA[0][1][:NTOP]); aB = np.array(resB[0][1][:NTOP])
    same = sum((aA[i] == aA[j]) == (aB[i] == aB[j]) for i in range(NTOP) for j in range(i + 1, NTOP)) / (NTOP * (NTOP - 1) / 2)
    rndpair = np.mean([sum((aA[i] == aA[j]) == (x[i] == x[j]) for i in range(NTOP) for j in range(i + 1, NTOP)) / (NTOP * (NTOP - 1) / 2)
                       for x in [np.random.RandomState(k).randint(0, L, NTOP) for k in range(20)]])
    P(f'\nPartition agreement between best A and best B: {same:.3f} of sign pairs co-assigned alike (random {rndpair:.3f})')
    # how many top-10 A solutions have the robust signs in the same block as in the permuted control?
    stC1 = stability(resC1); robustC1 = [(S['top'][i], LABELS[l], f) for i, l, f in stC1 if f > 0.8]
    P(f'Permuted-fact control: {len(robustC1)} signs reach >80% stability (same statistic), so stability alone is not evidence')

    # which fact does each robust cluster predict? drop-one-label test on best A
    def block_report(best, sbest, dbest, tag):
        P(f'\nWhich outside fact each label block predicts (best {tag} reading; held-out gain lost when that block is merged into UNKNOWN):')
        blocks = {}
        for lab in range(L):
            members = [S['top'][i] for i in range(NTOP) if best[i] == lab]
            if not members: continue
            a2 = best.copy(); a2[a2 == lab] = LABELS.index('UNKNOWN')
            s2, d2 = FS_held.score(a2, detail=True)
            loss = {f: round(dbest[f]['gain_bits'] - d2[f]['gain_bits'], 3) for f in dbest}
            topf = max(loss, key=loss.get)
            blocks[LABELS[lab]] = dict(members=members, loss_total=round(sbest - s2, 3), loss_by_fact=loss, predicts=topf)
            P(f'  {LABELS[lab]:<12} {len(members):2d} signs W{members[:12]}  total loss {sbest - s2:+.3f}  most for: {topf} ({loss[topf]:+.3f})')
        return blocks
    blocks = block_report(bestA, sA, dA, 'A'); blocksB = block_report(bestB, sB, dB, 'B')

    # ---- verdict
    gapA = sA - sC1; gapB = sB - sBp
    P('\nVERDICT')
    P(f'  Held-out-annealed reading: {sA:.3f} bits/text vs permuted-fact control {sC1:.3f} (gap {gapA:+.3f}); '
      f'raw-sign classifier {raw_s:.3f}; length-only {len_s:.3f}; random reading median {np.median(rand_scores):.3f}.')
    P(f'  Cross-city-annealed reading scored once on held-out: {sB:.3f} (top-10 median {np.median(hb):.3f}) vs its permuted control '
      f'{sBp:.3f} (top-10 max {max(hbp):.3f}); gap {gapB:+.3f}.')
    P(f'  Order-shuffled control: {sC2:.3f} (order contributes {sA - sC2:+.3f}).')
    ident = (sB > max(hbp)) and (sB > len_s + 0.05)   # beats every permuted-fact solution and the length-only floor
    P(f'  Identifiable? {"yes, weakly" if ident else "no"}: ' +
      ('the reading beats permuted facts and the length-only floor out of sample.' if ident else
       'the honest out-of-sample gain over permuted facts / length-only is within noise.'))
    P(f'  Does the 16-label reading add anything over raw sign identity? {"no" if sB <= raw_s else "yes"} '
      f'(raw {raw_s:.3f} vs reading {sB:.3f}); a reading is a coarsening, so at best it matches the raw ceiling.')
    P('  Note: the ontology names (COMMODITY, OFFICE ...) carry no meaning in an outside-fact-only scorer; labels are '
      'exchangeable clusters. What the search can identify is which signs behave alike for object type, material, emblem, '
      'region, size and period, i.e. genre blocks (S270), not word meanings.')
    P(f'  elapsed {time.time() - t0:.0f}s')
    J.update(results=results, stability_A=rowsA, robust_A=robustA, stability_B=rowsB, robust_B=robustB,
             robust_in_permuted_control=len(robustC1), partition_agreement_A_B=same, partition_agreement_random=rndpair,
             blocks_A=blocks, blocks_B=blocksB, best_A=dict(zip([str(w) for w in S['top']], [LABELS[x] for x in bestA[:NTOP]])),
             best_B=dict(zip([str(w) for w in S['top']], [LABELS[x] for x in bestB[:NTOP]])),
             verdict=dict(gap_A=gapA, gap_B=gapB, identifiable=bool(ident)))
    open(os.path.join(ROOT, f'data/derived/strat_anneal{suffix}.txt'), 'w').write('\n'.join(out) + '\n')
    json.dump(J, open(os.path.join(ROOT, f'data/derived/strat_anneal{suffix}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
