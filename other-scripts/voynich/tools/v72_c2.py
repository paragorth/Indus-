"""v72 cycle 2: freeze the extraction rule (hash), extract the payload stream of every page, test on the HOLDOUT
half (odd leaves; plants: odd pages), never used in cycle 1.

Per corpus (payload = frozen rule; also the raw surface for comparison):
  LAD   message-layer ladder trained on the discovery half, scored on the holdout half
  ARR   x4 word-arrow probes (1,000 random word-class probes, lags 1-4, survivors replicated on a page split),
        8,000-token whole-page subsample of the holdout half, 2 seeds
  GAP   x4/v33 generator-relative gap ratio G (stem x ending connectance vs 3 own-trigram resyntheses, 31 rules)
  SEC   section (illustration) information: naive Bayes on 20-token chunks, train discovery / test holdout,
        z against 20 section-label permutations of the training pages
  REC   recurrence: within-line ordered payload bigrams found on >= 2 holdout pages, vs 20 within-section
        shuffles of the holdout payload tokens (ratio, z)
  RECOV planted controls only: purity of payload type -> real word
Out: data/v72_ckpt/c2/<corpus>__<view>.json ; data/v72_ckpt/frozen.json
"""
import os, sys, json, time, random, math, hashlib
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L
sys.path.insert(0, os.path.join(os.path.dirname(L.ROOT), 'proto-elamite', 'tools'))

OUT = os.path.join(L.CK, 'c2'); os.makedirs(OUT, exist_ok=True)
_CACHE = {}


def freeze():
    fz = L.jload('frozen.json')
    if fz: return fz
    D = {}
    for f in os.listdir(os.path.join(L.CK, 'c1')):
        r = json.load(open(os.path.join(L.CK, 'c1', f))); D[(r['name'], r['rule'])] = r
    best, tab = None, {}
    for rule in L.RULES:
        if rule == 'E0_identity': continue
        if ('ZL', rule) not in D: continue
        g = [D[('ZL~' + k, rule)]['excess'] for k in L.GENS if ('ZL~' + k, rule) in D]
        crit = D[('ZL', rule)]['excess'] - max(g)
        tab[rule] = crit
        if best is None or crit > tab[best]: best = rule
    fz = dict(rule=best, params=L.RULES[best], hash=L.rule_hash(best), criterion=tab,
              note='chosen on the discovery half only: ZL message excess minus the largest generator excess')
    L.jsave('frozen.json', fz)
    return fz


def corpus(name):
    if name in _CACHE: return _CACHE[name]
    base, gen = name.split('~') if '~' in name else (name, None)
    plain = None
    if base in ('ZL', 'IT'):
        S = L.voynich('ZL3b' if base == 'ZL' else 'IT2a')
        half = [L.leaf_half(p['id']) for p in S]
    else:
        txt, mode = base.split('-')
        P = dict(BRU=L.brumati_plain, BRUL=L.brumati_plain, ISI=L.isidore_plain, DEU=L.german_plain)[txt]()
        if txt == 'BRUL':                 # list-like meaningful control: page topic kept, word order destroyed
            P = [dict(p, lines=[dict(l) for l in p['lines']]) for p in P]
            rng = random.Random(4)
            for p in P:
                ws = [w for l in p['lines'] for w in l['w']]; rng.shuffle(ws); it = iter(ws)
                for l in p['lines']: l['w'] = [next(it) for _ in l['w']]
        code = L.payload_code([w for p in P for l in p['lines'] for w in l['w']], mode=mode)
        S = L.surface(L.encode_payload(P, code)); plain = P
        half = [i % 2 for i in range(len(S))]
    if gen:
        S = L.GENS[gen](S, seed=2); plain = None
    _CACHE[name] = (S, plain, half)
    return _CACHE[name]


def docs_of(pages):
    return [[l['w'] for l in p['lines']] for p in pages]


def arrows(pages, seed):
    import x4_lib as X, x4_cycle1 as A
    docs = X.subsample(docs_of(pages), 8000, seed)
    docs = [d for d in docs if X.ntok([d])]
    gf = Counter(w for d in docs for w in X.stream(d)); alpha = sorted(gf)
    A_ = set(random.Random(5 + seed).sample(range(len(docs)), len(docs) // 2))
    surv, disc = A.probe_run([[X.stream(d)] for d in docs], alpha, A.word_classes(alpha, gf, None), range(1, 5),
                             1000, A_, random.Random(17 + seed))
    return dict(surv=sum(surv.values()), by=surv, disc=sum(disc.values()), ntok=X.ntok(docs))


def gap(pages, seed):
    import x4_lib as X, v33_lib as V
    docs = X.subsample(docs_of(pages), 8000, seed)
    toks = [w for d in docs for l in d for w in l if len(w) >= 2]
    rules = V.rule_set(n_rand=20); g = []
    base = []
    for i in range(3):
        T = V.Trigram([toks]); out = []
        rng = random.Random(seed * 10 + i)
        while len(out) < len(toks): out += [w for w in T.gen(len(toks), rng) if len(w) >= 2]
        base.append(out[:len(toks)])
    for rule in rules:
        rule.fit(toks); M = V.matrix(toks, rule, 200, 25); c = float(M.mean()) if M.size else float('nan')
        cb = []
        for b in base:
            rule.fit(b); Mb = V.matrix(b, rule, 200, 25); cb.append(float(Mb.mean()) if Mb.size else float('nan'))
        if np.mean(cb) > 0 and c == c: g.append(c / np.mean(cb))
    return dict(G=float(np.median(g)), G_iqr=[float(np.percentile(g, 25)), float(np.percentile(g, 75))], n=len(toks))


def chunks(p, n=20):
    t = [w for l in p['lines'] for w in l['w']]
    return [t[i:i + n] for i in range(0, len(t) - n + 1, n)]


def nb_acc(train, test, labels_train, alpha=0.3):
    cnt = defaultdict(Counter); tot = Counter(); prior = Counter()
    for p, y in zip(train, labels_train):
        prior[y] += len(chunks(p))
        for l in p['lines']:
            for w in l['w']: cnt[y][w] += 1; tot[y] += 1
    V = len({w for c in cnt.values() for w in c}) + 1
    labs = [k for k in prior if prior[k] > 0]; pt = sum(prior.values())
    hit = n = 0
    for p in test:
        if p['sec'] not in prior: continue
        for ch in chunks(p):
            best = max(labs, key=lambda k: math.log(prior[k] / pt) + sum(math.log(cnt[k][w] + alpha) for w in ch)
                       - len(ch) * math.log(tot[k] + alpha * V))
            hit += best == p['sec']; n += 1
    return hit / max(n, 1)


def sec_info(tr, te):
    a = nb_acc(tr, te, [p['sec'] for p in tr])
    rng = random.Random(9); nul = []
    for _ in range(20):
        lab = [p['sec'] for p in tr]; rng.shuffle(lab); nul.append(nb_acc(tr, te, lab))
    return dict(acc=a, null=float(np.mean(nul)), z=(a - np.mean(nul)) / (np.std(nul) + 1e-9))


def rec(te, nshuf=20):
    def count(pages):
        on = defaultdict(set)
        for i, p in enumerate(pages):
            for l in p['lines']:
                for a, b in zip(l['w'], l['w'][1:]): on[(a, b)].add(i)
        return sum(1 for v in on.values() if len(v) >= 2)
    real = count(te)
    nul = [count(L.gen_wshuf(te, seed=s)) for s in range(nshuf)]
    return dict(real=real, null=float(np.mean(nul)), ratio=real / max(np.mean(nul), 1e-9),
                z=(real - np.mean(nul)) / (np.std(nul) + 1e-9))


def run(job):
    name, view = job
    fn = os.path.join(OUT, f'{name}__{view}.json')
    if os.path.exists(fn): return job
    t0 = time.time()
    fz = freeze()
    S, plain, half = corpus(name)
    X = S if view == 'surface' else L.extract(S, L.RULES[fz['rule']])
    tr = [p for p, h in zip(X, half) if h == 0]; te = [p for p, h in zip(X, half) if h == 1]
    out = dict(name=name, view=view, rule=fz['rule'], hash=fz['hash'])
    lad = L.ladder(tr, te, W=20, seed=0); lad.pop('per_tok')
    out['LAD'] = {k: (v['bits'] if isinstance(v, dict) else v) for k, v in lad.items()}
    out['LAD']['excess'] = lad['gain_msg'] - lad['gain_msgsh']
    out['ARR'] = [arrows(te, s) for s in (0, 1)]
    out['GAP'] = [gap(te, s) for s in (0, 1)]
    out['SEC'] = sec_info(tr, te)
    out['REC'] = rec(te)
    toks = [w for p in te for l in p['lines'] for w in l['w']]
    out['types'] = len(set(toks)); out['ntok'] = len(toks); out['mean_len'] = sum(map(len, toks)) / len(toks)
    out['alphabet'] = len({c for w in toks for c in w})
    if plain is not None and view == 'payload':
        out['RECOV'] = L.recovery([p for p, h in zip(X, half) if h == 1], [p for p, h in zip(plain, half) if h == 1])
    out['sec'] = time.time() - t0
    json.dump(out, open(fn, 'w'), default=float)
    print(name, view, round(out['LAD']['excess'], 3), [a['surv'] for a in out['ARR']],
          [round(g['G'], 3) for g in out['GAP']], round(out['SEC']['z'], 1), round(out['REC']['ratio'], 2),
          round(out['sec']), flush=True)
    return job


def jobs():
    J = []
    for c in ['ZL', 'IT'] + ['ZL~' + g for g in L.GENS]:
        J.append((c, 'payload')); J.append((c, 'surface'))
    for txt in ['BRU', 'ISI', 'DEU', 'BRUL']:
        for mode in ['merge', 'verbose']:
            if txt == 'BRUL' and mode == 'verbose': continue
            b = f'{txt}-{mode}'
            J.append((b, 'payload')); J.append((b, 'surface'))
            if mode == 'merge':
                for g in L.GENS: J.append((b + '~' + g, 'payload'))
    return J


if __name__ == '__main__':
    print(json.dumps(freeze()), flush=True)
    J = jobs(); print(len(J), 'jobs', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.imap_unordered(run, J, chunksize=1): pass
