#!/usr/bin/env python3
"""la68 cycle 2: does a one-sign word on a tablet abbreviate a word written on the SAME tablet
(or on another tablet of the same deposit)?

Per document D: S = one-sign words (LA: syllabic signs standing alone on tablets, the la45/la49
sealing class and others; LB: free one-syllable words and syllabic ligature adjuncts), W = words of 2+ signs.
Rule = sign position chosen per word length (named uniform rules + NR random maps).
Hit = an S token equal to the rule's sign of some W word of the same document (scope 'doc') or of
another document in the same site x deposit (scope 'dep').
Nulls: B within-word sign shuffle (keeps each document's sign content, kills position);
       A one-sign words re-dealt among documents of the same site.
Held-out: rule picked on a random half of documents, scored on the other half (20 splits), z vs null B.
Planted: LA documents get one-sign words written by a hidden rule from their own words (rate f).
"""
import sys, os, json, re, collections, math, time, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la68_common as C

NR = int(os.environ.get('NR', 2000))
NNULL = int(os.environ.get('NNULL', 60))
SEED = int(os.environ.get('SEED', 682))


def la_docs():
    d = json.load(open(os.path.join(C.DATA, 'corpus.json')))
    fs = C.la_findspots()
    rec_single = collections.Counter()
    for r in d:
        if r['support'] in C.LA_R:
            for t in r['tokens']:
                if t['t'] == 'word' and len(t['s']) == 1:
                    rec_single[t['s'][0]] += 1
    out = []
    for r in d:
        if r['support'] not in C.LA_T:
            continue
        S, W = [], []
        for t in r['tokens']:
            if t['t'] == 'word':
                if len(t['s']) == 1:
                    S.append(t['s'][0])
                else:
                    W.append(tuple(t['s']))
        out.append(dict(id=r['id'], site=r['site'], dep=r['site'] + '|' + fs.get(r['id'], ''), S=S, W=W))
    return out, rec_single


def lb_docs():
    U = C.load_lb()   # reuse the parser: rebuild per document
    out = []
    raw = collections.defaultdict(lambda: dict(S=[], W=[]))
    # load_lb keeps words with doc ids; abbreviations per doc are in abbr_docs without ids, so re-parse
    sylls = set()
    for s in U:
        for w in U[s]['words']:
            sylls |= set(w['signs'])
    for l in open(os.path.join(C.DATA, 'damos_items.jsonl')):
        r = json.loads(l)
        h = r.get('heading') or ''
        m = re.match(r'([A-Z]+)\s+([A-Z][a-z]*)?', h)
        if not m or m.group(1) not in C.LB_SITES:
            continue
        txt = C._clean(r.get('content') or '')
        txt = re.sub(r'supra\s+sigillum(=[A-Z0-9=]+)?', ' ', txt)
        S, W = [], []
        for tok in re.split(r'\s+', txt):
            if re.fullmatch(r'[a-z]{2}[0-9]?', tok) and tok in sylls:
                S.append(tok)
            elif re.fullmatch(r'[A-Z*][A-Z0-9*]*(\+[A-Z0-9*]+)+', tok):
                S += [p.lower() for p in tok.split('+')[1:] if p.lower() in sylls]
            elif re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', tok):
                sg = tok.split('-')
                if all(x in sylls for x in sg):
                    W.append(tuple(sg))
        ser = m.group(2) or ''
        out.append(dict(id=h, site=m.group(1), dep=m.group(1) + '|' + ser, S=S, W=W))
    return out


def sign_of(w, o, glob):
    l = len(w)
    if o < 6:
        return [w[min(o, l - 1)]]
    if o == 6:
        return [w[-1]]
    if o == 7:
        return [w[-2]]
    if o == 8:
        return [w[min(range(l), key=lambda j: (glob[w[j]], j))]]
    if o == 9:
        return [w[max(range(l), key=lambda j: (glob[w[j]], -j))]]
    return list(w)


def hits(docs, rules, glob, scope, Sfilter=None):
    """returns [nrules, ndocs] hit counts (S tokens matched)."""
    deps = collections.defaultdict(list)
    for i, d in enumerate(docs):
        deps[d['dep']].append(i)
    H = np.zeros((len(rules), len(docs)))
    # cache per (doc, length, option) sign sets
    for i, d in enumerate(docs):
        S = [x for x in d['S'] if Sfilter is None or x in Sfilter]
        if not S:
            continue
        if scope == 'doc':
            Ws = d['W']
        else:
            Ws = [w for j in deps[d['dep']] if j != i for w in docs[j]['W']]
        if not Ws:
            continue
        opts = {}
        for w in Ws:
            l = min(len(w), C.LMAX)
            for o in range(C.NOPT):
                opts.setdefault((l, o), set()).update(sign_of(w, o, glob))
        for ri, (_, (pm, _, _)) in enumerate(rules):
            cov = set()
            for l in range(2, C.LMAX + 1):
                cov |= opts.get((l, int(pm[l])), set())
            H[ri, i] = sum(1 for x in S if x in cov)
    return H


def make_rules(seed):
    rng = np.random.default_rng(seed)
    R = []
    for o, nm in enumerate(C.OPT_NAMES):
        R.append((nm, (np.full(C.LMAX + 1, o, dtype=np.int32), 'all', 'token')))
    for _ in range(NR):
        r = C.random_rule(rng)
        R.append((C.rule_name(r).split('|')[0], r))
    return R


def shuffle_words(docs, rng):
    out = []
    for d in docs:
        W = []
        for w in d['W']:
            w = list(w)
            rng.shuffle(w)
            W.append(tuple(w))
        out.append(dict(d, W=W))
    return out


def redeal_S(docs, rng):
    by = collections.defaultdict(list)
    for i, d in enumerate(docs):
        by[d['site']].append(i)
    out = [dict(d) for d in docs]
    for s, ix in by.items():
        Ss = [docs[i]['S'] for i in ix]
        p = rng.permutation(len(ix))
        for k, i in enumerate(ix):
            out[i]['S'] = Ss[p[k]]
    return out


def job(args):
    script, scope, kind, k, filt = args
    rng = np.random.default_rng(SEED * 100003 + k * 7 + zlib.crc32((script + scope + kind).encode()) % 1000)
    docs, glob, Sf = DATA_[script]
    if filt == 'seal':
        Sfilter = Sf
    else:
        Sfilter = None
    if kind == 'B':
        docs = shuffle_words(docs, rng)
    elif kind == 'A':
        docs = redeal_S(docs, rng)
    H = hits(docs, RULES, glob, scope, Sfilter)
    return (script, scope, kind, k, filt, H.sum(1), H)


DATA_ = {}
RULES = make_rules(SEED)


def setup():
    la, rec = la_docs()
    lb = lb_docs()
    for name, docs in (('LA', la), ('LB', lb)):
        glob = collections.Counter()
        for d in docs:
            for w in d['W']:
                glob.update(w)
        DATA_[name] = (docs, glob, set(rec) if name == 'LA' else None)
    return la, lb, rec


def plant(f, k):
    """LA documents: each tablet with words gets its one-sign words replaced, at rate f, by the hidden rule's sign
    of a random word on the same tablet (rest kept)."""
    rng = np.random.default_rng(SEED * 31 + k)
    docs, glob, _ = DATA_['LA']
    if k % 2 == 0:
        o = [0, 6, 1, 8, 7][(k // 2) % 5]
        pm = np.full(C.LMAX + 1, o, dtype=np.int32)
    else:
        pm = C.random_rule(rng)[0]
    nd = []
    for d in docs:
        S = list(d['S'])
        if d['W']:
            for j in range(len(S)):
                if rng.random() < f:
                    w = d['W'][rng.integers(len(d['W']))]
                    o = int(pm[min(len(w), C.LMAX)])
                    c = sign_of(w, o, glob)
                    S[j] = c[rng.integers(len(c))]
        nd.append(dict(d, S=S))
    return nd, pm


def main():
    t0 = time.time()
    setup()
    res = {}
    jobs = []
    for script in ('LA', 'LB'):
        for scope in ('doc', 'dep'):
            for filt in (['all', 'seal'] if script == 'LA' else ['all']):
                jobs.append((script, scope, 'real', 0, filt))
                for kind in ('B', 'A'):
                    for k in range(NNULL if kind == 'B' else NNULL // 2):
                        jobs.append((script, scope, kind, k, filt))
    with Pool(2) as P:
        out = P.map(job, jobs, chunksize=2)
    split_rng = np.random.default_rng(SEED)
    for script in ('LA', 'LB'):
        ndocs = len(DATA_[script][0])
        splits = [split_rng.random(ndocs) < 0.5 for _ in range(20)]
        for scope in ('doc', 'dep'):
            for filt in (['all', 'seal'] if script == 'LA' else ['all']):
                real = [o for o in out if o[:3] == (script, scope, 'real') and o[4] == filt][0]
                Hr = real[6]
                tot = real[5]
                key = f'{script}|{scope}|{filt}'
                d = dict(n_S_tokens_matched_any=float(tot[10]))
                for kind in ('B', 'A'):
                    nl = [o for o in out if o[:3] == (script, scope, kind) and o[4] == filt]
                    NT = np.array([o[5] for o in nl])            # [nnull, nrules]
                    mu, sd = NT.mean(0), np.maximum(NT.std(0), 1.0)
                    z = (tot - mu) / sd
                    zn = (NT - mu) / sd
                    fw = float((1 + (zn.max(1) >= z.max()).sum()) / (1 + len(nl)))
                    named = {RULES[i][0]: dict(real=float(tot[i]), null=float(mu[i]), z=float(z[i]),
                                               P=float((1 + (NT[:, i] >= tot[i]).sum()) / (1 + len(nl))))
                             for i in range(C.NOPT)}
                    best = int(z.argmax())
                    # held-out: choose on half a, score z on half b (null per-doc hits from same nulls)
                    ho = []
                    NH = np.array([o[6] for o in nl])           # [nnull, nrules, ndocs]
                    for sp in splits:
                        a, b = sp, ~sp
                        za = (Hr[:, a].sum(1) - NH[:, :, a].sum(2).mean(0)) / np.maximum(NH[:, :, a].sum(2).std(0), 1.0)
                        bi = int(za.argmax())
                        nb = NH[:, bi, b].sum(1)
                        rb = Hr[bi, b].sum()
                        ho.append(dict(rule=RULES[bi][0], z_b=float((rb - nb.mean()) / max(nb.std(), 1.0)),
                                       P_b=float((1 + (nb >= rb).sum()) / (1 + len(nb)))))
                    d[kind] = dict(best=RULES[best][0], z_best=float(z.max()), P_familywise=fw, named=named,
                                   heldout_mean_z=float(np.mean([h['z_b'] for h in ho])),
                                   heldout_frac_P05=float(np.mean([h['P_b'] <= 0.05 for h in ho])),
                                   heldout_rules=collections.Counter(h['rule'] for h in ho).most_common(4))
                res[key] = d
                print(key, json.dumps({k: (v if k not in ('B', 'A') else {kk: vv for kk, vv in v.items() if kk != 'named'}) for k, v in d.items()}), flush=True)
                for kind in ('B', 'A'):
                    print('   ', kind, {n: (round(v['real']), round(v['null'], 1), round(v['z'], 2), round(v['P'], 3)) for n, v in d[kind]['named'].items()}, flush=True)
    json.dump(res, open(os.path.join(C.CK, f'c2_seed{SEED}.json'), 'w'), indent=1, default=str)
    print('secs', time.time() - t0)


def main_plant():
    setup()
    docs0 = DATA_['LA']
    rows = []
    for f in (1.0, 0.5, 0.25):
        for k in range(10):
            nd, pm = plant(f, k)
            glob = docs0[1]
            Hr = hits(nd, RULES, glob, 'doc').sum(1)
            rng = np.random.default_rng(k)
            NT = np.array([hits(shuffle_words(nd, rng), RULES, glob, 'doc').sum(1) for _ in range(15)])
            z = (Hr - NT.mean(0)) / np.maximum(NT.std(0), 1.0)
            zn = (NT - NT.mean(0)) / np.maximum(NT.std(0), 1.0)
            best = RULES[int(z.argmax())][1][0]
            from la68_c1 import canon
            m = sum(canon(best[l], l) == canon(pm[l], l) for l in (2, 3, 4))
            rows.append(dict(f=f, k=k, hidden=[C.OPT_NAMES[pm[l]] for l in range(2, 7)],
                             best=[C.OPT_NAMES[best[l]] for l in range(2, 7)], match=int(m), zmax=float(z.max()),
                             null_zmax95=float(np.quantile(zn.max(1), 0.95))))
            print(rows[-1], flush=True)
    summ = {f: dict(match3=float(np.mean([r['match'] == 3 for r in rows if r['f'] == f])),
                    sig=float(np.mean([r['zmax'] > r['null_zmax95'] for r in rows if r['f'] == f]))) for f in (1.0, 0.5, 0.25)}
    print(summ)
    json.dump(dict(rows=rows, summary=summ), open(os.path.join(C.CK, f'c2_plant_seed{SEED}.json'), 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'plant':
        main_plant()
    else:
        main()
