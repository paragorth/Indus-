"""v45 THE RESIDUAL IS THE MESSAGE: shared library.

A rule model (the v26 Scribe with the evolved minimal genome: rich junction key, line-initial chain,
paragraph drift; section x position tables; slot-template escape) is fitted on one half of a corpus's
pages and scores the other half (2-fold cross-fit). For every word it records what the rules could
not predict: surprisal, the model's entropy at that slot, the rank and the probability-integral
transform (PIT) of the chosen word among the allowed alternatives, and, per page, the expected count
of every word type (so observed minus expected = the page's residual word usage).

Corpora run through the identical pipeline:
  V    Voynich ZL3b paragraph text
  VI   Voynich IT2a
  GEN  output of the v26 minimal genome fitted on all of V (the rules alone; no message)
  PL   output of the richer v26 planted genome (copy-and-vary, ch/sh runs, m/g quota): rules the rule model
       does NOT contain, so its residual has misfit but no message
  LAw / LAl  Latin (Isidore XVI-XVII) pushed through a planted verbose encoding: every Latin word (w) or
       letter (l) fixes a bucket of Voynich word types, and the Voynich-fitted rule model picks the word
       inside that bucket. Same surface rules, message in the choices.
  GEw / GEl  the same for Gerard's Herball page text (woodcut features exist for these pages).
"""
import os, sys, re, json, math, random, html
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v26_lib import Scribe, split_half, tokens, OFF
import v21_lib

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v45_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

# v26 evolved winner (eval_V.json): junc=2 width=1 pi_chain=0.65 b_dr=3.64
G = dict(OFF, junc=2, width=1, pi_chain=0.6537355145882364, b_dr=3.643507599308341)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ metadata
def meta():
    from v8_lib import voynich_pages as vp8
    return {p['id']: dict(quire=p['quire'], bifolio=p['bifolio'], leafnum=p['leafnum'], lang=p['lang'],
                          hand=p['hand'], illus=p['illus'], order=p['order']) for p in vp8(min_tokens=1)}


# ------------------------------------------------------------------ the walker
W0 = (0.3, 0.3, 0.4)   # tuned on V cross-fit: -11.44 bits/word vs -12.26 unsmoothed


def smooth(S, g, st, W=W0):
    """Rule model with backoff: genome distribution (rich key, chain, drift), last-unit junction table and
    section unigram, mixed with weights W; duplicates merged."""
    idx, pr = S.dist(g, st)
    s = st['s']; parts = [(idx, pr * W[0])]
    if st['k'] > 0:
        pk = 'end' if st['pc'] == 'end' else 'ne'
        J = S.get(('J', s, pk, st['prev'][-1]), ('J', s, 'any', st['prev'][-1]), ('U', s, 'any'))
        parts.append((J[0], J[1] * W[1])); wu = W[2]
    else:
        wu = W[1] + W[2]
    U = S.get(('U', s, 'any'), ('U', '*', 'any'))
    parts.append((U[0], U[1] * wu))
    a = np.concatenate([x[0] for x in parts]); b = np.concatenate([x[1] for x in parts])
    u, inv = np.unique(a, return_inverse=True)
    q = np.bincount(inv, weights=b)
    return u, q / q.sum()


def walk(S, g, p, mode, rng=None, eps=0.01, bucket_of=None, msg=None, fallback=None, W=W0):
    """mode 'score': per-token residual records + expected counts. mode 'encode': choose each word from the
    model distribution restricted to bucket msg[t] (bucket_of: word index -> bucket)."""
    s = p['sec']
    if ('L', s, 'pf') not in S.T: s = '*'
    recs = []; E = defaultdict(float); paras = []; t = 0
    for qi, pa in enumerate(p['paras']):
        out = []; prevfirst = None; prevline = None
        for li, ws in enumerate(pa):
            n = len(ws)
            drift = (np.mean([S._q(w) for w in prevline], 0) - S.base_q) if (prevline is not None and g['b_dr']) else None
            line = []
            for k in range(n):
                pc = 'lf' if k == 0 else ('end' if k == n - 1 else ('p1' if k == 1 else ('p2' if k == 2 else 'mid')))
                st = dict(s=s, k=k, pc=pc, li=li, prevfirst=prevfirst, hist=None, lineno=0, cs=0.0, above_lu=None,
                          drift=drift, hasmg=False, room=None, cpool=None, upool=None, nrng=None)
                if k > 0:
                    pw = line[-1]
                    st.update(prev=pw, previ=S.w2i[pw], prevfu=S.u2i[pw[0]], prevlen=len(pw),
                              prev2f=line[-2][0] if k >= 2 else '^')
                idx, pr = smooth(S, g, st, W)
                cls = 'lf' if k == 0 else ('end' if pc == 'end' else 'mid')
                if mode == 'score':
                    w = ws[k]
                    wi = S.w2i.get(w, -1)
                    m = idx == wi
                    pm = float(pr[m].sum()) if wi >= 0 else 0.0
                    ps = math.exp(S.slot_lp(w, p['sec'], cls))
                    lp = math.log2((1 - eps) * pm + eps * ps + 1e-300)
                    H = float(-(pr * np.log2(pr + 1e-300)).sum())
                    if pm > 0:
                        above = float(pr[pr > pm + 1e-15].sum())
                        rank = int((pr > pm + 1e-15).sum()) + 1
                        u = above + random.random() * pm
                    else:
                        rank = len(idx) + 1; u = -1.0
                    recs.append((w, qi, li, k, n, lp, H, rank, u, pm, len(idx)))
                    for a, b in zip(idx.tolist(), pr.tolist()): E[a] += b
                else:
                    b = msg[t]; t += 1
                    sel = bucket_of[idx] == b
                    if sel.any() and pr[sel].sum() > 0:
                        q = pr[sel] / pr[sel].sum(); j = idx[sel][min(int(np.searchsorted(np.cumsum(q), rng.random())), sel.sum() - 1)]
                    else:
                        cand = fallback[b]; j = cand[rng.randrange(len(cand))]
                    w = S.words[j]
                S.wid(w); line.append(w)
            out.append(line); prevfirst = line[0][0]; prevline = line
        paras.append(out)
    if mode == 'score':
        return recs, {S.words[a]: v for a, v in E.items()}
    q = dict(p); q['paras'] = paras; return q


def crossfit(C, g=G, seed=45, W=W0):
    """2-fold cross-fit: returns {page id: (records, expected counts)} and eps per fold."""
    A, B = split_half(C, seed)
    out = {}
    for tr, te in ((A, B), (B, A)):
        S = Scribe(tr, ('C', 'S'))
        voc = set(S.words)
        eps = max(0.005, float(np.mean([w not in voc for p in te for w in tokens(p)])))
        for p in te:
            out[p['id']] = walk(S, g, p, 'score', eps=eps, W=W)
    return out


# ------------------------------------------------------------------ corpora
def lines_of(C):
    return [len(l) for p in C for pa in p['paras'] for l in pa]


def skeleton(msgs, rng, line_lens, plen=(5, 10)):
    """Message units -> page skeleton (paras of lines) with Voynich-like line lengths."""
    paras = []; i = 0; n = len(msgs)
    while i < n:
        pa = []
        for _ in range(rng.randint(*plen)):
            if i >= n: break
            L = min(max(3, line_lens[rng.randrange(len(line_lens))]), n - i)
            pa.append(['?'] * L); i += L
        if pa: paras.append(pa)
    return paras


def latin_units(width=48):
    """Isidore XVI-XVII pages with chapter ids (topical units)."""
    pages = []
    for book in ('16', '17'):
        t = open(os.path.join(SCR, 'v21', f'isid{book}.html'), encoding='latin-1').read()
        t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t)
        parts = re.split(r'\n\s*([IVXL]+)\.\s+(DE [A-Z ]+)\.', t)
        for k in range(1, len(parts) - 2, 3):
            body = re.sub(r'\(([^)]*)\)', ' ', parts[k + 2])
            ws = re.findall(r'[a-z]+', body.lower().replace('j', 'i').replace('v', 'u'))
            ch = f'{book}.{parts[k]}'
            for j in range(0, len(ws), 220):
                chunk = ws[j:j + 220]
                if len(chunk) >= 60: pages.append(dict(id=f'la{len(pages):03d}', sec='L' + book, chap=ch, words=chunk))
    return pages


def gerard_units(cap=260):
    from v38_cycle2 import gerard_setup
    os.environ.setdefault('V38_CACHE', os.path.join(SCR, 'v38img'))
    import v38_cycle2
    v38_cycle2.CACHE = os.path.join(SCR, 'v38img')
    keys, words, vis, conf = gerard_setup()
    return [dict(id='ge' + k, key=k, sec='G', words=ws[:cap]) for k, ws in zip(keys, words)]


def encode(units, kind, S, M, seed, vsec='HA', cap_letters=700):
    """Planted verbose encoding. kind 'w': one Voynich word per message word, M buckets; 'l': one per letter."""
    rng = random.Random(seed)
    nv = S.nvocab
    freq = Counter()
    for (key, tab) in S.T.items():
        if key[0] == 'U' and key[1] == '*' and key[2] == 'any':
            for a, c in zip(tab[0].tolist(), (tab[1] * tab[2]).tolist()): freq[a] += c
    order = sorted(range(nv), key=lambda a: -freq[a])
    perm = list(range(M)); rng.shuffle(perm)
    bucket_of = np.full(len(S.words) + 100000, -1, np.int64)
    for r, a in enumerate(order): bucket_of[a] = perm[r % M] if (r // M) % 2 == 0 else perm[M - 1 - r % M]
    fallback = defaultdict(list)
    for a in order[:max(M * 30, 3000)]: fallback[int(bucket_of[a])].append(a)
    if kind == 'w':
        mf = Counter(w for u in units for w in u['words'])
        mo = [w for w, _ in mf.most_common()]
        mb = {w: r % M for r, w in enumerate(mo)}
        msgs = [[mb[w] for w in u['words']] for u in units]
    else:
        letters = sorted(set(''.join(w for u in units for w in u['words'])))
        lb = {c: i % M for i, c in enumerate(letters)}
        msgs = [[lb[c] for c in ''.join(u['words'])][:cap_letters] for u in units]
    ll = [x for x in lines_of(v21_lib.voynich_pages('ZL3b')) if x >= 3]
    out = []
    for u, m in zip(units, msgs):
        sk = dict(id=u['id'], sec=vsec, paras=skeleton(m, rng, ll))
        q = walk(S, G, sk, 'encode', rng=rng, bucket_of=bucket_of, msg=m, fallback=fallback)
        q['sec'] = u['sec']
        for k in ('chap', 'key'):
            if k in u: q[k] = u[k]
        out.append(q)
    return out


def get_corpus(name):
    p = os.path.join(CK, f'corpus_{name}.json')
    if os.path.exists(p): return json.load(open(p))
    if name == 'V': C = v21_lib.voynich_pages('ZL3b')
    elif name == 'VI': C = v21_lib.voynich_pages('IT2a')
    elif name == 'PL': C = json.load(open(os.path.join(vlib.DATA, 'v26_ckpt', 'planted_corpus.json')))
    elif name.startswith('GEN'):
        V = get_corpus('V'); S = Scribe(V, ('C', 'S'))
        C = S.forge(V, G, random.Random(4500 + int(name[3:] or 0)))
    else:
        V = get_corpus('V'); S = Scribe(V, ('C', 'S'))
        units = latin_units() if name.startswith('LA') else gerard_units()
        kind = name[2]
        Mb = int(name[3:]) if name[3:] else (400 if kind == 'w' else 26)
        C = encode(units, kind, S, Mb, seed=451)
    json.dump(C, open(p, 'w'))
    return C


# ------------------------------------------------------------------ residual objects
def residual(name):
    p = os.path.join(CK, f'resid_{name}.json')
    if os.path.exists(p): return json.load(open(p))
    C = get_corpus(name)
    R = crossfit(C)
    out = {pid: dict(recs=r, E=e) for pid, (r, e) in R.items()}
    json.dump(out, open(p, 'w'))
    return out


def resid_matrix(C, R, min_tot=3, kind='pearson'):
    """Pages x types residual usage (observed - expected) / sqrt(expected + 0.5); also raw counts."""
    ids = [p['id'] for p in C if p['id'] in R]
    O = [Counter(r[0] for r in R[i]['recs']) for i in ids]
    tot = Counter()
    for c in O: tot.update(c)
    voc = [w for w, n in tot.items() if n >= min_tot]
    vi = {w: j for j, w in enumerate(voc)}
    Om = np.zeros((len(ids), len(voc))); Em = np.zeros_like(Om)
    for i, pid in enumerate(ids):
        for w, n in O[i].items():
            if w in vi: Om[i, vi[w]] = n
        for w, e in R[pid]['E'].items():
            if w in vi: Em[i, vi[w]] = e
    Rm = (Om - Em) / np.sqrt(Em + 0.5)
    return ids, voc, Om, Em, Rm


def cos(M):
    Z = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return Z @ Z.T


def tfidf(Om):
    df = (Om > 0).sum(0); n = len(Om)
    return np.log1p(Om) * np.log(n / np.maximum(df, 1))
