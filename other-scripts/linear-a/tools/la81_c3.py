"""la81 cycle 3: THE COPYIST'S EYE.
A scribe who copies a written model by eye tends to keep its line breaks; a scribe who writes from
dictation, memory or receipts re-flows the text.  For every pair of documents that share >= 2 sign-groups,
test whether a shared sign-group starts a physical line in both more often than the documents' own layout
allows.  Null: line-start flags permuted within token class (entry-initial word / other word) inside each
document, 20,000 times (keeps each document's line density and its entry structure).  Planted control: a
share of pairs given copied layout.  Linear B: DAMOS physical lines (KN, PY) through the same code, where
entries start lines by convention, so the class-preserving null is essential.
Time travel: pairs found by 1950 vs pairs with a member published later."""
import json, os, sys, re, random, itertools, collections, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from la81_common import CK, DATA, sha, load_phys
from la78_common import pub_year, _rights
from la15_common import _pub_source

NP = int(os.environ.get('NPERM', 20000))


def la_docs():
    P = load_phys(); src = _pub_source(); url = _rights()
    D = {}
    for did, d in P.items():
        recs = [r for r in d['recs'] if r['ok'] and r['lines']]
        toks = []
        for i, r in enumerate(recs):
            prev = recs[i - 1]['lines'][-1] if i else None
            start = (i == 0) or (r['lines'][0] != prev)
            if '-' in r['w'] and '+' not in r['w'] and set(r['cls']) == {'S'}:
                cl = int(i == 0 or recs[i - 1]['cls'][:1] in ('N', 'F'))
                toks.append([r['w'], int(start), cl])
        if toks:
            y = pub_year(did, src.get(did, 'blank'), url.get(did, ''))[0]
            D[did] = dict(site=d['site'], year=y, toks=toks, base=re.sub(r'[a-z]$', '', did))
    return D


def _clean_lb(t):
    t = unicodedata.normalize('NFD', t)
    t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    t = re.sub(r"[\[\]⟦⟧⌞⌟⸢⸣'\"?!<>{}]", '', t)
    return t.strip('-')


def lb_docs(sites=('KN', 'PY')):
    D = {}
    for line in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(line)
        h = x.get('heading') or ''
        if not x.get('content') or h[:2] not in sites: continue
        toks = []; prevnum = True
        for ln in x['content'].split('\n'):
            first = True
            for raw in ln.split():
                if raw.startswith('.') or raw in (',', '/', '//', ':'): continue
                t = _clean_lb(raw)
                if not t: continue
                if re.fullmatch(r'\d+', t): prevnum = True; first = False; continue
                if any(ch.islower() for ch in t) and re.fullmatch(r'[a-z0-9*\-]+', t) and '-' in t:
                    toks.append([t.upper(), int(first), int(prevnum)]); first = False; prevnum = False
                elif re.fullmatch(r"[A-Z*0-9+]+", t):
                    first = False
        if toks: D[h] = dict(site=h[:2], year=1, toks=toks, base=h)
    return D


def pairs_of(D, min_shared=2):
    ty = {k: {w for w, _, _ in v['toks']} for k, v in D.items()}
    inv = collections.defaultdict(set)
    for k, s in ty.items():
        for w in s: inv[w].add(k)
    cand = collections.Counter()
    for w, ks in inv.items():
        if len(ks) > 30: continue  # ubiquitous words (KU-RO etc.) do not define a pair
        for a, b in itertools.combinations(sorted(ks), 2): cand[(a, b)] += 1
    return [(a, b) for (a, b), n in cand.items() if n >= min_shared]


def concord(D, pairs, flags):
    """flags: dict doc -> list of line-start flags aligned with toks; count shared words starting a line in both"""
    tt = 0; n = 0
    for a, b in pairs:
        A = {}; B = {}
        for (w, _, _), f in zip(D[a]['toks'], flags[a]): A.setdefault(w, f)
        for (w, _, _), f in zip(D[b]['toks'], flags[b]): B.setdefault(w, f)
        for w in set(A) & set(B):
            tt += A[w] * B[w]; n += 1
    return tt, n


def perm_flags(D, docs, rng):
    out = {}
    for k in docs:
        T = D[k]['toks']; f = [x[1] for x in T]
        for c in (0, 1):
            idx = [i for i, x in enumerate(T) if x[2] == c]
            v = [f[i] for i in idx]; rng.shuffle(v)
            for i, x in zip(idx, v): f[i] = x
        out[k] = f
    return out


def test(D, pairs, nperm=NP, seed=0):
    docs = sorted({x for p in pairs for x in p})
    real = {k: [x[1] for x in D[k]['toks']] for k in docs}
    r, n = concord(D, pairs, real)
    rng = random.Random(seed); null = []
    for _ in range(nperm):
        null.append(concord(D, pairs, perm_flags(D, docs, rng))[0])
    null = np.array(null)
    return dict(pairs=len(pairs), shared=n, both_start=r, null_mean=float(null.mean()), null_sd=float(null.std()),
                z=float((r - null.mean()) / (null.std() + 1e-9)), p=float((null >= r).mean()))


def plant(D, pairs, share, seed):
    """copy layout from a to b for shared words in a share of pairs"""
    rng = random.Random(seed); D2 = json.loads(json.dumps(D))
    for a, b in pairs:
        if rng.random() < share:
            A = {w: f for w, f, _ in D2[a]['toks']}
            for t in D2[b]['toks']:
                if t[0] in A: t[1] = A[t[0]]
    return D2


def main():
    res = {}
    D = la_docs(); P = pairs_of(D)
    same = [(a, b) for a, b in P if D[a]['base'] == D[b]['base']]
    diff = [(a, b) for a, b in P if D[a]['base'] != D[b]['base']]
    early = [(a, b) for a, b in P if max(D[a]['year'], D[b]['year']) <= 1950]
    late = [(a, b) for a, b in P if max(D[a]['year'], D[b]['year']) > 1950]
    print('LA docs', len(D), 'pairs', len(P), 'same tablet', len(same), 'early', len(early), 'late', len(late))
    for nm, pp in [('all', P), ('same_tablet', same), ('other_tablet', diff), ('early', early), ('late', late)]:
        if pp: res['la_' + nm] = test(D, pp); print(nm, res['la_' + nm])
    # planted power
    for sh in (0.25, 0.5):
        res[f'plant_{sh}'] = [test(plant(D, P, sh, s), P, nperm=2000, seed=s) for s in range(5)]
        print('plant', sh, [round(x['z'], 2) for x in res[f'plant_{sh}']])
    # Linear B: same code (KN and PY separately)
    for site in ('KN', 'PY'):
        L = lb_docs((site,)); PL = pairs_of(L)
        rng = random.Random(1); PL = rng.sample(PL, min(len(PL), 3000))
        res['lb_' + site] = test(L, PL, nperm=500); print('LB', site, res['lb_' + site])
    # per-pair listing for the frozen copy list
    real = {k: [x[1] for x in D[k]['toks']] for k in D}
    listing = []
    for a, b in P:
        tt, n = concord(D, [(a, b)], real)
        listing.append(dict(a=a, b=b, shared=n, both_start=tt))
    res['pairs'] = sorted(listing, key=lambda x: -x['both_start'])
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'))


if __name__ == '__main__':
    main()
