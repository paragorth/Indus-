"""v96 cycle 3 (N8): ONE KEY OR TWO FACTORS? CROSS-STRATUM TRANSPLANT.
If the page-recurring rare words are products of the page's spelling key (W2), the spelling of the page's COMMONEST words
(the key's other products) tells which page they belong to. If they are the page's topic (W1), the common words say
nothing about them, even when the common words carry a page habit of their own (control PKC: topic + a key that touches
only common words). Query = all tokens of the page's recurring rare types (corpus count 2-20, on >= 2 lines of the page);
reference = glyph profile of the page's common-stratum tokens only (acc_RC), or of its middle stratum (acc_RM).
Random views on train folios select the views separating {W2, W12} from {W0, W1, PKC}; the top 20 are frozen and tested
once on held-out folios (leaf parity)."""
import os, sys, json, time, hashlib
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L

NV = int(os.environ.get('V96_NV3', '250'))
FZ = os.path.join(L.DATA, 'v96_frozen_c3.json')


def view(rng):
    v = L.random_view(rng, sorted(set('abcdefghijklmnopqrstuvwxyzCSTKPF')))
    v.update(cfrac=float(rng.choice([0.2, 0.3, 0.4])), hi=int(rng.choice([10, 20, 40])), mrec=int(rng.choice([2, 2, 3])))
    return v


def stats(pages, v, h, far=0):
    cnt = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    acc = 0; C = set(); tot = sum(cnt.values())
    for w, n in cnt.most_common():
        if acc >= v['cfrac'] * tot: break
        C.add(w); acc += n
    ff = L.featf(v); cache = {}
    def prof(ws):
        z = Counter()
        for w in ws:
            x = cache.get(w)
            if x is None: x = cache[w] = Counter(ff(w))
            z.update(x)
        return z
    rows = []
    for pi, p in enumerate(pages):
        if L.half(p['id']) != h: continue
        occ = defaultdict(set)
        for li, l in enumerate(p['lines']):
            for w in l['w']: occ[w].add(li)
        Rs = {w for w, s in occ.items() if w not in C and len(s) >= v['mrec'] and 2 <= cnt[w] <= v['hi']}
        ws = [w for l in p['lines'] for w in l['w']]
        q = [w for w in ws if w in Rs]; c = [w for w in ws if w in C]; m = [w for w in ws if w not in C and w not in Rs]
        if len(q) < 2 or len(c) < 10 or len(m) < 10: continue
        rows.append((L.group(p), prof(q), prof(c), prof(m), pi))
    if len(rows) < 12: return None
    feats = sorted({f for r in rows for z in r[1:4] for f in z}); fi = {f: i for i, f in enumerate(feats)}
    def vec(z):
        x = np.zeros(len(feats))
        for f, n in z.items(): x[fi[f]] = n
        return x
    Q = np.array([vec(r[1]) for r in rows]); Cr = np.array([vec(r[2]) for r in rows]); M = np.array([vec(r[3]) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r[0]].append(i)
    pos = np.array([r[4] for r in rows])
    aRC, n = L.rank_acc(Q, Cr, groups, v['lam'], pos, far); aRM, _ = L.rank_acc(Q, M, groups, v['lam'], pos, far)
    aCM, _ = L.rank_acc(Cr, M, groups, v['lam'], pos, far)
    return dict(aRC=aRC, aRM=aRM, aCM=aCM, n=n)


def names():
    n = []
    for X in L.TEXTS:
        n += ['P_' + X, 'D_' + X] + ['K_%s_%d' % (X, k) for k in range(L.NKEY)] + ['PK_%s_%d' % (X, k) for k in range(L.NKEY)] + \
             ['PKC_%s_%d' % (X, k) for k in range(L.NKEY)]
    return n


def world(n): return 'W1CK' if n.startswith('PKC_') else L.world(n)


def run(name):
    h = int(os.environ.get('V96_H', '0'))
    out = os.path.join(L.CK, 'c3_%s_h%d.json' % (name, h))
    if os.path.exists(out): return name, 0.0
    t0 = time.time(); pages = L.corpus3(name)
    if h == 0:
        rng = np.random.default_rng(96003); views = [view(rng) for _ in range(NV)]
        res = [stats(pages, v, 0) for v in views]
    else:
        views = json.load(open(FZ))['views']
        res = []
        for v in views:
            r = stats(pages, v, 1)
            if r:
                rf = stats(pages, v, 1, far=4); r['aRC_far'] = rf['aRC'] if rf else None
            res.append(r)
    json.dump(dict(name=name, h=h, res=res, secs=time.time() - t0), open(out, 'w'))
    return name, time.time() - t0


def select():
    rng = np.random.default_rng(96003); views = [view(rng) for _ in range(NV)]
    R = {n: json.load(open(os.path.join(L.CK, 'c3_%s_h0.json' % n)))['res'] for n in names()}
    sc = []
    for i in range(NV):
        pos = [R[n][i]['aRC'] for n in R if world(n) in ('W2', 'W12') and R[n][i]]
        neg = [R[n][i]['aRC'] for n in R if world(n) in ('W0', 'W1', 'W1CK') and R[n][i]]
        if len(pos) < 25 or len(neg) < 20: continue
        auc = float(np.mean([[p > q for q in neg] for p in pos]))
        cut = 0.5 * (min(pos) + max(neg)) if min(pos) > max(neg) else float(np.median(pos + neg))
        sc.append((auc, (min(pos) - max(neg)) / (np.std(pos + neg) + 1e-9), i, cut))
    sc.sort(key=lambda x: (-x[0], -x[1])); sel = sc[:20]
    fz = dict(views=[views[i] for *_, i, _ in sel], cut=[c for *_, c in sel], train_auc=[a for a, *_ in sel],
              predictions=['Q1 (W2 reading would support): in each of ZL3b, IT2a, GC2a the held-out median aRC-0.5 over the 20 '
                           'frozen views exceeds the largest held-out median of every W1, W0 and W1CK plant',
                           'Q2: >= 15 of 20 frozen views put Voynich held-out aRC above their cut, in all three transcriptions',
                           'Q3: Q1 still holds with candidate pages within 4 positions excluded',
                           'Q4 (method check): frozen cuts classify >= 90% of held-out plants (majority vote)',
                           'Kill for the single-key reading: Voynich aRC at or below the largest W1CK plant (two factors)'])
    s = json.dumps(fz, sort_keys=True, default=str); open(FZ, 'w').write(s)
    hx = hashlib.sha256(s.encode()).hexdigest(); open(FZ.replace('.json', '.sha256'), 'w').write(hx + '  v96_frozen_c3.json\n')
    print('scored', len(sc), 'top', [(round(a, 3), round(m, 2)) for a, m, *_ in sel[:6]], 'sha', hx)


def report():
    fz = json.load(open(FZ)); cut = np.array(fz['cut']); out = {}
    for n in names() + L.GENS + L.VOY + L.FRESH_W1:
        p = os.path.join(L.CK, 'c3_%s_h1.json' % n)
        if not os.path.exists(p): continue
        r = json.load(open(p))['res']
        g = lambda k: float(np.nanmedian([x[k] - 0.5 if x and x.get(k) is not None else np.nan for x in r]))
        a = np.array([x['aRC'] if x else np.nan for x in r])
        out[n] = dict(w=world(n), aRC=g('aRC'), aRM=g('aRM'), aCM=g('aCM'), far=g('aRC_far'), votes=int(np.nansum(a > cut)))
    for n, d in sorted(out.items(), key=lambda kv: (kv[1]['w'], kv[0])):
        print('%-14s %-4s aRC %.3f far %.3f aRM %.3f aCM %.3f votes %2d' % (n, d['w'], d['aRC'], d['far'], d['aRM'], d['aCM'], d['votes']))
    neg = [d for d in out.values() if d['w'] in ('W0', 'W1', 'W1CK', 'W1F')]
    mx = max(d['aRC'] for d in neg); mxf = max(d['far'] for d in neg)
    mck = max(d['aRC'] for d in out.values() if d['w'] == 'W1CK')
    cls = [(d['votes'] > 10) == (d['w'] in ('W2', 'W12')) for d in out.values() if d['w'] in ('W0', 'W1', 'W2', 'W12', 'W1CK', 'W1F')]
    P = {v: dict(Q1=out[v]['aRC'] > mx, Q2=out[v]['votes'] >= 15, Q3=out[v]['far'] > mxf, kill=out[v]['aRC'] <= mck) for v in L.VOY if v in out}
    print('neg max %.3f far %.3f W1CK max %.3f classification %d/%d' % (mx, mxf, mck, sum(cls), len(cls))); print(P)
    json.dump(dict(out=out, P=P, cls=[sum(cls), len(cls)]), open(os.path.join(L.CK, 'c3_report.json'), 'w'))


if __name__ == '__main__':
    if sys.argv[1:] == ['select']: select(); sys.exit()
    if sys.argv[1:] == ['report']: report(); sys.exit()
    from multiprocessing import Pool
    nm = sys.argv[1:] or (names() if os.environ.get('V96_H', '0') == '0' else names() + L.GENS + L.VOY + L.FRESH_W1)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, nm):
            print(n, '%.0fs' % s, flush=True)
