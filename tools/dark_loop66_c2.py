"""S-DARK-66 cycle 2: do the two readings' predictions hold?
(a) Held-out gain of a CLASS-PAIR model over the element unigram: each middle element predicted from the class of the
    nearest preceding middle element at distance >= 2 (skip context), P(e | c') = P(c | c') P(e | c), so the gain per
    token is log2 P(c | c') - log2 P(c) (Witten-Bell smoothed).  Compared with the ELEMENT skip-bigram model (idiosyncratic
    pairs) through the same code.  Fit Mohenjo-daro + Harappa; test (i) 5-fold within MD+H, (ii) held-out Wells sites,
    (iii) the 324 IM77-only texts in Mahadevan numbers (classes mapped W -> M through bridge_extended + loop-27 proposals,
    one-to-one entries only; M unigram fitted on the IM77 texts Wells also saw).  Control: 300 random partitions of the same
    elements into classes of the same sizes (does a SHAPE-based partition beat an arbitrary one?).  Bootstrap CIs over texts.
(b) Symmetry: directed class table (earlier class -> later class), correlation of z(A->B) with z(B->A).
(c) Transitivity at the element level: the attraction graph (S366 pairs, die regime, MD+H; also relaxed count >= 3, O/E >= 2),
    clustering coefficient vs 1,000 degree-preserving rewirings; share of triangles whose three members fall in one class.
Usage: python3 tools/dark_loop66_c2.py <seq_raw|seq_strong|seq_all>
"""
import sys, json, collections, random, math, time
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop66_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rnd = random.Random(662); rng = np.random.default_rng(662); T0 = time.time()
out = [f'# S-DARK-66 cycle 2 ({LV}) {time.strftime("%Y-%m-%dT%H:%M")}']
def P(s): out.append(s); print(s, flush=True)

T = parse_all(load_wells(LV))
CJ = json.load(open(DARK + f'loop66_classes_{LV}.json'))
CLS = {k: {int(w): c for w, c in v.items()} for k, v in CJ.items() if k != 'desc_text'}
midtok = collections.Counter(x for t in T for x in t['mid'])
ELEM = sorted(w for w, c in midtok.items() if c >= 5)

# ---------------------------------------------------------------- (a) skip-context prediction
def skip_events(t, field='midpos'):
    """(prev element, element) with prev = nearest preceding middle element at distance >= 2"""
    s = t['seq']; pos = t[field]; ev = []
    for ii in range(len(pos)):
        prev = None
        for jj in range(ii - 1, -1, -1):
            if pos[ii] - pos[jj] >= 2: prev = s[pos[jj]]; break
        if prev is not None: ev.append((prev, s[pos[ii]]))
    return ev

class WB:
    """Witten-Bell bigram over symbols with unigram backoff and an UNK mass"""
    def __init__(self, pairs, unis, V):
        self.big = collections.defaultdict(collections.Counter)
        for a, b in pairs: self.big[a][b] += 1
        self.uni = collections.Counter(unis); self.N = sum(self.uni.values()); self.V = V
        self.T = len(self.uni)
    def pu(self, b):
        # Witten-Bell unigram with escape to uniform over V
        lam = self.N / (self.N + self.T)
        return lam * self.uni.get(b, 0) / self.N + (1 - lam) / self.V
    def pb(self, a, b):
        c = self.big.get(a)
        if not c: return self.pu(b)
        n = sum(c.values()); t = len(c); lam = n / (n + t)
        return lam * c.get(b, 0) / n + (1 - lam) * self.pu(b)

def gain_class(train_ev, test_ev, cls, V):
    """per-token gain in bits of P(c|c') over P(c), on test events where both signs are classed"""
    pairs = [(cls[a], cls[b]) for a, b in train_ev if a in cls and b in cls]
    unis = [cls[b] for a, b in train_ev if b in cls]
    m = WB(pairs, unis, V)
    g = []
    for a, b in test_ev:
        if a in cls and b in cls: g.append(math.log2(m.pb(cls[a], cls[b])) - math.log2(m.pu(cls[b])))
    return g

def gain_elem(train_ev, test_ev, V):
    m = WB(train_ev, [b for a, b in train_ev], V)
    return [math.log2(m.pb(a, b)) - math.log2(m.pu(b)) for a, b in test_ev]

def boot_ci(g, nb=1000):
    g = np.asarray(g)
    if len(g) == 0: return float('nan'), float('nan'), float('nan')
    bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(nb)]
    return float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def random_partitions(cls, n):
    """permute class labels among the classed signs (sizes kept)"""
    keys = list(cls); labs = [cls[w] for w in keys]
    for _ in range(n):
        rnd.shuffle(labs); yield dict(zip(keys, labs))

FIT = [t for t in T if t['site'] in BIG]; HO = [t for t in T if t['site'] not in BIG]
ev_fit = [e for t in FIT for e in skip_events(t)]; ev_ho = [e for t in HO for e in skip_events(t)]
P(f'texts {len(T)}: fit MD+H {len(FIT)} ({len(ev_fit)} skip events), held-out sites {len(HO)} ({len(ev_ho)} events)')
RES = {}
# 5-fold within MD+H
folds = [[] for _ in range(5)]
for i, t in enumerate(FIT): folds[i % 5].append(t)
for name, cls in CLS.items():
    k = len(set(cls.values()))
    g5 = []
    for f in range(5):
        tr = [e for j in range(5) if j != f for t in folds[j] for e in skip_events(t)]
        te = [e for t in folds[f] for e in skip_events(t)]
        g5 += gain_class(tr, te, cls, k)
    gho = gain_class(ev_fit, ev_ho, cls, k)
    # random partitions
    rp5 = []; rpho = []
    for rc in random_partitions(cls, NR):
        gg = []
        for f in range(5):
            tr = [e for j in range(5) if j != f for t in folds[j] for e in skip_events(t)]
            te = [e for t in folds[f] for e in skip_events(t)]
            gg += gain_class(tr, te, rc, k)
        rp5.append(np.mean(gg)); rpho.append(np.mean(gain_class(ev_fit, ev_ho, rc, k)))
    m5, lo5, hi5 = boot_ci(g5); mh, loh, hih = boot_ci(gho)
    P(f'\n## {name} ({k} classes): class-pair gain over unigram, bits per classed token')
    P(f'   5-fold MD+H: {m5:+.4f} [{lo5:+.4f}, {hi5:+.4f}] on {len(g5)} tokens; random partitions {np.mean(rp5):+.4f} [{np.percentile(rp5, 2.5):+.4f}, {np.percentile(rp5, 97.5):+.4f}] P = {pval(m5, rp5):.3f}')
    P(f'   held-out sites: {mh:+.4f} [{loh:+.4f}, {hih:+.4f}] on {len(gho)} tokens; random partitions {np.mean(rpho):+.4f} [{np.percentile(rpho, 2.5):+.4f}, {np.percentile(rpho, 97.5):+.4f}] P = {pval(mh, rpho):.3f}')
    RES[name] = dict(k=k, g5=[m5, lo5, hi5], g5_rand=[float(np.mean(rp5)), float(np.percentile(rp5, 2.5)), float(np.percentile(rp5, 97.5))], g5_P=pval(m5, rp5),
                     gho=[mh, loh, hih], gho_rand=[float(np.mean(rpho)), float(np.percentile(rpho, 2.5)), float(np.percentile(rpho, 97.5))], gho_P=pval(mh, rpho), n5=len(g5), nho=len(gho))
# element skip-bigram for comparison (unigram bits also reported)
V = len(ELEM) + 1
ge5 = []
for f in range(5):
    tr = [e for j in range(5) if j != f for t in folds[j] for e in skip_events(t)]
    te = [e for t in folds[f] for e in skip_events(t)]
    ge5 += gain_elem(tr, te, V)
geh = gain_elem(ev_fit, ev_ho, V)
m, lo, hi = boot_ci(ge5); mh, loh, hih = boot_ci(geh)
uni = WB([], [b for a, b in ev_fit], V); ubits = np.mean([-math.log2(uni.pu(b)) for a, b in ev_ho])
P(f'\n## ELEMENT skip-bigram (idiosyncratic pairs) gain: 5-fold {m:+.4f} [{lo:+.4f}, {hi:+.4f}]; held-out sites {mh:+.4f} [{loh:+.4f}, {hih:+.4f}] (unigram {ubits:.2f} bits/token on held-out)')
RES['ELEMENT'] = dict(g5=[m, lo, hi], gho=[mh, loh, hih], unigram_bits_ho=float(ubits))

# ---------------------------------------------------------------- (a iii) IM77-only new texts
I = parse_im77(load_im77())
newkeys = set(tuple(x) for x in json.load(open(DARK + 'loop27_sets.json'))['new'])
INEW = [t for t in I if tuple(t['key']) in newkeys]; IOLD = [t for t in I if tuple(t['key']) not in newkeys]
m2, src = w2m_map()
ev_new = [e for t in INEW for e in skip_events(t)]; ev_old = [e for t in IOLD for e in skip_events(t)]
P(f'\n## IM77: {len(I)} texts, new (never seen by Wells) {len(INEW)} with {len(ev_new)} skip events; old {len(IOLD)} with {len(ev_old)}')
for name, cls in CLS.items():
    k = len(set(cls.values()))
    clsM, srcM = m2w_classes(cls, m2, src)
    # transition fitted on Wells MD+H (in W), mapped to M via class labels: fit class pairs in W space, test in M space
    pairs = [(cls[a], cls[b]) for a, b in ev_fit if a in cls and b in cls]; unis = [cls[b] for a, b in ev_fit if b in cls]
    mdl = WB(pairs, unis, k)
    g = []; nprop = 0
    for a, b in ev_new:
        if a in clsM and b in clsM:
            g.append(math.log2(mdl.pb(clsM[a], clsM[b])) - math.log2(mdl.pu(clsM[b])))
            if srcM.get(a) == 'proposal' or srcM.get(b) == 'proposal': nprop += 1
    gb = []
    for a, b in ev_old:
        if a in clsM and b in clsM: gb.append(math.log2(mdl.pb(clsM[a], clsM[b])) - math.log2(mdl.pu(clsM[b])))
    # random partitions through the same mapping
    rp = []
    for rc in random_partitions(cls, NR):
        rcM, _ = m2w_classes(rc, m2, src)
        pr = [(rc[a], rc[b]) for a, b in ev_fit if a in rc and b in rc]; un = [rc[b] for a, b in ev_fit if b in rc]
        mr = WB(pr, un, k)
        gg = [math.log2(mr.pb(rcM[a], rcM[b])) - math.log2(mr.pu(rcM[b])) for a, b in ev_new if a in rcM and b in rcM]
        rp.append(np.mean(gg) if gg else float('nan'))
    mn, lon, hin = boot_ci(g); mo, loo, hio = boot_ci(gb)
    P(f'   {name}: new texts {mn:+.4f} [{lon:+.4f}, {hin:+.4f}] on {len(g)} tokens ({nprop} rest on a proposed bridge entry); random partitions {np.nanmean(rp):+.4f} [{np.nanpercentile(rp, 2.5):+.4f}, {np.nanpercentile(rp, 97.5):+.4f}] P = {pval(mn, [x for x in rp if x == x]):.3f}; '
      f'IM77 old texts (transcription-robust only) {mo:+.4f} [{loo:+.4f}, {hio:+.4f}] on {len(gb)}')
    RES[name]['gnew'] = [mn, lon, hin]; RES[name]['gnew_rand'] = [float(np.nanmean(rp)), float(np.nanpercentile(rp, 2.5)), float(np.nanpercentile(rp, 97.5))]
    RES[name]['gnew_P'] = pval(mn, [x for x in rp if x == x]); RES[name]['nnew'] = len(g); RES[name]['gold'] = [mo, loo, hio]

# ---------------------------------------------------------------- (b) symmetry of the directed class table
P('\n## (b) Symmetry: directed class table (earlier -> later element, distance >= 2), z vs 1,000 permutations within site x type x midlen')
def directed_pairs(t, seq=None):
    s = seq if seq is not None else t['seq']; pos = t['midpos']; out = []
    for ii in range(len(pos)):
        for jj in range(ii + 1, len(pos)):
            if pos[jj] - pos[ii] >= 2 and s[pos[ii]] != s[pos[jj]]: out.append((s[pos[ii]], s[pos[jj]]))
    return out
def dtable(T, seqs, cls, classes):
    ci = {c: i for i, c in enumerate(classes)}; M = np.zeros((len(classes), len(classes)))
    for t, s in zip(T, seqs):
        for a, b in directed_pairs(t, s):
            if a in cls and b in cls: M[ci[cls[a]], ci[cls[b]]] += 1
    return M
SYM = {}
for name, cls in CLS.items():
    classes = sorted(set(cls.values())); O = dtable(T, [t['seq'] for t in T], cls, classes)
    N = np.array([dtable(T, permute_middles(T, rnd), cls, classes) for _ in range(1000)])
    Z = (O - N.mean(0)) / (N.std(0) + 1e-9)
    iu = np.triu_indices(len(classes), 1); m = (N.mean(0)[iu] >= 3) & (N.mean(0).T[iu] >= 3)
    za = Z[iu][m]; zb = Z.T[iu][m]
    r = float(np.corrcoef(za, zb)[0, 1]) if len(za) > 3 else float('nan')
    agree = float(np.mean(np.sign(za) == np.sign(zb))) if len(za) else float('nan')
    strong = [(classes[i], classes[j], Z[i, j], Z[j, i]) for i, j in zip(*iu) if max(abs(Z[i, j]), abs(Z[j, i])) > 2.5]
    P(f'   {name}: {m.sum()} off-diagonal cell pairs with E >= 3 both ways; corr(z(A->B), z(B->A)) = {r:+.2f}, sign agreement {agree:.2f}; '
      f'cells with |z| > 2.5 in either direction: ' + '; '.join(f'{a}->{b} {z1:+.1f} / {b}->{a} {z2:+.1f}' for a, b, z1, z2 in strong[:10]))
    SYM[name] = dict(r=r, agree=agree, n=int(m.sum()))

# ---------------------------------------------------------------- (c) transitivity of the attraction graph
P('\n## (c) Transitivity: attraction graph among signs (distance >= 2 co-occurrence, die regime, MD+H) vs degree-preserving rewiring')
TD = [t for t in load_wells_die(LV) if t['site'] in BIG]
def graph_stats(edges, cls_all):
    import itertools
    adj = collections.defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)
    tri = 0; trip = 0; same = 0
    for v, nb in adj.items():
        d = len(nb)
        if d < 2: continue
        trip += d * (d - 1) / 2
        for x, y in itertools.combinations(nb, 2):
            if y in adj[x]: tri += 1
    tri //= 3; trip = trip
    cc = 3 * tri / trip if trip else float('nan')
    return cc, tri, adj
def rewire(edges, rnd, nswap=None):
    E = [tuple(e) for e in edges]; S = set(frozenset(e) for e in E); n = len(E); nswap = nswap or 10 * n
    for _ in range(nswap):
        i, j = rnd.randrange(n), rnd.randrange(n)
        if i == j: continue
        a, b = E[i]; c, d = E[j]
        if rnd.random() < 0.5: c, d = d, c
        if a == d or c == b or frozenset((a, d)) in S or frozenset((c, b)) in S: continue
        S.discard(frozenset((a, b))); S.discard(frozenset((c, d))); S.add(frozenset((a, d))); S.add(frozenset((c, b)))
        E[i] = (a, d); E[j] = (c, b)
    return E
TRANS = {}
for label, mc, ratio in (('S366 (count >= 5, O/E >= 3)', 5, 3.0), ('relaxed (count >= 3, O/E >= 2)', 3, 2.0)):
    att, both, has = attraction_pairs([t['seq'] for t in TD], all_nonadj_pairs, mc, ratio)
    edges = list(att); cc, tri, adj = graph_stats(edges, None)
    nulls = []
    for _ in range(1000):
        c2, _, _ = graph_stats(rewire(edges, rnd), None); nulls.append(c2)
    nulls = [x for x in nulls if x == x]
    P(f'   {label}: {len(edges)} edges among {len(adj)} signs, triangles {tri}, clustering {cc:.3f} vs rewired {np.mean(nulls):.3f} [{np.percentile(nulls, 2.5):.3f}, {np.percentile(nulls, 97.5):.3f}] P = {pval(cc, nulls):.3f}')
    # triangles: same-class share under each classification
    import itertools
    tris = set()
    for v, nb in adj.items():
        for x, y in itertools.combinations(nb, 2):
            if y in adj[x]: tris.add(tuple(sorted((v, x, y))))
    for name, cls in CLS.items():
        ct = [t for t in tris if all(w in cls for w in t)]
        if not ct: continue
        same3 = sum(1 for t in ct if len(set(cls[w] for w in t)) == 1); any2 = sum(1 for t in ct if len(set(cls[w] for w in t)) <= 2)
        # expectation: triangles of random classed sign triples with the same class frequencies among graph nodes
        nodes = [w for w in adj if w in cls]; labs = [cls[w] for w in nodes]
        e3 = []; e2 = []
        for _ in range(500):
            rnd.shuffle(labs); rc = dict(zip(nodes, labs))
            e3.append(sum(1 for t in ct if len(set(rc[w] for w in t)) == 1)); e2.append(sum(1 for t in ct if len(set(rc[w] for w in t)) <= 2))
        P(f'      {name}: {len(ct)} classed triangles; all three same class {same3} (expected {np.mean(e3):.1f}, P = {pval(same3, e3):.3f}); <= 2 classes {any2} (expected {np.mean(e2):.1f})')
    if label.startswith('S366'):
        P('      edges: ' + ', '.join(f'{a}-{b}' for a, b in sorted(edges, key=lambda e: -att[e][0])[:60]))
    TRANS[label] = dict(edges=len(edges), nodes=len(adj), tri=tri, cc=cc, cc_null=float(np.mean(nulls)), P=pval(cc, nulls))

json.dump(dict(level=LV, res=RES, sym=SYM, trans=TRANS), open(DARK + f'loop66_c2_{LV}.json', 'w'))
open(DARK + f'loop66_c2_{LV}.txt', 'w').write('\n'.join(out) + '\n')
P(f'done {time.time() - T0:.0f}s')
