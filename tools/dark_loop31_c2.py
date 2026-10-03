"""S-DARK-31 cycle 2: what do the recurring offsets encode?
(a) Known relations as offsets: for each documented relation kind (numeral step, plain -> marked fish, marked/unmarked jar,
    seal/tablet allograph, closer -> its qualifier, opener <-> opener, closer <-> closer, suffix pair), take its pairs within the
    150-sign vocabulary and measure the mean pairwise cosine of their offset vectors (E_a - E_b). Null: the same number of pairs
    drawn at random from the vocabulary with the same frequency bins (2,000x). If a relation is a 'direction' in the space, the
    cosine is high.
(b) Offset classes of cycle 1 (rule 4, real seq_raw): how many of their member pairs are known relations, vs the share of known
    pairs among all vocabulary pairs; classes with >= 2 known pairs of one kind are 'interpretable'; their other members are
    NEW relation candidates.
(c) New candidates tested outside the fitting corpus: held-out Wells sites (everything but Mohenjo-daro + Harappa) and IM77 (all
    sites, via bridge_extended + loop 27 proposals): context-profile cosine (left and right neighbour distributions) of c and d
    vs 500 frequency-matched random pairs, and adjacency count c~d vs expectation.
Runs on seq_raw / seq_strong / seq_all (python3 tools/dark_loop31_c2.py <level>).
"""
import json, sys, random, collections, math, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop31 as D
HERE = D.HERE; OUT = D.OUT
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
rnd = random.Random(312)
fo = open(OUT + f'loop31_c2_{LV}.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); fo.write(s + '\n'); fo.flush()

objs = D.load_indus(LV)
vocab, E = D.embed(objs)
VI = {s: i for i, s in enumerate(vocab)}
freq = collections.Counter(x for o in objs for x in o['seq'])
log(f'== S-DARK-31 cycle 2, level {LV}: {len(objs)} MD+H texts, vocab {len(vocab)} (min tokens {min(freq[v] for v in vocab)})')

# ---------------- known relations ----------------
lev = json.load(open(os.path.join(HERE, 'data/derived/sign_allographs_levels.json')))
allo = [(m['form'], m['into']) for m in lev['merges'] if m['level'] in ('strong', 'probable')]
KNOWN = {
    'numeral step (short strokes 1->3->4->5)': [(1, 3), (3, 4), (4, 5), (16, 17), (17, 18)],
    'numeral step (tall strokes 1->2->3->4)': [(31, 32), (32, 33), (33, 34)],
    'short <-> tall numeral of equal value': [(1, 31), (3, 33), (4, 34), (16, 32), (17, 33), (18, 34)],
    'plain fish -> marked fish': [(220, 233), (220, 240), (220, 235), (220, 231), (220, 226)],
    'marked fish <-> marked fish': [(233, 240), (233, 235), (233, 231), (240, 235), (240, 231), (235, 231)],
    'jar -> marked jar': [(740, 741), (740, 742), (740, 745)],
    'seal/tablet or medium allograph (S262, S268)': [(390, 405), (154, 158), (817, 861), (817, 820), (861, 820), (803, 806), (705, 706)] + allo,
    'closer -> its qualifier (S289/S303)': [(154, 806), (158, 806), (527, 550), (527, 555), (520, 220), (520, 33), (520, 240), (156, 3), (226, 32), (617, 142), (740, 760), (740, 100), (740, 176), (700, 33), (700, 34), (700, 32)],
    'opener <-> opener': [(817, 861), (817, 820), (817, 920), (817, 692), (861, 820), (861, 920), (861, 692), (820, 920), (820, 692), (920, 692)],
    'closer <-> closer (paradigm)': [(a, b) for i, a in enumerate(D.CL) for b in D.CL[i + 1:]],
    'suffix pair 400/90': [(400, 90)],
    'pre-jar title <-> title': [(176, 100), (176, 760), (100, 760), (176, 923), (760, 923), (690, 760), (482, 176)],
    'connective pair 2/60': [(2, 60)],
}
def dedup(prs):
    seen = set(); out = []
    for a, b in prs:
        if a in VI and b in VI and a != b and (a, b) not in seen and (b, a) not in seen:
            seen.add((a, b)); out.append((a, b))
    return out
KNOWN = {k: dedup(v) for k, v in KNOWN.items()}
known_pairs = {}
for k, prs in KNOWN.items():
    for a, b in prs: known_pairs[frozenset((a, b))] = k

fbin = {s: int(math.log2(freq[s])) for s in vocab}
by_bin = collections.defaultdict(list)
for s in vocab: by_bin[fbin[s]].append(s)
def matched_pair(a, b):
    """random pair with the same frequency bins as (a, b)"""
    for _ in range(50):
        x = rnd.choice(by_bin[fbin[a]]); y = rnd.choice(by_bin[fbin[b]])
        if x != y: return x, y
    return rnd.sample(vocab, 2)

def offcos(prs):
    O = np.array([E[VI[a]] - E[VI[b]] for a, b in prs]); On = O / np.linalg.norm(O, axis=1, keepdims=True)
    cs = On @ On.T; m = len(prs)
    return float((cs.sum() - m) / (m * (m - 1)))
def offcos_abs(prs):
    """direction-free: |cos|, so that a reversed pair counts as the same offset"""
    O = np.array([E[VI[a]] - E[VI[b]] for a, b in prs]); On = O / np.linalg.norm(O, axis=1, keepdims=True)
    cs = np.abs(On @ On.T); m = len(prs)
    return float((cs.sum() - m) / (m * (m - 1)))

log('\n-- (a) known relations as offset directions: mean pairwise offset cosine of the relation pairs vs 2,000 frequency-matched random pair sets')
rowsA = {}
for k, prs in KNOWN.items():
    if len(prs) < 2:
        log(f'  {k}: {len(prs)} pair(s) in vocab, not testable'); continue
    real = offcos(prs); real_abs = offcos_abs(prs)
    null = []; null_abs = []
    for _ in range(2000):
        rp = [matched_pair(a, b) for a, b in prs]
        null.append(offcos(rp)); null_abs.append(offcos_abs(rp))
    P = (sum(1 for x in null if x >= real) + 1) / 2001; Pa = (sum(1 for x in null_abs if x >= real_abs) + 1) / 2001
    sim = float(np.mean([E[VI[a]] @ E[VI[b]] for a, b in prs]))
    rowsA[k] = dict(n=len(prs), offcos=real, null_med=float(np.median(null)), P=P, offcos_abs=real_abs, P_abs=Pa, mean_sim=sim)
    log(f'  {k}: n = {len(prs)}, offset cos {real:.2f} (null median {np.median(null):.2f}, 95% {np.percentile(null,95):.2f}, P = {P:.4f}); '
        f'|cos| {real_abs:.2f} (P = {Pa:.4f}); mean member similarity cos(a,b) {sim:.2f}')
    # per pair: nearest offset neighbour among the other known pairs of the same kind
    if len(prs) >= 3:
        O = np.array([E[VI[a]] - E[VI[b]] for a, b in prs]); On = O / np.linalg.norm(O, axis=1, keepdims=True); cs = On @ On.T
        np.fill_diagonal(cs, -2)
        det = ', '.join(f'{D.fmt_sign(a)}-{D.fmt_sign(b)} best {cs[i].max():.2f}' for i, (a, b) in enumerate(prs))
        log(f'     per pair best within-kind offset cos: {det}')

# ---------------- (b) offset classes from cycle 1 vs known relations ----------------
log('\n-- (b) cycle-1 offset classes (rule 4) scored against the known-relation list')
NN = D.nn_tensor(E)
res = {}
for rule in (4, 2):
    links, _, _ = D.mine(E, rule, NN=NN)
    st = D.offset_stats(E, links)
    allpairs = len(vocab) * (len(vocab) - 1) / 2
    base = len(known_pairs) / allpairs
    nk = 0; ntot = 0; interp = []
    for c in st['classes']:
        kinds = collections.Counter()
        for a, b in c['pairs']:
            ntot += 1
            k = known_pairs.get(frozenset((vocab[a], vocab[b])))
            if k: nk += 1; kinds[k] += 1
        if kinds and max(kinds.values()) >= 2: interp.append((c, kinds))
    # null share: random pairs from the vocabulary
    log(f'  rule {rule}: {st["n_classes"]} classes, {ntot} member pairs, {nk} are known relations ({nk/max(ntot,1):.3f}) vs {base:.3f} of all vocabulary pairs '
        f'(expected {base*ntot:.1f}); classes with >= 2 known pairs of one kind: {len(interp)}')
    # also: known pairs that appear in ANY link at all
    linked_known = collections.Counter()
    for p, q in links:
        for a, b in (p, q):
            k = known_pairs.get(frozenset((vocab[a], vocab[b])))
            if k: linked_known[k] += 1
    log(f'  rule {rule}: known pairs taking part in any link, by kind: {dict(linked_known)}')
    for c, kinds in interp[:25]:
        prs = ', '.join(f'{D.fmt_sign(vocab[a])}:{D.fmt_sign(vocab[b])}' + (f' [{known_pairs[frozenset((vocab[a],vocab[b]))][:18]}]' if frozenset((vocab[a], vocab[b])) in known_pairs else '') for a, b in c['pairs'][:14])
        log(f'    class ({c["n_pairs"]} pairs, {c["disjoint"]} disjoint, cos {c["meancos"]:.2f}) kinds {dict(kinds)}: {prs}')
    res[f'rule{rule}'] = dict(n_classes=st['n_classes'], member_pairs=ntot, known_members=nk, base_share=base, interpretable=len(interp),
                              interp=[dict(pairs=[[vocab[a], vocab[b]] for a, b in c['pairs']], kinds=dict(kinds), meancos=c['meancos'], disjoint=c['disjoint']) for c, kinds in interp])

# ---------------- (c) new candidates on held-out sites and IM77 ----------------
log('\n-- (c) new relation candidates. Blind classes carry no known relation, so the probes are the relation directions that passed (a):')
log('   for each passing kind, o = mean unit offset of its pairs; for every other vocabulary sign c, d = NN(E_c + o) if cos >= 0.6 and d != NN1(c);')
log('   test: context-profile cosine of c and d (and, for closer -> qualifier, the count of d directly before c) on held-out Wells sites and IM77 vs frequency-matched random pairs')
probes = {}
for k, prs in KNOWN.items():
    r = rowsA.get(k)
    if not r or r['P'] >= 0.01 or k.startswith('closer <-> closer') or k.startswith('opener <-> opener'): continue
    O = np.array([E[VI[a]] - E[VI[b]] for a, b in prs]); On = O / np.linalg.norm(O, axis=1, keepdims=True)
    probes[k] = (On.mean(0) * np.linalg.norm(O, axis=1).mean(), {x for a, b in prs for x in (a, b)})
cs_all = E @ E.T; nn1 = (cs_all - 2 * np.eye(len(vocab), dtype=np.float32)).argmax(1)
cands = []
for k, (o, used) in probes.items():
    for ci, c in enumerate(vocab):
        if c in used: continue
        q = E[ci] + o; q = q / np.linalg.norm(q)
        sc = E @ q; sc[ci] = -2
        di = int(sc.argmax())
        if sc[di] < 0.6 or di == nn1[ci]: continue
        # the pair order follows the relation: a - b = o means a = b + o, so c + o ~ d plays 'a' and c plays 'b'
        cands.append((vocab[di], c, k, float(sc[di])))
log(f'  {len(cands)} candidate pairs (predicted a, given b) from {len(probes)} probe directions: ' + ', '.join(f"{k[:24]}: {sum(1 for x in cands if x[2]==k)}" for k in probes))
for a, b, k, sc in sorted(cands, key=lambda x: -x[3])[:40]:
    known = known_pairs.get(frozenset((a, b)), '')
    log(f'     {D.fmt_sign(a)} <- {D.fmt_sign(b)} + [{k[:28]}] cos {sc:.2f} {("KNOWN: "+known) if known else ""}')

def ctx_profiles(texts):
    L = collections.defaultdict(collections.Counter); R = collections.defaultdict(collections.Counter); f = collections.Counter(); adj = collections.Counter(); before = collections.Counter()
    for s in texts:
        for i, x in enumerate(s):
            f[x] += 1
            if i > 0: L[x][s[i - 1]] += 1; adj[frozenset((x, s[i - 1]))] += 1; before[(s[i - 1], x)] += 1
            if i < len(s) - 1: R[x][s[i + 1]] += 1
    return L, R, f, adj, before
def cos_c(c1, c2):
    keys = set(c1) | set(c2)
    if not keys: return 0.0
    v1 = np.array([c1.get(k, 0) for k in keys], float); v2 = np.array([c2.get(k, 0) for k in keys], float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / n) if n else 0.0
def ctxsim(L, R, a, b):
    return 0.5 * (cos_c(L[a], L[b]) + cos_c(R[a], R[b]))

def test_candidates(texts, cands_local, label, nrand=500):
    L, R, f, adj, before = ctx_profiles(texts)
    signs = [s for s, n in f.items() if n >= 3]
    fb = {s: int(math.log2(f[s])) for s in signs}; bb = collections.defaultdict(list)
    for s in signs: bb[fb[s]].append(s)
    hits = 0; tested = 0; details = []; adj_hits = 0; adj_tested = 0
    for a, b, kind, sc in cands_local:
        if f.get(a, 0) < 3 or f.get(b, 0) < 3: continue
        tested += 1
        real = ctxsim(L, R, a, b)
        null = []; nulladj = []
        for _ in range(nrand):
            x = rnd.choice(bb[fb[a]]); y = rnd.choice(bb[fb[b]])
            if x == y: continue
            null.append(ctxsim(L, R, x, y)); nulladj.append(before[(y, x)])
        P = (sum(1 for v in null if v >= real) + 1) / (len(null) + 1)
        if P < 0.05: hits += 1
        Padj = None
        if kind.startswith('closer'):
            # relation a = closer, b = qualifier: b should stand directly before a
            adj_tested += 1
            Padj = (sum(1 for v in nulladj if v >= before[(b, a)]) + 1) / (len(nulladj) + 1)
            if Padj < 0.05 and before[(b, a)] >= 2: adj_hits += 1
        details.append((a, b, kind, f[a], f[b], real, float(np.median(null)), P, before[(b, a)], Padj))
    log(f'  {label}: {tested} candidates testable (both signs >= 3 tokens), {hits} with context similarity above 95% of frequency-matched pairs (expected {0.05*tested:.1f})'
        + (f'; closer -> qualifier adjacency (qualifier directly before closer, >= 2x and P < 0.05): {adj_hits} of {adj_tested} (expected {0.05*adj_tested:.1f})' if adj_tested else ''))
    for a, b, kind, fa, fb_, real, nm, P, bef, Padj in sorted(details, key=lambda d: d[7])[:14]:
        log(f'     {a} <- {b} [{kind[:26]}] n {fa}/{fb_} ctx cos {real:.2f} vs null {nm:.2f} P {P:.3f}; b-before-a {bef}x' + (f' P {Padj:.3f}' if Padj is not None else ''))
    return tested, hits, adj_tested, adj_hits, details

C_all = D.C
held = [r[LV] for r in C_all if r[LV] and len(r[LV]) >= 2 and r['site'] not in ('Mohenjo-daro', 'Harappa') and r['complete'] == 'Y']
log(f'  held-out Wells texts (not MD/H, complete, >= 2 signs): {len(held)}')
r1 = test_candidates(held, cands, 'held-out Wells sites')

# IM77 via bridge + proposals
BR = D.BR; prop = json.load(open(os.path.join(HERE, 'data/derived/dark/bridge_proposals.json')))
W2M = {int(w): list(m) for w, m in BR.items()}
for p in prop['proposals']:
    W2M.setdefault(p['W'], [])
    if p['M'] not in W2M[p['W']]: W2M[p['W']].append(p['M'])
for p in prop.get('corrections', []): W2M[p['W']] = list(p['new'])
for p in prop.get('contextual_not_alignment_validated', []): W2M.setdefault(p['W'], [p['M']])
im = D.load_im77()
im_texts = [o['seq'] for o in im]
cands_m = []
for a, b, kind, sc in cands:
    if a in W2M and b in W2M:
        ma, mb = W2M[a][0], W2M[b][0]
        if ma != mb: cands_m.append((ma, mb, kind, sc))
log(f'  IM77 texts {len(im_texts)}; candidates bridged to Mahadevan numbers: {len(cands_m)} of {len(cands)}')
r2 = test_candidates(im_texts, cands_m, 'IM77 (all sites, bridged)')
# control: random vocabulary pairs with the same kind labels, same procedure
ctrl = []
for a, b, kind, sc in cands:
    x, y = rnd.sample(vocab, 2); ctrl.append((x, y, kind, 0.0))
r3 = test_candidates(held, ctrl, 'CONTROL random vocabulary pairs, held-out sites')
ctrl_m = [(W2M[a][0], W2M[b][0], kind, 0.0) for a, b, kind, _ in ctrl if a in W2M and b in W2M and W2M[a][0] != W2M[b][0]]
r4 = test_candidates(im_texts, ctrl_m, 'CONTROL random vocabulary pairs, IM77')
# the known pairs themselves on held-out / IM77, for calibration of the test's power
kn = [(a, b, k, 1.0) for k, prs in KNOWN.items() for a, b in prs if k in probes]
r5 = test_candidates(held, kn, 'CALIBRATION known relation pairs, held-out sites')
kn_m = [(W2M[a][0], W2M[b][0], k, 1.0) for a, b, k, _ in kn if a in W2M and b in W2M and W2M[a][0] != W2M[b][0]]
r6 = test_candidates(im_texts, kn_m, 'CALIBRATION known relation pairs, IM77')
res['candidates'] = dict(n=len(cands), list=[list(c) for c in cands], heldout=r1[:4], im77=r2[:4], control_heldout=r3[:4], control_im77=r4[:4], known_heldout=r5[:4], known_im77=r6[:4])
res['known_offsets'] = rowsA
json.dump(res, open(OUT + f'loop31_c2_{LV}.json', 'w'), default=str)
log('done')
