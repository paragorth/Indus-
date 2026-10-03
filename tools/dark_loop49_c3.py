"""S-DARK-49 cycle 3: CYCLE STRUCTURE. If a candidate set of elements were a calendar (months) or a serial counter, then
(a) inside one text two members should be mutually exclusive (one date per text), (b) across the faces of one multi-sided object neighbouring
members (i, i+1; cyclically for a calendar) should pair above chance (a tablet series written month after month, a counter advancing), and
(c) where two members do co-occur in one text their relative order should be fixed (i before i+1) rather than free.
Candidate sets: stroke numerals by value (S204 values; W1/W2/W31 kept as separate elements), fish words in the S296 order, the closer paradigm
(S289), the openers, the 705/706/255/435/690 title set, the PO3 350-798-415 chain, the marked jar / leaf family, and (control) the
10 commonest NAME-slot signs. Orderings: natural value order for numerals; S296 order for fish; for the rest frequency rank and the
S-DARK-19 chain depth are both tried (a cyclic adjacency would show under the true order, so the best of the two orderings is also reported
against a null that takes the best of two random orderings).
Nulls: (a) sign identities permuted across texts within site x object type (500x) for the within-text co-occurrence O/E;
(b) faces reassigned among objects of the same site x type (500x; S-DARK-37 null C) for cross-face pairs; (c) binomial for order.
Objects/faces: data/raw/inscriptions.csv via tools/dark_loop37.load_faces (three merge levels), identical moulded face-sets collapsed.
Usage: python3 tools/dark_loop49_c3.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import sys, json, math, collections, random
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import load_faces, multi, collapse, OPEN, CL, FISH, NUM, learn_qual, make_parser, pval
ROOT = '/home/user/Indus-/'
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rnd = random.Random(493)
OUT = open(ROOT + f'data/derived/dark/loop49_c3_{LV}.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n'); OUT.flush()

objs = load_faces(LV)
faces = [(o['site'], o['ot'], f['seq']) for o in objs.values() for f in o['faces'] if f['seq']]
texts = [f[2] for f in faces]
parse = make_parser(learn_qual(texts))
tokfreq = collections.Counter(w for s in texts for w in s)
name_cnt = collections.Counter(w for s in texts for w, l in zip(s, parse(s)) if l == 'NAME')
# S-DARK-19 longest chain (loop19_cycle2): chain depth as an ordering
CHAIN = [320, 920, 60, 741, 2, 803, 32, 350, 798, 415, 220, 233, 705, 33, 740, 400]
VAL = {1: 1, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8, 31: 1, 32: 2, 33: 3, 34: 4, 55: 12, 56: 24, 2: 2}
SETS = collections.OrderedDict([
    ('numerals by value (short 1,3-8 & tall 1-4; 12, 24)', ({1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}, lambda w: (VAL[w], w))),
    ('short-stroke numerals 3-8 only', ({3, 4, 5, 16, 17, 18}, lambda w: VAL[w])),
    ('tall-stroke numerals 1-4 only', ({31, 32, 33, 34}, lambda w: VAL[w])),
    ('fish words (S296 order hat>whiskers>bar>stroke>plain)', (set(FISH), lambda w: [235, 240, 233, 231, 220].index(w))),
    ('closer paradigm', (set(CL), None)),
    ('openers', (set(OPEN), None)),
    ('title set 705/706/255/435/690', ({705, 706, 255, 435, 690}, None)),
    ('PO3 350-798-415', ({350, 798, 415}, None)),
    ('marked jar / leaf family 741/742/745/803/806', ({741, 742, 745, 803, 806}, None)),
    ('CONTROL: 10 commonest NAME-slot signs', (set(w for w, n in name_cnt.most_common(10)), None)),
])
say(f'# S-DARK-49 cycle 3, level {LV}, nperm {NPERM}: cycle structure. Faces with signs {len(texts)}; objects {len(objs)}')

# ---------------------------------------------------------------- (a) within-text co-occurrence and order
def within(S):
    n2 = 0; pairs = collections.Counter(); order = collections.Counter(); adj = 0; npair = 0
    for s in texts:
        mem = [(i, w) for i, w in enumerate(s) if w in S]
        d = set(w for _, w in mem)
        if len(d) >= 2: n2 += 1
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                (i, wa), (j, wb) = mem[a], mem[b]
                if wa == wb: continue
                pairs[frozenset((wa, wb))] += 1; order[(wa, wb)] += 1; npair += 1
                if j == i + 1: adj += 1
    return n2, pairs, order, adj, npair
def null_within(S):
    """sign identities permuted across texts within site x type, text lengths kept"""
    groups = collections.defaultdict(list)
    for k, (site, ot, s) in enumerate(faces): groups[(site, ot)].append(k)
    out = []
    for it in range(NPERM):
        newtexts = [None] * len(texts)
        for g in groups.values():
            pool = [w for k in g for w in texts[k]]; rnd.shuffle(pool); p = 0
            for k in g:
                L = len(texts[k]); newtexts[k] = pool[p:p + L]; p += L
        n2 = 0
        for s in newtexts:
            d = set(w for w in s if w in S)
            if len(d) >= 2: n2 += 1
        out.append(n2)
    return out
# ---------------------------------------------------------------- (b) cross-face neighbour pairs
M = multi(objs, 1); Mc, dropped = collapse(M)
say(f'multi-sided objects: {len(M)}, after collapsing identical moulded face-sets {len(Mc)} (dropped {dropped})')
def cross_face_pairs(objlist, S):
    """for each object: set of members per face; cross-face pairs of DISTINCT members (unordered), one per object pair of faces"""
    P = []
    for o in objlist:
        fm = [set(w for w in f['seq'] if w in S) for f in o['faces']]
        for a in range(len(fm)):
            for b in range(a + 1, len(fm)):
                for wa in fm[a]:
                    for wb in fm[b]:
                        if wa != wb: P.append((wa, wb))
    return P
def dist1_share(P, rank, K, cyclic):
    if not P: return None
    d1 = 0
    for wa, wb in P:
        d = abs(rank[wa] - rank[wb])
        if cyclic: d = min(d, K - d)
        if d == 1: d1 += 1
    return d1 / len(P)
def null_faces(objlist, S, rank, K, cyclic, best_of_random=False):
    """faces reassigned among objects of the same site x type (S-DARK-37 null C); returns null shares"""
    groups = collections.defaultdict(list)
    for o in objlist: groups[(o['site'], o['ot'])].append(o)
    out = []; outn = []
    elems = sorted(S)
    for it in range(NPERM):
        fake = []
        for g in groups.values():
            pool = [f for o in g for f in o['faces']]; rnd.shuffle(pool); p = 0
            for o in g:
                k = len(o['faces']); fake.append(dict(faces=pool[p:p + k])); p += k
        P = cross_face_pairs(fake, S)
        if best_of_random:
            best = -1
            for r in range(2):
                perm = elems[:]; rnd.shuffle(perm); rk = {w: i for i, w in enumerate(perm)}
                sh = dist1_share(P, rk, K, cyclic); best = max(best, sh if sh is not None else -1)
            out.append(best)
        else:
            sh = dist1_share(P, rank, K, cyclic); out.append(sh if sh is not None else 0.0)
        outn.append(len(P))
    return out, outn

for name, (S, keyf) in SETS.items():
    S = set(w for w in S if tokfreq[w] > 0)
    K = len(S)
    say(f'\n== {name}: K={K}, tokens {sum(tokfreq[w] for w in S)}; members {sorted(S)}')
    n2, pairs, order, adj, npair = within(S)
    ntexts_with = sum(1 for s in texts if any(w in S for w in s))
    nl = null_within(S); E = sum(nl) / len(nl)
    say(f'  (a) texts with >= 2 distinct members: {n2} of {ntexts_with} carrying the set; null (signs permuted within site x type) {E:.1f} '
        f'[{sorted(nl)[int(0.025*NPERM)]}-{sorted(nl)[int(0.975*NPERM)-1]}], O/E {n2/E if E else float("nan"):.2f}, P_lo={pval(n2, nl, "lo"):.3f} P_hi={pval(n2, nl):.3f}'
        + (f'; upper bound < {3/ntexts_with:.3f}' if n2 == 0 else ''))
    # order fixity among co-occurring pairs with >= 5 instances
    fixed = free = 0; lines = []
    for pr, n in pairs.items():
        if n < 5: continue
        a, b = sorted(pr); ab, ba = order[(a, b)], order[(b, a)]
        maj = max(ab, ba); p = sum(math.comb(n, k) for k in range(maj, n + 1)) / 2 ** n * 2
        (lines.append(f'{a}>{b} {ab}:{ba}'))
        if p < 0.05: fixed += 1
        else: free += 1
    say(f'      co-occurring pairs with >= 5 instances: {fixed + free}; fixed order {fixed}, free {free}; adjacent in {adj}/{npair} co-occurrences. ' + ', '.join(lines[:14]))
    # (b) cross-face
    orderings = []
    if keyf is not None: orderings.append(('natural', {w: i for i, w in enumerate(sorted(S, key=keyf))}))
    orderings.append(('frequency rank', {w: i for i, w in enumerate(sorted(S, key=lambda w: -tokfreq[w]))}))
    orderings.append(('S-DARK-19 chain depth', {w: i for i, w in enumerate(sorted(S, key=lambda w: (CHAIN.index(w) if w in CHAIN else 99, -tokfreq[w])))}))
    P = cross_face_pairs(Mc, S)
    nobj = sum(1 for o in Mc if sum(1 for f in o['faces'] if any(w in S for w in f['seq'])) >= 2)
    say(f'  (b) multi-sided objects with the set on >= 2 faces: {nobj}; cross-face distinct-member pairs: {len(P)}' + (f'; upper bound < {3/len(Mc):.4f} per object' if nobj == 0 else ''))
    if P:
        for oname, rank in orderings:
            for cyclic in (False, True):
                sh = dist1_share(P, rank, K, cyclic); nl, nln = null_faces(Mc, S, rank, K, cyclic)
                say(f'      ordering {oname:22s} {"cyclic" if cyclic else "linear"} neighbour (|i-j|=1) share {sh:.2f} vs faces-reassigned null {sum(nl)/len(nl):.2f} '
                    f'[{sorted(nl)[int(0.025*NPERM)]:.2f}-{sorted(nl)[int(0.975*NPERM)-1]:.2f}] (null pairs {sum(nln)/len(nln):.0f}) P_hi={pval(sh, nl):.3f}')
        if keyf is None:
            best = max(dist1_share(P, r, K, True) for _, r in orderings)
            nl, _ = null_faces(Mc, S, None, K, True, best_of_random=True)
            say(f'      best of the orderings tried (cyclic) {best:.2f} vs best-of-2-random-orderings null {sum(nl)/len(nl):.2f} P_hi={pval(best, nl):.3f}')
        cc = collections.Counter(tuple(sorted(p)) for p in P)
        say(f'      commonest cross-face pairs: {cc.most_common(8)}')
        # serial-counter signature for numerals: value on face 2 = value on face 1 + 1
        if keyf is not None and 'numeral' in name:
            plus1 = sum(1 for wa, wb in P if VAL[wb] - VAL[wa] == 1 or VAL[wa] - VAL[wb] == 1)
            same = sum(1 for o in Mc for a in range(len(o['faces'])) for b in range(a + 1, len(o['faces']))
                       for wa in set(o['faces'][a]['seq']) & S for wb in set(o['faces'][b]['seq']) & S if wa == wb)
            say(f'      numeral faces: same value on two faces {same}, values differing by exactly 1: {plus1} of {len(P)} differing pairs')
