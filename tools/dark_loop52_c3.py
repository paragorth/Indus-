"""Loop 52, cycle 3: does the grammar survive maximal merging, and do the merged classes predict anything outside?

(a) Frame check on seq_raw, canonical all, contextual-merged (k>=2) and maximal-merged (k>=1): opener initial rate,
    connective second-position rate, closer-final rates, closer mutual exclusion (two closers in one text vs shuffle),
    pairwise slot order (opener -> connective -> closer -> suffix); slot purity of every merged class (chi-square of
    position bins among members; a class that mixes an opener with a closer is a wrong merge).
(b) Outside test 1: Mahadevan's one-to-many bridge facts (one M sign <- several Wells signs; loop27 cycle 3, 45 M signs).
    Classes rebuilt WITHOUT the transcriber evidence (E1-E3 only, k>=2 and k>=1): how many lumps fall inside one class,
    vs classes of the same size structure placed at random among signs of matched frequency (1,000x).
(c) Outside test 2: the 324 IM77 texts Wells never transcribed (loop27_sets 'new'). Seen sample = IM77 texts matched to
    Wells (overlap+strict). Unseen M types and tokens in the 324 vs Good-Turing prediction from the seen sample, under
    the raw M inventory and under the merged inventory (M signs merged when their bridged Wells signs share a class).
Writes data/derived/dark/loop52_cycle3.txt.
"""
import json, collections, random, csv, itertools
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
rng = np.random.default_rng(523); random.seed(523)
corpus = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
merges = json.load(open(OUT + 'loop52_merges.json'))
levels = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']
bridge = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
props = json.load(open(OUT + 'bridge_proposals.json'))['proposals']
sets27 = json.load(open(OUT + 'loop27_sets.json'))
tok = collections.Counter(s for r in corpus for s in r['seq_raw'])
signs = sorted(tok)


class UF:
    def __init__(self): self.p = {}
    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def partition(level, k, exclude=()):
    allowed = {'raw': set(), 'strong': {'strong'}, 'all': {'strong', 'probable'}}[level]
    uf = UF()
    for s in signs: uf.f(s)
    for m in levels:
        if m['level'] in allowed: uf.u(m['form'], m['into'])
    for r in merges['pairs']:
        if r['veto']: continue
        e = {t: (v and t not in exclude) for t, v in r['ev'].items()}
        if k == 'p':
            if (e['E3'] or e['E4']) and (e['E1'] or e['E2']): uf.u(r['a'], r['b'])
        elif sum(e.values()) >= k: uf.u(r['a'], r['b'])
    cls = collections.defaultdict(set)
    for s in signs: cls[uf.f(s)].add(s)
    return {min(v, key=lambda s: -tok[s]): v for v in cls.values()}


def mapping(cls):
    return {s: h for h, v in cls.items() for s in v}


out = ['# Loop 52 cycle 3: grammar after merging, and two outside tests']

# ------------------------------------------------------------------ (a) frame check
OPENERS = {817, 861, 820, 920, 692}
CONNECT = {2, 60}
CLOSERS = {740, 520, 156, 154, 527, 226, 617, 151, 236, 700, 595}
SUFFIX = {400}


def frame_stats(texts, m):
    """texts in raw numbers, m = sign -> class head; slot sets mapped through m."""
    O = {m[s] for s in OPENERS}; K = {m[s] for s in CONNECT}; C = {m[s] for s in CLOSERS}; X = {m[s] for s in SUFFIX}
    mt = [[m[s] for s in t] for t in texts if len(t) >= 2]
    tokc = collections.Counter(s for t in mt for s in t)
    o_init = sum(1 for t in mt if t[0] in O); o_tok = sum(tokc[s] for s in O)
    k_second = sum(1 for t in mt if len(t) >= 2 and t[1] in K); k_tok = sum(tokc[s] for s in K)
    c_final = sum(1 for t in mt if t[-1] in C); c_tok = sum(tokc[s] for s in C)
    two_closers = sum(1 for t in mt if sum(1 for s in t if s in C) >= 2)
    # shuffle null for two closers: permute tokens across texts keeping lengths (20x)
    flat = [s for t in mt for s in t]; nulls = []
    for _ in range(20):
        random.shuffle(flat); i = 0; c2 = 0
        for t in mt:
            seg = flat[i:i + len(t)]; i += len(t)
            if sum(1 for s in seg if s in C) >= 2: c2 += 1
        nulls.append(c2)
    # pairwise order
    def order(A, B):
        ok = bad = 0
        for t in mt:
            ia = [i for i, s in enumerate(t) if s in A]; ib = [i for i, s in enumerate(t) if s in B]
            if ia and ib:
                if min(ia) < min(ib): ok += 1
                else: bad += 1
        return ok, bad
    return {'opener_init': (o_init, o_tok), 'conn_second': (k_second, k_tok), 'closer_final': (c_final, c_tok),
            'two_closers': (two_closers, float(np.mean(nulls))), 'O<C': order(O, C), 'O<K': order(O, K), 'K<C': order(K, C), 'C<X': order(C, X),
            'class_sizes': {'O': len(O), 'K': len(K), 'C': len(C)}}


texts_raw = [r['seq_raw'] for r in corpus]
steps = [('raw Wells', partition('raw', 99)), ('canonical all', partition('all', 99)), ('contextual-merged (principled)', partition('all', 'p')),
         ('permissive k>=2', partition('all', 2)), ('maximal-merged k>=1', partition('all', 1)), ('maximal-merged, no E4', partition('all', 1, exclude=('E4',)))]
out.append('\n## (a) frame statistics at each ladder step (texts >= 2 signs; slot sets mapped through the merge)')
out.append('| step | classes | opener initial / opener tokens | connective 2nd / tokens | closer final / tokens | texts with 2 closers vs shuffle | O<C | O<K | K<C | C<X | distinct heads O/K/C |')
for label, cls in steps:
    m = mapping(cls); fs = frame_stats(texts_raw, m)
    out.append('| %s | %d | %d / %d (%.2f) | %d / %d (%.2f) | %d / %d (%.2f) | %d vs %.0f | %d:%d | %d:%d | %d:%d | %d:%d | %d/%d/%d |' % (
        label, len(cls), *fs['opener_init'], fs['opener_init'][0] / fs['opener_init'][1], *fs['conn_second'], fs['conn_second'][0] / fs['conn_second'][1],
        *fs['closer_final'], fs['closer_final'][0] / fs['closer_final'][1], *fs['two_closers'], *fs['O<C'], *fs['O<K'], *fs['K<C'], *fs['C<X'],
        fs['class_sizes']['O'], fs['class_sizes']['K'], fs['class_sizes']['C']))

# closer paradigm per unit: final rate and jar co-occurrence for each closing sign after maximal merge
out.append('\n## (a2) closer paradigm (S289) after maximal merging: final rate and co-occurrence with the jar class (obs/exp)')
m_max = mapping(steps[4][1])
mt = [[m_max[s] for s in t] for t in texts_raw if len(t) >= 2]
tokc = collections.Counter(s for t in mt for s in t); ntexts = len(mt)
jar = m_max[740]
has = collections.defaultdict(int); fin = collections.Counter(); withjar = collections.Counter()
for t in mt:
    st = set(t)
    for s in st:
        has[s] += 1
        if jar in st and s != jar: withjar[s] += 1
    fin[t[-1]] += 1
for s in sorted(CLOSERS, key=lambda s: -tok[s]):
    h = m_max[s]
    exp = has[h] * has[jar] / ntexts
    out.append('  W%d -> class %d (%d members): final %d/%d = %.2f; with jar %d vs %.1f expected (%.2fx)' % (
        s, h, len(steps[4][1][h]), fin[h], tokc[h], fin[h] / max(1, tokc[h]), withjar[h], exp, withjar[h] / exp if exp else 0))

# slot purity of merged classes
out.append('\n## (a3) slot purity of merged classes (position bins initial / medial / final, texts >= 2; chi-square among members with >= 5 tokens)')


def posbins(sign):
    b = np.zeros(3)
    for t in texts_raw:
        if len(t) < 2: continue
        for i, s in enumerate(t):
            if s == sign:
                b[0 if i == 0 else (2 if i == len(t) - 1 else 1)] += 1
    return b


from scipy.stats import chi2_contingency
for label, cls in steps[2:5]:
    impure = []; tested = 0
    for h, v in cls.items():
        mem = [s for s in v if tok[s] >= 5]
        if len(mem) < 2: continue
        tab = np.array([posbins(s) for s in mem])
        tab = tab[:, tab.sum(0) > 0]
        if tab.shape[1] < 2: continue
        tested += 1
        chi, p, _, _ = chi2_contingency(tab)
        if p < 0.01: impure.append((h, mem, p))
    out.append('  %s: %d classes tested, %d impure (P < 0.01): %s' % (label, tested, len(impure), '; '.join('%s P=%.1e' % ('/'.join(map(str, mem)), p) for h, mem, p in impure[:25])))

# ------------------------------------------------------------------ (b) Mahadevan lumps
lumps = collections.defaultdict(set)
for w, ms in bridge.items():
    for mm in ms: lumps[mm].add(int(w))
for p in props:
    lumps[p['M']].add(p['W'])
lumps = {mm: {w for w in ws if tok[w] > 0} for mm, ws in lumps.items()}
lumps = {mm: ws for mm, ws in lumps.items() if len(ws) >= 2}
out.append('\n## (b) outside test 1: Mahadevan lumps (one M sign <- >= 2 attested Wells signs; bridge + proposals): %d M signs, %d Wells signs' % (len(lumps), sum(len(v) for v in lumps.values())))


def lump_score(m):
    inside = 0; pairs_in = 0; pairs_all = 0
    for mm, ws in lumps.items():
        heads = collections.Counter(m[w] for w in ws)
        if max(heads.values()) >= 2: inside += 1
        for a, b in itertools.combinations(ws, 2):
            pairs_all += 1
            if m[a] == m[b]: pairs_in += 1
    return inside, pairs_in, pairs_all


# frequency-stratified null: permute class labels among signs within log-frequency bins
def null_scores(cls, n=1000):
    m = mapping(cls)
    bins = collections.defaultdict(list)
    for s in signs: bins[int(np.log2(tok[s]))].append(s)
    res = []
    for _ in range(n):
        perm = {}
        for b, ss in bins.items():
            sh = ss[:]; random.shuffle(sh)
            for s, s2 in zip(ss, sh): perm[s] = m[s2]
        res.append(lump_score(perm)[:2])
    return np.array(res)


for label, cls in [('E1-E3 only, principled (E3 & (E1|E2))', partition('all', 'p', exclude=('E4',))), ('graph only (no canonical), E1-E3 principled', partition('raw', 'p', exclude=('E4',))),
                   ('E1-E3 only, k>=2', partition('all', 2, exclude=('E4',))), ('E1-E3 only, k>=1', partition('all', 1, exclude=('E4',))),
                   ('graph only (no canonical), E1-E3 k>=2', partition('raw', 2, exclude=('E4',))), ('canonical all (S268) alone', partition('all', 99)),
                   ('with E4 (circular), principled', partition('all', 'p'))]:
    m = mapping(cls); ins, pin, pall = lump_score(m)
    nl = null_scores(cls, 1000)
    out.append('  %s (%d classes): lumps inside one class %d of %d (null %.1f +/- %.1f, P %.3f); lump pairs merged %d of %d (null %.1f, P %.3f)' % (
        label, len(cls), ins, len(lumps), nl[:, 0].mean(), nl[:, 0].std(), (np.sum(nl[:, 0] >= ins) + 1) / 1001, pin, pall, nl[:, 1].mean(), (np.sum(nl[:, 1] >= pin) + 1) / 1001))
    print(out[-1], flush=True)
# which lumps are recovered / missed at k>=2 no-E4
m = mapping(partition('raw', 'p', exclude=('E4',)))
rec = [(mm, sorted(ws)) for mm, ws in lumps.items() if max(collections.Counter(m[w] for w in ws).values()) >= 2]
mis = [(mm, sorted(ws)) for mm, ws in lumps.items() if max(collections.Counter(m[w] for w in ws).values()) < 2]
out.append('  recovered: ' + '; '.join('M%d=%s' % (mm, ws) for mm, ws in sorted(rec)))
out.append('  missed: ' + '; '.join('M%d=%s' % (mm, ws) for mm, ws in sorted(mis)))

# ------------------------------------------------------------------ (c) 324 IM77-only texts
rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
texts_im = collections.defaultdict(list)
for r in rows:
    key = (r['text_no'], r['side'])
    texts_im[key].extend(int(s) for s in r['signs_clean'].split() if s != '0')
new_keys = {tuple(k) for k in sets27['new']}
seen_keys = {tuple(k) for k in sets27['overlap']} | {tuple(k) for k in sets27['strict']}
new_texts = [t for k, t in texts_im.items() if k in new_keys and t]
seen_texts = [t for k, t in texts_im.items() if k in seen_keys and t]
all_other = [t for k, t in texts_im.items() if k not in new_keys and t]
out.append('\n## (c) outside test 2: the %d IM77 texts Wells never saw (%d tokens) vs the seen IM77 sample (%d matched texts, %d tokens; also all %d non-new texts)' % (
    len(new_texts), sum(map(len, new_texts)), len(seen_texts), sum(map(len, seen_texts)), len(all_other)))

# M-level merge maps from W classes through the bridge (+ proposals)
w2m = collections.defaultdict(set)
for w, ms in bridge.items():
    for mm in ms: w2m[int(w)].add(mm)
for p in props: w2m[p['W']].add(p['M'])


def m_mapping(cls):
    uf = UF()
    for h, v in cls.items():
        ms = set()
        for w in v: ms |= w2m.get(w, set())
        ms = sorted(ms)
        for a in ms[1:]: uf.u(a, ms[0])
    return lambda mm: uf.f(mm)


def unseen_report(label, mapf):
    seen = collections.Counter(mapf(s) for t in seen_texts for s in t)
    new = [[mapf(s) for s in t] for t in new_texts]
    ntok = sum(map(len, new)); ntypes = len(set(s for t in new for s in t))
    un_tok = sum(1 for t in new for s in t if s not in seen); un_typ = len({s for t in new for s in t if s not in seen})
    N = sum(seen.values()); f1 = sum(1 for v in seen.values() if v == 1)
    gt = f1 / N
    # Good-Turing expected new types in a sample of ntok tokens, by subsampling the seen set itself as a control
    exp_types = []
    flat_texts = seen_texts
    for _ in range(200):
        idx = rng.permutation(len(flat_texts)); hold = []; cnt = 0
        for i in idx:
            if cnt >= ntok: break
            hold.append(i); cnt += len(flat_texts[i])
        hs = set(hold)
        tr = collections.Counter(mapf(s) for i, t in enumerate(flat_texts) if i not in hs for s in t)
        te = [mapf(s) for i in hold for s in flat_texts[i]]
        exp_types.append((sum(1 for s in te if s not in tr), len({s for s in te if s not in tr})))
    e = np.array(exp_types)
    out.append('  %s: seen inventory %d types; new texts %d tokens / %d types; unseen tokens %d (%.1f%%), unseen types %d; GT rate from seen %.3f -> %.1f tokens; S309 1.7%% -> %.1f tokens; hold-out of the seen sample at the same size: %.1f +/- %.1f unseen tokens, %.1f +/- %.1f types' % (
        label, len(seen), ntok, ntypes, un_tok, 100 * un_tok / ntok, un_typ, gt, gt * ntok, 0.017 * ntok, e[:, 0].mean(), e[:, 0].std(), e[:, 1].mean(), e[:, 1].std()))
    return un_tok, un_typ


unseen_report('raw M inventory', lambda s: s)
for label, cls in [('canonical all', partition('all', 99)), ('contextual-merged principled (no E4)', partition('all', 'p', exclude=('E4',))), ('permissive k>=2 (no E4)', partition('all', 2, exclude=('E4',))), ('maximal-merged k>=1 (no E4)', partition('all', 1, exclude=('E4',)))]:
    unseen_report(label, m_mapping(cls))
# which M signs are the unseen ones
seen = set(s for t in seen_texts for s in t)
un = collections.Counter(s for t in new_texts for s in t if s not in seen)
out.append('  unseen M signs in the 324 (raw): ' + ', '.join('M%d x%d' % (s, c) for s, c in un.most_common()))
out.append('  of these, bridged to a Wells sign: %d of %d types' % (sum(1 for s in un if any(s in ms for ms in w2m.values())), len(un)))

open(OUT + 'loop52_cycle3.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
