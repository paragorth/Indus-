"""Loop 4 (arrow in the dark): are the UNITS wrong? Search over segmentations and ask whether any makes the corpus
simultaneously MORE predictable out of sample and more name-like / lexicon-like than the sign level.

Segmentation families
  bpe   : merges of adjacent pairs learned on the fit set (MD + Harappa), greedy or with random tie-breaking among
          the top-T pairs, k = 5..60 merges, applied in order to fit and test sets.
  allo  : sign-identity level: seq_raw / seq_strong / seq_all, plus random subsets of the merge list (each merge on/off).
  sub   : sub-sign decomposition: marked sign -> [base, MARK:kind] (41 glyph-containment pairs, strat_shape.txt),
          ligature -> [A, B] (13 ligatures, strat_shape.txt (c)); all / marks / ligatures / random subsets.
Statistics (all on the held-out sites unless said)
  bits  : bits per ORIGINAL sign token, interpolated absolute-discounting bigram over units (fit set), unseen units
          spelled by the sign-level model (fair to merges).
  uniq  : uniqueness of distinct texts (>=2 units) / uniqueness under a bigram null over units (S321 statistic). <1 = texts
          recur more than chaining predicts (name-like in the S321 sense).
  gram  : share of original sign tokens that lie in units that are slot-pure under the S310 parser (frame units kept).
  rep   : share of texts (>=3 units) with a repeated unit / iid expectation from unit frequencies (S9/S16 ratio).
Null for arrows fired: the whole search re-run on corpora generated from the sign-level bigram model of the fit set
(same lengths, same split); the best improvement in each statistic over all segmentations is recorded per replicate.
Controls: Ur III seal legends (sign = grapheme, true word boundaries known) and a planted logo-syllabic corpus.
Usage: python3 loop4_seg.py <cycle> [fast]"""
import json, random, math, collections, sys, re, os
sys.path.insert(0, '/home/user/Indus-')
ROOT = '/home/user/Indus-/'
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '1'
FAST = 'fast' in sys.argv
OUT = open(ROOT + f'data/derived/dark/loop4_cycle{CYCLE}.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ------------------------------------------------------------------ corpus
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
AL = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
def M(w):
    v = BR.get(str(w)); return 'M' + '/'.join(str(x) for x in v) if v else '-'
FIT_SITES = {'Mohenjo-daro', 'Harappa'}
def indus(level):
    seen = set(); fit = []; test = []
    for r in C:
        s = r[level]
        if not s or r['site'] == 'Unknown': continue
        key = (r['site'], tuple(s))
        if key in seen: continue
        seen.add(key); (fit if r['site'] in FIT_SITES else test).append([int(a) for a in s])
    return fit, test

# ------------------------------------------------------------------ S310 parser (sign level, Wells numbers)
OPEN = {817, 861, 820}; MARK = {2, 60}; SUF = {400, 90}; CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
def learn_qual(fit):
    left = collections.defaultdict(collections.Counter)
    for s in fit:
        s = s[:]
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q
def parse(s, QUAL):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK: lab[1] = 'MARKER'; i = 2
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab

# ------------------------------------------------------------------ segmentations
# a text is a list of signs; a segmented text is a list of units; a unit is a tuple of atoms (atoms = signs, or sub-sign parts)
def bpe_learn(fit, k, T, rng):
    seqs = [[(a,) for a in s] for s in fit]; merges = []
    for _ in range(k):
        cnt = collections.Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]): cnt[(a, b)] += 1
        if not cnt: break
        top = cnt.most_common(T)
        pair = rng.choice(top)[0] if T > 1 else top[0][0]
        if cnt[pair] < 3: break
        merges.append(pair); seqs = [bpe_apply1(s, pair) for s in seqs]
    return merges
def bpe_apply1(s, pair):
    out = []; i = 0
    while i < len(s):
        if i + 1 < len(s) and (s[i], s[i + 1]) == pair: out.append(s[i] + s[i + 1]); i += 2
        else: out.append(s[i]); i += 1
    return out
def bpe_apply(texts, merges):
    seqs = [[(a,) for a in s] for s in texts]
    for pair in merges: seqs = [bpe_apply1(s, pair) for s in seqs]
    return seqs
def identity(texts): return [[(a,) for a in s] for s in texts]

# sub-sign decomposition from strat_shape.txt
MARKS = {}; LIGS = {}
for line in open(ROOT + 'data/derived/strat_shape.txt'):
    m = re.match(r'\s+(\d+)\(.*?\) ->\s+(\d+)\(.*?\)\s+mark=(\w+)', line)
    if m: MARKS[int(m.group(2))] = (int(m.group(1)), 'MARK_' + m.group(3))
    if 'ligatures tested:' in line:
        for a, b, c in re.findall(r'(\d+)=(\d+)\+(\d+)', line): LIGS[int(a)] = (int(b), int(c))
def sub_apply(texts, marks, ligs):
    out = []
    for s in texts:
        u = []
        for a in s:
            if a in ligs: u.append((ligs[a][0],)); u.append((ligs[a][1],))
            elif a in marks: u.append((marks[a][0],)); u.append((marks[a][1],))
            else: u.append((a,))
        out.append(u)
    return out

# ------------------------------------------------------------------ statistics
class Bigram:
    def __init__(self, seqs, D=0.75):
        self.D = D; self.big = collections.defaultdict(collections.Counter); self.uni = collections.Counter()
        for s in seqs:
            p = 'S'
            for u in list(s) + ['E']: self.big[p][u] += 1; self.uni[u] += 1; p = u
        self.N = sum(self.uni.values()); self.n1 = sum(1 for v in self.uni.values() if v == 1) or 1
        self.V = len(self.uni)
        # atom-level model for spelling unseen units
        atoms = collections.Counter(a for s in seqs for u in s for a in u)
        self.aN = sum(atoms.values()); self.an1 = sum(1 for v in atoms.values() if v == 1) or 1; self.atoms = atoms
    def p_atom(self, a):
        if a in self.atoms: return self.atoms[a] / (self.aN + self.an1)
        return self.an1 / (self.aN + self.an1) / 1000.0
    def p_uni(self, u):
        if u in self.uni: return self.uni[u] / (self.N + self.n1)
        spell = 1.0
        for a in u: spell *= self.p_atom(a)
        return self.n1 / (self.N + self.n1) * spell * (0.5 ** (len(u) - 1))  # length prior for the spelled form
    def p(self, u, prev):
        pu = self.p_uni(u); c = self.big.get(prev)
        if not c: return pu
        tot = sum(c.values()); lam = self.D * len(c) / tot
        return max(c.get(u, 0) - self.D, 0) / tot + lam * pu
    def bits(self, seqs, nsigns):
        b = 0.0
        for s in seqs:
            p = 'S'
            for u in list(s) + ['E']: b -= math.log2(self.p(u, p)); p = u
        return b / nsigns
    def sample(self, n, rng):
        if not hasattr(self, '_cum'):
            import itertools
            self._cum = {}
            for p, c in self.big.items():
                ks = [k for k in c if k != 'E'] or ['E']; ws = [c[k] for k in ks]
                self._cum[p] = (ks, list(itertools.accumulate(ws)))
        import bisect
        out = []; p = 'S'
        while len(out) < n:
            ks, cw = self._cum.get(p) or self._cum['S']
            u = ks[bisect.bisect(cw, rng.random() * cw[-1])]
            if u == 'E': p = 'S'; continue
            out.append(u); p = u
        return out

def uniq(ms):
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms)
def uniq_ratio(seqs, rng, reps=30):
    S = [tuple(s) for s in seqs if len(s) >= 2]
    if len(S) < 50: return float('nan')
    o = uniq(S); m = Bigram(S); L = [len(s) for s in S]
    nulls = sorted(uniq([tuple(m.sample(n, rng)) for n in L]) for _ in range(reps))
    return o / nulls[len(nulls) // 2]
def rep_ratio(seqs, rng, reps=30):
    S = [s for s in seqs if len(s) >= 3]
    if len(S) < 30: return float('nan')
    o = sum(1 for s in S if len(set(s)) < len(s)) / len(S)
    freq = collections.Counter(u for s in S for u in s); ks = list(freq.keys()); ws = list(freq.values())
    e = []
    for _ in range(reps):
        e.append(sum(1 for s in S if len(set(rng.choices(ks, ws, k=len(s)))) < len(s)) / len(S))
    return o / (sum(e) / len(e)) if sum(e) else float('nan')
def gram_share(seg_texts, sign_texts, QUAL):
    """share of sign tokens inside slot-pure units (unit labels from the sign-level parse). Works for merges only
    (units = tuples of signs); for sub-sign decomposition slot-purity is preserved by construction."""
    tok = ok = 0
    for u_seq, s in zip(seg_texts, sign_texts):
        lab = parse(s, QUAL); i = 0
        for u in u_seq:
            n = len(u); labs = set(lab[i:i + n]); i += n; tok += n
            if len(labs) == 1 and labs != {'NAME'}: ok += n
    return ok / tok

def evaluate(fit_seg, test_seg, nsigns_test, rng, test_signs=None, QUAL=None, gram=True):
    m = Bigram(fit_seg)
    r = {'bits': m.bits(test_seg, nsigns_test),
         'uniq': uniq_ratio(fit_seg + test_seg, rng),
         'rep': rep_ratio(fit_seg + test_seg, rng),
         'units': m.V}
    if gram and test_signs is not None: r['gram'] = gram_share(test_seg, test_signs, QUAL)
    return r

# ------------------------------------------------------------------ the search
def search(fit, test, rng, label, fams=('bpe', 'sub'), allo=None, QUAL=None, verbose=True, gram=True):
    """fit/test: sign-level texts. allo: optional (fit_raw, test_raw, merges) for the allograph family.
    Returns dict name -> stats; baseline under 'sign'."""
    nsigns = sum(len(s) for s in test)
    res = {}
    base = evaluate(identity(fit), identity(test), nsigns, rng, test, QUAL, gram); res['sign'] = base
    if verbose: P(f'  [{label}] baseline sign level: ' + fmt(base))
    if 'bpe' in fams:
        ks = (5, 10, 20, 40, 60) if not FAST else (10, 40)
        seeds = range(3 if not FAST else 1)
        for T in (1, 5):
            for k in ks:
                for sd in (seeds if T > 1 else [0]):
                    r2 = random.Random(1000 * k + sd)
                    merges = bpe_learn(fit, k, T, r2)
                    st = evaluate(bpe_apply(fit, merges), bpe_apply(test, merges), nsigns, rng, test, QUAL, gram)
                    st['merges'] = merges; res[f'bpe k={k} T={T} s={sd}'] = st
    if 'sub' in fams:
        variants = {'sub all': (MARKS, LIGS), 'sub marks': (MARKS, {}), 'sub ligs': ({}, LIGS)}
        r2 = random.Random(77)
        for i in range(6 if not FAST else 2):
            mk = {k: v for k, v in MARKS.items() if r2.random() < 0.5}; lg = {k: v for k, v in LIGS.items() if r2.random() < 0.5}
            variants[f'sub rand{i}'] = (mk, lg)
        for name, (mk, lg) in variants.items():
            fs = sub_apply(fit, mk, lg); ts = sub_apply(test, mk, lg)
            st = evaluate(fs, ts, nsigns, rng, None, None, False); st['gram'] = base.get('gram'); res[name] = st
    if allo is not None:
        fit_raw, test_raw, merges = allo
        r2 = random.Random(5)
        subsets = {'allo strong': [m for m in merges if m['level'] == 'strong'],
                   'allo strong+probable': [m for m in merges if m['level'] in ('strong', 'probable')],
                   'allo all incl doubtful': merges}
        for i in range(8 if not FAST else 2):
            subsets[f'allo rand{i}'] = [m for m in merges if r2.random() < 0.5]
        for name, ms in subsets.items():
            mp = {m['form']: m['into'] for m in ms}
            f2 = [[mp.get(a, a) for a in s] for s in fit_raw]; t2 = [[mp.get(a, a) for a in s] for s in test_raw]
            st = evaluate(identity(f2), identity(t2), nsigns, rng, t2, QUAL, gram); res[name] = st
    return res
def fmt(st):
    return ' '.join(f'{k}={st[k]:.3f}' if isinstance(st[k], float) else f'{k}={st[k]}' for k in ('bits', 'uniq', 'rep', 'gram', 'units') if k in st)
def best(res, key, lower=True):
    cand = [(v[key], k) for k, v in res.items() if k != 'sign' and isinstance(v.get(key), float) and not math.isnan(v[key])]
    if not cand: return None
    return (min if lower else max)(cand)
def report(res, label):
    b = res['sign']
    P(f'  [{label}] best per statistic (delta vs sign level):')
    for key, lower, what in (('bits', True, 'bits/sign lower=more predictable'), ('uniq', True, 'uniq ratio lower=texts recur more than bigram'),
                             ('rep', True, 'within-text repetition ratio'), ('gram', False, 'grammar-covered token share')):
        bb = best(res, key, lower)
        if bb: P(f'    {key:5s} {bb[1]:28s} {bb[0]:.3f}  (sign {b[key]:.3f}, delta {bb[0] - b[key]:+.3f})  {what}')
    joint = [(k, v) for k, v in res.items() if k != 'sign' and v['bits'] < b['bits'] and (v['uniq'] < b['uniq'] or v['rep'] < b['rep'])]
    P(f'    segmentations beating sign level on bits AND (uniq or rep): {len(joint)} of {len(res) - 1}: ' + ', '.join(k for k, _ in joint[:12]))
    return joint

if __name__ == '__main__':
    import datetime
    P(f'# loop4 cycle {CYCLE} ({datetime.datetime.now().isoformat(timespec="minutes")}) fast={FAST}')
    P('MARK pairs:', len(MARKS), 'ligatures:', len(LIGS))
    rng = random.Random(int(CYCLE) if CYCLE.isdigit() else 1)
    fit_raw, test_raw = indus('seq_raw')
    P(f'Indus: fit (MD+Harappa) {len(fit_raw)} texts, test (other named sites) {len(test_raw)} texts, '
      f'{sum(len(s) for s in test_raw)} test sign tokens')
    ALL = {}
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        fit, test = indus(level); QUAL = learn_qual(fit)
        P(f'\n## Indus level {level}')
        res = search(fit, test, rng, level, fams=('bpe', 'sub'), allo=(fit_raw, test_raw, AL['merges']) if level == 'seq_raw' else None, QUAL=QUAL)
        ALL[level] = res
        for k, v in sorted(res.items(), key=lambda kv: kv[1]['bits']):
            P(f'    {k:28s} ' + fmt(v))
        joint = report(res, level)
        # show the merges of the best-bits BPE
        bb = best({k: v for k, v in res.items() if k.startswith('bpe')}, 'bits')
        if bb:
            ms = res[bb[1]]['merges'][:12]
            P('    first merges of best BPE: ' + '; '.join('+'.join(str(a) for u in pr for a in u) + ' (' + '+'.join(M(a) for u in pr for a in u) + ')' for pr in ms))
    json.dump({lvl: {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in res.items()} for lvl, res in ALL.items()},
              open(ROOT + f'data/derived/dark/loop4_cycle{CYCLE}_indus.json', 'w'))
