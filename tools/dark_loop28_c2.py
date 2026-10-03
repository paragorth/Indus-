"""S-DARK-28 cycle 2: is a graphically derived sign functionally a transform of its base?

Input: data/derived/dark/loop28_edges.json (cycle 1).  For every edge (base b -> derived d)
with enough tokens, profiles of b and d are compared: position in the text (first / second /
middle / penultimate / last / alone), frame label (S310 parser), object type, site, left and
right neighbours, numeral-before rate.  Distance = Jensen-Shannon divergence.
Null: 1,000 draws of a random sign r with frequency within x2 of b (not graphically linked
to d); the percentile of JSD(b,d) among JSD(r,d) says whether d sits closer to its base
than to a frequency-matched stranger.  Per derivation type: mean percentile, mean shift
vs the null mean (1,000 permutations), directional shifts (relative position, last-slot
share, tablet share, numeral-before rate), co-occurrence of b and d in one text vs
expectation, and minimal pairs (texts identical except b <-> d).
Train = Mohenjo-daro + Harappa; confirm = held-out sites; IM77 via the bridge.
usage: python3 tools/dark_loop28_c2.py seq_raw|seq_strong|seq_all
"""
import sys, json, random, collections, csv, time
import numpy as np
random.seed(28); np.random.seed(28)
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = 1000
OUT = 'data/derived/dark/'
EDGES = json.load(open(OUT + 'loop28_edges_all.json'))
M = json.load(open('data/derived/merged-corpus-canonical.json'))
BR = {int(k): v for k, v in json.load(open('data/derived/bridge_extended.json')).items()}
# S-DARK-24.1 corrections
BR[798] = [53]; BR[806] = [389]
NUM = {1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 55, 56}
NUM_M = set(range(86, 122))

def otype(t):
    t = t.split(':')[0]
    return 'SEAL' if t == 'SEAL' else ('TAB' if t == 'TAB' else ('POT' if t == 'POT' else 'OTHER'))

TEXTS = []
for r in M:
    s = [x for x in (r.get(LV) or []) if x != 999 and x != 0]
    if not s or r['complete'] != 'Y': continue
    TEXTS.append(dict(seq=s, site=r['site'], ot=otype(r['type']), big=r['site'] in ('Mohenjo-daro', 'Harappa')))
# ---------------- frame parser (as in dark_loop26 / parse_all) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}; CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}
left = collections.defaultdict(collections.Counter)
for o in TEXTS:
    s = o['seq'][:]
    while len(s) > 1 and s[-1] in SUF: s.pop()
    if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
QUAL = {}
for c, cnt in left.items():
    tot = sum(cnt.values()); acc = 0; q = set()
    for a, n in cnt.most_common():
        if acc / tot >= 0.6: break
        q.add(a); acc += n
    QUAL[c] = q
def parse(s):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
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
for o in TEXTS: o['lab'] = parse(o['seq'])
LABS = ['OPENER', 'MARKER', 'TITLE', 'CLOSER', 'SUFFIX', 'NAME', 'COUNT']
POS = ['alone', 'first', 'second', 'middle', 'penult', 'last']
OTS = ['SEAL', 'TAB', 'POT', 'OTHER']

def posbin(i, n):
    if n == 1: return 'alone'
    if i == 0: return 'first'
    if i == n - 1: return 'last'
    if i == 1: return 'second'
    if i == n - 2: return 'penult'
    return 'middle'

class Corpus:
    """token-level profiles for every sign"""
    def __init__(self, texts, numset, topk=60, labels=True):
        self.texts = texts; self.numset = numset
        self.freq = collections.Counter(x for t in texts for x in t['seq'])
        self.top = [w for w, _ in self.freq.most_common(topk)]
        self.ti = {w: i for i, w in enumerate(self.top)}
        self.sites = sorted(set(t['site'] for t in texts))
        self.si = {s: i for i, s in enumerate(self.sites)}
        self.prof = {}
        P = collections.defaultdict(lambda: dict(pos=np.zeros(6), lab=np.zeros(7), ot=np.zeros(4), site=np.zeros(len(self.sites)),
                                                 L=np.zeros(topk + 2), R=np.zeros(topk + 2), numbefore=0.0, relpos=[], n=0, texts=set()))
        for ti, t in enumerate(texts):
            s = t['seq']; n = len(s)
            for i, x in enumerate(s):
                p = P[x]; p['n'] += 1; p['texts'].add(ti)
                p['pos'][POS.index(posbin(i, n))] += 1
                if labels: p['lab'][LABS.index(t['lab'][i])] += 1
                p['ot'][OTS.index(t['ot'])] += 1
                p['site'][self.si[t['site']]] += 1
                l = s[i - 1] if i > 0 else None; r = s[i + 1] if i < n - 1 else None
                p['L'][self.ti.get(l, topk) if l is not None else topk + 1] += 1
                p['R'][self.ti.get(r, topk) if r is not None else topk + 1] += 1
                if l is not None and l in numset: p['numbefore'] += 1
                if n > 1: p['relpos'].append(i / (n - 1))
        for x, p in P.items():
            p['numbefore'] /= p['n']; p['relpos'] = float(np.mean(p['relpos'])) if p['relpos'] else np.nan
            p['last'] = p['pos'][5] / p['n']; p['tab'] = p['ot'][1] / p['n']
        self.P = P
    def cooc(self, a, b):
        return len(self.P[a]['texts'] & self.P[b]['texts'])
    def minimal_pairs(self, a, b):
        """distinct texts (>= 2 signs) that are identical except a <-> b at one position"""
        A = set(tuple(self.texts[i]['seq']) for i in self.P[a]['texts']); B = set(tuple(self.texts[i]['seq']) for i in self.P[b]['texts'])
        k = 0
        for s in A:
            if len(s) < 2: continue
            for i, x in enumerate(s):
                if x == a:
                    t = s[:i] + (b,) + s[i + 1:]
                    if t in B: k += 1
        return k

def jsd(p, q, eps=0.5):
    p = p + eps; q = q + eps; p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

PROFS = ['pos', 'lab', 'ot', 'site', 'L', 'R']
def dist(C, a, b, which):
    return jsd(C.P[a][which], C.P[b][which])

def run(C, edges, label, minfreq, nperm=NPERM, profs=PROFS, keymap=None, rep=None):
    """keymap maps a Wells number to this corpus' sign id (identity for Wells, bridge for IM77)"""
    km = keymap or (lambda w: w)
    linked = collections.defaultdict(set)
    for e in EDGES:
        linked[e['derived']].add(e['base']); linked[e['base']].add(e['derived'])
    rows = []
    for e in edges:
        b = km(e['base']); d = km(e['derived'])
        if b is None or d is None or b == d or b not in C.P or d not in C.P: continue
        if C.P[b]['n'] < minfreq or C.P[d]['n'] < minfreq: continue
        fb = C.P[b]['n']
        pool = [r for r in C.P if r != d and r != b and fb / 2 <= C.P[r]['n'] <= fb * 2 and r not in linked.get(r, set()) and r not in {km(x) for x in linked[e['derived']] if km(x)}]
        if len(pool) < 5: continue
        obs = {w: dist(C, b, d, w) for w in profs}
        nul = {w: np.array([dist(C, r, d, w) for r in random.choices(pool, k=nperm)]) for w in profs}
        pct = {w: float(np.mean(nul[w] <= obs[w])) for w in profs}
        drel = C.P[d]['relpos'] - C.P[b]['relpos']
        rels = np.array([C.P[d]['relpos'] - C.P[r]['relpos'] for r in random.choices(pool, k=200)])
        row = dict(edge=e, b=b, d=d, nb=fb, nd=C.P[d]['n'], obs=obs, pct=pct, mean_pct=float(np.mean(list(pct.values()))),
                   drel=drel, drel_null=float(np.nanmean(rels)), dlast=C.P[d]['last'] - C.P[b]['last'], dtab=C.P[d]['tab'] - C.P[b]['tab'],
                   dnum=C.P[d]['numbefore'] - C.P[b]['numbefore'], numb=C.P[b]['numbefore'], numd=C.P[d]['numbefore'],
                   cooc=C.cooc(b, d), cooc_exp=fb * C.P[d]['n'] / len(C.texts) * 0 + len(C.P[b]['texts']) * len(C.P[d]['texts']) / len(C.texts),
                   cooc_null=float(np.mean([C.cooc(r, d) for r in random.sample(pool, min(len(pool), 50))])),
                   mp=C.minimal_pairs(b, d), mp_null=float(np.mean([C.minimal_pairs(r, d) for r in random.sample(pool, min(len(pool), 50))])),
                   labb=LABS[int(np.argmax(C.P[b]['lab']))], labd=LABS[int(np.argmax(C.P[d]['lab']))],
                   posb=POS[int(np.argmax(C.P[b]['pos']))], posd=POS[int(np.argmax(C.P[d]['pos']))], pool=len(pool))
        rows.append(row)
    rep.append(f'\n### {label}: {len(rows)} edges with both signs >= {minfreq} tokens')
    if not rows: return rows
    # per type summaries
    types = ['enclosure', 'roof', 'strokes', 'doubling', 'ligature', 'family']
    rep.append(f'{"type":10s} {"n":>3s} | mean pct of JSD(base,derived) in freq-matched null (0 = derived is the base\'s twin, 0.5 = stranger)')
    rep.append(f'{"":10s} {"":>3s} |  pos   lab    obj   site   left  right | drel(obs/null) dlast  dtab  dnum | cooc obs/exp/null | minpairs obs/null')
    for t in types + ['ALL']:
        rs = [r for r in rows if t == 'ALL' or r['edge']['type'] == t]
        if not rs: continue
        pm = {w: np.mean([r['pct'][w] for r in rs]) for w in profs}
        # one-sided permutation: mean percentile vs uniform (expected 0.5); simulate by averaging uniform draws
        sim = np.mean(np.random.rand(nperm, len(rs)), axis=1)
        pval = {w: float(np.mean(sim <= pm[w])) for w in profs}
        rep.append(f'{t:10s} {len(rs):3d} | ' + ' '.join(f'{pm[w]:.2f}{"*" if pval[w] < 0.05 else " "}' for w in profs) +
                   f' | {np.nanmean([r["drel"] for r in rs]):+.3f}/{np.nanmean([r["drel_null"] for r in rs]):+.3f} {np.mean([r["dlast"] for r in rs]):+.3f} {np.mean([r["dtab"] for r in rs]):+.3f} {np.mean([r["dnum"] for r in rs]):+.3f}' +
                   f' | {sum(r["cooc"] for r in rs)}/{sum(r["cooc_exp"] for r in rs):.1f}/{sum(r["cooc_null"] for r in rs):.1f} | {sum(r["mp"] for r in rs)}/{sum(r["mp_null"] for r in rs):.1f}')
    # directional permutation test for drel per type: observed mean drel vs mean of (relpos(d) - relpos(r)) draws
    rep.append('directional test, relative position: P(null |mean shift| >= observed) per type; sign of the shift (+ = derived sits further right than its base)')
    for t in types:
        rs = [r for r in rows if r['edge']['type'] == t and not np.isnan(r['drel'])]
        if len(rs) < 3: continue
        obs = np.mean([r['drel'] for r in rs])
        null = np.mean([r['drel_null'] for r in rs])
        rep.append(f'  {t:10s} n={len(rs):2d} mean shift {obs:+.3f} (vs stranger {null:+.3f}); share of edges with |shift| < 0.1: {np.mean([abs(r["drel"]) < 0.1 for r in rs]):.2f}')
    rep.append('edge detail (base -> derived: n_b n_d | label/pos base -> derived | pct pos lab L R | drel dnum | cooc obs/exp | minpairs):')
    for r in sorted(rows, key=lambda r: (r['edge']['type'], -r['nd'])):
        e = r['edge']
        rep.append(f"  {e['type']:9s} W{e['base']:<4d}->W{e['derived']:<4d} {r['nb']:4d} {r['nd']:4d} | {r['labb']}/{r['posb']} -> {r['labd']}/{r['posd']} | {r['pct']['pos']:.2f} {r['pct'].get('lab',float('nan')):.2f} {r['pct']['L']:.2f} {r['pct']['R']:.2f} | {r['drel']:+.2f} {r['dnum']:+.2f} ({r['numb']:.2f}->{r['numd']:.2f}) | {r['cooc']}/{r['cooc_exp']:.1f} | {r['mp']}")
    return rows

rep = [f'# LOOP 28 cycle 2 ({LV}): distribution of derived signs vs their bases ({time.strftime("%Y-%m-%dT%H:%M")})']
rep.append(f'edges {len(EDGES)}; texts {len(TEXTS)} complete; nperm {NPERM}; null = random sign with frequency within x2 of the base, not graphically linked to the derived sign')
ALL = Corpus(TEXTS, NUM)
BIG = Corpus([t for t in TEXTS if t['big']], NUM)
HELD = Corpus([t for t in TEXTS if not t['big']], NUM)
good = [e for e in EDGES if e['conf'] != 'low']
rep.append(f'edges used: {len(good)} with confidence mid/high (low dropped)')
rows_all = run(ALL, good, 'ALL SITES', 5, rep=rep)
rows_big = run(BIG, good, 'TRAIN: Mohenjo-daro + Harappa', 5, rep=rep)
# confirm on held-out: the edges that were significant (mean pct < 0.2) on train
sig = [r['edge'] for r in rows_big if r['mean_pct'] < 0.2]
rep.append(f'\n## held-out confirmation: {len(sig)} edges with mean percentile < 0.2 on train; held-out sites (min 3 tokens each)')
rows_held = run(HELD, sig, 'HELD-OUT SITES (train-significant edges)', 3, rep=rep)
if rows_held:
    rep.append(f'   held-out: {sum(r["mean_pct"] < 0.5 for r in rows_held)} of {len(rows_held)} train-significant edges have derived closer to base than to the median stranger; mean pct {np.mean([r["mean_pct"] for r in rows_held]):.2f}')
# IM77 through the bridge
IM = []
with open('data/im77/im77_corpus_lines.csv') as f:
    for r in csv.DictReader(f):
        s = [int(x) for x in r['signs_clean'].split() if x.isdigit() and int(x) > 0]
        if s: IM.append(dict(seq=s, site=r['site'], ot=('SEAL' if r['object_type'] == 'seal' else ('TAB' if 'tablet' in r['object_type'] else ('POT' if 'pot' in r['object_type'] else 'OTHER'))), big=r['site_code'] in ('MD', 'H'), lab=None))
for t in IM: t['lab'] = ['NAME'] * len(t['seq'])
IMC = Corpus(IM, NUM_M, labels=False)
def km(w):
    v = BR.get(w)
    return v[0] if v and len(v) == 1 else None
rev = collections.Counter(km(w) for w in BR if km(w))
def km1(w):
    m = km(w)
    return m if m and rev[m] == 1 else None
rep.append(f'\n## IM77 (Mahadevan) through the bridge, one-to-one mappings only; {len(IM)} lines')
rows_im = run(IMC, good, 'IM77', 5, profs=['pos', 'ot', 'site', 'L', 'R'], keymap=km1, rep=rep)
json.dump(dict(all=[{k: v for k, v in r.items() if k != 'edge'} | dict(base=r['edge']['base'], derived=r['edge']['derived'], type=r['edge']['type'], sub=r['edge'].get('sub')) for r in rows_all],
               im77=[{k: v for k, v in r.items() if k != 'edge'} | dict(base=r['edge']['base'], derived=r['edge']['derived'], type=r['edge']['type']) for r in rows_im]),
          open(OUT + f'loop28_c2_{LV}.json', 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
open(OUT + f'loop28_cycle2_{LV}.txt', 'w').write('\n'.join(rep) + '\n')
print('\n'.join(rep))
