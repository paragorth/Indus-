#!/usr/bin/env python3
"""Loop 71, cycle 2: the Lothal co-impression network.
Unit: sealing x die (seal identity, cycle-1 assignment) incidence, built twice: Wells faces (W numbers) and IM77 sides
(M numbers; same objects read twice -> transcription-robust, not replicated).
(a) Network vs a COUNT-PRESERVING null: curveball swaps of the sealing x die incidence (every sealing keeps its number
    of distinct dies, every die its number of sealings; no die twice on one sealing), 2,000 null matrices.
    Statistics: distinct co-impression pairs, pairs repeated on >= 2 sealings, size of the largest connected team,
    distinct partners of the hub, dies used both alone and in company.
(b) Do co-impressed dies (team members) share a head (closer slot), an opener-initial position, or a middle (NAME /
    TITLE / COUNT slot) sign? Observed edge rates vs the same rates on the edges of the null networks (so the
    comparison holds degree fixed).
(c) Does the back type (Frenez & Tosi) predict the seal or its head? Wells-linked sealings; back label permuted over
    sealings 5,000x; statistic = mutual information (bits) of back x die (recurrent dies only) and back x head;
    with and without the elephant boxes.
Usage: python3 tools/dark_loop71_c2.py [seq_raw|seq_strong|seq_all] [nperm]"""
import sys, csv, json, collections, random, math
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop71_common import *
from dark_loop37 import learn_qual, make_parser, IM77 as IM77_PATH
from dark_loop65 import head_of, M_HEAD, M_SUF, M_OPEN

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rnd = random.Random(71)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

FT = {r[0]: dict(nimp=r[1], back=r[2], ctx=r[3]) for r in json.load(open(FTJ))['rows']}

# ---------------- incidence ----------------
def wells_incidence():
    W = wells_lothal(LV)
    imps = [(f['id'], f['toks']) for k, fs in W.items() for f in fs]
    lab = assign_dies(imps)
    inc = collections.OrderedDict()
    for c, fs in W.items():
        ds = []
        for f in fs:
            if not [x for x in f['toks'] if x]: continue
            d = lab[f['id']]
            if d not in ds: ds.append(d)
        if ds: inc[c] = ds
    return inc

def im77_incidence():
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(IM77_PATH)):
        if r['site'] != 'Lothal' or r['object_type'] != 'sealing' or r['line'] == '9' or not r['signs_clean'].strip(): continue
        raw[r['text_no']][int(r['side'])].extend(int(x) for x in r['signs_clean'].split())
    imps = [(f'{k}.{s}', t) for k, sides in raw.items() for s, t in sides.items()]
    lab = assign_dies(imps)
    inc = collections.OrderedDict()
    for k, sides in raw.items():
        ds = []
        for s, t in sides.items():
            if not [x for x in t if x]: continue
            d = lab[f'{k}.{s}']
            if d not in ds: ds.append(d)
        # a fragment die ('u:') fully contained in another die on the same object = the same seal stamped again
        ds2 = []
        for d in ds:
            if d[0] == 'u' and any(e != d and fits(list(d[1:]), [x for x in (e[1:] if e[0] == 'u' else e)]) for e in ds): continue
            ds2.append(d)
        inc[k] = ds2
    return inc

def toks(d): return [x for x in (d[1:] if d[0] == 'u' else d) if x]

def curveball(inc_lists, n_swaps):
    """inc_lists: list of sets (dies per sealing). Curveball trades between random pairs of sealings."""
    L = [set(s) for s in inc_lists]
    n = len(L)
    for _ in range(n_swaps):
        i, j = rnd.randrange(n), rnd.randrange(n)
        if i == j: continue
        a, b = L[i], L[j]
        ua = list(a - b); ub = list(b - a)
        if not ua or not ub: continue
        pool = ua + ub; rnd.shuffle(pool)
        na = len(ua)
        L[i] = (a & b) | set(pool[:na]); L[j] = (a & b) | set(pool[na:])
    return L

def stats(L, hub):
    pairs = collections.Counter()
    for s in L:
        s = sorted(s, key=str)
        for x in range(len(s)):
            for y in range(x + 1, len(s)): pairs[(s[x], s[y])] += 1
    # largest connected component among dies on multi-impression sealings
    adj = collections.defaultdict(set)
    for (a, b) in pairs: adj[a].add(b); adj[b].add(a)
    seen = set(); big = 0
    for v in adj:
        if v in seen: continue
        st = [v]; comp = 0; seen.add(v)
        while st:
            u = st.pop(); comp += 1
            for w in adj[u]:
                if w not in seen: seen.add(w); st.append(w)
        big = max(big, comp)
    alone = collections.Counter(); comp_ = collections.Counter()
    for s in L:
        for d in s:
            (alone if len(s) == 1 else comp_)[d] += 1
    both = sum(1 for d in set(alone) | set(comp_) if alone[d] and comp_[d])
    return dict(edges=len(pairs), rep=sum(1 for v in pairs.values() if v >= 2), maxw=max(pairs.values(), default=0),
                big=big, hubp=len(adj.get(hub, ())), both=both), pairs

def run(name, inc, hub, headf, openset, parse, markset):
    P(f'\n-- {name}: {len(inc)} sealings with a legible die; {sum(1 for v in inc.values() if len(v) >= 2)} with >= 2 distinct dies; '
      f'{len({d for v in inc.values() for d in v})} distinct dies; recurrent (>= 2 sealings): '
      f'{sum(1 for d, n in collections.Counter(d for v in inc.values() for d in v).items() if n >= 2)}')
    L0 = [set(v) for v in inc.values()]
    obs, pairs = stats(L0, hub)
    P('   observed: ' + ', '.join(f'{k}={v}' for k, v in obs.items()))
    rep_pairs = [(a, b, n) for (a, b), n in pairs.items() if n >= 2]
    for a, b, n in sorted(rep_pairs, key=lambda x: -x[2]): P(f'      repeated pair x{n}: {fmt(a)} + {fmt(b)}')
    for k, v in inc.items():
        if len(v) >= 2: P(f'      {k}: ' + ' + '.join(fmt(d) for d in v))
    nulls = collections.defaultdict(list); nshare = collections.defaultdict(list)
    def share(pairs_):
        e = list(pairs_)
        if not e: return dict(head=0, open=0, frame=0, mid=0, any=0)
        h = sum(1 for a, b in e if headf(toks(a)) == headf(toks(b)) != 'none') / len(e)
        o = sum(1 for a, b in e if toks(a) and toks(b) and toks(a)[0] in openset and toks(b)[0] in openset) / len(e)
        fr = sum(1 for a, b in e if len(toks(a)) > 1 and len(toks(b)) > 1 and toks(a)[1] in markset and toks(b)[1] in markset) / len(e)
        def mid(t):
            lab = parse(t)
            if len(t) > 1 and t[1] in markset: lab = ['OPENER', 'MARKER'] + lab[2:]   # any sign + marker = frame opening (S286)
            return {w for w, l in zip(t, lab) if l in ('NAME', 'TITLE', 'COUNT')}
        m = sum(1 for a, b in e if mid(toks(a)) & mid(toks(b))) / len(e)
        an = sum(1 for a, b in e if set(toks(a)) & set(toks(b))) / len(e)
        return dict(head=h, open=o, frame=fr, mid=m, any=an)
    sh_obs = share(pairs)
    for _ in range(NP):
        Ln = curveball(L0, 20 * sum(len(s) for s in L0))
        st, pn = stats(Ln, hub)
        for k, v in st.items(): nulls[k].append(v)
        for k, v in share(pn).items(): nshare[k].append(v)
    P('   count-preserving null (curveball), mean / P_hi / P_lo:')
    for k in obs:
        nv = nulls[k]
        P(f'      {k:6s} obs {obs[k]:3d}  null {sum(nv)/len(nv):6.2f}  P_hi {pval(obs[k], nv):.3f}  P_lo {pval(obs[k], nv, "lo"):.3f}')
    P('   (b) team members share ... (share of co-impressed die pairs; null = same rate on null-network pairs;')
    P('       head = same closer head; open = both opener-initial (fixed opener set); frame = both have a marker in position 2;')
    P('       mid = a shared sign outside initial / marker / head / suffix):')
    for k in sh_obs:
        nv = nshare[k]
        P(f'      {k:5s} obs {sh_obs[k]:.2f}  null {sum(nv)/len(nv):.2f}  P_hi {pval(sh_obs[k], nv):.3f}  P_lo {pval(sh_obs[k], nv, "lo"):.3f}')
    # team composition: heads and openers of the recurrent dies
    cnt = collections.Counter(d for v in inc.values() for d in v)
    P('   recurrent dies: text [head, opener-initial?] sealings alone / with others')
    alone = collections.Counter(d for v in inc.values() if len(v) == 1 for d in v)
    for d, n in cnt.most_common():
        if n < 2: break
        t = toks(d)
        P(f'      {fmt(d):34s} [{headf(t)}, {"opener" if t and t[0] in openset else "-"}, {"X+marker" if len(t) > 1 and t[1] in markset else "-"}] {alone[d]} / {n - alone[d]}')
    return obs

# ---------------- Wells ----------------
canon = json.load(open(ROOT + '/data/derived/merged-corpus-canonical.json'))
parseW = make_parser(learn_qual([r[LV] for r in canon if r[LV]]))
incW = wells_incidence()
obsW = run('Wells', incW, HUB, head_of, OPEN, parseW, {2, 60})
# ---------------- IM77 ----------------
from dark_loop37 import load_im77
im = load_im77()
MOPEN = M_OPEN
def headM(t): return head_of(t, M_HEAD, M_SUF, M_OPEN)
# IM77 frame parser: same rules with M-space sets (openers M267/391/293/150, markers M99/100/123)
def parseM(t):
    lab = ['NAME'] * len(t); i = 0; j = len(t)
    if t and t[0] in MOPEN:
        lab[0] = 'OPENER'; i = 1
        if len(t) > 1 and t[1] in (99, 100, 123): lab[1] = 'MARKER'; i = 2
    while j - 1 > i and t[j - 1] in M_SUF: lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and t[j - 1] in M_HEAD: lab[j - 1] = 'CLOSER'
    return lab
incM = im77_incidence()
obsM = run('IM77', incM, (336, 209, 343, 98, 121, 59, 342, 1), headM, MOPEN, parseM, {99, 100, 123})

# ---------------- (c) back type ----------------
P('\n-- (c) back type (Frenez & Tosi) vs seal and head, Wells-linked sealings')
def mi(pairs_):
    n = len(pairs_); cx = collections.Counter(a for a, b in pairs_); cy = collections.Counter(b for a, b in pairs_); cxy = collections.Counter(pairs_)
    return sum(v / n * math.log2(v * n / (cx[a] * cy[b])) for (a, b), v in cxy.items())
recW = {d for d, n in collections.Counter(d for v in incW.values() for d in v).items() if n >= 2}
for excl_ele in (False, True):
    items = []
    for c, ds in incW.items():
        if c not in FT or FT[c]['back'] in ('undiagnostic',): continue
        if excl_ele and any(len(toks(d)) == 4 and toks(d)[1:] == [2, 48, 740] for d in ds): continue   # elephant die at any merge level
        for d in ds: items.append((c, FT[c]['back'], d))
    tag = 'without elephant' if excl_ele else 'all'
    for what in ('die', 'head'):
        if what == 'die': pr = [(b, fmt(d) if d in recW else 'other') for c, b, d in items]
        else: pr = [(b, head_of(toks(d))) for c, b, d in items]
        o = mi(pr)
        # permute back labels over sealings
        secs = collections.OrderedDict()
        for c, b, d in items: secs.setdefault(c, [b, []])[1].append(d)
        backs = [v[0] for v in secs.values()]; null = []
        for _ in range(5000):
            rnd.shuffle(backs)
            prn = []
            for (c, (b0, ds)), b in zip(secs.items(), backs):
                for d in ds: prn.append((b, (fmt(d) if d in recW else 'other') if what == 'die' else head_of(toks(d))))
            null.append(mi(prn))
        P(f'   {tag:16s} back x {what:4s}: n = {len(pr)} impressions on {len(secs)} sealings, MI {o:.3f} bits vs null {sum(null)/len(null):.3f} (P = {pval(o, null):.3f})')
P('   back types per recurrent die (Wells):')
for d in sorted(recW, key=lambda d: -sum(1 for v in incW.values() if d in v)):
    bs = collections.Counter(FT[c]['back'] for c, v in incW.items() if d in v and c in FT)
    P(f'      {fmt(d):34s} ' + ', '.join(f'{b} {n}' for b, n in bs.most_common()))
open(OUT + f'loop71_c2_{LV}.txt', 'w').write('\n'.join(LOG) + '\n')
