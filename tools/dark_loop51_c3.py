#!/usr/bin/env python3
"""Loop 51 cycle 3: is there a foreign register (a reduced export code)?

Sets (distinct site x text, >= 2 signs): FA = round Gulf-type seals found abroad; FS = square (+ cylinder) seals
found abroad; FO = pots / sealing / tablet found abroad; HR = round and cylinder seals found at home sites;
HS = home square seals; HO = home pots/sealings/tablets; UR = round/cylinder seals of unknown provenance.
Per-text features: opener-initial, closer-final, any frame, person sign present, person first, numeral present,
share of tokens in the home top-50 signs, share of rare tokens (home frequency < 5 texts), share of adjacent pairs
attested at home (outside the text itself), type-token ratio.
Set features: distinct signs used, Simpson concentration, share of tokens in the set's own top-10 signs.
Null: for each foreign text, a home text of the same object shape class and the same length (1,000 set draws);
when the shape stratum is too thin (round seals at home, n ~ 20) the null is also drawn from home square seals of
the same length, and the two are reported side by side. Direct contrast FA vs FS: label permutation 1,000x.
usage: python3 tools/dark_loop51_c3.py LEVEL
"""
import sys, os, json, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dark_loop51_common import *
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NPERM = 1000
rnd = random.Random(53)
OUT = f'data/derived/dark/loop51_c3_{LV}.txt'; LOG = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

objs = load(LV)
def sh(o): return 'round' if o['shape'] in ('round', 'cylinder') else o['shape']
home = dedup([o for o in objs if not o['foreign'] and not o['border'] and o['site'] != 'Unknown' and len(o['seq']) >= 2])
unk = dedup([o for o in objs if o['site'] == 'Unknown' and len(o['seq']) >= 2])
foreign = dedup([o for o in objs if o['foreign'] and len(o['seq']) >= 2])
FA = [o for o in foreign if sh(o) == 'round']; FS = [o for o in foreign if sh(o) == 'square']
FO = [o for o in foreign if sh(o) not in ('round', 'square')]
HR = [o for o in home if sh(o) == 'round']; HS = [o for o in home if sh(o) == 'square']
HO = [o for o in home if sh(o) not in ('round', 'square')]
UR = [o for o in unk if sh(o) == 'round']
P(f'# S-DARK-51 cycle 3, level {LV}: foreign register. FA {len(FA)} FS {len(FS)} FO {len(FO)} | HR {len(HR)} HS {len(HS)} HO {len(HO)} | UR {len(UR)}')

# home sign and bigram tables (home distinct texts)
tfreq = collections.Counter(); bg = collections.Counter()
for o in home:
    for x in set(o['seq']): tfreq[x] += 1
    for a, b in zip(o['seq'], o['seq'][1:]): bg[(a, b)] += 1
TOP50 = {x for x, _ in tfreq.most_common(50)}
def feats(o, self_in_home):
    s = o['seq']; fr = frame(s)
    pairs = list(zip(s, s[1:]))
    sub = 1 if self_in_home else 0
    att = sum(1 for p in pairs if bg[p] - sub > 0) / len(pairs) if pairs else float('nan')
    return dict(opener=fr['opener'], closer=fr['closer'], frame=fr['opener'] or fr['closer'], person=fr['person'],
                person_first=fr['person_first'], num=fr['num'], top50=sum(x in TOP50 for x in s) / len(s),
                rare=sum((tfreq[x] - sub) < 5 for x in s) / len(s), att=att, ttr=len(set(s)) / len(s), L=len(s))
FEATS = ['opener', 'closer', 'frame', 'person', 'person_first', 'num', 'top50', 'rare', 'att', 'ttr']
def set_stats(texts, self_in_home):
    F = [feats(o, self_in_home) for o in texts]
    out = {k: sum(f[k] for f in F if f[k] == f[k]) / max(1, sum(1 for f in F if f[k] == f[k])) for k in FEATS}
    toks = [x for o in texts for x in o['seq']]; c = collections.Counter(toks)
    out['distinct'] = len(c); out['simpson'] = sum((v / len(toks)) ** 2 for v in c.values())
    out['top10own'] = sum(v for _, v in c.most_common(10)) / len(toks)
    out['ntok'] = len(toks)
    return out

def draw_matched(target, pool, fallback):
    """one home text per target text, same length (nearest length within the pool; fallback pool if none)"""
    by_len = collections.defaultdict(list)
    for o in pool: by_len[len(o['seq'])].append(o)
    fb = collections.defaultdict(list)
    for o in fallback: fb[len(o['seq'])].append(o)
    out = []
    for o in target:
        L = len(o['seq']); cands = by_len.get(L) or fb.get(L)
        if not cands:
            for dl in range(1, 6):
                cands = by_len.get(L - dl) or by_len.get(L + dl) or fb.get(L - dl) or fb.get(L + dl)
                if cands: break
        out.append(rnd.choice(cands))
    return out

def compare(name, target, pool, poolname, fallback=None, self_in_home=True):
    obs = set_stats(target, False)
    null = collections.defaultdict(list)
    for _ in range(NPERM):
        d = draw_matched(target, pool, fallback or pool)
        st = set_stats(d, self_in_home)
        for k, v in st.items(): null[k].append(v)
    P(f'\n-- {name} (n = {len(target)}, {obs["ntok"]} tokens) vs {poolname} matched on length ({NPERM} draws)')
    for k in FEATS + ['distinct', 'simpson', 'top10own']:
        nv = sorted(null[k]); mu = sum(nv) / len(nv)
        lo, hi = nv[int(0.025 * len(nv))], nv[int(0.975 * len(nv))]
        ph = pval(obs[k], nv); pl = pval(obs[k], nv, 'lo')
        flag = ' <<' if min(ph, pl) < 0.025 else ''
        P(f'   {k:12s} obs {obs[k]:.3f} | null {mu:.3f} [{lo:.3f}-{hi:.3f}] P_hi {ph:.3f} P_lo {pl:.3f}{flag}')
    return obs

P('\n== lengths: ' + '; '.join(f'{n} mean {sum(len(o["seq"]) for o in S)/max(1,len(S)):.2f}' for n, S in
                              [('FA', FA), ('FS', FS), ('FO', FO), ('HR', HR), ('HS', HS), ('HO', HO), ('UR', UR)]))
P('home round/cylinder seals: ' + '; '.join(f'{o["site"]} {"-".join(map(str,o["seq"]))}' for o in HR))
compare('FA round seals abroad', FA, HR, 'HOME ROUND seals (shape-matched; thin stratum, fallback to home square at missing lengths)', fallback=HS)
compare('FA round seals abroad', FA, HS, 'HOME SQUARE seals')
compare('FS square seals abroad', FS, HS, 'HOME SQUARE seals')
compare('FO pots/sealing/tablet abroad', FO, HO, 'HOME pots/sealings/tablets')
compare('HR home round seals', HR, HS, 'HOME SQUARE seals (control: shape effect at home)')
compare('UR unknown-provenance round seals', UR, HS, 'HOME SQUARE seals')
compare('ALL foreign', foreign, home, 'ALL HOME (shape ignored)')

# ---- direct contrast FA vs FS, label permutation
P(f'\n== direct contrast: round seals abroad (FA, n={len(FA)}) vs square seals abroad (FS, n={len(FS)}); label permutation {NPERM}x')
a = set_stats(FA, False); b = set_stats(FS, False)
allf = FA + FS; na = len(FA)
null = collections.defaultdict(list)
for _ in range(NPERM):
    rnd.shuffle(allf); sa = set_stats(allf[:na], False); sb = set_stats(allf[na:], False)
    for k in FEATS: null[k].append(sa[k] - sb[k])
for k in FEATS:
    d = a[k] - b[k]; nv = null[k]
    p2 = (sum(1 for v in nv if abs(v) >= abs(d)) + 1) / (NPERM + 1)
    P(f'   {k:12s} FA {a[k]:.3f} FS {b[k]:.3f} diff {d:+.3f} two-sided P {p2:.3f}')

# ---- inventory: which signs does the foreign register use, and does it reuse them?
P('\n== inventory of the foreign-found texts (FA + FS + FO)')
c = collections.Counter(x for o in foreign for x in o['seq'])
P(f'   {sum(c.values())} tokens, {len(c)} distinct signs; used >= 2 times: ' +
  ', '.join(f'W{x} x{n} (home texts {tfreq[x]})' for x, n in c.most_common() if n >= 2))
P(f'   signs unseen at home: {[x for x in c if tfreq[x] == 0]}')
cA = collections.Counter(x for o in FA for x in o['seq'])
P(f'   FA alone: {sum(cA.values())} tokens, {len(cA)} distinct; >= 2: ' + ', '.join(f'W{x} x{n}' for x, n in cA.most_common() if n >= 2))
json.dump(dict(level=LV, FA=[list(o['seq']) for o in FA], FS=[list(o['seq']) for o in FS], FO=[list(o['seq']) for o in FO]),
          open(OUT.replace('.txt', '.json'), 'w'))
P(f'\nwritten {OUT}')
