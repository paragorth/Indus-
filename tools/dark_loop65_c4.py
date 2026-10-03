"""S-DARK-65 cycle 4: replication of cycle 2 inside IM77 (Mahadevan records sealing sides), in M space.
Sides that are Harappa voucher counts (numeral + M328, S93) are removed; sides identical to another side of the same
object, or a fragment contained in one, are one die; identical side-sets within a site are one object read twice
(IM77 1623 = 2847 etc., S-DARK-13). Nulls as in cycle 2: N1 same-site IM77 sealing sides, N1L length-matched,
N2 same-site IM77 seal texts. Then a bridge check: which Wells multi-impression Lothal texts recur in IM77's sides.
Usage: python3 tools/dark_loop65_c4.py [nperm] [seq_raw|seq_strong|seq_all for the bridge check]"""
import sys, collections, itertools, random, json
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop65 import *
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
rnd = random.Random(654)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'== S-DARK-65 cycle 4 (IM77, nperm {NP}): two-party tests on IM77 sealing sides')
# ---- M-space frame parser (mirror of the loop-37 Wells parser) ----
M_FISH = {59, 65, 67, 70, 72, 60}; M_SUFS = {176, 1}
def learn_qual_M(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in M_SUFS: s.pop()
        if len(s) >= 2 and s[-1] in M_HEAD: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q
def make_parser_M(QUAL):
    def parse(s):
        if not s: return []
        lab = ['NAME'] * len(s); i = 0; j = len(s)
        if s[0] in M_OPEN:
            lab[0] = 'OPENER'; i = 1
            if len(s) > 1 and s[1] in M_MARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 293 and len(s) > 2 and s[2] in (343, 344, 345): lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in M_SUFS and j >= 2 and (s[j - 2] in M_HEAD or s[j - 2] in M_SUFS): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in M_HEAD:
            c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
            if c == 211:
                if j - 2 >= i and s[j - 1] == 89 and s[j - 2] == 336: lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
                while j - 1 >= i and s[j - 1] in M_FISH: lab[j - 1] = 'TITLE'; j -= 1
            elif c == 342:
                if j - 1 >= i and s[j - 1] == 8: lab[j - 1] = 'TITLE'; j -= 1
                if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in M_NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
        for k in range(i, j - 1):
            if s[k] in M_NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
        for k in range(i, j):
            if s[k] in M_NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
        return lab
    return parse
im = load_im77()
allM = [f['seq'] for o in im.values() for f in o['faces'] if f['seq']]
parseM = make_parser_M(learn_qual_M(allM)); middleM = middle_fn(parseM); nameM = middle_fn(parseM, ('NAME',))
headM = lambda s: head_of(s, M_HEAD, M_SUF, M_OPEN)
def contained(a, b):
    a, b = list(a), list(b)
    return len(a) < len(b) and any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))
# IM77 'sealing' objects -> distinct non-count impressions
SE = collections.OrderedDict(); seen = set()
ncount = 0
for o in im.values():
    if o['type'] != 'sealing': continue
    faces = [f['seq'] for f in o['faces'] if f['seq']]
    cnt = [f for f in faces if is_count_face_M(f)]; ncount += len(cnt)
    faces = [f for f in faces if not is_count_face_M(f)]
    imps = []
    for f in faces:
        if any(f == g or contained(f, g) for g in imps): continue
        imps = [g for g in imps if not contained(g, f)] + [f]
    if not imps: continue
    key = (o['site'], tuple(sorted(tuple(i) for i in imps)))
    if len(imps) >= 2 and key in seen: continue
    seen.add(key)
    frag = {tuple(f['seq']): (0 in [int(x) for r in [] for x in r]) for f in o['faces']}
    SE[o['oid']] = dict(oid=o['oid'], site=o['site'], imps=[dict(seq=i, complete=True) for i in imps])
multi = [o for o in SE.values() if len(o['imps']) >= 2]
P(f'IM77 sealings with signs: {sum(1 for o in im.values() if o["type"]=="sealing")}; count sides removed: {ncount}; objects kept {len(SE)}; '
  f'with >= 2 distinct non-count impressions: {len(multi)} by site {dict(collections.Counter(o["site"] for o in multi))}')
P('  (Harappa "sealings" with a count side are the two-sided voucher tablets of S93; what remains at Harappa is listed below)')
for o in multi:
    if o['site'] == 'Harappa': P('   H', o['oid'], ' | '.join(' '.join(map(str, i['seq'])) for i in o['imps']))
pairs_all = [(o, a, b) for o in multi for a, b in itertools.combinations(o['imps'], 2)]
pairs_strict = [(o, a, b) for o, a, b in pairs_all if len(a['seq']) >= 2 and len(b['seq']) >= 2]
pool_tag = collections.defaultdict(list)
for o in SE.values():
    for f in o['imps']: pool_tag[o['site']].append((o['oid'], f))
pool_seal = collections.defaultdict(list)
for o in im.values():
    if o['type'] == 'seal':
        for f in o['faces']:
            if f['seq']: pool_seal[o['site']].append(dict(seq=f['seq'], complete=True))
other_seals = [f for s, L in pool_seal.items() if len(L) < 10 for f in L]
def seal_pool(site):
    L = pool_seal.get(site, []); return L if len(L) >= 10 else other_seals + L
def draw_tag(o, f, lenmatch=False):
    cand = [g for oid, g in pool_tag[o['site']] if oid != o['oid']]
    if lenmatch:
        c2 = [g for g in cand if len(g['seq']) == len(f['seq'])]
        if len(c2) < 3: c2 = [g for g in cand if abs(len(g['seq']) - len(f['seq'])) <= 1]
        cand = c2 or cand
    return rnd.choice(cand) if cand else None
# seal-text match in IM77: exact or contained in an IM77 seal text of any site
seal_set = set(tuple(f['seq']) for L in pool_seal.values() for f in L)
seal_sub = set()
for t in seal_set:
    for L in range(2, len(t)):
        for i in range(len(t) - L + 1): seal_sub.add(t[i:i + L])
def matchM(s): return tuple(s) in seal_set or (len(s) >= 2 and tuple(s) in seal_sub)
KEYS = ['shared_any', 'shared_mid', 'shared_name', 'same_mid', 'jacc', 'same_head', 'both_jar', 'lendiff', 'short_long',
        'frame_only_one', 'frame_only_both', 'opener_one', 'opener_both', 'match_one', 'match_both', 'match_none']
MEAN = {'jacc', 'lendiff'}
def stats(a, b):
    st = pair_stats(a, b, middleM, nameM, headM)
    st['opener_one'] = (a[0] in M_OPEN) != (b[0] in M_OPEN); st['opener_both'] = a[0] in M_OPEN and b[0] in M_OPEN
    xa, xb = matchM(a), matchM(b)
    st['match_one'] = xa != xb; st['match_both'] = xa and xb; st['match_none'] = not (xa or xb)
    return st
def run(name, pairs):
    n = len(pairs)
    if n < 4: P(f'\n-- {name}: {n} pairs, too few'); return
    obs = collections.Counter(); hp = collections.Counter()
    for o, a, b in pairs:
        st = stats(a['seq'], b['seq'])
        for k in KEYS: obs[k] += st[k]
        hp[st['heads']] += 1
    nulls = {}
    for nm, mode in (('N1 same-site sealing sides', 'tag'), ('N1L length-matched', 'tagL'), ('N2 same-site seal pairs', 'seal')):
        null = collections.defaultdict(list); hpn = collections.Counter()
        for _ in range(NP):
            c = collections.Counter()
            for o, a, b in pairs:
                if mode == 'seal':
                    pl = seal_pool(o['site']); fa, fb = rnd.choice(pl), rnd.choice(pl)
                else:
                    fa = a; fb = draw_tag(o, b, lenmatch=(mode == 'tagL'))
                if fb is None: continue
                st = stats(fa['seq'], fb['seq'])
                for k in KEYS: c[k] += st[k]
                hpn[st['heads']] += 1
            for k in KEYS: null[k].append(c[k])
        nulls[nm] = (null, hpn)
    P(f'\n-- {name}: {len(set(o["oid"] for o, _, _ in pairs))} sealings, {n} impression pairs')
    P(f'   {"statistic":16s} {"obs":>9s} | ' + ' | '.join(f'{nm[:26]:>26s}' for nm in nulls))
    for k in KEYS:
        o_ = obs[k]; cells = []
        for nm, (null, _) in nulls.items():
            nl = null[k]; mu = sum(nl) / NP
            cells.append(f'{mu/n:6.3f} P+={pval(o_, nl):.3f} P-={pval(o_, nl, "lo"):.3f}')
        P(f'   {k:16s} {o_/n:6.3f} ({o_ if k not in MEAN else round(o_,1):>3}) | ' + ' | '.join(cells))
    null, hpn = nulls['N1 same-site sealing sides']
    P('   head x head pairs (obs vs N1 expected):')
    for hpair, c in sorted(hp.items(), key=lambda x: -x[1]):
        P(f'      {hpair[0]:>11s} + {hpair[1]:<11s} {c:3d}  vs {hpn[hpair]/NP:5.2f}')
    exp = {k: v / NP for k, v in hpn.items()}
    chi = sum((hp[k] - exp.get(k, 0)) ** 2 / exp[k] for k in hp if exp.get(k, 0) > 0)
    chin = []
    for _ in range(min(NP, 300)):
        c = collections.Counter()
        for o, a, b in pairs:
            fb = draw_tag(o, b)
            if fb is None: continue
            c[stats(a['seq'], fb['seq'])['heads']] += 1
        chin.append(sum((c[k] - exp.get(k, 0)) ** 2 / exp[k] for k in c if exp.get(k, 0) > 0))
    P(f'   head-pair chi2 vs N1 table: {chi:.2f}, null mean {sum(chin)/len(chin):.2f}, P = {pval(chi, chin):.3f}')
# Harappa IM77 'sealings' are Vats' moulded tablets (count side + text side, S93): kept out of the main run
pairs_H = [p for p in pairs_all if p[0]['site'] == 'Harappa']
pairs_all = [p for p in pairs_all if p[0]['site'] != 'Harappa']; pairs_strict = [p for p in pairs_strict if p[0]['site'] != 'Harappa']
run('IM77 all pairs without Harappa (true sealings)', pairs_all)
run('IM77 strict (both >= 2 signs, no Harappa)', pairs_strict)
run('IM77 Harappa two-sided "sealings" (= tablets, S93; shown for completeness)', pairs_H)
run('IM77 Lothal', [p for p in pairs_all if p[0]['site'] == 'Lothal'])
run('IM77 Mohenjo-daro', [p for p in pairs_all if p[0]['site'] == 'Mohenjodaro'])
run('IM77 without Lothal or Harappa', [p for p in pairs_all if p[0]['site'] != 'Lothal'])
P('\n-- IM77 pairs sharing a middle sign:')
for o, a, b in pairs_all:
    st = stats(a['seq'], b['seq'])
    if st['shared_mid']: P(f'   {o["site"]:12s} {o["oid"]:5s} {" ".join(map(str,a["seq"])):28s} | {" ".join(map(str,b["seq"])):28s} shared {sorted(set(middleM(a["seq"])) & set(middleM(b["seq"])))}')
# ---- bridge check: Wells multi-impression Lothal/Kalibangan texts -> M -> found among IM77 sides? ----
BR = json.load(open(BRIDGE)); PR = json.load(open(OUT + 'bridge_proposals.json'))
w2m = {int(k): v[0] for k, v in BR.items() if v}
prop = {}
for p in PR['proposals']: prop.setdefault(p['W'], p['M'])
objs, S = load_sealings(LV)
im_sides = collections.defaultdict(set)
for o in im.values():
    if o['type'] == 'sealing':
        for f in o['faces']: im_sides[o['site']].add(tuple(f['seq']))
P(f'\n-- bridge check ({LV}): Wells impressions on multi-impression sealings translated W -> M (bridge_extended, then S-DARK-27 proposals marked *):')
found = tot = 0
for o in S.values():
    if len(o['imps']) < 2: continue
    site = {'Mohenjo-daro': 'Mohenjodaro'}.get(o['site'], o['site'])
    for f in o['imps']:
        m = []; used_prop = False; ok = True
        for w in f['seq']:
            if w in w2m: m.append(w2m[w])
            elif w in prop: m.append(prop[w]); used_prop = True
            else: ok = False; break
        if not ok: P(f'   {o["cisi"] or o["oid"]:8s} {fmt(f["seq"]):30s} -> unbridged sign'); continue
        tot += 1
        hit = any(tuple(m) == s or (len(m) >= 2 and any(s[i:i + len(m)] == tuple(m) for i in range(len(s) - len(m) + 1))) for s in im_sides.get(site, ()))
        found += hit
        P(f'   {o["cisi"] or o["oid"]:8s} {fmt(f["seq"]):30s} -> M {" ".join(map(str, m)):28s}{"*" if used_prop else " "} {"found in IM77 " + site + " sides" if hit else "not found"}')
P(f'   {found}/{tot} bridged Wells impressions found among IM77 sealing sides of the same site')
open(OUT + 'loop65_c4_im77.txt', 'w').write('\n'.join(out) + '\n')
