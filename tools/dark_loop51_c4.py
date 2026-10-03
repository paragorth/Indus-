#!/usr/bin/env python3
"""Loop 51 cycle 4: replication on IM77's West Asian finds (text numbers 9801-9905, M numbers).

(1) Link each IM77 WA text to a Wells foreign text through the bridge (W -> M; unbridged W signs cannot match;
    accept unique best with cost <= 1 + 0.15 L, the loop 24 rule) -> matched set and IM77-only set.
(2) Origin assignment in M space with the IM77 home populations (Mohenjo-daro, Harappa, Lothal, Kalibangan,
    Chanhu-daro, other sites) using the same per-site boundary bigram LR model; LOO calibration on IM77 home texts;
    null = 1,000 length-matched draws of IM77 home texts (corpus mix and uniform mix).
(3) For the matched texts: does the IM77-based assignment agree with the Wells-based one (cycle 1, seq_raw W)?
    chance = pairing permuted 1,000x and the product of marginals.
(4) Register features of the WA texts vs IM77 home seals matched on length (IM77 frame: openers M267/M391/M293,
    closers M342/M211/M15/M12/M254/M162/M169, person M1 (= W90) and M3 (= W93, proposed)).
usage: python3 tools/dark_loop51_c4.py
"""
import sys, os, json, csv, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dark_loop51_common import *
rnd = random.Random(54); NPERM = 1000
OUT = 'data/derived/dark/loop51_c4.txt'; LOG = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.write(s + '\n'); LOG.flush()

# ---- IM77 texts: one text per text_no, lines concatenated in side/line order, 0 dropped
rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
by_text = collections.defaultdict(list)
for r in rows: by_text[r['text_no']].append(r)
IM = []
SITE = {'MD': 'Mohenjo-daro', 'HP': 'Harappa', 'LL': 'Lothal', 'KB': 'Kalibangan', 'CD': 'Chanhu-daro', 'OS': 'other-Indus', 'WA': 'WA'}
for tn, rs in by_text.items():
    rs.sort(key=lambda r: (int(r['side']), int(r['line'])))
    seq = []
    for r in rs:
        seq += [int(x) for x in r['signs_clean'].split() if x.strip() and int(x) != 0]
    if not seq: continue
    IM.append(dict(id=tn, site=SITE[rs[0]['site_code']], ot=rs[0]['object_type'], seq=tuple(seq), nlines=len(rs),
                   fs=rs[0]['fs_description']))
WA = [o for o in IM if o['site'] == 'WA']
IMH = [o for o in IM if o['site'] != 'WA']
LAB = ['Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Chanhu-daro', 'other-Indus']
P(f'# S-DARK-51 cycle 4: IM77 West Asian finds. {len(WA)} WA texts with signs; IM77 home texts {len(IMH)} '
  f'{dict(collections.Counter(o["site"] for o in IMH))}')

# ---- (1) link to Wells foreign texts
objs = load('seq_raw')
WF = dedup([o for o in objs if o['foreign'] and len(o['seq']) >= 1])
fM = mapper('M')
def mseq(s): return tuple(x[1] if x[0] == 'M' else ('W', x[1]) for x in fM(s))
P('\n== (1) IM77 WA texts and their Wells link (bridge W->M; unbridged W signs cannot match)')
links = {}
for w in WA:
    best = []
    for o in WF:
        d = edit(w['seq'], mseq(o['seq']))
        best.append((d, o))
    best.sort(key=lambda x: x[0])
    d0, o0 = best[0]; L = max(len(w['seq']), len(o0['seq']))
    ok = d0 <= 1 + 0.15 * L and (len(best) < 2 or best[1][0] > d0)
    links[w['id']] = o0 if ok else None
    P(f"  {w['id']} {w['ot']:8s} M {'-'.join(map(str, w['seq'])):28s} -> {'LINK' if ok else 'no link'} "
      f"d={d0} {o0['site']} W {'-'.join(map(str, o0['seq']))} (M-space {'-'.join(str(x) if not isinstance(x, tuple) else 'W'+str(x[1]) for x in mseq(o0['seq']))})")
matched = [w for w in WA if links[w['id']]]; only = [w for w in WA if not links[w['id']]]
P(f'  linked {len(matched)} of {len(WA)}; IM77-only (not in Wells) {len(only)}; Wells foreign texts not reached by IM77: '
  f'{len(WF) - len({id(links[w["id"]]) for w in matched})}')

# ---- (2) origin assignment in M space on IM77 home populations
by_site = collections.defaultdict(list)
for o in IMH: by_site[o['site']].append(o['seq'])
model = SiteModel({s: by_site[s] for s in LAB}, k=10.0)
copies = collections.Counter((o['site'], o['seq']) for o in IMH)
def assign(t, exclude_site=None):
    lr = {s: model.loglr(s, t, exclude=(t if exclude_site == s else None)) for s in LAB}
    b = max(LAB, key=lambda s: lr[s]); v = sorted(lr.values(), reverse=True)
    return b, v[0] - v[1], lr
P('\n== (2) origin assignment of the WA texts (IM77 home populations, M space)')
wa_assign = {}
for w in WA:
    b, m, lr = assign(w['seq']); wa_assign[w['id']] = b
    cp = {s: copies[(s, w['seq'])] for s in LAB if copies[(s, w['seq'])]}
    P(f"  {w['id']} {'-'.join(map(str, w['seq'])):28s} -> {b:12s} margin {m:.2f} | " + ' '.join('%s:%+.1f' % (s[:4], lr[s]) for s in LAB) + f" | copies {cp or '-'}")
cnt = collections.Counter(wa_assign.values())
P(f'  counts all WA: {dict(cnt)}; IM77-only: {dict(collections.Counter(wa_assign[w["id"]] for w in only))}; linked: {dict(collections.Counter(wa_assign[w["id"]] for w in matched))}')
# LOO calibration
home2 = [o for o in IMH if len(o['seq']) >= 2]
loo = []
for o in home2:
    b, m, _ = assign(o['seq'], exclude_site=o['site']); loo.append((o['site'], b, len(o['seq'])))
acc = sum(1 for r in loo if r[0] == r[1]) / len(loo)
marg = collections.Counter(r[1] for r in loo); true = collections.Counter(r[0] for r in loo)
chance = sum(true[s] * marg[s] for s in LAB) / len(loo) ** 2
P(f'  LOO (object removed) accuracy {acc:.3f} vs chance {chance:.3f}; assigned marginal {dict(marg)}')
for s in LAB:
    rs = [r for r in loo if r[0] == s]
    if rs: P(f'     true {s:12s} n={len(rs):5d}: ' + ' '.join(f'{u[:4]} {sum(1 for r in rs if r[1]==u)/len(rs):.2f}' for u in LAB))
by_len = collections.defaultdict(list); by_len_site = collections.defaultdict(list)
for r in loo: by_len[r[2]].append(r); by_len_site[(r[0], r[2])].append(r)
def near(L, store, key=lambda L: L):
    for dl in range(0, 10):
        for LL in (L - dl, L + dl):
            if store.get(key(LL)): return store[key(LL)]
    return []
for name, S in [('all WA', WA), ('IM77-only WA', only)]:
    if not S: continue
    lens = [len(w['seq']) for w in S]; obs = collections.Counter(wa_assign[w['id']] for w in S)
    nc = {s: [] for s in LAB}; nu = {s: [] for s in LAB}; top = []
    for _ in range(NPERM):
        c = collections.Counter(rnd.choice(near(L, by_len))[1] for L in lens)
        cu = collections.Counter()
        for L in lens:
            s = rnd.choice(LAB[:5]); rs = near(L, by_len_site, key=lambda LL: (s, LL)) or near(L, by_len)
            cu[rnd.choice(rs)[1]] += 1
        for s in LAB: nc[s].append(c[s]); nu[s].append(cu[s])
        top.append(max(c.values()) / len(S))
    P(f'  [{name}, n = {len(S)}] top-site share {max(obs.values())/len(S):.2f} vs corpus-mix null {sum(top)/NPERM:.2f} P(>=) {pval(max(obs.values())/len(S), top):.3f}')
    for s in LAB:
        v = sorted(nc[s]); vu = sorted(nu[s])
        P(f'     {s:12s} obs {obs[s]:2d} | mix {sum(v)/NPERM:5.2f} [{v[int(0.025*NPERM)]}-{v[int(0.975*NPERM)]}] P_hi {pval(obs[s], v):.3f} P_lo {pval(obs[s], v, "lo"):.3f} '
          f'| uniform {sum(vu)/NPERM:5.2f} P_hi {pval(obs[s], vu):.3f} P_lo {pval(obs[s], vu, "lo"):.3f}')

# ---- (3) agreement with the Wells-based assignment (cycle 1, seq_raw W)
c1 = json.load(open('data/derived/dark/loop51_c1_seq_raw_W.json'))
wells_assign = {(r['site'], tuple(r['seq'])): r['best'] for r in c1['rows']}
pairs = []
for w in matched:
    o = links[w['id']]; k = (o['site'], tuple(o['seq']))
    if k in wells_assign: pairs.append((w['id'], wells_assign[k], wa_assign[w['id']]))
agree = sum(1 for _, a, b in pairs if a == b)
P(f'\n== (3) linked texts with both assignments: {len(pairs)}; agree {agree}/{len(pairs)}')
for pid, a, b in pairs: P(f'     {pid}: Wells-based {a:12s} IM77-based {b:12s} {"AGREE" if a == b else ""}')
if pairs:
    A = [a for _, a, _ in pairs]; B = [b for _, _, b in pairs]
    null = []
    for _ in range(NPERM):
        rnd.shuffle(B); null.append(sum(1 for a, b in zip(A, B) if a == b))
    P(f'     pairing permuted: expected {sum(null)/NPERM:.2f}, P(>= {agree}) = {pval(agree, null):.3f}; '
      f'rule-of-three bound on disagreement if 0: < 3/{len(pairs)} = {3/len(pairs):.2f}')

# ---- (4) register features in IM77
OPEN_M = {267, 391, 293}; CLS_M = {342, 211, 15, 12, 254, 162, 169}; SUF_M = {176, 1}; PERSON_M = {1, 3}
def ffeat(s, bg, tf, sub):
    pairs = list(zip(s, s[1:]))
    return dict(opener=s[0] in OPEN_M, closer=s[-1] in CLS_M or (len(s) > 1 and s[-1] in SUF_M and s[-2] in CLS_M),
                person=any(x in PERSON_M for x in s), person_first=s[0] in PERSON_M,
                att=(sum(1 for p in pairs if bg[p] - sub > 0) / len(pairs)) if pairs else float('nan'),
                rare=sum((tf[x] - sub) < 5 for x in s) / len(s))
hd = dedup([dict(o, site=o['site']) for o in IMH])
bg = collections.Counter(); tf = collections.Counter()
for o in hd:
    for a, b in zip(o['seq'], o['seq'][1:]): bg[(a, b)] += 1
    for x in set(o['seq']): tf[x] += 1
seals = [o for o in hd if o['ot'] == 'seal' and len(o['seq']) >= 2]
WA2 = [w for w in WA if len(w['seq']) >= 2]
FE = ['opener', 'closer', 'person', 'person_first', 'att', 'rare']
def sstat(S, sub):
    F = [ffeat(o['seq'], bg, tf, sub) for o in S]
    return {k: sum(f[k] for f in F if f[k] == f[k]) / max(1, sum(1 for f in F if f[k] == f[k])) for k in FE} | {'distinct': len({x for o in S for x in o['seq']})}
obs = sstat(WA2, 0)
bl = collections.defaultdict(list)
for o in seals: bl[len(o['seq'])].append(o)
null = collections.defaultdict(list)
for _ in range(NPERM):
    d = [rnd.choice(near(len(w['seq']), bl)) for w in WA2]
    st = sstat(d, 1)
    for k, v in st.items(): null[k].append(v)
P(f'\n== (4) WA register (n = {len(WA2)}) vs IM77 home seals matched on length')
for k in FE + ['distinct']:
    v = sorted(null[k]); P(f'   {k:12s} obs {obs[k]:.3f} | null {sum(v)/NPERM:.3f} [{v[int(0.025*NPERM)]:.3f}-{v[int(0.975*NPERM)]:.3f}] P_hi {pval(obs[k], v):.3f} P_lo {pval(obs[k], v, "lo"):.3f}')
only2 = [w for w in only if len(w['seq']) >= 2]
if only2:
    o2 = sstat(only2, 0)
    P(f'   IM77-only WA texts (n = {len(only2)}): ' + ', '.join(f'{k} {o2[k]:.2f}' for k in FE) + f'; opener-initial {sum(w["seq"][0] in OPEN_M for w in only2)}/{len(only2)} (bound < 3/{len(only2)} = {3/len(only2):.2f} if 0)')
json.dump(dict(wa=[dict(w, seq=list(w['seq']), link=(links[w['id']]['site'] + ' ' + '-'.join(map(str, links[w['id']]['seq']))) if links[w['id']] else None, assign=wa_assign[w['id']]) for w in WA],
               pairs=pairs, agree=agree), open(OUT.replace('.txt', '.json'), 'w'), indent=0)
P(f'\nwritten {OUT}')
