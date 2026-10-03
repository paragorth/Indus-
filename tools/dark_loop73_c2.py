"""Loop 73 cycle 2: within-tablet structure and cross-document recurrence of the designation.
(2a) Within-tablet element sharing (the analogue of S-DARK-64's room test): every pair of entries on one PE tablet whose
     middles differ; share of pairs sharing >= 1 element, split by element frequency class (rare < 5 entries,
     mid 5-19, frequent >= 20, counted over all middles); identical middles on one tablet counted separately;
     adjacent entries vs non-adjacent. Null: middles permuted across tablets keeping each tablet's entry count,
     within the numeral-system stratum of the entry (500x). Calibration: Linear B personnel names on one tablet
     (syllables as elements), same code and null (within series).
(2b) Recurrence across documents: share of designation tokens (>= 2 elements) that also occur in another document
     (PE: another tablet; Indus: another object row, seals, seals + sealings; Ur III: another legend; Linear B: another
     tablet), at common n = 800 tokens (40 draws); ratio to an element-bigram generator trained on the same
     designations with the same lengths and document assignment (the test-e null). For recurring PE middles: share
     that recur with a different class sign / numeral system (same designation, different commodity).
Usage: python3 tools/dark_loop73_c2.py [nperm]
"""
import sys
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import *

R = {}
P(f'== Loop 73 cycle 2: within-tablet sharing and cross-document recurrence; nperm {NPERM}')

def pair_stats(groups, fcls):
    """groups: list of lists of (pos, mid). returns counts."""
    tot = sh = 0; rare = mid = freq = 0; ident = 0; adj_t = adj_s = 0; nadj_t = nadj_s = 0
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = g[i][1]; b = g[j][1]
                if a == b: ident += 1; continue
                tot += 1; S = set(a) & set(b)
                isadj = abs(g[i][0] - g[j][0]) == 1
                if isadj: adj_t += 1
                else: nadj_t += 1
                if S:
                    sh += 1
                    if isadj: adj_s += 1
                    else: nadj_s += 1
                    cl = {fcls[x] for x in S}
                    if 'rare' in cl: rare += 1
                    if 'mid' in cl: mid += 1
                    if 'freq' in cl: freq += 1
    return dict(pairs=tot, share=sh / tot if tot else float('nan'), rare=rare / tot if tot else 0, mid=mid / tot if tot else 0,
                freq=freq / tot if tot else 0, ident=ident, adj=adj_s / adj_t if adj_t else float('nan'),
                nadj=nadj_s / nadj_t if nadj_t else float('nan'), adj_n=adj_t)

def within(label, docs, minel=1):
    """docs: dict doc -> list of (pos, mid, stratum)."""
    items = [(d, p, m, s) for d, L in docs.items() for p, m, s in L if len(m) >= minel]
    cnt = collections.Counter(x for _, _, m, _ in items for x in set(m))
    fcls = {x: 'rare' if c < 5 else 'mid' if c < 20 else 'freq' for x, c in cnt.items()}
    def groups(its):
        g = collections.defaultdict(list)
        for d, p, m, s in its: g[d].append((p, m))
        return [v for v in g.values() if len(v) >= 2]
    obs = pair_stats(groups(items), fcls)
    r = random.Random(731); byS = collections.defaultdict(list)
    for i, (d, p, m, s) in enumerate(items): byS[s].append(i)
    nulls = []
    for _ in range(NPERM):
        mids = [m for _, _, m, _ in items]
        for s, idx in byS.items():
            vals = [mids[i] for i in idx]; r.shuffle(vals)
            for i, v in zip(idx, vals): mids[i] = v
        nulls.append(pair_stats(groups([(d, p, mm, s) for (d, p, _, s), mm in zip(items, mids)]), fcls))
    out = dict(obs=obs)
    line = f'  [{label}] docs with >= 2 designations {len(groups(items))}, non-identical pairs {obs["pairs"]}, identical pairs {obs["ident"]} (null {sum(x["ident"] for x in nulls)/NPERM:.1f}); '
    for k in ['share', 'rare', 'mid', 'freq', 'adj', 'nadj']:
        v = [x[k] for x in nulls if not math.isnan(x[k])]; mu = sum(v) / len(v)
        p = sum(1 for x in v if x >= obs[k]) / len(v)
        out[k] = dict(obs=obs[k], null=mu, ratio=obs[k] / mu if mu else float('nan'), p=p)
        line += f'{k} {obs[k]:.3f} vs {mu:.3f} (x{obs[k]/mu if mu else float("nan"):.2f}, P {p:.3f}); '
    vi = [x['ident'] for x in nulls]; out['ident'] = dict(obs=obs['ident'], null=sum(vi) / len(vi), p=sum(1 for x in vi if x >= obs['ident']) / len(vi))
    P(line + f'adjacent pairs {obs["adj_n"]}')
    return out

# ---------------- (2a) PE within-tablet ----------------
E = pe_entries()
by = collections.defaultdict(list)
for i, e in enumerate(E):
    by[e['tablet']].append((i, e['mid'], e['system'] or 'none'))
P('\n##### (2a) within-tablet element sharing (pairs of entries with different middles)')
R['PE_within_mid1'] = within('PE tablets, middles >= 1 element, null within numeral system', by, 1)
R['PE_within_mid2'] = within('PE tablets, middles >= 2 elements', by, 2)
by2 = collections.defaultdict(list)
for i, e in enumerate(E): by2[e['tablet']].append((i, e['mid'], 'all'))
R['PE_within_mid1_flat'] = within('PE tablets, middles >= 1, unstratified null', by2, 1)
# which elements carry the within-tablet sharing
pairs = collections.Counter()
for d, L in by.items():
    for a in range(len(L)):
        for b in range(a + 1, len(L)):
            if L[a][1] != L[b][1]:
                for x in set(L[a][1]) & set(L[b][1]): pairs[x] += 1
P('  PE elements most shared within tablets:', pairs.most_common(12))
# Linear B calibration
LB = linb_names_tab()
lbd = collections.defaultdict(list)
for i, x in enumerate(LB): lbd[x['tablet']].append((i, x['name'], x['series']))
R['LinB_within'] = within('Linear B tablets, personnel names (syllables as elements), null within series', lbd, 2)
P('  Indus comparison (S-DARK-64.1, same-room non-identical seal pairs, Mohenjo-daro): share >= 1 element 7.3-7.8% vs 4.1-4.8% null (1.6-1.9x), rare 0/218 (< 1.4%), Harappa tablets 0.7% vs 0.9-1.3%')

# ---------------- (2b) recurrence across documents ----------------
P('\n##### (2b) recurrence of the whole designation (>= 2 elements) across documents, n = 800 tokens x 40 draws; generator = element bigram (same lengths)')
def gen_bigram(des, r):
    big = collections.defaultdict(list); st = []
    for d in des:
        st.append(d[0])
        for a, b in zip(d, d[1:]): big[a].append(b)
    out = []
    for d in des:
        s = [r.choice(st)]
        while len(s) < len(d):
            nx = big.get(s[-1]); s.append(r.choice(nx) if nx else r.choice(st))
        out.append(tuple(s))
    return out
def recur(toks):
    """toks: list of (doc, des). share of tokens whose des occurs in another doc."""
    docs = collections.defaultdict(set)
    for d, m in toks: docs[m].add(d)
    return sum(1 for d, m in toks if len(docs[m]) >= 2) / len(toks)
def recurrence(label, toks, n=800, ndraw=40):
    obs = []; gen = []
    for b in range(ndraw):
        r = random.Random(7700 + b); sub = r.sample(toks, min(n, len(toks)))
        obs.append(recur(sub)); g = gen_bigram([m for _, m in sub], r); gen.append(recur([(d, m) for (d, _), m in zip(sub, g)]))
    o = q(obs, 0.5); g = q(gen, 0.5)
    P(f'  [{label}] tokens {len(toks)}, docs {len(set(d for d,_ in toks))}: recurring in another document {o:.3f} [{q(obs,0.025):.3f},{q(obs,0.975):.3f}]; bigram generator {g:.3f} [{q(gen,0.025):.3f},{q(gen,0.975):.3f}]; ratio {o/g if g else float("nan"):.2f}')
    return dict(obs=o, lo=q(obs, 0.025), hi=q(obs, 0.975), gen=g, glo=q(gen, 0.025), ghi=q(gen, 0.975), ratio=o / g if g else None, n=len(toks))
pe_t = [(e['tablet'], e['mid']) for e in E if len(e['mid']) >= 2]
R['rec_PE'] = recurrence('PE entry middles, document = tablet', pe_t)
R['rec_PE_mid3'] = recurrence('PE entry middles >= 3 elements', [(d, m) for d, m in pe_t if len(m) >= 3])
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    O = D56.load_indus(LV, dedup=False)
    for i, o in enumerate(O): o['doc'] = i
    seals = [(o['doc'], o['mid']) for o in O if o['ot'] == 'seal' and len(o['mid']) >= 2]
    R[f'rec_Indus_{LV}_seals'] = recurrence(f'Indus {LV} seals, document = object row', seals)
    ss = [(o['doc'], o['mid']) for o in O if o['ot'] in ('seal', 'sealing') and len(o['mid']) >= 2]
    R[f'rec_Indus_{LV}_seals_sealings'] = recurrence(f'Indus {LV} seals + sealings', ss)
    al = [(o['doc'], o['mid']) for o in O if len(o['mid']) >= 2]
    R[f'rec_Indus_{LV}_all'] = recurrence(f'Indus {LV} all objects', al)
    m3 = [(d, m) for d, m in seals if len(m) >= 3]
    R[f'rec_Indus_{LV}_seals_mid3'] = recurrence(f'Indus {LV} seals, middles >= 3', m3)
    # same designation in a different whole text (different frame) among seals
    full = collections.defaultdict(set)
    for o in O:
        if o['ot'] == 'seal' and len(o['mid']) >= 2: full[o['mid']].add(tuple(o['seq']))
    rec = [m for m, s in full.items() if len(s) >= 2]
    multi = sum(1 for o in O if o['ot'] == 'seal' and len(o['mid']) >= 2 and len(full[o['mid']]) >= 2)
    P(f'    {LV}: seal middles found in >= 2 DIFFERENT whole texts (same designation, different frame): {len(rec)} middles, {multi} seal rows; examples {rec[:6]}')
    R[f'rec_Indus_{LV}_diffframe'] = dict(middles=len(rec), rows=multi)
# IM77
O = D56.im77_objects()
seals = [(i, o['mid']) for i, o in enumerate(O) if o['ot'] == 'seal' and len(o['mid']) >= 2]
R['rec_IM77_seals_dedup'] = recurrence('IM77 seals (deduplicated per site x type x text: recurrence = across sites only)', seals)
U = []
for i, s in enumerate(jl(DARK + 'loop32_corpora/ur3_words.jsonl')):
    w = s[0]
    if w in UR3_TITLE or w.startswith('_') or 'x' in w.split('-') or '...' in w or '$' in w: continue
    t = tuple(x for x in w.split('-') if x)
    if len(t) >= 2: U.append((i, t))
R['rec_UrIII'] = recurrence('Ur III owner names, document = distinct legend', U)
lbt = [(x['tablet'], x['name']) for x in LB if len(x['name']) >= 2]
R['rec_LinB'] = recurrence('Linear B personnel names, document = tablet', lbt)
# PE: recurring middles with a different class sign / system
docs = collections.defaultdict(list)
for e in E:
    if len(e['mid']) >= 2: docs[e['mid']].append(e)
recm = {m: L for m, L in docs.items() if len(set(x['tablet'] for x in L)) >= 2}
dcls = sum(1 for L in recm.values() if len(set(x['cls'] for x in L)) >= 2); dsys = sum(1 for L in recm.values() if len(set(x['system'] for x in L)) >= 2)
P(f'  PE middles (>= 2 elements) on >= 2 tablets: {len(recm)} of {len(docs)} distinct; with a different class sign {dcls}, different numeral system {dsys}; commonest {sorted(((len(set(x["tablet"] for x in L)), m) for m, L in recm.items()), reverse=True)[:10]}')
R['PE_recurring'] = dict(n=len(recm), distinct=len(docs), diffcls=dcls, diffsys=dsys)
save('loop73_c2', R)
