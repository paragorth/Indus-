"""S-DARK-76 cycle 3: THE TEST. Person or label?
(A) Person test: if the designation names a person (one seal each, many acts), sealings (acts) must show MORE recurrence
    per designation than the seals of the same site. Statistic per site = recur(sealings) - recur(seals); null = the
    kind label shuffled among the site's seal + sealing documents (1,000x, document counts kept); summed over sites
    (weights = sealing tokens). Same for tablets vs seals (Harappa, Mohenjo-daro). Act and die level, loose / strict.
    Ur III control: sealings (owner over tablets) vs seals (owner over distinct legends) per site, same code.
(B) What does recurrence follow? Over all pairs of documents sharing a designation: share with the same HEAD (closer
    group), the same FRAME (opener/marker + head, i.e. the whole text minus the designation), the same SITE.
    Null N1 = designations shuffled among documents within site x type (Wells type code), 1,000x -> expected same head /
    frame; null N2 = within type only -> expected same site. ratio = observed / null mean.
    Comparators with the same code: Linear B personnel lines (frame = ideogram, stratum = site x series, DAMOS via
    tools/dark_loop73_common.linb_lines), Ur III administrative names (frame = formula slot, stratum = site), Ur III seals
    (frame = rest of the legend, head = second word; stratum = site), 200x for the comparators.
Usage: python3 tools/dark_loop76_c3.py [nperm]   (writes data/derived/dark/loop76_c3.txt)"""
import sys; sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop76_common import *
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

# ---------------- (A) ----------------
def site_diff(docs, ka, kb, r=None, labels=None):
    """recur(kind ka) - recur(kind kb) on one site's docs; labels = optional permuted kinds."""
    ta = []; tb = []
    for i, d in enumerate(docs):
        k = labels[i] if labels else d['kind']
        seen = set()
        for f in d['faces']:
            m = f['mid']
            if len(m) < 2 or m in seen or f['count_face']: continue
            seen.add(m)
            (ta if k == ka else tb).append((i, m))
    return recur_share(ta) - recur_share(tb), len(ta), len(tb), recur_share(ta), recur_share(tb)

def person_test(label, D, ka, kb, sites):
    tot_o = 0; tot_n = []; rows = []
    per = {}
    for s in sites:
        Ds = [d for d in D if d['site'] == s and d['kind'] in (ka, kb)]
        o, na, nb, ra, rb = site_diff(Ds, ka, kb)
        if na < 5 or nb < 5 or math.isnan(o): continue
        lab = [d['kind'] for d in Ds]; r = random.Random(76)
        null = []
        for _ in range(NP):
            r.shuffle(lab); null.append(site_diff(Ds, ka, kb, labels=lab)[0])
        null = [x for x in null if not math.isnan(x)]
        p = (1 + sum(1 for x in null if x >= o)) / (1 + len(null))
        mu = sum(null) / len(null)
        per[s] = dict(obs=o, null=mu, p=p, na=na, nb=nb, ra=ra, rb=rb, nulls=null)
        rows.append(f'{s} {ka} {ra:.3f} (n {na}) vs {kb} {rb:.3f} (n {nb}): diff {o:+.3f} vs null {mu:+.3f}, P(>=) {p:.3f}')
    if per:
        W = sum(v['na'] for v in per.values())
        obs = sum(v['obs'] * v['na'] for v in per.values()) / W
        L = min(len(v['nulls']) for v in per.values())
        nul = [sum(v['nulls'][j] * v['na'] for v in per.values()) / W for j in range(L)]
        p = (1 + sum(1 for x in nul if x >= obs)) / (1 + L)
        P(f'  [{label}] pooled (weights = {ka} tokens) diff {obs:+.3f} vs null {sum(nul)/L:+.3f}, P {p:.3f}; ' + ' | '.join(rows))
        return dict(obs=obs, null=sum(nul) / L, p=p, per={k: {a: b for a, b in v.items() if a != 'nulls'} for k, v in per.items()})
    P(f'  [{label}] too few'); return None

# ---------------- (B) ----------------
def pair_rates(items):
    """items: list of dict(des, head, frame, site). Over all pairs sharing a designation: share same head / frame / site."""
    g = collections.defaultdict(list)
    for x in items: g[x['des']].append(x)
    tot = 0; sh = 0; sf = 0; ss = 0
    for L in g.values():
        n = len(L)
        if n < 2: continue
        tot += n * (n - 1) // 2
        for key, acc in (('head', 0), ('frame', 1), ('site', 2)):
            c = collections.Counter(x[key] for x in L); v = sum(k * (k - 1) // 2 for k in c.values())
            if acc == 0: sh += v
            elif acc == 1: sf += v
            else: ss += v
    return (sh / tot, sf / tot, ss / tot, tot) if tot else (float('nan'),) * 3 + (0,)

def follow_test(label, items, nperm):
    """items need des, head, frame, site, stratum1 (site x type), stratum2 (type)."""
    o = pair_rates(items)
    def shuffled(key, r):
        by = collections.defaultdict(list)
        for i, x in enumerate(items): by[x[key]].append(i)
        des = [x['des'] for x in items]
        for idx in by.values():
            vals = [des[i] for i in idx]; r.shuffle(vals)
            for i, v in zip(idx, vals): des[i] = v
        return [dict(x, des=d) for x, d in zip(items, des)]
    r = random.Random(7676); n1 = []; n2 = []
    for _ in range(nperm):
        n1.append(pair_rates(shuffled('stratum1', r))); n2.append(pair_rates(shuffled('stratum2', r)))
    def stat(j, nl):
        v = [x[j] for x in nl if not math.isnan(x[j])]
        mu = sum(v) / len(v) if v else float('nan')
        p = (1 + sum(1 for x in v if x >= o[j])) / (1 + len(v))
        return mu, p
    hm, hp = stat(0, n1); fm, fp = stat(1, n1); sm, sp = stat(2, n2)
    P(f'  [{label}] docs {len(items)}, pairs sharing a designation {o[3]}: same head {o[0]:.3f} vs {hm:.3f} (x{o[0]/hm if hm else float("nan"):.2f}, P {hp:.3f}); '
      f'same frame {o[1]:.3f} vs {fm:.3f} (x{o[1]/fm if fm else float("nan"):.2f}, P {fp:.3f}); same site {o[2]:.3f} vs {sm:.3f} (x{o[2]/sm if sm else float("nan"):.2f}, P {sp:.3f})')
    return dict(pairs=o[3], head=o[0], head_null=hm, head_p=hp, frame=o[1], frame_null=fm, frame_p=fp, site=o[2], site_null=sm, site_p=sp)

def indus_items(D, kinds):
    out = []
    for i, m, f in tokens(D, 2, kinds):
        d = D[i]
        out.append(dict(des=m, head=f['hgroup'], frame=(f['pre'], f['head']), site=d['site'], stratum1=(d['site'], d['type']), stratum2=d['type']))
    return out

R = {}
P(f'== S-DARK-76 cycle 3: person test and what recurrence follows; nperm {NP}')
P('#### (A) person test: sealings (acts) vs seals (instruments) of the same site; tablets vs seals')
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    for filt in ['loose', 'strict']:
        D0 = wells_docs(LV, filt)
        for lev in ['act', 'die']:
            D = collapse_level(D0, lev)
            R[f'A_{LV}_{filt}_{lev}_sealing'] = person_test(f'Wells {LV} {filt} {lev}: sealing vs seal', D, 'sealing', 'seal', ['Lothal', 'Kalibangan', 'Mohenjo-daro', 'Harappa', 'Dholavira'])
            R[f'A_{LV}_{filt}_{lev}_sealing_noMDH'] = person_test(f'Wells {LV} {filt} {lev}: sealing vs seal, Lothal + Kalibangan + Dholavira only', D, 'sealing', 'seal', ['Lothal', 'Kalibangan', 'Dholavira'])
            for tk in ('tablet_m', 'tablet_i'):
                R[f'A_{LV}_{filt}_{lev}_{tk}'] = person_test(f'Wells {LV} {filt} {lev}: {tk} vs seal', D, tk, 'seal', ['Mohenjo-daro', 'Harappa'])
I = im77_docs('strict')
R['A_im77'] = person_test('IM77 strict act: sealing vs seal (Lothal, Kalibangan; IM77 Harappa / MD "sealings" are moulded objects)', I, 'sealing', 'seal', ['Lothal', 'Kalibangan'])
# Ur III control, per site (sealings over tablets vs distinct legends)
SL = ur3_sealings(); SE = ur3_seals()
uD = []
for x in SL: uD.append(dict(site=x['site'], kind='sealing', faces=[dict(mid=('U',) + (x['des'],), count_face=False)]))
for x in SE: uD.append(dict(site=x['site'], kind='seal', faces=[dict(mid=('U',) + (x['des'],), count_face=False)]))
# subsample each site to Lothal-like sizes so the test has the Indus power: 60 sealings + 60 seals per site, 5 sites
rr = random.Random(5)
sub_u = []
for s in ['Umma', 'Girsu', 'Puzriš-Dagan', 'Nippur', 'Garšana']:
    a = [d for d in uD if d['site'] == s and d['kind'] == 'sealing']; b = [d for d in uD if d['site'] == s and d['kind'] == 'seal']
    rr.shuffle(a); rr.shuffle(b); sub_u += a[:60] + b[:60]
R['A_ur3_small'] = person_test('Ur III CONTROL, 60 sealings + 60 seals per site (Indus-sized)', sub_u, 'sealing', 'seal', ['Umma', 'Girsu', 'Puzriš-Dagan', 'Nippur', 'Garšana'])

P('\n#### (B) what recurrence follows: head / frame (null N1 within site x type) and site (null N2 within type)')
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    for filt in ['loose', 'strict']:
        D0 = wells_docs(LV, filt)
        for lev in ['act', 'die']:
            D = collapse_level(D0, lev)
            for k in [('seal',), ('sealing',), ('tablet_m',), ('tablet_i',), ('seal', 'sealing', 'tablet_m', 'tablet_i')]:
                it = indus_items(D, set(k))
                if sum(1 for _ in it) < 20: continue
                R[f'B_{LV}_{filt}_{lev}_{"+".join(k)}'] = follow_test(f'Wells {LV} {filt} {lev} {"+".join(k)}', it, NP if LV == 'seq_raw' else NP // 5)
        if LV == 'seq_raw' and filt == 'loose':
            D = collapse_level(D0, 'die')
            for nm, sub_ in [('seals MD + Harappa', lambda d: d['big']), ('seals other sites', lambda d: not d['big'])]:
                it = indus_items([d for d in D if sub_(d)], {'seal'})
                R[f'B_seal_{nm}'] = follow_test(f'Wells seq_raw loose die {nm}', it, NP)
it = indus_items([d for d in I if not (d['kind'] == 'sealing' and d['site'] not in ('Lothal', 'Kalibangan'))], {'seal'})
R['B_im77_seal'] = follow_test('IM77 strict seals', it, NP)
it = indus_items(I, {'tablet_m', 'tablet_c'}) + indus_items([d for d in I if d['kind'] == 'sealing' and d['site'] == 'Harappa'], {'sealing'})
R['B_im77_tab'] = follow_test('IM77 strict tablets (miniature + copper + Harappa moulded "sealings")', it, NP)

P('  -- comparators (200x)')
import dark_loop73_common as L73
LB = L73.linb_lines()
it = []
seen = set()
for x in LB:
    ser = x['tablet'].split()[1] if len(x['tablet'].split()) > 1 else '?'
    ser = re.sub(r'\(.*', '', ser)
    k = (x['tablet'], x['name'])
    if k in seen: continue
    seen.add(k)
    ideo = next((t for t in x['frame'] if t.isupper() and t != 'NUM'), 'none')
    it.append(dict(des=x['name'], head=ideo, frame=x['frame'], site=x['site'], stratum1=(x['site'], ser), stratum2=ser))
R['B_linb'] = follow_test('Linear B personnel lines (head = ideogram, frame = rest of the line, stratum site x series)', it, 200)
A = ur3_admin(); r = random.Random(3); A = r.sample(A, 20000)
it = [dict(des=x['des'], head=x['frame'][0], frame=x['frame'], site=x['site'], stratum1=x['site'], stratum2='all') for x in A]
R['B_ur3_admin'] = follow_test('Ur III administrative names (random 20,000; head = frame = formula slot)', it, 200)
it = [dict(des=x['des'], head=(x['frame'][0] if x['frame'] else 'none'), frame=x['frame'], site=x['site'], stratum1=x['site'], stratum2='all') for x in SE]
R['B_ur3_seals'] = follow_test('Ur III seals (distinct legends; head = 2nd word, frame = rest of legend)', it, 200)
json.dump(R, open(DARK + 'loop76_c3.json', 'w'), indent=1, default=str)
save('loop76_c3')
