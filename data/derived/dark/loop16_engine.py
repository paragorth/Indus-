"""Dark loop 16: USAGE INTENSITY. Clay sealings (TAG*) as impressions of seals.
Cycles: 1 impressions-per-text spectrum + heavy-use tail; 2 ghost issuers; 3 dead seals;
4 condition as wear; 5 multi-sided tablets. Usage: python3 loop16_engine.py <cycle> [level] [nperm]
Writes data/derived/dark/loop16_cycle<N>_<level>.txt
"""
import csv, json, re, sys, random, collections, statistics as st, math
ROOT = '/home/user/Indus-'
CYCLE = int(sys.argv[1]); LEVEL = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
random.seed(16)
OUT = open(f'{ROOT}/data/derived/dark/loop16_cycle{CYCLE}_{LEVEL}.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')

# ---------- load + align ----------
rows = list(csv.DictReader(open(f'{ROOT}/data/raw/inscriptions.csv')))
C = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
def parse(t): return tuple(int(x) for x in re.findall(r'\d+', t))
def norm(s): return tuple(x for x in s if x != 0)
idx = collections.defaultdict(list)
for i, r in enumerate(rows):
    p = norm(parse(r['text']))
    idx[(r['cisi'], r['type'], p)].append(i); idx[(r['cisi'], r['type'], p[::-1])].append(i)
used = set(); R = []
for c in C:
    k = (c['cisi'], c['type'], norm(tuple(c['seq_raw'])))
    cands = [i for i in idx.get(k, []) if i not in used]
    assert cands, k
    used.add(cands[0]); raw = rows[cands[0]]
    rec = dict(c); rec['raw'] = raw
    rec['obj'] = raw['id'].split('.')[0]; rec['line'] = int(raw['id'].split('.')[1])
    rec['sides'] = int(raw['sides']) if raw['sides'].isdigit() else 1
    cond = raw['condition'].strip().lower()
    rec['cond'] = {'poor': 0, 'fair': 1, 'good': 2, 'fine': 3}.get(cond, None)
    pres = raw['preservation'].strip().lower()
    rec['pres'] = pres
    def fl(x):
        try: v = float(x); return v if v > 0 else None
        except: return None
    rec['h'] = fl(raw['horizontal(mm)']); rec['v'] = fl(raw['vertical(mm)'])
    rec['area'] = rec['h'] * rec['v'] if rec['h'] and rec['v'] else None
    rec['mat'] = raw['material'].strip().capitalize()
    rec['emb'] = raw['symbol']
    rec['cls'] = raw['class']
    rec['t'] = tuple(c[LEVEL])
    rec['otype'] = c['type'].split(':')[0]  # SEAL, TAB, TAG, POT ...
    R.append(rec)

NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34}; TREE = {390, 405, 407}
OPEN = {817, 861, 820, 920, 692}
JAR = {740, 741, 742, 745}
def feats(t):
    f = {}
    f['len'] = len(t)
    f['jar_final'] = int(t[-1] in JAR)
    f['arrow_final'] = int(t[-1] == 520)
    f['opener_first'] = int(t[0] in OPEN)
    f['any_numeral'] = int(any(x in NUM for x in t))
    f['stock'] = int(any(t[i] in TREE and t[i - 1] in NUM for i in range(1, len(t))))
    f['voucher700'] = int(700 in t)
    f['fish'] = int(any(x in {220, 231, 232, 233, 235, 240} for x in t))
    return f
FEATS = ['len', 'jar_final', 'arrow_final', 'opener_first', 'any_numeral', 'stock', 'voucher700', 'fish']

def perm_p(obs, sims):
    n = len(sims); lo = sum(1 for s in sims if s <= obs); hi = sum(1 for s in sims if s >= obs)
    return (lo + 1) / (n + 1), (hi + 1) / (n + 1)
def fmt_p(obs, sims):
    lo, hi = perm_p(obs, sims); m = st.mean(sims)
    return f'{obs:+.3f} (null {m:+.3f}, p_lo {lo:.3f}, p_hi {hi:.3f})'

TAG = [r for r in R if r['otype'] == 'TAG' and len(r['t']) >= 1]
SEAL = [r for r in R if r['otype'] == 'SEAL' and len(r['t']) >= 1]
TAB = [r for r in R if r['otype'] == 'TAB' and len(r['t']) >= 1]
P(f'== loop16 cycle {CYCLE} level={LEVEL} nperm={NPERM}; lines TAG={len(TAG)} (objects {len({r["obj"] for r in TAG})}) SEAL={len(SEAL)} TAB={len(TAB)}')
BIG = ['Lothal', 'Dholavira', 'Kalibangan', 'Harappa', 'Mohenjo-daro']

def spectrum(texts):
    c = collections.Counter(texts); f = collections.Counter(c.values())
    return c, f
def zipf_slope(c):
    fr = sorted(c.values(), reverse=True)
    if len(fr) < 5 or fr[0] < 2: return None
    xs = [math.log(i + 1) for i in range(len(fr))]; ys = [math.log(v) for v in fr]
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
def chao1(c):
    f = collections.Counter(c.values()); f1, f2 = f[1], f[2]; S = len(c)
    return S + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)

# =========================== CYCLE 1 ===========================
if CYCLE == 1:
    P('# (a) impressions per distinct text on sealings; spectrum vs seals/tablets subsampled to same n within site')
    for minlen in (1, 2):
        P(f'\n-- text identity: >= {minlen} signs')
        for site in BIG:
            tg = [r['t'] for r in TAG if r['site'] == site and len(r['t']) >= minlen]
            if len(tg) < 10: continue
            c, f = spectrum(tg)
            spec = ' '.join(f'f{k}={f[k]}' for k in sorted(f))
            P(f'{site:13s} TAG lines={len(tg)} distinct={len(c)} singletons={f[1]} max={max(c.values())} chao1={chao1(c):.0f} zipf={zipf_slope(c) or float("nan"):.2f}  [{spec}]')
            P(f'   top: ' + '; '.join(f'{"-".join(map(str,t))} x{n}' for t, n in c.most_common(4)))
            for lab, pool in (('SEAL', SEAL), ('TAB', TAB)):
                pl = [r['t'] for r in pool if r['site'] == site and len(r['t']) >= minlen]
                if len(pl) < len(tg): P(f'   {lab}: pool {len(pl)} < TAG n, skip'); continue
                obs_d, obs_s, obs_max = len(c), f[1] / len(c), max(c.values())
                sims = [spectrum(random.sample(pl, len(tg)))[0] for _ in range(1000)]
                sd = [len(s) for s in sims]; ss = [collections.Counter(s.values())[1] / len(s) for s in sims]; sm = [max(s.values()) for s in sims]
                P(f'   vs {lab} (pool {len(pl)}, sub-sampled to {len(tg)}): distinct {obs_d} vs {st.mean(sd):.1f} p_lo={perm_p(obs_d,sd)[0]:.3f}; singleton share {obs_s:.2f} vs {st.mean(ss):.2f} p_lo={perm_p(obs_s,ss)[0]:.3f}; max copies {obs_max} vs {st.mean(sm):.1f} p_hi={perm_p(obs_max,sm)[1]:.3f}')
    P('\n# heavy-use tail: features of distinct TAG texts (>=2 signs) by copy count; null = copy counts shuffled among distinct texts within site')
    arrows = 0; results = collections.defaultdict(dict)
    sites_used = []
    for site in BIG + ['ALL']:
        tg = [r['t'] for r in TAG if (site == 'ALL' or r['site'] == site) and len(r['t']) >= 2]
        c = spectrum(tg)[0]
        texts = list(c); counts = [c[t] for t in texts]
        if sum(1 for x in counts if x >= 2) < 2: P(f'{site}: fewer than 2 repeated texts ({sum(1 for x in counts if x>=2)}), skip'); continue
        sites_used.append(site)
        F = [feats(t) for t in texts]
        def stat(cnt, name):  # mean feature in repeated minus mean in singletons
            a = [F[i][name] for i in range(len(texts)) if cnt[i] >= 2]; b = [F[i][name] for i in range(len(texts)) if cnt[i] == 1]
            return st.mean(a) - st.mean(b) if a and b else 0
        P(f'{site}: distinct={len(texts)} repeated={sum(1 for x in counts if x>=2)} (lines in repeated {sum(x for x in counts if x>=2)})')
        for name in FEATS:
            obs = stat(counts, name)
            sims = []
            for _ in range(NPERM):
                sh = counts[:]; random.shuffle(sh); sims.append(stat(sh, name))
            results[name][site] = (obs, perm_p(obs, sims))
            arrows += 1
            P(f'   {name:13s} repeated-minus-single: {fmt_p(obs, sims)}')
    P(f'\n arrows fired {arrows}; Bonferroni alpha 0.05/{arrows} = {0.05/arrows:.4f}')
    P(' held-out sign agreement across sites (direction of repeated-minus-single):')
    for name in FEATS:
        signs = {s: ('+' if v[0] > 0 else '-' if v[0] < 0 else '0') for s, v in results[name].items() if s != 'ALL'}
        P(f'   {name:13s} ' + ' '.join(f'{s[:3]}:{signs[s]}' for s in signs) + f"  ALL p_lo/p_hi={results[name].get('ALL',(0,(1,1)))[1]}")

# =========================== CYCLE 2 ===========================
if CYCLE == 2:
    P('# (b) ghost issuers: distinct sealing texts with no matching seal text anywhere')
    seal_texts = collections.defaultdict(set)  # text -> sites
    for r in SEAL:
        if len(r['t']) >= 2: seal_texts[r['t']].add(r['site'])
    seal_list = list(seal_texts)
    tab_texts = {r['t'] for r in TAB if len(r['t']) >= 2}
    def contained(t):
        n = len(t)
        for s in seal_list:
            if len(s) > n and any(s[i:i + n] == t for i in range(len(s) - n + 1)): return True
        return False
    for minlen in (2, 3):
        P(f'\n-- sealing texts >= {minlen} signs')
        tot = collections.Counter()
        for site in BIG + ['other', 'ALL']:
            tg = [r for r in TAG if len(r['t']) >= minlen and (site == 'ALL' or (site == 'other' and r['site'] not in BIG) or r['site'] == site)]
            dist = {}
            for r in tg: dist.setdefault(r['t'], []).append(r['site'])
            if not dist: continue
            n = len(dist)
            ex = sum(1 for t in dist if t in seal_texts)
            same = sum(1 for t in dist if t in seal_texts and seal_texts[t] & set(dist[t]))
            cont = sum(1 for t in dist if t not in seal_texts and contained(t))
            ontab = sum(1 for t in dist if t not in seal_texts and t in tab_texts)
            ghost = n - ex
            P(f'{site:13s} distinct={n} lines={len(tg)} exact-seal-match={ex} (same-site {same}) ghost={ghost} ({100*ghost/n:.0f}%); of ghosts: inside a longer seal text {cont}, on a tablet {ontab}, nowhere {ghost-cont-ontab}')
        # chance-match control: seal texts scrambled within text (as S324) -> expected exact matches
        tgd = {r['t'] for r in TAG if len(r['t']) >= minlen}
        sims = []
        for _ in range(300):
            scr = set()
            for s in seal_list:
                l = list(s); random.shuffle(l); scr.add(tuple(l))
            sims.append(sum(1 for t in tgd if t in scr))
        P(f'   chance control (seal texts scrambled, 300x): exact matches expected {st.mean(sims):.1f} (max {max(sims)}) vs observed {sum(1 for t in tgd if t in seal_texts)}')
    # Lincoln-Petersen two-sample estimate of distinct seal-text population
    P('\n# capture-recapture (Lincoln-Petersen, Chapman form) on distinct texts >= 2 signs: N = (S+1)(T+1)/(m+1) - 1')
    for scope in ['ALL', 'within-site']:
        if scope == 'ALL':
            S = len(seal_list); T = len({r['t'] for r in TAG if len(r['t']) >= 2}); m = sum(1 for t in {r['t'] for r in TAG if len(r['t']) >= 2} if t in seal_texts)
            P(f'ALL (any site): S={S} T={T} m={m} -> N={(S+1)*(T+1)/(m+1)-1:.0f}  (S309 Chao1 texts; corpus distinct texts 3599)')
        else:
            for site in BIG:
                S_ = {r['t'] for r in SEAL if r['site'] == site and len(r['t']) >= 2}; T_ = {r['t'] for r in TAG if r['site'] == site and len(r['t']) >= 2}
                m_ = len(S_ & T_)
                P(f'{site:13s} S={len(S_)} T={len(T_)} m={m_} -> N={(len(S_)+1)*(len(T_)+1)/(m_+1)-1:.0f} (m=0 -> lower bound only)')
    P('\n# ghosts vs matched: features; null = matched label permuted among distinct sealing texts within site')
    arrows = 0; res = collections.defaultdict(dict)
    for site in BIG + ['ALL']:
        dist = list({r['t'] for r in TAG if len(r['t']) >= 2 and (site == 'ALL' or r['site'] == site)})
        lab = [int(t in seal_texts) for t in dist]
        if sum(lab) < 3 or len(lab) - sum(lab) < 3: P(f'{site}: matched={sum(lab)} of {len(lab)}, too few, skip'); continue
        F = [feats(t) for t in dist]
        def stat(l, name):
            a = [F[i][name] for i in range(len(dist)) if l[i] == 0]; b = [F[i][name] for i in range(len(dist)) if l[i] == 1]
            return st.mean(a) - st.mean(b)
        P(f'{site}: distinct={len(dist)} matched={sum(lab)} ghost={len(lab)-sum(lab)}')
        for name in FEATS:
            obs = stat(lab, name); sims = []
            for _ in range(NPERM):
                sh = lab[:]; random.shuffle(sh); sims.append(stat(sh, name))
            res[name][site] = (obs, perm_p(obs, sims)); arrows += 1
            P(f'   {name:13s} ghost-minus-matched: {fmt_p(obs, sims)}')
    P(f'\n arrows fired {arrows}; Bonferroni alpha = {0.05/max(arrows,1):.4f}')
    for name in FEATS:
        P(f'   {name:13s} ' + ' '.join(f'{s[:3]}:{"+" if v[0]>0 else "-" if v[0]<0 else "0"}' for s, v in res[name].items()))
    # vocabulary: signs over-represented in ghost texts vs matched (ALL)
    dist = list({r['t'] for r in TAG if len(r['t']) >= 2})
    g = collections.Counter(x for t in dist if t not in seal_texts for x in set(t)); mt = collections.Counter(x for t in dist if t in seal_texts for x in set(t))
    ng = sum(1 for t in dist if t not in seal_texts); nm = len(dist) - ng
    P(f'\n# sign presence in ghost ({ng}) vs matched ({nm}) distinct sealing texts (signs with >= 5 ghost texts):')
    for x, n in g.most_common(25):
        if n >= 5: P(f'   W{x}: ghost {n}/{ng} ({100*n/ng:.0f}%) matched {mt[x]}/{nm} ({100*mt[x]/max(nm,1):.0f}%)')

# =========================== CYCLE 3 ===========================
if CYCLE == 3:
    P('# (c) dead seals: seal lines whose text (>=2 signs) never appears on any TAG or TAB line (any site)')
    use_texts = {r['t'] for r in TAG + TAB if len(r['t']) >= 2}
    seals = [r for r in SEAL if len(r['t']) >= 2 and r['type'] in ('SEAL:S', 'SEAL:R', 'SEAL')]
    for r in seals: r['used'] = int(r['t'] in use_texts)
    P(f'seal lines {len(seals)}; with attested use {sum(r["used"] for r in seals)}')
    for site in BIG:
        s = [r for r in seals if r['site'] == site]
        P(f'   {site:13s} seals={len(s)} used={sum(r["used"] for r in s)} ({100*sum(r["used"] for r in s)/max(len(s),1):.1f}%)  used-on-TAG {sum(1 for r in s if r["t"] in {q["t"] for q in TAG})} used-on-TAB {sum(1 for r in s if r["t"] in {q["t"] for q in TAB})}')
    def uni(e): return int(e.startswith('Bull1'))
    def noemb(e): return int(e in ('', '-', 'None'))
    def numfeat(r):
        f = feats(r['t'])
        f['area_mm2'] = r['area']; f['h_mm'] = r['h']
        f['cond'] = r['cond']; f['chipped_or_frag'] = int(r['pres'] not in ('complete', '-', 'complsete'))
        f['unicorn'] = uni(r['emb']); f['no_emblem'] = noemb(r['emb'])
        f['steatite'] = int(r['mat'] == 'Steatite'); f['rect_bar'] = int(r['type'] == 'SEAL:R')
        return f
    NF = FEATS + ['area_mm2', 'h_mm', 'cond', 'chipped_or_frag', 'unicorn', 'no_emblem', 'steatite', 'rect_bar']
    F = [numfeat(r) for r in seals]
    strata = collections.defaultdict(list)
    for i, r in enumerate(seals): strata[(r['site'], r['type'])].append(i)
    def stat(lab, name):
        a = [F[i][name] for i in range(len(seals)) if lab[i] == 1 and F[i][name] is not None]
        b = [F[i][name] for i in range(len(seals)) if lab[i] == 0 and F[i][name] is not None]
        return (st.mean(a) - st.mean(b), len(a), len(b)) if a and b else (0, len(a), len(b))
    lab = [r['used'] for r in seals]
    P('\n# used-minus-dead mean difference; null = used label permuted within site x seal type')
    arrows = 0
    for name in NF:
        obs, na, nb = stat(lab, name); sims = []
        for _ in range(NPERM):
            sh = lab[:]
            for ids in strata.values():
                vals = [sh[i] for i in ids]; random.shuffle(vals)
                for i, v in zip(ids, vals): sh[i] = v
            sims.append(stat(sh, name)[0])
        arrows += 1
        P(f'   {name:15s} (n used {na}, dead {nb}): {fmt_p(obs, sims)}')
    P(f' arrows {arrows}; Bonferroni alpha {0.05/arrows:.4f}')
    P('\n# held-out: same stats per site (Mohenjo-daro, Harappa, rest) for len, jar_final, any_numeral, area_mm2, cond, unicorn, no_emblem')
    for grp, sel in (('Mohenjo-daro', lambda r: r['site'] == 'Mohenjo-daro'), ('Harappa', lambda r: r['site'] == 'Harappa'), ('other sites', lambda r: r['site'] not in ('Mohenjo-daro', 'Harappa'))):
        ids = [i for i, r in enumerate(seals) if sel(r)]
        if sum(lab[i] for i in ids) < 3: P(f'   {grp}: used {sum(lab[i] for i in ids)}, skip'); continue
        out = []
        for name in ['len', 'jar_final', 'any_numeral', 'area_mm2', 'cond', 'unicorn', 'no_emblem']:
            a = [F[i][name] for i in ids if lab[i] == 1 and F[i][name] is not None]; b = [F[i][name] for i in ids if lab[i] == 0 and F[i][name] is not None]
            out.append(f'{name} {st.mean(a)-st.mean(b):+.2f}' if a and b else f'{name} na')
        P(f'   {grp:13s} used={sum(lab[i] for i in ids)} dead={len(ids)-sum(lab[i] for i in ids)}: ' + ', '.join(out))
    P('\n# the used seals (text, site, type, emblem, material, size, condition, where used):')
    tag_sites = collections.defaultdict(set); tab_sites = collections.defaultdict(set)
    for r in TAG: tag_sites[r['t']].add(r['site'])
    for r in TAB: tab_sites[r['t']].add(r['site'])
    for r in sorted((r for r in seals if r['used']), key=lambda r: (r['site'], r['t'])):
        P(f"   {'-'.join(map(str,r['t'])):28s} {r['cisi']:8s} {r['site'][:12]:12s} {r['type']:7s} emb={r['emb'] or '-':9s} {r['mat'][:8]:8s} {r['h'] or 0:.0f}x{r['v'] or 0:.0f}mm cond={r['raw']['condition'].strip()} TAG@{','.join(sorted(tag_sites[r['t']])) or '-'} TAB@{','.join(sorted(tab_sites[r['t']])) or '-'}")

# =========================== CYCLE 4 ===========================
if CYCLE == 4:
    P('# (d) condition as wear: seals (SEAL:S, SEAL:R), condition Poor=0 Fair=1 Good=2 Fine=3; preservation; text class; strata = site x material')
    seals = [r for r in SEAL if len(r['t']) >= 1 and r['type'] in ('SEAL:S', 'SEAL:R') and r['cond'] is not None]
    def tclass(t):
        if any(t[i] in TREE and t[i - 1] in NUM for i in range(1, len(t))): return 'stock'
        if t[-1] in JAR: return 'office_jar'
        if t[-1] == 520: return 'arrow'
        if any(x in NUM for x in t): return 'counted_other'
        return 'other'
    for r in seals: r['tc'] = tclass(r['t']); r['dam'] = int(r['pres'] not in ('complete', '-', 'complsete'))
    P(f'seals {len(seals)}; class counts {collections.Counter(r["tc"] for r in seals)}')
    P(f'condition counts {collections.Counter(r["cond"] for r in seals)}; damaged(pres not complete) {sum(r["dam"] for r in seals)}')
    P('\n# raw means by class (ALL):')
    for tc in ['office_jar', 'stock', 'counted_other', 'arrow', 'other']:
        s = [r for r in seals if r['tc'] == tc]
        if s: P(f'   {tc:14s} n={len(s)} cond={st.mean(r["cond"] for r in s):.2f} damaged={st.mean(r["dam"] for r in s):.2f} len={st.mean(len(r["t"]) for r in s):.2f}')
    strata = collections.defaultdict(list)
    for i, r in enumerate(seals): strata[(r['site'], r['mat'], r['type'])].append(i)
    def mean_by(lab, vals, key):
        a = [vals[i] for i in range(len(seals)) if lab[i] == key]
        return st.mean(a) if a else 0
    arrows = 0
    for ycol in ['cond', 'dam']:
        y = [r[ycol] for r in seals]
        labs = [r['tc'] for r in seals]
        P(f'\n# outcome {ycol}: class mean minus rest, null = class labels permuted within site x material x type ({NPERM}x)')
        for tc in ['office_jar', 'stock', 'counted_other', 'arrow']:
            def stat(l):
                a = [y[i] for i in range(len(seals)) if l[i] == tc]; b = [y[i] for i in range(len(seals)) if l[i] != tc]
                return st.mean(a) - st.mean(b) if a and b else 0
            obs = stat(labs); sims = []
            for _ in range(NPERM):
                sh = labs[:]
                for ids in strata.values():
                    vals = [sh[i] for i in ids]; random.shuffle(vals)
                    for i, v in zip(ids, vals): sh[i] = v
                sims.append(stat(sh))
            arrows += 1
            P(f'   {tc:14s} n={labs.count(tc)}: {fmt_p(obs, sims)}')
        # length: Spearman-ish via mean condition per length bin and permutation of length within strata
        L = [len(r['t']) for r in seals]
        def corr(l):
            ml, my = st.mean(l), st.mean(y); sx = st.pstdev(l); sy = st.pstdev(y)
            return sum((a - ml) * (b - my) for a, b in zip(l, y)) / (len(l) * sx * sy) if sx and sy else 0
        obs = corr(L); sims = []
        for _ in range(NPERM):
            sh = L[:]
            for ids in strata.values():
                vals = [sh[i] for i in ids]; random.shuffle(vals)
                for i, v in zip(ids, vals): sh[i] = v
            sims.append(corr(sh))
        arrows += 1
        P(f'   text length r with {ycol}: {fmt_p(obs, sims)}')
    P(f'\n arrows {arrows}; Bonferroni alpha {0.05/arrows:.4f}')
    P('\n# held-out by site: office_jar minus rest for cond and dam, within material (raw differences)')
    for site in ['Mohenjo-daro', 'Harappa', 'Dholavira', 'Lothal', 'Kalibangan', 'Chanhu-daro']:
        s = [r for r in seals if r['site'] == site]
        a = [r for r in s if r['tc'] == 'office_jar']; b = [r for r in s if r['tc'] != 'office_jar']
        if len(a) >= 5 and len(b) >= 5:
            P(f'   {site:13s} n={len(s)} office={len(a)} cond diff {st.mean(r["cond"] for r in a)-st.mean(r["cond"] for r in b):+.3f}  dam diff {st.mean(r["dam"] for r in a)-st.mean(r["dam"] for r in b):+.3f}  len-cond r {corr([len(r["t"]) for r in s]) if False else "":}')
    P('\n# condition by site and material (shows the excavation/site effect the strata remove):')
    for (site, mat), ids in sorted(collections.Counter((r['site'], r['mat']) for r in seals).items(), key=lambda kv: -kv[1])[:10]:
        s = [r for r in seals if r['site'] == site and r['mat'] == mat]
        P(f'   {site:13s} {mat:10s} n={len(s)} mean cond {st.mean(r["cond"] for r in s):.2f} damaged {st.mean(r["dam"] for r in s):.2f}')

# =========================== CYCLE 5 ===========================
if CYCLE == 5:
    P('# (e) multi-sided objects: are the lines on one object the same text (counting device) or different (ledger)?')
    objs = collections.defaultdict(list)
    for r in R:
        if len(r['t']) >= 1: objs[(r['obj'], r['site'], r['type'])].append(r)
    multi = {k: v for k, v in objs.items() if len(v) >= 2}
    P(f'objects with >= 2 inscribed lines: {len(multi)}; by type {collections.Counter(k[2] for k in multi).most_common(12)}')
    def tclass(t):
        if 700 in t and all(x in NUM or x == 700 for x in t): return 'voucher_N700'
        if any(t[i] in TREE and t[i - 1] in NUM for i in range(1, len(t))): return 'stock'
        if t[-1] in JAR: return 'office_jar'
        if t[-1] == 520: return 'arrow'
        if t[0] in OPEN: return 'opener'
        if any(x in NUM for x in t): return 'counted_other'
        return 'other'
    for typ in ['TAB:B', 'TAB:I', 'TAB:C', 'TAG', 'TAG:L', 'SEAL:S', 'SEAL:R', 'IMPL']:
        ks = [k for k in multi if k[2] == typ]
        if len(ks) < 3: continue
        same_all = sum(1 for k in ks if len({r['t'] for r in multi[k]}) == 1)
        any_pair = sum(1 for k in ks if len({r['t'] for r in multi[k]}) < len(multi[k]))
        # null: for each object draw the same number of lines from the pool of lines of same site x type
        pools = collections.defaultdict(list)
        for k, v in objs.items():
            if k[2] == typ:
                for r in v: pools[k[1]].append(r['t'])
        sims = []
        for _ in range(1000):
            s = 0
            for k in ks:
                pool = pools[k[1]]
                draw = random.sample(pool, len(multi[k])) if len(pool) >= len(multi[k]) else pool
                if len(set(draw)) < len(draw): s += 1
            sims.append(s)
        P(f'\n{typ}: objects={len(ks)} all-sides-identical={same_all} any-two-identical={any_pair} ({100*any_pair/len(ks):.0f}%) vs null (random lines from same site x type) {st.mean(sims):.1f} p_hi={perm_p(any_pair,sims)[1]:.3f} p_lo={perm_p(any_pair,sims)[0]:.3f}')
        P(f'   sites: {collections.Counter(k[1] for k in ks).most_common(5)}; lines per object {collections.Counter(len(multi[k]) for k in ks)}')
        # class composition of line pairs
        pairs = collections.Counter()
        for k in ks:
            cl = sorted(tclass(r['t']) for r in multi[k])
            pairs[tuple(cl)] += 1
        P('   class combinations: ' + '; '.join(f'{"+".join(c)} x{n}' for c, n in pairs.most_common(8)))
        # for different-text objects: do the two texts share a sign? vs null
        diff = [k for k in ks if len({r['t'] for r in multi[k]}) == len(multi[k])]
        if len(diff) >= 5:
            def share(lines):
                sets = [set(t) for t in lines]
                return int(any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))))
            obs = sum(share([r['t'] for r in multi[k]]) for k in diff)
            sims = []
            for _ in range(1000):
                s = 0
                for k in diff:
                    pool = pools[k[1]]; draw = random.sample(pool, len(multi[k]))
                    if len(set(draw)) == len(draw): s += share(draw)
                    else: s += share(draw)  # keep same n
                sims.append(s)
            P(f'   different-text objects {len(diff)}: lines share >= 1 sign in {obs} vs null {st.mean(sims):.1f} p_hi={perm_p(obs,sims)[1]:.3f} p_lo={perm_p(obs,sims)[0]:.3f}')
        if typ.startswith('TAG'):
            P('   TAG multi-line objects (site, cisi, lines):')
            for k in ks:
                P(f'      {k[1][:12]:12s} {multi[k][0]["cisi"]:8s} ' + ' | '.join('-'.join(map(str, r['t'])) for r in multi[k]))
    # voucher-count structure on TAB:B two-sided: count side vs text side
    P('\n# TAB:B/TAB:I Harappa two-line objects: when one line is a voucher count (N . W700), what is the other line?')
    for typ in ['TAB:B', 'TAB:I']:
        other = collections.Counter(); nn = 0
        for k, v in multi.items():
            if k[2] != typ: continue
            cls = [tclass(r['t']) for r in v]
            if 'voucher_N700' in cls:
                nn += 1
                for r, c in zip(v, cls):
                    if c != 'voucher_N700': other['-'.join(map(str, r['t']))] += 1
        P(f'   {typ}: objects with a voucher line {nn}; the companion texts: ' + '; '.join(f'{t} x{n}' for t, n in other.most_common(10)))
OUT.close()
