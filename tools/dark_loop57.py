"""S-DARK-57: the founder set. Which seal signs descend from the pre-seal pottery marks, and are they the frame?
Cycle 1: founder set (signs attested on pre-Mature pottery, loop57_earlymarks.csv + inscriptions.csv Period 1-2 pots)
         vs the Mature seal inventory by slot role; nulls = frequency-matched and complexity-matched random draws, 1,000x.
Cycle 2: chronology of individual signs: role on early pots -> Mature pots -> seals (continuity vs recruitment).
Cycle 3: held-out prediction: Mature pot marks at Lothal / Kalibangan / Dholavira draw from the founder set more than
         Mature seals do (token shares with bootstrap CI and rule-of-three bounds), against MD + Harappa pots.
Usage: python3 tools/dark_loop57.py <1|2|3> [nperm]
"""
import csv, json, collections, re, sys, math
import numpy as np
CYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(57)
ROOT = '/home/user/Indus-/'
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
RAW = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
P = json.load(open(ROOT + 'data/derived/parsed_texts.json'))
CPLX = {int(k): v for k, v in json.load(open(ROOT + 'data/derived/sign-complexity.json')).items()}
LEV = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))
MERGE = {lv: {m['form']: m['into'] for m in LEV['merges'] if (lv == 'seq_all' and m['level'] in ('strong', 'probable')) or (lv == 'seq_strong' and m['level'] == 'strong')} for lv in ('seq_raw', 'seq_strong', 'seq_all')}
NUM = {1, 2, 3, 4, 5, 6, 7, 31, 32, 33, 34, 35, 36, 37}
OPEN = {817, 861, 820, 920}; MARK = {2, 60}
HEADS = {692, 575, 125, 416, 413, 920, 495}          # name-initial signs (S310-S311)
OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)
def write(c):
    open(ROOT + f'data/derived/dark/loop57_cycle{c}_detail.txt', 'w').write('\n'.join(OUT) + '\n')
def parse(t):
    return [int(x) for x in re.split(r'[-/]', t.replace(']', '').replace('[', '').replace('+', '')) if x.isdigit() and int(x) > 0]
def rule3(k, n): return f'{k}/{n}' + (f' (< {3/n:.2f})' if k == 0 and n else '')

# ---------------------------------------------------------------- early marks
EM = list(csv.DictReader(open(ROOT + 'data/derived/dark/loop57_earlymarks.csv')))
def early_tokens(min_conf='B', sources=None):
    """(sign, weight, source, site, phase) for every Wells-numbered early mark with confidence >= min_conf."""
    order = {'A': 0, 'B': 1, 'C': 2}
    toks = []
    for r in EM:
        if sources and r['source'] not in sources: continue
        if r['signs_W'] == 'non-Wells': continue
        signs = r['signs_W'].split('-'); confs = r['confidence_per_sign'].split('-')
        if len(confs) < len(signs): confs = confs * len(signs)
        for s, c in zip(signs, confs):
            if not s.isdigit() or c not in order or order[c] > order[min_conf]: continue
            toks.append((int(s), int(r['n_marks']), r['source'], r['site'], r['phase']))
    return toks
def corpus_early():
    """inscriptions.csv pots of Harappa Period 1-2 and Kanmer IIa (the loop 43 set), as (sign, 1, 'CSV', site, phase)."""
    toks = []
    for r in RAW:
        if r['type'].split(':')[0] != 'POT': continue
        t = r['time']; per = r['period'].strip(); site = r['site']
        if t in ('Period 1', 'Period 2', 'Period 2A', 'Period 2B') or (per in ('1', '2', 'IIa', 'IIb') and site in ('Harappa', 'Kanmer')):
            for x in parse(r['text']): toks.append((x, 1, 'CSV', site, f'{site} {t or per}'))
    return toks

# ---------------------------------------------------------------- Mature roles
def seal_roles(level='seq_raw'):
    """modal parse slot per sign on seals (parsed_texts.json is seq_raw numbering; merges applied by level)."""
    mp = MERGE[level]
    slot = collections.defaultdict(collections.Counter); pot = collections.defaultdict(collections.Counter)
    for o in P:
        tc = o['type'].split(':')[0]
        for s, l in zip(o['seq'], o['slots']):
            s = mp.get(s, s)
            if tc == 'SEAL': slot[s][l] += 1
            elif tc == 'POT': pot[s][l] += 1
    tok = collections.Counter()
    for o in C:
        if o['type'].split(':')[0] == 'SEAL':
            for x in o[level]: tok[x] += 1
    def role(s):
        if s not in tok: return 'unattested'
        if s in NUM: return 'numeral'
        if s in OPEN: return 'opener'
        if s in MARK: return 'marker'
        m = slot[s].most_common(1)[0][0] if slot[s] else 'NAME'
        if m == 'COUNT': return 'numeral'
        if m == 'CLOSER': return 'closer'
        if m == 'TITLE': return 'qualifier'
        if m == 'SUFFIX': return 'suffix'
        if m == 'OPENER': return 'opener'
        if m == 'MARKER': return 'marker'
        return 'head' if s in HEADS else 'middle'
    return role, tok, slot, pot
FRAME = ('opener', 'marker', 'closer', 'suffix'); UNIT = ('numeral',)
ROLES = ['opener', 'marker', 'numeral', 'closer', 'suffix', 'head', 'qualifier', 'middle', 'unattested']

if CYCLE == 1:
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        say(f'\n===== level {level} =====')
        mp = MERGE[level]; role, tok, slot, pot = seal_roles(level)
        inv = sorted(tok); freq = np.array([tok[s] for s in inv], float)
        cp = np.array([CPLX.get(s, np.nan) for s in inv])
        for label, minconf, srcs in [('new transcriptions A+B', 'B', None), ('new transcriptions A+B+C', 'C', None),
                                     ('corpus Period 1-2 pots only (loop 43 set)', None, 'CSV'),
                                     ('ALL early (new A+B + corpus)', 'B', 'ALL'), ('Harappa only (K06 + corpus)', 'B', 'HP'),
                                     ('Mehrgarh only (Q80)', 'B', 'Q80')]:
            if srcs == 'CSV': toks = corpus_early()
            elif srcs == 'ALL': toks = early_tokens(minconf) + corpus_early()
            elif srcs == 'HP': toks = early_tokens(minconf, {'K06'}) + corpus_early()
            elif srcs == 'Q80': toks = early_tokens(minconf, {'Q80'})
            else: toks = early_tokens(minconf)
            toks = [(mp.get(s, s), w, src, site, ph) for s, w, src, site, ph in toks]
            founder = sorted({s for s, *_ in toks}); wt = collections.Counter()
            for s, w, *_ in toks: wt[s] += w
            n_tok = sum(wt.values())
            rc = collections.Counter(role(s) for s in founder); rw = collections.Counter()
            for s in founder: rw[role(s)] += wt[s]
            say(f'\n[{label}] founder signs {len(founder)}, early tokens {n_tok}')
            say('  signs: ' + ', '.join(f'W{s}({wt[s]},{role(s)})' for s in founder))
            say('  by seal role (types): ' + ', '.join(f'{k}={rc[k]}' for k in ROLES if rc[k]))
            say('  by seal role (early-token weighted): ' + ', '.join(f'{k}={rw[k]}' for k in ROLES if rw[k]))
            att = [s for s in founder if role(s) != 'unattested']
            if not att: continue
            obs_fu = np.mean([role(s) in FRAME + UNIT for s in att]); obs_f = np.mean([role(s) in FRAME for s in att])
            obs_n = np.mean([role(s) in UNIT for s in att]); obs_c = np.mean([role(s) == 'closer' for s in att])
            obs_cp = np.nanmean([CPLX.get(s, np.nan) for s in att])
            # null 1: frequency-matched draw of len(att) distinct seal signs, p ~ seal token frequency
            k = len(att); p = freq / freq.sum()
            N1 = []; N2 = []
            for _ in range(NPERM):
                d = rng.choice(len(inv), k, replace=False, p=p)
                rs = [role(inv[i]) for i in d]
                N1.append((np.mean([r in FRAME + UNIT for r in rs]), np.mean([r in FRAME for r in rs]), np.mean([r in UNIT for r in rs]), np.mean([r == 'closer' for r in rs]), np.nanmean(cp[d])))
            # null 2: complexity-matched: for each founder sign draw a seal sign of the same complexity bin (+-1 stroke unit), p ~ frequency within bin
            bins = {}
            for s in att:
                c0 = CPLX.get(s)
                cand = [i for i, s2 in enumerate(inv) if c0 is not None and CPLX.get(s2) is not None and abs(CPLX[s2] - c0) <= 1] or list(range(len(inv)))
                bins[s] = (np.array(cand), freq[cand] / freq[cand].sum())
            for _ in range(NPERM):
                rs = [role(inv[rng.choice(bins[s][0], p=bins[s][1])]) for s in att]
                N2.append((np.mean([r in FRAME + UNIT for r in rs]), np.mean([r in FRAME for r in rs]), np.mean([r in UNIT for r in rs]), np.mean([r == 'closer' for r in rs])))
            N1 = np.array(N1); N2 = np.array(N2)
            say(f'  attested on seals: {len(att)} of {len(founder)} ({len(founder)-len(att)} unattested); mean complexity {obs_cp:.2f} vs frequency-null {N1[:,4].mean():.2f}')
            for j, (nm, ob) in enumerate([('frame+unit share', obs_fu), ('frame share (opener/marker/closer/suffix)', obs_f), ('numeral share', obs_n), ('closer share', obs_c)]):
                p1 = np.mean(N1[:, j] >= ob - 1e-12); p2 = np.mean(N2[:, j] >= ob - 1e-12)
                say(f'  {nm}: {ob:.2f} | frequency-null {N1[:,j].mean():.2f} (P one-sided {p1:.3f}) | complexity-matched null {N2[:,j].mean():.2f} (P {p2:.3f})')
            # which frame signs are NOT in the founder set
            missing = [s for s in inv if role(s) in FRAME and tok[s] >= 20 and s not in founder]
            say('  frequent frame signs (>= 20 seal tokens) absent from the founder set: ' + ', '.join(f'W{s}({role(s)},{tok[s]})' for s in sorted(missing, key=lambda s: -tok[s])))
            present = [s for s in founder if role(s) in FRAME]
            say('  frame signs present in the founder set: ' + ', '.join(f'W{s}({role(s)},{tok[s]})' for s in present))
    write(1)

elif CYCLE == 2:
    role, tok, slot, pot = seal_roles('seq_raw')
    toks = early_tokens('C') + corpus_early()
    wt = collections.Counter(); src = collections.defaultdict(set); conf = {}
    for s, w, sr, site, ph in toks: wt[s] += w; src[s].add(sr)
    for r in EM:
        if r['signs_W'] == 'non-Wells': continue
        ss = r['signs_W'].split('-'); cc = r['confidence_per_sign'].split('-')
        if len(cc) < len(ss): cc = cc * len(ss)
        for s, c in zip(ss, cc):
            if s.isdigit(): conf[int(s)] = min(conf.get(int(s), 'C'), c)
    # modal role on Mature pots (all sites), on held-out pots, on seals, plus position shares on Mature pots from inscriptions.csv
    first = collections.Counter(); last = collections.Counter(); ptok = collections.Counter(); lone = collections.Counter()
    for r in RAW:
        if r['type'].split(':')[0] != 'POT': continue
        t = r['time']; per = r['period'].strip(); site = r['site']
        if t in ('Period 1', 'Period 2', 'Period 2A', 'Period 2B') or (per in ('1', '2', 'IIa', 'IIb') and site in ('Harappa', 'Kanmer')): continue
        s = parse(r['text'])
        if not s: continue
        if r['dir.'].strip() in ('R/L', 'L/R'): s = s[::-1]
        for x in s: ptok[x] += 1
        if len(s) == 1: lone[s[0]] += 1
        elif r['complete'] == 'Y' and r['dir.'].strip() in ('R/L', 'L/R'): first[s[0]] += 1; last[s[-1]] += 1
    say('Chronology of each founder sign: early (pre-Mature pots) -> Mature pots -> Mature seals')
    say('columns: sign | early n (sources, best conf) | Mature-pot tokens, lone-sign share, first/last share among complete directed multi-sign pots, modal parse slot on pots | seal tokens, modal seal slot -> role | verdict')
    cont = rec = 0; recruited = []
    for s in sorted(wt, key=lambda s: -wt[s]):
        pm = pot[s].most_common(1)[0][0] if pot[s] else '-'
        sm = slot[s].most_common(1)[0][0] if slot[s] else '-'
        rl = role(s)
        nm = ptok[s]; fl = f'{first[s]}/{last[s]}' if nm else '-'
        if rl == 'unattested': v = 'early only (not on seals)'
        elif not nm: v = 'early -> seals, skips Mature pots'
        elif pm == sm or (pm == 'COUNT' and rl == 'numeral') or (pm == 'CLOSER' and rl == 'closer'): v = 'CONTINUITY'; cont += 1
        else: v = f'RECRUITED (pots {pm} -> seals {sm})'; rec += 1; recruited.append((s, pm, sm, rl))
        say(f'  W{s:<4d} | early {wt[s]:4d} ({"/".join(sorted(src[s]))}, {conf.get(s,"csv")}) | pots {nm:4d} lone {lone[s]:3d} first/last {fl:7s} slot {pm:7s} | seals {tok[s]:4d} slot {sm:7s} -> {rl:10s} | {v}')
    say(f'\ncontinuity {cont}, recruited {rec}; recruited signs: ' + ', '.join(f'W{s} ({a}->{b}, seal role {r})' for s, a, b, r in recruited))
    # the frame signs: do they ever occur on pre-Mature pots?
    say('\nFrame signs on pre-Mature pots (new transcriptions + corpus): ' + ', '.join(f'W{s}: {wt.get(s,0)}' for s in sorted(OPEN | MARK | {740, 520, 400, 176, 90, 806, 154, 527, 156, 151, 226, 617, 236, 595})))
    n_all = sum(wt.values())
    say(f'opener/marker tokens among {n_all} early tokens: {rule3(sum(wt[s] for s in OPEN|MARK), n_all)}; jar W740 {rule3(wt[740], n_all)}; arrow W520 {rule3(wt[520], n_all)}; comb W400/401 {wt[400]+wt[401]}')
    write(2)

elif CYCLE == 3:
    HELD = {'Lothal', 'Kalibangan', 'Dholavira'}
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        say(f'\n===== level {level} =====')
        mp = MERGE[level]; role, tok, slot, pot = seal_roles(level)
        for label, minconf in [('founder A+B', 'B'), ('founder A+B+C', 'C')]:
            F = {mp.get(s, s) for s, *_ in early_tokens(minconf) + corpus_early()}
            Fn = F - NUM
            say(f'\n[{label}] founder set {len(F)} signs ({len(Fn)} non-numeral)')
            groups = collections.defaultdict(list)       # group -> list of token lists
            for r in RAW:
                if r['type'].split(':')[0] != 'POT': continue
                t = r['time']; per = r['period'].strip(); site = r['site']
                if t in ('Period 1', 'Period 2', 'Period 2A', 'Period 2B') or (per in ('1', '2', 'IIa', 'IIb') and site in ('Harappa', 'Kanmer')): continue
                s = [mp.get(x, x) for x in parse(r['text'])]
                if not s: continue
                g = 'POT held-out (Lothal/Kalibangan/Dholavira)' if site in HELD else ('POT MD+Harappa' if site in ('Mohenjo-daro', 'Harappa') else 'POT other sites')
                groups[g].append(s)
                groups[f'POT {site}' if site in HELD else '_'].append(s)
            for o in C:
                tc = o['type'].split(':')[0]
                if tc not in ('SEAL', 'TAB'): continue
                g = f'{tc} held-out' if o['site'] in HELD else (f'{tc} MD+Harappa' if o['site'] in ('Mohenjo-daro', 'Harappa') else f'{tc} other sites')
                groups[g].append(o[level])
            res = {}
            for g in sorted(groups):
                if g == '_': continue
                S = groups[g]; T = [x for s in S for x in s]
                if not T: continue
                inF = [x in F for x in T]; inFn = [x in Fn for x in T if x not in NUM]
                boot = [np.mean(rng.choice(inF, len(inF))) for _ in range(500)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                res[g] = (np.mean(inF), len(T))
                say(f'  {g:45s} texts {len(S):5d} tokens {len(T):6d} | founder share {np.mean(inF):.3f} [{lo:.3f}-{hi:.3f}] | non-numeral tokens in founder {np.mean(inFn) if inFn else float("nan"):.3f} (n {len(inFn)}) | tokens: numerals {np.mean([x in NUM for x in T]):.2f}')
            a = res.get('POT held-out (Lothal/Kalibangan/Dholavira)'); b = res.get('SEAL held-out'); c = res.get('SEAL MD+Harappa'); d = res.get('POT MD+Harappa')
            if a and b:
                # permutation: tokens pooled between held-out pots and held-out seals
                Tp = [x for s in groups['POT held-out (Lothal/Kalibangan/Dholavira)'] for x in s]; Ts = [x for s in groups['SEAL held-out'] for x in s]
                pool = np.array([x in F for x in Tp + Ts]); obs = pool[:len(Tp)].mean() - pool[len(Tp):].mean()
                null = []
                for _ in range(NPERM):
                    rng.shuffle(pool); null.append(pool[:len(Tp)].mean() - pool[len(Tp):].mean())
                say(f'  PREDICTION held-out pots > held-out seals: {a[0]:.3f} vs {b[0]:.3f}, diff {obs:+.3f}, permutation P {np.mean(np.array(null) >= obs - 1e-12):.3f} (n {len(Tp)} vs {len(Ts)} tokens)')
            if a and c: say(f'  held-out pots {a[0]:.3f} vs MD+Harappa seals {c[0]:.3f}; MD+Harappa pots {d[0]:.3f} (the fitting data)')
    write(3)
