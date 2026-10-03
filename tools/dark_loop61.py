"""S-DARK-61: IS THE HEAD A TITLE OR THE LAST ELEMENT OF A NAME?
S-DARK-56 found that Indus middles stripped of head + adjacent qualifier have no closed edge set (top-10 finals cover 0.27, random
0.26) whereas every one-per-person name list has 0.29-0.63. Arrow: the head IS the closed final-element set (like Sumerian / Akkadian
theophoric or kinship finals), the adjacent qualifier the penultimate name element, and the 'title' reading is wrong.
Cycle 1  edge-set, slot-binding, MI and predictability on WHOLE Indus texts with only openers (+ their connective), numerals and the
         trailing tablet suffix W400/W90 removed (head kept), vs Ur III (a) name alone (signs; elements; elements minus final),
         (b) name + title as final element (all line-2 words; profession titles only), (c) names with a theophoric final, Linear B
         and Latin whole names; frequency-matched random strings as the 'no closed set' floor.
Cycle 2  penultimate -> final binding: MI (bits, and as % of H(final)), share of penultimate types (>= 5 tokens) loyal (>= 80%) to one
         final, share of testable (penultimate, final) pairs with |log2 O/E| > 1; null = finals permuted among strings (1,000x).
         Indus qualifier -> head (texts ending in a paradigm head), Indus any final; Ur III element -> final element, element -> title,
         whole name -> title; Linear B, Latin final element.
Cycle 3  (3) heads in non-final position vs name finals in compound names and vs titles; two heads in one text; head-only texts vs
         Ur III title-only legends; (4) replication on held-out Wells sites and IM77, all three merge levels.
Usage: python3 tools/dark_loop61.py <1|2|3> [ndraw] [nperm]
"""
import json, sys, random, collections, math, csv, os, re
C = json.load(open('data/derived/merged-corpus-canonical.json'))
BR = json.load(open('data/derived/bridge_extended.json'))
PROP = json.load(open('data/derived/dark/bridge_proposals.json'))
C56 = 'data/derived/dark/loop56_corpora/'; C61 = 'data/derived/dark/loop61_corpora/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rnd = random.Random(61)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ---------------- frame parser (S310/S331, identical to tools/dark_loop56.py) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
HEADS = set(CL) | {595}
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
NUMALL = set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(32, 40)) | {1, 2, 31, 55, 56}
def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t, 'other')
def build_qual(objs):
    left = collections.defaultdict(collections.Counter)
    for o in objs:
        s = o['seq'][:]
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
def annotate(objs):
    QUAL = build_qual(objs)
    for o in objs:
        o['lab'] = parse(o['seq'], QUAL)
        # WITH-HEAD text: drop opener + its connective, every numeral sign, the trailing suffix; keep everything else in order
        keep = [a for a, l in zip(o['seq'], o['lab']) if l not in ('OPENER', 'MARKER', 'SUFFIX') and a not in NUMALL]
        o['wh'] = tuple(keep)
        o['mid'] = tuple(a for a, l in zip(o['seq'], o['lab']) if l in ('NAME', 'COUNT'))
        o['closer'] = next((a for a, l in zip(o['seq'], o['lab']) if l == 'CLOSER'), None)
    return objs
def load_indus(LV):
    objs = []; seen = set()
    for r in C:
        s = r[LV]
        if not s or len(s) < 1 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
        key = (r['site'], otype(r['type']), tuple(s))
        if key in seen: continue
        seen.add(key)
        objs.append(dict(cisi=r['cisi'], site=r['site'], ot=otype(r['type']), seq=list(s), big=r['site'] in ('Mohenjo-daro', 'Harappa')))
    return annotate(objs)
def im77_objects():
    M2W = {}
    for w, ms in BR.items():
        for m in ms: M2W.setdefault(m, []).append(int(w))
    for p in PROP['proposals']: M2W.setdefault(p['M'], []).append(p['W'])
    for m in M2W: M2W[m] = min(M2W[m])
    rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[r['text_no']].append(r)
    objs = []; seen = set()
    for tn, rs in by.items():
        rs = sorted(rs, key=lambda r: (int(r['side']), int(r['line'])))
        rs0 = [r for r in rs if r['side'] == '0'] or rs[:1]
        ms = [int(x) for r in rs0 for x in r['signs_clean'].split() if x.strip() and x != '0']
        if len(ms) < 1: continue
        seq = [M2W.get(m, 10000 + m) for m in ms]
        site = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa'}.get(rs0[0]['site'], rs0[0]['site'])
        ot = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}.get(rs0[0]['object_type'], 'other')
        key = (site, ot, tuple(seq))
        if key in seen: continue
        seen.add(key)
        objs.append(dict(cisi='IM' + tn, site=site, ot=ot, seq=seq, big=site in ('Mohenjo-daro', 'Harappa')))
    return annotate(objs)
_OBJ = {}
def objs_for(LV):
    if LV not in _OBJ: _OBJ[LV] = load_indus(LV) if LV != 'im77' else im77_objects()
    return _OBJ[LV]
def indus_strings(LV, field='wh', minlen=2, subset=None):
    objs = objs_for(LV)
    if subset: objs = [o for o in objs if subset(o)]
    return sorted(set(o[field] for o in objs if len(o[field]) >= minlen))

# ---------------- reference lists ----------------
def clean_tok(t): return bool(t) and '$' not in t and 'blank' not in t and t not in ('(', ')')
def jl(path, minlen=2):
    out = set()
    for l in open(path):
        s = tuple(t for t in json.loads(l)['seq'] if clean_tok(t))
        if len(s) >= minlen: out.add(s)
    return sorted(out)
REFS = {
    'ur3_name_signs': C61 + 'ur3_name_signs.jsonl', 'ur3_name_elems': C61 + 'ur3_name_elems.jsonl',
    'ur3_name_minus_final': C61 + 'ur3_name_minus_final.jsonl', 'ur3_name_title': C61 + 'ur3_name_title.jsonl',
    'ur3_name_proftitle': C61 + 'ur3_name_proftitle.jsonl', 'ur3_name_theo': C61 + 'ur3_name_theo.jsonl',
    'linb_names': C56 + 'linb_personnel_dedup.jsonl', 'latin_names': C56 + 'latin_names_dedup.jsonl',
    'ob_name_signs': C56 + 'ob_names_dedup.jsonl'}

# ---------------- statistics ----------------
def H(cnt):
    t = sum(cnt.values()); return -sum(v / t * math.log2(v / t) for v in cnt.values() if v > 0) if t else 0.0
def raw_stats(names, freq_min=5):
    long = [n for n in names if len(n) >= 2]
    pos = collections.defaultdict(lambda: [0, 0, 0])
    for n in long:
        L = len(n)
        for i, a in enumerate(n): pos[a][0 if i == 0 else 2 if i == L - 1 else 1] += 1
    freq = {a: v for a, v in pos.items() if sum(v) >= freq_min}
    skew = {a: abs(v[0] - v[2]) / sum(v) for a, v in freq.items()}
    pos_mean = sum(skew.values()) / len(skew) if skew else float('nan')
    bg = collections.Counter((n[i], n[i + 1]) for n in long for i in range(len(n) - 1)); nb = sum(bg.values())
    A = collections.Counter(); B = collections.Counter(); byfirst = collections.defaultdict(collections.Counter)
    for (a, b), v in bg.items(): A[a] += v; B[b] += v; byfirst[a][b] += v
    mi = sum(v / nb * math.log2(v * nb / (A[a] * B[b])) for (a, b), v in bg.items()) if nb else float('nan')
    hnext = H(B); hcond = sum(A[a] / nb * H(byfirst[a]) for a in A) if nb else 0
    pred = 1 - hcond / hnext if hnext else float('nan')
    ini = collections.Counter(n[0] for n in long); fin = collections.Counter(n[-1] for n in long); N = len(long)
    top_i = sum(v for _, v in ini.most_common(10)) / N; top_f = sum(v for _, v in fin.most_common(10)) / N
    hi = H(ini) / math.log2(len(ini)) if len(ini) > 1 else 0; hf = H(fin) / math.log2(len(fin)) if len(fin) > 1 else 0
    return dict(pos_mean=pos_mean, skew=skew, mi=mi, pred=pred, top_i=top_i, top_f=top_f, hnorm_i=hi, hnorm_f=hf, k_i=len(ini), k_f=len(fin))
def shuffle_names(names, r, mode='shuf'):
    lens = [len(n) for n in names]
    if mode == 'shuf':
        toks = [a for n in names for a in n]; r.shuffle(toks)
    else:
        pool = [a for n in names for a in n]; toks = [r.choice(pool) for _ in range(len(pool))]
    out = []; i = 0
    for L in lens: out.append(tuple(toks[i:i + L])); i += L
    return out
def metrics(names, r=None, nnull=20):
    r = r or rnd
    names = [tuple(n) for n in names]
    obs = raw_stats(names)
    nulls = [raw_stats(shuffle_names(names, r, 'shuf')) for _ in range(nnull)]
    fnull = [raw_stats(shuffle_names(names, r, 'freq')) for _ in range(nnull)]
    def nm(ns, k):
        v = [x[k] for x in ns if not math.isnan(x[k])]; return sum(v) / len(v) if v else float('nan')
    bound = 0; tot = 0
    for a, s in obs['skew'].items():
        nv = sorted(x['skew'].get(a, 0.0) for x in nulls)
        if len(nv) >= 10:
            tot += 1
            if s > nv[int(0.95 * len(nv)) - 1]: bound += 1
    el = collections.Counter(a for n in names for a in n)
    return dict(n=len(names), k=len(el), meanL=sum(el.values()) / len(names),
                top_i=obs['top_i'], top_f=obs['top_f'], top_i_rand=nm(fnull, 'top_i'), top_f_rand=nm(fnull, 'top_f'),
                top_f_ex=obs['top_f'] - nm(fnull, 'top_f'), top_i_ex=obs['top_i'] - nm(fnull, 'top_i'),
                hnorm_i=obs['hnorm_i'], hnorm_f=obs['hnorm_f'], hnorm_f_rand=nm(fnull, 'hnorm_f'), k_f=obs['k_f'],
                pos_excess=obs['pos_mean'] - nm(nulls, 'pos_mean'), pos_bound=bound / tot if tot else float('nan'), nfreq=tot,
                mi_ex=obs['mi'] - nm(nulls, 'mi'), pair_pred=obs['pred'] - nm(nulls, 'pred'))
KEYS = ['n', 'k', 'meanL', 'top_i', 'top_f', 'top_i_rand', 'top_f_rand', 'top_f_ex', 'hnorm_f', 'hnorm_f_rand', 'k_f', 'pos_excess', 'pos_bound', 'mi_ex', 'pair_pred']
LAD = ['top_i', 'top_f', 'top_f_rand', 'top_f_ex', 'hnorm_f', 'hnorm_f_rand', 'pos_excess', 'pos_bound', 'mi_ex', 'pair_pred']
def fmt(m, keys=KEYS): return ' '.join(f'{k}={m[k]:.3f}' if isinstance(m[k], float) else f'{k}={m[k]}' for k in keys)
def q(vals, p):
    v = sorted(x for x in vals if not (isinstance(x, float) and math.isnan(x)))
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')
def draws(pool, n, label, ndraw=NB):
    runs = []
    for b in range(ndraw):
        r = random.Random(6100 + b); sub = r.sample(pool, min(n, len(pool))); runs.append(metrics(sub, r, nnull=10))
    med = {k: q([x[k] for x in runs], 0.5) for k in KEYS}; lo = {k: q([x[k] for x in runs], 0.025) for k in KEYS}; hi = {k: q([x[k] for x in runs], 0.975) for k in KEYS}
    P(f'  [{label}] pool={len(pool)} draw n={med["n"]} x{ndraw}'); P('    median ' + fmt(med))
    P('    CI: ' + '; '.join(f'{k} {med[k]:.3f} [{lo[k]:.3f},{hi[k]:.3f}]' for k in LAD))
    return dict(med=med, lo=lo, hi=hi)
def boot(names, label, nboot=NB, frac=0.8):
    m = metrics(names, nnull=20); runs = []
    for b in range(nboot):
        r = random.Random(1000 + b); sub = r.sample(names, int(frac * len(names))); runs.append(metrics(sub, r, nnull=10))
    lo = {}; hi = {}
    for k in KEYS:
        v = [x[k] for x in runs]; md = q(v, 0.5)
        lo[k] = m[k] + q(v, 0.025) - md if isinstance(m[k], float) else m[k]; hi[k] = m[k] + q(v, 0.975) - md if isinstance(m[k], float) else m[k]
    P(f'  [{label}] n={len(names)}'); P('    ' + fmt(m))
    P('    CI(0.8-subsample): ' + '; '.join(f'{k} {m[k]:.3f} [{lo[k]:.3f},{hi[k]:.3f}]' for k in LAD))
    return dict(med=m, lo=lo, hi=hi)
def save(name, obj):
    json.dump(obj, open(f'data/derived/dark/{name}.json', 'w'), indent=1)
    open(f'data/derived/dark/{name}_log.txt', 'w').write('\n'.join(LOG) + '\n')

# ---------------- penultimate -> final binding ----------------
def pf_stats(pairs):
    pen = collections.Counter(a for a, b in pairs); fin = collections.Counter(b for a, b in pairs); jt = collections.Counter(pairs); N = len(pairs)
    mi = sum(v / N * math.log2(v * N / (pen[a] * fin[b])) for (a, b), v in jt.items())
    hf = H(fin); hp = H(pen)
    byp = collections.defaultdict(collections.Counter)
    for a, b in pairs: byp[a][b] += 1
    freqp = [a for a, v in pen.items() if v >= 5]
    loyal = sum(1 for a in freqp if byp[a].most_common(1)[0][1] / pen[a] >= 0.8) / len(freqp) if freqp else float('nan')
    # attraction / repulsion among testable (E >= 3) cells
    test = 0; dep = 0; att = 0; rep = 0
    for a in pen:
        for b in fin:
            E = pen[a] * fin[b] / N
            if E >= 3:
                test += 1; O = jt.get((a, b), 0)
                if O > 2 * E: dep += 1; att += 1
                elif O < E / 2: dep += 1; rep += 1
    return dict(mi=mi, mi_hf=mi / hf if hf else float('nan'), mi_hmin=mi / min(hf, hp) if min(hf, hp) else float('nan'), loyal=loyal, nfreqp=len(freqp),
                dep=dep / test if test else float('nan'), att=att, rep=rep, ntest=test, hf=hf, k_f=len(fin), k_p=len(pen), N=N,
                top10p=sum(v for _, v in pen.most_common(10)) / N, top10f=sum(v for _, v in fin.most_common(10)) / N)
def pf_test(pairs, label, nperm=NPERM, seed=7):
    r = random.Random(seed); obs = pf_stats(pairs); fins = [b for a, b in pairs]; pens = [a for a, b in pairs]
    nul = collections.defaultdict(list)
    for _ in range(nperm):
        r.shuffle(fins); s = pf_stats(list(zip(pens, fins)))
        for k in ('mi', 'mi_hf', 'mi_hmin', 'loyal', 'dep'): nul[k].append(s[k])
    out = dict(obs=obs)
    for k in ('mi', 'mi_hf', 'mi_hmin', 'loyal', 'dep'):
        v = sorted(x for x in nul[k] if not math.isnan(x)); mu = sum(v) / len(v) if v else float('nan')
        pv = (sum(1 for x in v if x >= obs[k]) + 1) / (len(v) + 1)
        out[k] = dict(obs=obs[k], null=mu, q95=q(v, 0.95), ex=obs[k] - mu, P=pv)
    P(f'  [{label}] pairs N={obs["N"]} k_pen={obs["k_p"]} k_fin={obs["k_f"]} H(final)={obs["hf"]:.2f} top10 pen {obs["top10p"]:.2f} fin {obs["top10f"]:.2f}')
    P('    ' + '; '.join(f'{k} {out[k]["obs"]:.3f} (null {out[k]["null"]:.3f}, 95% {out[k]["q95"]:.3f}, excess {out[k]["ex"]:+.3f}, P {out[k]["P"]:.3f})' for k in ('mi', 'mi_hf', 'mi_hmin', 'loyal', 'dep')))
    P(f'    testable cells {obs["ntest"]}: attraction {obs["att"]}, repulsion {obs["rep"]}; frequent penultimates {obs["nfreqp"]}')
    return out
def last2(names): return [(n[-2], n[-1]) for n in names if len(n) >= 2]

# ================= cycles =================
if CY == 1:
    P(f'== S-DARK-61 cycle 1: whole Indus texts WITH HEAD (openers, numerals, trailing suffix removed) vs name lists with their final element; ndraw {NB}')
    RES = {}
    for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
        names = indus_strings(LV, 'wh', 2)
        heads = sum(1 for n in names if n[-1] in HEADS) / len(names)
        P(f'\n##### Indus {LV}: {len(names)} distinct with-head strings >= 2 elements; final is a paradigm head in {heads:.3f}')
        RES['indus_wh_' + LV] = boot(names, f'Indus {LV} WITH HEAD')
        if LV == 'seq_raw':
            sub = [n for n in names if n[-1] in HEADS]; RES['indus_wh_headfinal_raw'] = boot(sub, 'Indus seq_raw WITH HEAD, head-final texts only')
            RES['indus_mid_raw'] = boot(indus_strings(LV, 'mid', 2), 'Indus seq_raw MIDDLE only (S-DARK-56 field, head + qualifier stripped)')
            seals = indus_strings(LV, 'wh', 2, subset=lambda o: o['ot'] == 'seal'); RES['indus_wh_seals_raw'] = boot(seals, 'Indus seq_raw WITH HEAD, seals only')
            # suffix kept variant
            objs = objs_for(LV); sk = sorted(set(tuple(a for a, l in zip(o['seq'], o['lab']) if l not in ('OPENER', 'MARKER') and a not in NUMALL) for o in objs))
            sk = [n for n in sk if len(n) >= 2]; RES['indus_wh_sufkept_raw'] = boot(sk, 'Indus seq_raw WITH HEAD, W400/W90 suffix kept')
    n0 = RES['indus_wh_seq_raw']['med']['n']
    P(f'\n##### reference lists (one per distinct string) drawn to n={n0}')
    for nm, path in REFS.items():
        RES[nm] = draws(jl(path), n0, nm)
    P('\n=== cycle 1 summary (top_f_rand = top-10 final coverage of frequency-matched random strings with the same lengths)')
    P('  corpus                          ' + ' '.join(f'{k:>10s}' for k in LAD) + '      n     k  meanL')
    for k, v in RES.items():
        m = v['med']; P(f'  {k:32s} ' + ' '.join(f'{m[x]:10.3f}' for x in LAD) + f'   {m["n"]:5d} {m["k"]:5d} {m["meanL"]:.2f}')
    save('loop61_cycle1', RES)

if CY == 2:
    P(f'== S-DARK-61 cycle 2: penultimate -> final binding, {NPERM} permutations of finals among strings')
    RES = {}
    for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
        names = indus_strings(LV, 'wh', 2)
        hf = [n for n in names if n[-1] in HEADS]
        P(f'\n##### Indus {LV}')
        RES[f'indus_qual_head_{LV}'] = pf_test(last2(hf), f'Indus {LV} qualifier -> HEAD (head-final texts, n={len(hf)})')
        RES[f'indus_any_final_{LV}'] = pf_test(last2(names), f'Indus {LV} penultimate -> any final (n={len(names)})')
        if LV == 'seq_raw':
            seals = [n for n in indus_strings(LV, 'wh', 2, subset=lambda o: o['ot'] == 'seal') if n[-1] in HEADS]
            RES['indus_qual_head_seals_raw'] = pf_test(last2(seals), f'Indus seq_raw seals qualifier -> HEAD (n={len(seals)})')
            # whole middle (everything before the head) -> head, head-final texts of >= 3 elements: is the head chosen by the string?
            trip = [(n[:-1], n[-1]) for n in hf if len(n) >= 3]
            RES['indus_wholemid_head_raw'] = pf_test(trip, f'Indus seq_raw WHOLE pre-head string -> head (len >= 3, n={len(trip)})')
            # qualifier -> head with the jar removed (the default head dominates H(final))
            nj = [(a, b) for a, b in last2(hf) if b != 740]
            RES['indus_qual_head_nojar_raw'] = pf_test(nj, f'Indus seq_raw qualifier -> HEAD, jar texts removed (n={len(nj)})')
            # position two steps back: antepenultimate -> head (should be weaker than adjacent for both readings)
            ap = [(n[-3], n[-1]) for n in hf if len(n) >= 3]
            RES['indus_antepen_head_raw'] = pf_test(ap, f'Indus seq_raw antepenultimate -> HEAD (n={len(ap)})')
    P('\n##### Ur III')
    el = jl(REFS['ur3_name_elems']); RES['ur3_elem_final'] = pf_test(last2(el), 'Ur III name elements: penultimate -> final element')
    sg = jl(REFS['ur3_name_signs']); RES['ur3_sign_final'] = pf_test(last2(sg), 'Ur III name signs: penultimate -> final sign')
    th = jl(REFS['ur3_name_theo']); RES['ur3_theo_final'] = pf_test(last2(th), 'Ur III names with theophoric final: penultimate -> {d}DN')
    nt = jl(REFS['ur3_name_title']); RES['ur3_elem_title'] = pf_test(last2(nt), 'Ur III last name element -> TITLE (all line-2 words)')
    pt = jl(REFS['ur3_name_proftitle']); RES['ur3_elem_proftitle'] = pf_test(last2(pt), 'Ur III last name element -> PROFESSION title')
    # whole name -> title (legend-level, deduplicated legends; also non-dedup impressions for reference)
    recs = [json.loads(l) for l in open(C61 + 'ur3_legend_fields.jsonl')]
    wn = [(tuple(r['name_elems']), 'T:' + r['title']) for r in recs if r['title']]
    RES['ur3_wholename_title'] = pf_test(wn, 'Ur III WHOLE name -> title (distinct legends)')
    fe = [(r['name_elems'][0], 'T:' + r['title']) for r in recs if r['title'] and len(r['name_elems']) >= 2]
    RES['ur3_firstelem_title'] = pf_test(fe, 'Ur III FIRST name element -> title')
    P('\n##### Linear B, Latin, Old Babylonian')
    RES['linb_final'] = pf_test(last2(jl(REFS['linb_names'])), 'Linear B names: penultimate -> final syllabogram')
    RES['latin_final'] = pf_test(last2(jl(REFS['latin_names'])), 'Latin names: penultimate -> final word')
    RES['ob_final'] = pf_test(last2(jl(REFS['ob_name_signs'])), 'Old Babylonian name signs: penultimate -> final sign')
    P('\n=== cycle 2 summary: MI excess (bits), MI/H(final) excess, loyal share obs (null), dependent-cell share obs (null)')
    for k, v in RES.items():
        P(f'  {k:34s} MI {v["mi"]["obs"]:.3f} (+{v["mi"]["ex"]:.3f}, P {v["mi"]["P"]:.3f})  MI/Hf {v["mi_hf"]["obs"]:.3f} (+{v["mi_hf"]["ex"]:.3f})  MI/Hmin {v["mi_hmin"]["obs"]:.3f} (+{v["mi_hmin"]["ex"]:.3f})  loyal {v["loyal"]["obs"]:.3f} ({v["loyal"]["null"]:.3f}, P {v["loyal"]["P"]:.3f})  dep {v["dep"]["obs"]:.3f} ({v["dep"]["null"]:.3f})  N={v["obs"]["N"]} k_f={v["obs"]["k_f"]}')
    save('loop61_cycle2', RES)

if CY == 3:
    P(f'== S-DARK-61 cycle 3: positional predictions of the two readings + replication on held-out sites and IM77')
    RES = {}
    def nonfinal_share(names, finals, label, minlen=3):
        """share of tokens of the given 'final-class' elements that stand in NON-final position, strings of >= minlen."""
        long = [n for n in names if len(n) >= minlen]
        tot = collections.Counter(); nf = collections.Counter(); first = collections.Counter()
        for n in long:
            for i, a in enumerate(n):
                if a in finals:
                    tot[a] += 1
                    if i < len(n) - 1: nf[a] += 1
                    if i == 0: first[a] += 1
        T = sum(tot.values()); NF = sum(nf.values()); F = sum(first.values())
        two = sum(1 for n in long if sum(1 for a in n if a in finals) >= 2)
        P(f'  [{label}] strings >= {minlen}: {len(long)}; final-class tokens {T}, non-final {NF} ({NF / T if T else float("nan"):.3f}), string-initial {F} ({F / T if T else float("nan"):.3f}); strings with >= 2 final-class elements {two} / {len(long)} ({two / len(long) if long else float("nan"):.3f})' + (f'  [< {3 / len(long):.3f}]' if two == 0 and long else ''))
        per = {a: (tot[a], nf[a]) for a in tot}
        return dict(n=len(long), T=T, NF=NF, share=NF / T if T else None, first=F, two=two, two_share=two / len(long) if long else None, per={str(k): v for k, v in per.items()})
    def top_finals(names, k=10): return [a for a, _ in collections.Counter(n[-1] for n in names if len(n) >= 2).most_common(k)]
    P('\n##### (3a) heads / name finals in NON-final position, and two in one string')
    for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
        names = indus_strings(LV, 'wh', 2)
        RES[f'indus_heads_nonfinal_{LV}'] = nonfinal_share(names, HEADS, f'Indus {LV}: paradigm HEADS (12 signs)')
        RES[f'indus_top10finals_nonfinal_{LV}'] = nonfinal_share(names, set(top_finals(names)), f'Indus {LV}: 10 commonest finals')
        if LV == 'seq_raw':
            # head co-occurrence vs chance: heads permuted among positions within strings? simpler: expected under independence of head counts per string
            long = [n for n in names if len(n) >= 3]; r = random.Random(3); toks = [a for n in long for a in n]; two_null = []
            for _ in range(200):
                r.shuffle(toks); i = 0; c = 0
                for n in long:
                    s = toks[i:i + len(n)]; i += len(n); c += sum(1 for a in s if a in HEADS) >= 2
                two_null.append(c)
            P(f'    two heads in one string: observed {RES["indus_heads_nonfinal_seq_raw"]["two"]} vs token-shuffle null {sum(two_null) / 200:.1f} (2.5% {q(two_null, 0.025)}, 97.5% {q(two_null, 0.975)})')
            RES['indus_two_heads_null_raw'] = dict(obs=RES['indus_heads_nonfinal_seq_raw']['two'], null=sum(two_null) / 200, lo=q(two_null, 0.025), hi=q(two_null, 0.975))
            # which heads stand non-final, and with what follows
            long = [n for n in names if len(n) >= 3]; foll = collections.Counter()
            for n in long:
                for i, a in enumerate(n[:-1]):
                    if a in HEADS: foll[(a, n[-1] if n[-1] in HEADS else 'nonhead-final')] += 1
            P('    non-final head -> what ends the string:', foll.most_common(20))
    for nm in ['ur3_name_elems', 'ur3_name_signs', 'linb_names', 'latin_names', 'ob_name_signs']:
        nl = jl(REFS[nm]); RES[f'{nm}_top10finals_nonfinal'] = nonfinal_share(nl, set(top_finals(nl)), f'{nm}: 10 commonest name-final elements')
    th = jl(REFS['ur3_name_elems']); theo = {a for n in th for a in n if a.startswith('{d}')}
    RES['ur3_theophoric_nonfinal'] = nonfinal_share(th, theo, 'Ur III names: theophoric {d}DN elements')
    nt = jl(REFS['ur3_name_title']); titles = {a for n in nt for a in n if a.startswith('T:')}
    RES['ur3_titles_nonfinal'] = nonfinal_share(nt, titles, 'Ur III name + title: TITLE words')
    # Ur III legends with >= 2 title-kind lines (titles stack: dub-sar + dumu PN)
    recs = [json.loads(l) for l in open(C61 + 'ur3_legend_fields.jsonl')]
    TW = {'dumu', 'dub-sar', 'arad2', 'lu2', 'aga3-us2', 'ugula', 'dam-gar3', 'sagi', 'dam', 'ensi2', 'sipa', 'gudu4', 'ma2-lah5', 'szabra', 'nu-banda3', 'nu-banda3-gu4', 'muhaldim', 'dumu-munus', 'ra2-gaba', 'lugal', 'sa12-du5', 'szagina', 'kiszib3', 'sanga', 'sukkal', 'simug', 'nagar', 'aszgab', 'kuruszda'}
    multi = sum(1 for r in recs if sum(1 for l in r['lines'][1:] if l.split() and l.split()[0].lower() in TW) >= 2)
    P(f'  Ur III distinct legends with >= 2 title/kin lines: {multi} / {len(recs)} ({multi / len(recs):.3f}) -- titles stack; name finals do not')
    RES['ur3_multi_title'] = dict(multi=multi, n=len(recs))
    P('\n##### (3b) head-only texts (no middle) vs Ur III title-only legends')
    for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
        objs = objs_for(LV)
        for ot in ['all', 'seal']:
            oo = [o for o in objs if (ot == 'all' or o['ot'] == ot)]
            withhead = [o for o in oo if o['wh'] and o['wh'][-1] in HEADS]
            only1 = sum(1 for o in withhead if len(o['wh']) == 1)
            only2 = sum(1 for o in withhead if len(o['wh']) == 2 and o['lab'][[i for i, a in enumerate(o['seq']) if a == o['wh'][0]][0]] == 'TITLE')
            nomid = sum(1 for o in withhead if all(l != 'NAME' for l in o['lab']))
            P(f'  Indus {LV} {ot}: texts with a head {len(withhead)}; head alone {only1} ({only1 / len(withhead):.3f}); head + its qualifier only {only2} ({only2 / len(withhead):.3f}); no NAME element at all {nomid} ({nomid / len(withhead):.3f})')
            RES[f'indus_headonly_{LV}_{ot}'] = dict(withhead=len(withhead), head_alone=only1, head_qual=only2, nomid=nomid)
    P('  Ur III: title-only legends 3 of 17,964 complete legends (0.0002; from loop61_prep); a title without a name is < 3/n.')
    P('\n##### (4) replication: cycle-1 and cycle-2 statistics on held-out Wells sites (not MD/H) and on IM77, three merge levels')
    for LV in ['seq_raw', 'seq_strong', 'seq_all']:
        ho = indus_strings(LV, 'wh', 2, subset=lambda o: not o['big'])
        RES[f'heldout_wh_{LV}'] = boot(ho, f'held-out sites {LV} WITH HEAD', nboot=max(10, NB // 2))
        hf = [n for n in ho if n[-1] in HEADS]
        RES[f'heldout_qual_head_{LV}'] = pf_test(last2(hf), f'held-out sites {LV} qualifier -> HEAD (n={len(hf)})', nperm=min(NPERM, 1000))
        mdh = indus_strings(LV, 'wh', 2, subset=lambda o: o['big'])
        RES[f'mdh_wh_{LV}'] = boot(mdh, f'Mohenjo-daro + Harappa {LV} WITH HEAD', nboot=max(10, NB // 2))
    im = indus_strings('im77', 'wh', 2); imho = indus_strings('im77', 'wh', 2, subset=lambda o: not o['big'])
    RES['im77_heldout_wh'] = boot(imho, 'IM77 held-out sites WITH HEAD', nboot=max(10, NB // 2))
    hf = [n for n in imho if n[-1] in HEADS]; RES['im77_heldout_qual_head'] = pf_test(last2(hf), f'IM77 held-out sites qualifier -> HEAD (n={len(hf)})', nperm=min(NPERM, 1000))
    save('loop61_cycle3', RES)
