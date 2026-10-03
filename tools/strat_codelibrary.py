"""Strategy: which known NON-LINGUISTIC sign system does the Indus corpus statistically resemble?

Instead of asking which language the signs encode, build a library of machine-readable corpora of known sign
systems (heraldic blazons, Inka khipu, Mesopotamian seal legends, Proto-Elamite entries) and language controls
(Latin inscriptions, Linear B tablet lines), compute one battery of statistics identically for every system at a
matched text-length distribution, standardise, and rank systems by distance from Indus. Bootstrap the ranking.

Stages:
  python3 tools/strat_codelibrary.py prep   # build data/codelib/<system>.jsonl from the raw sources
  python3 tools/strat_codelibrary.py run    # battery, distances, bootstrap -> data/derived/strat_codelibrary.txt

Sources (see data/codelib/SOURCES.txt): Rietstap, Armorial general (archive.org OCR, French blazons);
Open Khipu Repository 2.1.0 (Zenodo 18025748, SQLite); CDLI ATF @seal sections (lang sux); CDLI Proto-Elamite
ATF; EDH API (Latin inscriptions, Italy); DAMOS Linear B (other-scripts/linear-a/data/damos_items.jsonl);
Indus: data/derived/merged-corpus-canonical.json (seq_raw, seq_strong, seq_all).
"""
import sys, os, re, json, math, random, collections, sqlite3, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, 'data', 'codelib')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
OUT = os.path.join(ROOT, 'data', 'derived', 'strat_codelibrary.txt')
MAXLEN = 13


def dump(name, texts, note):
    os.makedirs(LIB, exist_ok=True)
    with open(os.path.join(LIB, name + '.jsonl'), 'w') as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({'id': i, 'seq': list(t)}, ensure_ascii=False) + '\n')
    toks = sum(len(t) for t in texts)
    types = len({s for t in texts for s in t})
    print(f'{name}: {len(texts)} texts, {toks} tokens, {types} types. {note}')
    return f'{name}: {len(texts)} texts, {toks} tokens, {types} types. {note}'


# ----------------------------------------------------------------------------------------------- prep
def prep_indus():
    d = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    out = {}
    for key in ('seq_raw', 'seq_strong', 'seq_all'):
        out[key] = [tuple(f'W{s}' for s in x[key]) for x in d if x.get(key)]
    seals = [tuple(f'W{s}' for s in x['seq_raw']) for x in d if x.get('seq_raw') and x['type'].startswith('SEAL')]
    return out, seals


def prep_heraldry():
    txt = ''
    for fn in ('rietstap1.txt', 'rietstap2.txt'):
        txt += open(os.path.join(SCR, 'codelib', fn), errors='ignore').read()
    txt = txt.replace('-\n', '').replace('- \n', '')
    paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', txt)]
    ents = []
    for p in paras:
        if not p:
            continue
        if re.search(r' — ', p) or re.match(r'^\d+\.\s', p):
            ents.append(p)
        elif ents and len(p) < 300:
            ents[-1] += ' ' + p
    TINC = (r"(D'or|D'argent|D'azur|De gueules|De sable|De sinople|De pourpre|Parti|Coupé|Tranché|Taillé|Écartelé|"
            r"Ecartelé|Tiercé|Burelé|Fascé|Palé|Bandé|Losangé|Échiqueté|Echiqueté|Gironné|Vairé|De vair|D'hermine|"
            r"Chevronné|Contre)")
    bl = []
    for e in ents:
        m = re.search(r' — [^.]*\.\s*(' + TINC + r'.*)$', e)
        if m:
            bl.append(m.group(1))
    STOP = {'de', 'à', 'et', 'au', 'un', 'une', 'en', 'du', 'la', 'le', 'les', 'l', 'd', 'aux', 'a', 'i', 'c', 'p',
            'v', 't', 'des', 'qu', 'ne', 's', 'n', 'e', 'j', 'r', 'u', 'o', 'm', 'f', 'g', 'h', 'b', 'k', 'x', 'y', 'z'}
    seqs = []
    for b in bl:
        ws = []
        for w in re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç'\-]+", b.lower()):
            w = re.sub(r"^(d'|l'|qu')", '', w).strip("'-")
            if w and w not in STOP:
                ws.append(w)
        if ws:
            seqs.append(ws)
    cnt = collections.Counter(w for s in seqs for w in s)
    seqs = [tuple(w for w in s if cnt[w] >= 5) for s in seqs]   # drop OCR-noise hapax forms
    seqs = [s for s in seqs if s]
    return seqs


def prep_khipu():
    c = sqlite3.connect(os.path.join(SCR, 'codelib/open-khipu-repository-2.1.0/data/khipu.db'))
    col = {}
    for cid, cc in c.execute("select CORD_ID, COLOR_CD_1 from ascher_cord_color where PCORD_FLAG=0"):
        col.setdefault(cid, (cc or '').strip() or '?')
    val = collections.defaultdict(float)
    for cid, v in c.execute("select CORD_ID, TOTAL_VALUE from knot_cluster"):
        try:
            val[cid] += float(v or 0)
        except (TypeError, ValueError):
            pass

    def vclass(v):
        if v <= 0:
            return '0'
        if v < 10:
            return '1'
        if v < 100:
            return '10'
        if v < 1000:
            return '100'
        return '1000'
    clusters = collections.defaultdict(list)
    for kid, cid, clid, co, lvl in c.execute(
            "select KHIPU_ID, CORD_ID, CLUSTER_ID, CORD_ORDINAL, CORD_LEVEL from cord where CORD_LEVEL=1"):
        clusters[(kid, clid)].append((co, cid))
    seqs = []
    for k, cords in clusters.items():
        cords.sort()
        seqs.append(tuple(f'{col.get(cid, "?")}:{vclass(val.get(cid, 0))}' for _, cid in cords))
    return seqs


def prep_ur3():
    legs = []
    cur = None
    lang = None
    for l in open(os.path.join(SCR, 'cdli.atf'), errors='ignore'):
        if l.startswith('&'):
            cur = None
            lang = None
            continue
        if l.startswith('#atf: lang'):
            lang = l.split()[-1]
        if l.startswith('@seal'):
            cur = []
            legs.append((lang, cur))
            continue
        if l[:1] in '@$#>':
            if l.startswith('@'):
                cur = None
            continue
        if cur is not None and re.match(r"^\d+'?\.", l):
            parts = l.split(None, 1)
            toks = parts[1] if len(parts) > 1 else ''
            toks = re.sub(r'[\[\]#?!<>*]', '', toks).lower().split()
            cur.extend(t for t in toks if t not in ('x', '...', '($', 'blank)'))
    L = {tuple(c) for l, c in legs if l == 'sux' and 2 <= len(c) <= 20}   # one copy per distinct legend
    return sorted(L)


def prep_latin():
    seqs = []
    for o in range(0, 6000, 1000):
        for it in json.load(open(os.path.join(SCR, f'codelib/edh_{o}.json')))['items']:
            t = it.get('transcription') or ''
            t = re.sub(r'\[-+\]|\[\.\.\.\]|\[---\]', ' ', t)
            t = re.sub(r'[\[\]()<>{}!?]', '', t)       # expand abbreviations, keep restored text
            t = t.replace('/', ' ')
            ws = [w for w in re.findall(r"[a-zA-Z]+", t.lower())]
            if ws:
                seqs.append(tuple(ws))
    return seqs


def prep_linearb():
    seqs = []
    for l in open(os.path.join(ROOT, 'other-scripts/linear-a/data/damos_items.jsonl')):
        it = json.loads(l)
        for line in it['content'].split('\n'):
            line = re.sub(r'^\s*\.?\d+[a-z]?\s+', ' ', line)
            if 'vac' in line or 'vest' in line:
                continue
            toks = []
            for w in line.split():
                w = w.strip("[],'/⟦⟧").replace('[', '').replace(']', '')
                if not w or w in ('[', ']', ',', '/', "'"):
                    continue
                if re.fullmatch(r'\d+', w):
                    toks.append('NUM')
                else:
                    toks.append(w)
            if toks:
                seqs.append(tuple(toks))
    return seqs


def prep_pe():
    seqs = []
    for l in open(os.path.join(SCR, 'pe.atf'), errors='ignore'):
        if not re.match(r"^\d+'?\.", l):
            continue
        parts = l.split(None, 1)
        if len(parts) < 2:
            continue
        toks = []
        for w in parts[1].split():
            if w in (',', ):
                continue
            m = re.match(r'^[\d/]+\((N\d+[A-Z]?)\)', w)
            if m:
                toks.append(m.group(1))
                continue
            w = re.sub(r'[#?!\[\]]', '', w)
            if w in ('x', '...', ''):
                continue
            toks.append(w)
        if toks:
            seqs.append(tuple(toks))
    return seqs


def bigram_gen(texts, seed=0):
    rnd = random.Random(seed)
    big = collections.defaultdict(collections.Counter)
    for t in texts:
        p = '<s>'
        for s in t:
            big[p][s] += 1
            p = s
    cache = {k: (list(v.keys()), list(v.values())) for k, v in big.items()}
    out = []
    for t in texts:
        p = '<s>'
        g = []
        for _ in range(len(t)):
            ks, ws = cache.get(p) or cache['<s>']
            s = rnd.choices(ks, ws)[0]
            g.append(s)
            p = s
        out.append(tuple(g))
    return out


def prep():
    notes = []
    ind, seals = prep_indus()
    notes.append(dump('indus_raw', ind['seq_raw'], 'all inscribed objects, Wells numbers (seq_raw)'))
    notes.append(dump('indus_strong', ind['seq_strong'], 'strong merges'))
    notes.append(dump('indus_all', ind['seq_all'], 'strong+probable merges'))
    notes.append(dump('indus_seals', seals, 'seals only, seq_raw'))
    notes.append(dump('indus_bigram', bigram_gen(ind['seq_raw']), 'control: bigram generator trained on indus_raw, same lengths'))
    notes.append(dump('heraldry', prep_heraldry(), 'Rietstap Armorial general (2 vols, archive.org OCR): blazon words after function words removed; forms seen <5 times dropped as OCR noise'))
    notes.append(dump('khipu', prep_khipu(), 'OKR khipu.db: text = one cord cluster (pendant cords, level 1, in order); sign = Ascher colour code : value class (0, 1-9, 10-99, 100-999, 1000+)'))
    notes.append(dump('ur3_legends', prep_ur3(), 'CDLI @seal legends, lang sux, one copy per distinct legend; words (name, title, patronym) as signs'))
    notes.append(dump('latin_edh', prep_latin(), 'EDH Italy, 6,000 inscriptions, abbreviations expanded, words as signs'))
    notes.append(dump('linear_b', prep_linearb(), 'DAMOS Linear B: text = one tablet line; sign groups, ideograms, NUM'))
    notes.append(dump('proto_elamite', prep_pe(), 'CDLI Proto-Elamite ATF: text = one entry line; signs plus numeral-sign type (count dropped)'))
    with open(os.path.join(LIB, 'SOURCES.txt'), 'w') as f:
        f.write('Code library for tools/strat_codelibrary.py. One JSONL per system; each line {"id", "seq"}.\n\n')
        f.write('\n'.join(notes) + '\n')
        f.write('\nRaw sources: Rietstap Armorial general vols 1-2 (archive.org armorialgnra01rietuoft, armorialgnra02rietuoft, djvu OCR);\n'
                'Open Khipu Repository 2.1.0 (https://doi.org/10.5281/zenodo.5037551, record 18025748, data/khipu.db);\n'
                'CDLI ATF dump (github cdli-gh/data) @seal sections; CDLI Proto-Elamite ATF; EDH API\n'
                '(https://edh.ub.uni-heidelberg.de/data/api/inschrift/suche?land=it, offsets 0-5000); DAMOS Linear B via other-scripts/linear-a.\n'
                'Not obtained: RGZM Samian (names on terra sigillata) needs a login; no machine-readable merchants or masons marks corpus found\n'
                '(Zenodo search gave only scanned articles); GitHub API search returned no blazon dataset; DrawShield has no bulk catalogue.\n')


# ----------------------------------------------------------------------------------------------- battery
def load(name):
    return [tuple(json.loads(l)['seq']) for l in open(os.path.join(LIB, name + '.jsonl'))]


def match_lengths(texts, target, rnd, cap=3000):
    """Subsample texts so that the length histogram matches target (Indus) as far as the supply allows."""
    by = collections.defaultdict(list)
    for t in texts:
        L = min(len(t), MAXLEN)
        by[L].append(t)
    tot = sum(target.values())
    p = {L: n / tot for L, n in target.items()}
    n = min(cap, min(int(len(by[L]) / p[L]) for L in p if p[L] > 0.01))
    out = []
    for L, pl in p.items():
        k = round(n * pl)
        pool = by.get(L, [])
        if k and pool:
            out += rnd.sample(pool, min(k, len(pool)))
    return out


def H(counter):
    n = sum(counter.values())
    return -sum(v / n * math.log2(v / n) for v in counter.values()) if n else 0.0


def n80(counter):
    n = sum(counter.values())
    acc = 0
    for i, (_, v) in enumerate(counter.most_common(), 1):
        acc += v
        if acc >= 0.8 * n:
            return i
    return len(counter)


def battery(texts, rnd):
    texts = [t for t in texts if t]
    n = len(texts)
    L = [len(t) for t in texts]
    uni = collections.Counter(s for t in texts for s in t)
    V = len(uni)
    N = sum(uni.values())
    r = {}
    r['mean_len'] = sum(L) / n
    r['median_len'] = st.median(L)
    r['single_share'] = sum(1 for l in L if l == 1) / n
    multi = [t for t in texts if len(t) >= 2]
    r['unique_share'] = len(set(multi)) / len(multi) if multi else 0
    # within-text repetition vs unigram shuffle null
    def rep(ts):
        return sum(1 for t in ts if len(t) >= 2 and len(set(t)) < len(t)) / max(1, sum(1 for t in ts if len(t) >= 2))
    ks, ws = zip(*uni.items())
    null = [tuple(rnd.choices(ks, ws, k=len(t))) for t in texts]
    r['repeat_ratio'] = rep(texts) / rep(null) if rep(null) else 0
    # positional entropies (texts >=3 signs), relative to log2 V
    first = collections.Counter(t[0] for t in texts if len(t) >= 3)
    last = collections.Counter(t[-1] for t in texts if len(t) >= 3)
    mid = collections.Counter(s for t in texts if len(t) >= 3 for s in t[1:-1])
    lv = math.log2(V) if V > 1 else 1
    r['H_first/H_mid'] = H(first) / H(mid) if H(mid) else 0
    r['H_last/H_mid'] = H(last) / H(mid) if H(mid) else 0
    r['H_uni_norm'] = H(uni) / lv
    # closed class at the ends: types covering 80% of first/last tokens, relative to all tokens
    r['n80_first/n80_all'] = n80(first) / n80(uni)
    r['n80_last/n80_all'] = n80(last) / n80(uni)
    # Zipf slope
    fr = sorted(uni.values(), reverse=True)
    xs = [math.log(i + 1) for i in range(len(fr))]
    ys = [math.log(f) for f in fr]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    r['zipf_slope'] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(1e-9, sum((x - mx) ** 2 for x in xs))
    r['hapax_type_share'] = sum(1 for v in uni.values() if v == 1) / V
    # bigram MI and conditional entropy (within texts)
    big = collections.Counter((t[i], t[i + 1]) for t in texts for i in range(len(t) - 1))
    nb = sum(big.values())
    if nb:
        a = collections.Counter()
        b = collections.Counter()
        for (x, y), v in big.items():
            a[x] += v
            b[y] += v
        mi = sum(v / nb * math.log2(v / nb / (a[x] / nb) / (b[y] / nb)) for (x, y), v in big.items())
        h2 = H(big) - H(a)
        r['bigram_MI/H1'] = mi / H(b) if H(b) else 0
        r['h2/H1'] = h2 / H(b) if H(b) else 0
    else:
        r['bigram_MI/H1'] = r['h2/H1'] = 0
    # uniqueness vs bigram generator (S321 statistic), texts >= 2
    gen = bigram_gen(multi, seed=rnd.randrange(10 ** 9))
    ug = len(set(gen)) / len(gen) if gen else 1
    r['uniq_vs_bigram'] = r['unique_share'] / ug if ug else 0
    # nesting: share of texts (2-5 signs) that occur contiguously inside a longer text
    short = [t for t in texts if 2 <= len(t) <= 5]
    longer = collections.defaultdict(set)
    for t in texts:
        for k in range(2, 6):
            if len(t) > k:
                for i in range(len(t) - k + 1):
                    longer[k].add(t[i:i + k])
    r['nest_share'] = sum(1 for t in short if t in longer[len(t)]) / len(short) if short else 0
    # frames
    fr2 = collections.Counter((t[0], t[-1]) for t in multi)
    r['frames_per_text'] = len(fr2) / len(multi) if multi else 0
    r['top10_frame_share'] = sum(v for _, v in fr2.most_common(10)) / len(multi) if multi else 0
    # type-token ratio at 2,000 tokens (rarefaction)
    toks = [s for t in texts for s in t]
    rnd.shuffle(toks)
    k = min(2000, len(toks))
    r['types_per_2000tok'] = len(set(toks[:k])) * 2000 / k
    return r


STATS = ['mean_len', 'single_share', 'unique_share', 'repeat_ratio', 'H_first/H_mid', 'H_last/H_mid', 'H_uni_norm',
         'n80_first/n80_all', 'n80_last/n80_all', 'zipf_slope', 'hapax_type_share', 'bigram_MI/H1', 'h2/H1',
         'uniq_vs_bigram', 'nest_share', 'frames_per_text', 'top10_frame_share', 'types_per_2000tok']
# 'median_len' kept in the table but not in the distance (redundant with mean_len)
SYSTEMS = ['heraldry', 'khipu', 'ur3_legends', 'proto_elamite', 'latin_edh', 'linear_b', 'indus_bigram']


def run():
    rnd = random.Random(1)
    lines = []
    P = lines.append
    P('strat_codelibrary: which known sign system does the Indus corpus statistically resemble?')
    P('Battery computed identically per system on a sample length-matched to the Indus length histogram (1..13 signs, cap 3,000 texts).')
    P('Distance: statistics z-scored across all systems (incl. Indus), Euclidean. Bootstrap: texts resampled with replacement, 200x.')
    P('')
    data = {s: load(s) for s in SYSTEMS}
    indus = {k: load(k) for k in ('indus_raw', 'indus_strong', 'indus_all', 'indus_seals')}
    target = collections.Counter(min(len(t), MAXLEN) for t in indus['indus_raw'])
    P('Library: ' + ', '.join(f'{s} n={len(v)}' for s, v in list(indus.items()) + list(data.items())))
    P('')
    for ind_key in ('indus_raw', 'indus_strong', 'indus_all', 'indus_seals'):
        tgt = collections.Counter(min(len(t), MAXLEN) for t in indus[ind_key])
        matched = {s: match_lengths(v, tgt, rnd) for s, v in data.items()}
        matched[ind_key] = indus[ind_key]
        names = [ind_key] + SYSTEMS
        full = {s: battery(matched[s], rnd) for s in names}
        P(f'=== Indus version: {ind_key} (n={len(indus[ind_key])}); matched sample sizes: ' + ', '.join(f'{s}={len(matched[s])}' for s in SYSTEMS))
        # table
        P('stat'.ljust(20) + ''.join(s[:13].rjust(14) for s in names))
        for k in ['median_len'] + STATS:
            P(k.ljust(20) + ''.join(f'{full[s][k]:14.3f}' for s in names))
        # z-scores and distance
        mu = {k: st.mean(full[s][k] for s in names) for k in STATS}
        sd = {k: st.pstdev([full[s][k] for s in names]) or 1e-9 for k in STATS}
        z = {s: {k: (full[s][k] - mu[k]) / sd[k] for k in STATS} for s in names}
        dist = {s: math.sqrt(sum((z[s][k] - z[ind_key][k]) ** 2 for k in STATS)) for s in SYSTEMS}
        order = sorted(dist, key=dist.get)
        P('')
        P('Distance from Indus (z-space): ' + ', '.join(f'{s} {dist[s]:.2f}' for s in order))
        near = order[0]
        contrib = sorted(((abs(z[near][k] - z[ind_key][k]), k) for k in STATS))
        P(f'Nearest: {near}. Statistics on which it is closest to Indus (|dz|): ' + ', '.join(f'{k} {d:.2f}' for d, k in contrib[:6]))
        P(f'Statistics on which {near} differs most from Indus: ' + ', '.join(f'{k} {d:.2f}' for d, k in contrib[-5:]))
        # residue: statistics where Indus lies outside the range of every other system
        res = []
        for k in STATS:
            vals = [full[s][k] for s in SYSTEMS]
            v = full[ind_key][k]
            if v > max(vals) or v < min(vals):
                gap = (v - max(vals)) if v > max(vals) else (v - min(vals))
                res.append((k, v, min(vals), max(vals), gap / sd[k]))
        P('Residue (Indus outside the range of all library systems): ' + ('; '.join(
            f'{k}={v:.3f} vs [{lo:.3f},{hi:.3f}] ({g:+.2f} sd)' for k, v, lo, hi, g in res) or 'none'))
        # per-statistic nearest system
        P('Per-statistic nearest system: ' + ', '.join(
            f'{k}->{min(SYSTEMS, key=lambda s: abs(z[s][k] - z[ind_key][k]))}' for k in STATS))
        # bootstrap
        if ind_key in ('indus_raw', 'indus_seals'):
            B = 200
            wins = collections.Counter()
            ranks = collections.defaultdict(list)
            for b in range(B):
                rb = random.Random(1000 + b)
                bz = {}
                for s in names:
                    smp = [rb.choice(matched[s]) for _ in range(len(matched[s]))]
                    r = battery(smp, rb)
                    bz[s] = {k: (r[k] - mu[k]) / sd[k] for k in STATS}
                db = {s: math.sqrt(sum((bz[s][k] - bz[ind_key][k]) ** 2 for k in STATS)) for s in SYSTEMS}
                ob = sorted(db, key=db.get)
                wins[ob[0]] += 1
                for i, s in enumerate(ob):
                    ranks[s].append(i + 1)
            P(f'Bootstrap ({B}x): nearest-system share: ' + ', '.join(f'{s} {wins[s] / B:.2f}' for s in order))
            P('Bootstrap mean rank: ' + ', '.join(f'{s} {st.mean(ranks[s]):.2f}' for s in order))
        P('')
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1:2] == ['prep']:
        prep()
    else:
        run()
