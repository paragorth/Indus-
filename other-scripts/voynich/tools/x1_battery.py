"""X1: 'undeciphered signature' hunt. Corpus loaders + a broad statistic battery.

A corpus = list of texts; a text = list of sign tokens (str). Text unit = line / inscription / entry line.
Word dividers are dropped everywhere (Indus and Proto-Elamite have none).

Classes:
  U  undeciphered: indus (canonical seq_raw, read-only), linear_a (lines), proto_elamite (entry lines),
     voynich (ZL3b paragraph lines, glyph units ch/sh/cth/ckh/cph/cfh merged)
  D  deciphered writing + designed codes (data/derived/dark/loop32_corpora + Gutenberg char-level texts)
  T  test-only (khipu: numeric code understood, wider system not; Voynich word-level; Indus variants)
"""
import os, sys, json, math, re, random, zlib, bz2, lzma
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OS = os.path.join(REPO, 'other-scripts')
LIB = os.path.join(REPO, 'data', 'derived', 'dark', 'loop32_corpora')
RES = os.path.join(OS, 'voynich', 'data', 'results')


def _jsonl(name):
    out = []
    for line in open(os.path.join(LIB, name + '.jsonl')):
        s = json.loads(line)['seq']
        if s:
            out.append([str(x) for x in s])
    return out


def load_indus(level='seq_raw'):
    d = json.load(open(os.path.join(REPO, 'data', 'derived', 'merged-corpus-canonical.json')))
    return [[str(x) for x in r[level]] for r in d if r.get(level)]


def load_linear_a():
    d = json.load(open(os.path.join(OS, 'linear-a', 'data', 'corpus.json')))
    texts = []
    for ins in d:
        cur = []
        for t in ins['tokens']:
            if t['t'] == 'nl':
                if cur:
                    texts.append(cur)
                cur = []
            elif t['t'] == 'word':
                cur += t['s']
            elif t['t'] == 'num':
                cur.append('NUM')
                for f in t.get('frac', []) or []:
                    cur.append('FR:' + str(f))
            elif t['t'] == 'logo':
                cur.append('L:' + t['v'])
        if cur:
            texts.append(cur)
    return texts


def load_voynich_glyph():
    lines = vlib.load_voynich('ZL3b', ltypes=('P',), drop_uncertain=True)
    out = []
    for L in lines:
        s = [g for w in L['words'] for g in vlib.glyphs(w) if g != '?']
        if s:
            out.append(s)
    return out


def load_voynich_words():
    lines = vlib.load_voynich('ZL3b', ltypes=('P',), drop_uncertain=True)
    return [L['words'] for L in lines if L['words']]


def load_gutenberg_chars(key, max_tokens=250000):
    lines = vlib.load_ref(key)
    out, n = [], 0
    for L in lines:
        s = [c for w in L['words'] for c in w]
        if len(s) >= 3:
            out.append(s); n += len(s)
        if n >= max_tokens:
            break
    return out


def load_gutenberg_words(key, max_tokens=60000):
    lines = vlib.load_ref(key)
    out, n = [], 0
    for L in lines:
        out.append(L['words']); n += len(L['words'])
        if n >= max_tokens:
            break
    return out


def chars_from_words(texts):
    return [[c for w in t for c in w] for t in texts if t]


CORPORA = {
    # name: (class, granularity, loader)
    'indus': ('U', 'sign', lambda: load_indus('seq_raw')),
    'linear_a': ('U', 'sign', load_linear_a),
    'proto_elamite': ('U', 'sign', lambda: _jsonl('proto_elamite')),
    'voynich': ('U', 'sign', load_voynich_glyph),
    # deciphered, sign level
    'ur3_syll': ('D', 'sign', lambda: _jsonl('ur3_syll')),
    'ur3_names_syll': ('D', 'sign', lambda: _jsonl('ur3_names_syll')),
    'linb_syll': ('D', 'sign', lambda: _jsonl('linb_syll')),
    'proto_cuneiform': ('D', 'sign', lambda: _jsonl('proto_cuneiform')),
    'latin_edh_chars': ('D', 'sign', lambda: chars_from_words(_jsonl('latin_edh'))),
    'runes_chars': ('D', 'sign', lambda: chars_from_words(_jsonl('runes_words'))),
    'lat_caesar_ch': ('D', 'sign', lambda: load_gutenberg_chars('Latin-Caesar')),
    'lat_descartes_ch': ('D', 'sign', lambda: load_gutenberg_chars('Latin-Descartes')),
    'ita_manzoni_ch': ('D', 'sign', lambda: load_gutenberg_chars('Italian-Manzoni')),
    'ita_dante_ch': ('D', 'sign', lambda: load_gutenberg_chars('Italian-Dante')),
    'ger_kafka_ch': ('D', 'sign', lambda: load_gutenberg_chars('German-Kafka')),
    'spa_cervantes_ch': ('D', 'sign', lambda: load_gutenberg_chars('Spanish-Cervantes')),
    'icd10': ('D', 'sign', lambda: _jsonl('icd10')),
    'hts': ('D', 'sign', lambda: _jsonl('hts')),
    'aircraft_reg': ('D', 'sign', lambda: _jsonl('aircraft_reg')),
    'chess_eco': ('D', 'sign', lambda: _jsonl('chess_eco')),
    'chords': ('D', 'sign', lambda: _jsonl('chords')),
    # deciphered, word level
    'latin_edh_words': ('D', 'word', lambda: _jsonl('latin_edh')),
    'ur3_words': ('D', 'word', lambda: _jsonl('ur3_words')),
    'linb_words': ('D', 'word', lambda: _jsonl('linb_words')),
    'unicode_names': ('D', 'word', lambda: _jsonl('unicode_names')),
    'heraldry': ('D', 'word', lambda: _jsonl('heraldry')),
    'runes_words': ('D', 'word', lambda: _jsonl('runes_words')),
    'spa_cervantes_w': ('D', 'word', lambda: load_gutenberg_words('Spanish-Cervantes')),
    # test-only
    'khipu': ('T', 'sign', lambda: _jsonl('khipu')),
    'voynich_words': ('T', 'word', load_voynich_words),
    'indus_strong': ('T', 'sign', lambda: load_indus('seq_strong')),
    'indus_im77': ('T', 'sign', lambda: _jsonl('indus_im77')),
}


def window(texts, budget, rng):
    """Contiguous run of whole texts starting at a random text, about `budget` tokens (wraps around)."""
    total = sum(len(t) for t in texts)
    if budget is None or budget >= total:
        return list(texts)
    n = len(texts); i = rng.randrange(n); out = []; s = 0
    while s < budget:
        t = texts[i % n]; out.append(t); s += len(t); i += 1
    return out


# ------------------------------------------------------------------ helpers
def H(counter):
    n = sum(counter.values())
    if n == 0:
        return float('nan')
    p = np.array(list(counter.values()), float) / n
    return float(-(p * np.log2(p)).sum())


def cond_H(pairs):
    """H(Y|X) from Counter of (x,y)."""
    joint = H(pairs)
    xs = Counter()
    for (x, y), c in pairs.items():
        xs[x] += c
    return joint - H(xs)


def MI_pairs(pairs):
    xs, ys = Counter(), Counter()
    for (x, y), c in pairs.items():
        xs[x] += c; ys[y] += c
    return H(xs) + H(ys) - H(pairs)


def jsd(c1, c2):
    keys = set(c1) | set(c2)
    n1, n2 = sum(c1.values()), sum(c2.values())
    p = np.array([c1.get(k, 0) / n1 for k in keys]); q = np.array([c2.get(k, 0) / n2 for k in keys])
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def kl_smooth(c, ref, alpha=0.5):
    keys = set(ref) | set(c)
    n, nr = sum(c.values()), sum(ref.values())
    k = len(keys)
    v = 0.0
    for key in keys:
        p = (c.get(key, 0) + alpha) / (n + alpha * k)
        q = (ref.get(key, 0) + alpha) / (nr + alpha * k)
        v += p * math.log2(p / q)
    return v


def to_bytes(ids, sep=None):
    a = np.asarray(ids, dtype='>u2')
    return a.tobytes()


COMPRESSORS = {
    'zlib': lambda b: len(zlib.compress(b, 9)),
    'bz2': lambda b: len(bz2.compress(b, 9)),
    'lzma': lambda b: len(lzma.compress(b, preset=6)),
}


def lz76(seq):
    """Lempel-Ziv 1976 complexity (number of phrases)."""
    n = len(seq); i, c = 0, 0
    s = seq
    while i < n:
        l = 1
        while i + l <= n:
            sub = s[i:i + l]
            # search in prefix s[0:i+l-1]
            found = False
            pref = s[:i + l - 1]
            m = len(sub)
            for j in range(0, len(pref) - m + 1):
                if pref[j:j + m] == sub:
                    found = True; break
            if not found:
                break
            l += 1
        c += 1; i += l
    return c


def lz78_phrases(seq):
    d = set(); w = (); c = 0
    for x in seq:
        w2 = w + (x,)
        if w2 in d:
            w = w2
        else:
            d.add(w2); c += 1; w = ()
    return c + (1 if w else 0)


def dfa_hurst(x, rng_scales=None):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    N = len(y)
    scales = [s for s in (4, 8, 16, 32, 64, 128, 256, 512) if s <= N // 4]
    if len(scales) < 3:
        return float('nan')
    F = []
    for s in scales:
        nseg = N // s
        segs = y[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        res = []
        for seg in segs:
            c = np.polyfit(t, seg, 1)
            res.append(np.mean((seg - np.polyval(c, t)) ** 2))
        F.append(math.sqrt(np.mean(res)))
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


# ------------------------------------------------------------------ battery
def battery(texts, seed=0):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    F = {}
    texts = [t for t in texts if t]
    toks = [x for t in texts for x in t]
    N = len(toks)
    cnt = Counter(toks)
    V = len(cnt)
    rank = {k: i + 1 for i, (k, _) in enumerate(cnt.most_common())}
    ids = [rank[x] for x in toks]
    lens = np.array([len(t) for t in texts], float)
    freqs = np.array(sorted(cnt.values(), reverse=True), float)
    p = freqs / N

    # --- size / shape (family 'size' and 'length')
    F['size_ntexts'] = len(texts); F['size_ntokens'] = N; F['size_types'] = V
    F['size_log_types'] = math.log(V)
    F['len_mean'] = lens.mean(); F['len_median'] = float(np.median(lens)); F['len_sd'] = lens.std()
    F['len_cv'] = lens.std() / lens.mean(); F['len_frac1'] = float((lens == 1).mean())
    F['len_frac2'] = float((lens == 2).mean()); F['len_frac_le3'] = float((lens <= 3).mean())
    F['len_q90_over_median'] = float(np.quantile(lens, 0.9) / max(1, np.median(lens)))
    F['len_max_over_mean'] = lens.max() / lens.mean()
    F['len_skew'] = float(((lens - lens.mean()) ** 3).mean() / (lens.std() ** 3 + 1e-9))
    F['len_log_mean'] = math.log(lens.mean())

    # --- lexicon / Zipf / Heaps
    F['lex_ttr'] = V / N; F['lex_herdan_c'] = math.log(V) / math.log(N)
    h1 = int((freqs == 1).sum()); h2c = int((freqs == 2).sum())
    F['lex_hapax_types'] = h1 / V; F['lex_hapax_tokens'] = h1 / N; F['lex_dis_types'] = h2c / V
    F['lex_top1'] = p[0]; F['lex_top5'] = p[:5].sum(); F['lex_top10'] = p[:10].sum()
    F['lex_top10pct_types_cover'] = p[:max(1, V // 10)].sum()
    F['lex_yule_k'] = 1e4 * (np.sum(freqs ** 2) - N) / N ** 2
    F['lex_simpson'] = float(np.sum(freqs * (freqs - 1)) / (N * (N - 1)))
    F['lex_n50_frac_types'] = float(np.searchsorted(np.cumsum(p), 0.5) + 1) / V
    F['lex_n80_frac_types'] = float(np.searchsorted(np.cumsum(p), 0.8) + 1) / V
    r = np.arange(1, V + 1)
    m2 = freqs >= 2
    if m2.sum() >= 3:
        c = np.polyfit(np.log(r[m2]), np.log(freqs[m2]), 1)
        F['zipf_slope_f2'] = c[0]
        pred = np.polyval(c, np.log(r[m2]))
        ss = np.sum((np.log(freqs[m2]) - pred) ** 2); st = np.sum((np.log(freqs[m2]) - np.log(freqs[m2]).mean()) ** 2)
        F['zipf_r2_f2'] = 1 - ss / st
        c2 = np.polyfit(np.log(r[m2]), np.log(freqs[m2]), 2)
        F['zipf_curv_f2'] = c2[0]
    k = min(10, V)
    F['zipf_slope_top10'] = np.polyfit(np.log(r[:k]), np.log(freqs[:k]), 1)[0]
    rel = r / V
    mm = (rel >= 0.1) & (rel <= 0.5) & (freqs >= 1)
    if mm.sum() >= 3:
        F['zipf_slope_mid'] = np.polyfit(np.log(r[mm]), np.log(freqs[mm]), 1)[0]
    # Heaps
    seen = set(); pts = []
    checkpoints = set(np.unique(np.geomspace(20, N, 20).astype(int)))
    for i, x in enumerate(toks, 1):
        seen.add(x)
        if i in checkpoints:
            pts.append((i, len(seen)))
    pts = np.array(pts, float)
    F['heaps_beta'] = np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)[0]
    # heaps under text-order shuffle -> vocabulary locality
    tt = texts[:]; rng.shuffle(tt)
    seen = set(); pts2 = []
    for i, x in enumerate((x for t in tt for x in t), 1):
        seen.add(x)
        if i in checkpoints:
            pts2.append((i, len(seen)))
    pts2 = np.array(pts2, float)
    F['heaps_locality'] = float(np.mean(np.log(pts2[:, 1]) - np.log(pts[:, 1])))
    F['heaps_types_at_half'] = len(set(toks[:N // 2])) / V
    # bigram Zipf
    bg = Counter((t[i], t[i + 1]) for t in texts for i in range(len(t) - 1))
    if len(bg) >= 10:
        bf = np.array(sorted(bg.values(), reverse=True), float)
        mb = bf >= 2
        if mb.sum() >= 3:
            F['zipf_bigram_slope'] = np.polyfit(np.log(np.arange(1, len(bf) + 1)[mb]), np.log(bf[mb]), 1)[0]
        F['bigram_hapax_types'] = float((bf == 1).mean())
        F['bigram_types_per_token'] = len(bg) / bf.sum()
        F['bigram_fill_log'] = math.log(len(bg)) / math.log(V * V) if V > 1 else float('nan')

    # --- entropy profile
    H1 = H(cnt); F['ent_h1'] = H1
    F['ent_h1_mm'] = H1 + (V - 1) / (2 * N * math.log(2))
    F['ent_h1_norm'] = H1 / math.log2(V) if V > 1 else float('nan')
    tri = Counter(((t[i], t[i + 1]), t[i + 2]) for t in texts for i in range(len(t) - 2))
    bgp = Counter(((a,), b) for (a, b), c in bg.items() for _ in [0])
    bgp = Counter({((a,), b): c for (a, b), c in bg.items()})
    h2 = cond_H(bgp) if bg else float('nan'); h3 = cond_H(tri) if tri else float('nan')
    F['ent_h2'] = h2; F['ent_h3'] = h3
    F['ent_h2_over_h1'] = h2 / H1 if H1 > 0 else float('nan')
    F['ent_h3_over_h1'] = h3 / H1 if H1 > 0 else float('nan')
    F['ent_h3_over_h2'] = h3 / h2 if h2 and h2 > 0 else float('nan')
    back = Counter({((b,), a): c for (a, b), c in bg.items()})
    F['ent_back_minus_fwd'] = (cond_H(back) - h2) / H1 if bg else float('nan')
    # shuffled-within-text baseline
    sh_texts = []
    for t in texts:
        u = t[:]; rng.shuffle(u); sh_texts.append(u)
    bg_s = Counter({((t[i],), t[i + 1]): 1 for t in []})
    bg_s = Counter()
    for t in sh_texts:
        for i in range(len(t) - 1):
            bg_s[((t[i],), t[i + 1])] += 1
    tri_s = Counter(((t[i], t[i + 1]), t[i + 2]) for t in sh_texts for i in range(len(t) - 2))
    if bg:
        F['ent_order_gain2'] = (cond_H(bg_s) - h2) / H1
    if tri:
        F['ent_order_gain3'] = (cond_H(tri_s) - h3) / H1
    # global shuffle baseline (sign identity spread over corpus)
    allt = toks[:]; rng.shuffle(allt)
    gt, k0 = [], 0
    for t in texts:
        gt.append(allt[k0:k0 + len(t)]); k0 += len(t)
    bg_g = Counter()
    for t in gt:
        for i in range(len(t) - 1):
            bg_g[((t[i],), t[i + 1])] += 1
    if bg:
        F['ent_glob_gain2'] = (cond_H(bg_g) - h2) / H1
        F['ent_text_cohesion'] = (cond_H(bg_g) - cond_H(bg_s)) / H1  # bag-of-signs per text information
    # block entropies on stream (within texts)
    for n in (2, 3, 4):
        ng = Counter(tuple(t[i:i + n]) for t in texts for i in range(len(t) - n + 1))
        if ng:
            F[f'ent_block{n}_per_sign'] = H(ng) / n / H1
            F[f'ngram{n}_type_token'] = len(ng) / sum(ng.values())
            F[f'ngram{n}_hapax_types'] = sum(1 for v in ng.values() if v == 1) / len(ng)

    q4 = Counter(((t[i], t[i + 1], t[i + 2]), t[i + 3]) for t in texts for i in range(len(t) - 3))
    if q4:
        F['ent_h4'] = cond_H(q4); F['ent_h4_over_h1'] = F['ent_h4'] / H1
    cmap = lambda x: min(rank[x], 16)
    c1 = Counter(cmap(x) for x in toks)
    c2 = Counter({((cmap(a),), cmap(b)): 0 for (a, b) in []})
    c2 = Counter()
    for (a, b), c in bg.items():
        c2[((cmap(a),), cmap(b))] += c
    if c2:
        F['ent_h2_coarse16_over_h1'] = cond_H(c2) / H(c1)
    if V > 1 and h2c > 0:
        F['lex_chao1_ratio'] = (V + h1 * h1 / (2 * h2c)) / V
    else:
        F['lex_chao1_ratio'] = float('nan')
    F['lex_types_ge10_frac'] = float((freqs >= 10).mean())
    lc = Counter(lens.astype(int).tolist())
    F['len_entropy'] = H(lc); F['len_mode_share'] = lc.most_common(1)[0][1] / len(lens)
    # --- positional effects
    long3 = [t for t in texts if len(t) >= 3]
    long2 = [t for t in texts if len(t) >= 2]
    init = Counter(t[0] for t in long2); fin = Counter(t[-1] for t in long2)
    mid = Counter(x for t in long3 for x in t[1:-1])
    if init:
        F['pos_h_init'] = H(init) / H1; F['pos_h_final'] = H(fin) / H1
        F['pos_top1_init'] = init.most_common(1)[0][1] / sum(init.values())
        F['pos_top1_final'] = fin.most_common(1)[0][1] / sum(fin.values())
        F['pos_kl_init'] = kl_smooth(init, cnt); F['pos_kl_final'] = kl_smooth(fin, cnt)
        F['pos_jsd_init_final'] = jsd(init, fin)
        F['pos_init_types_frac'] = len(init) / V; F['pos_final_types_frac'] = len(fin) / V
    if long3:
        pairs = Counter()
        for t in long3:
            for i, x in enumerate(t):
                pairs[(0 if i == 0 else (2 if i == len(t) - 1 else 1), x)] += 1
        F['pos_mi_ifm'] = MI_pairs(pairs) / H1
        # within-text shuffle baseline for same stat
        pairs_s = Counter()
        for t in long3:
            u = t[:]; rng.shuffle(u)
            for i, x in enumerate(u):
                pairs_s[(0 if i == 0 else (2 if i == len(u) - 1 else 1), x)] += 1
        F['pos_mi_ifm_excess'] = (MI_pairs(pairs) - MI_pairs(pairs_s)) / H1
        sec = Counter(t[1] for t in long3); pen = Counter(t[-2] for t in long3)
        F['pos_kl_second'] = kl_smooth(sec, cnt); F['pos_kl_penult'] = kl_smooth(pen, cnt)
        F['pos_kl_mid'] = kl_smooth(mid, cnt) if mid else float('nan')
        # position-restricted types
        tc = Counter(); ti = Counter(); tf = Counter()
        for t in long3:
            for i, x in enumerate(t):
                tc[x] += 1
                if i == 0: ti[x] += 1
                if i == len(t) - 1: tf[x] += 1
        common = [x for x in tc if tc[x] >= 5]
        if common:
            F['pos_frac_init_locked'] = sum(1 for x in common if ti[x] / tc[x] >= 0.8) / len(common)
            F['pos_frac_final_locked'] = sum(1 for x in common if tf[x] / tc[x] >= 0.8) / len(common)
            F['pos_frac_edge_avoid'] = sum(1 for x in common if (ti[x] + tf[x]) / tc[x] <= 0.05) / len(common)
            F['pos_frac_edge_only'] = sum(1 for x in common if (ti[x] + tf[x]) / tc[x] >= 0.9) / len(common)
            F['pos_mean_edge_share_sd'] = float(np.std([(ti[x] + tf[x]) / tc[x] for x in common]))
    long5 = [t for t in texts if len(t) >= 5]
    if len(long5) >= 20:
        pr = Counter()
        for t in long5:
            L = len(t)
            for i, x in enumerate(t):
                pr[(min(4, int(5 * i / L)), x)] += 1
        F['pos_mi_relpos5'] = MI_pairs(pr) / H1
        pr_s = Counter()
        for t in long5:
            u = t[:]; rng.shuffle(u); L = len(u)
            for i, x in enumerate(u):
                pr_s[(min(4, int(5 * i / L)), x)] += 1
        F['pos_mi_relpos5_excess'] = (MI_pairs(pr) - MI_pairs(pr_s)) / H1
    # length interactions
    if len(long2) >= 20:
        lb = lambda L: min(L, 8)
        F['edge_mi_final_len'] = MI_pairs(Counter((lb(len(t)), t[-1]) for t in long2)) / H1
        F['edge_mi_init_len'] = MI_pairs(Counter((lb(len(t)), t[0]) for t in long2)) / H1
        shl = [t[:] for t in long2]
        for u in shl: rng.shuffle(u)
        F['edge_mi_final_len_excess'] = F['edge_mi_final_len'] - MI_pairs(Counter((lb(len(t)), t[-1]) for t in shl)) / H1
        F['edge_first_last_same'] = float(np.mean([t[0] == t[-1] for t in long3])) if long3 else float('nan')
        F['edge_h_last_given_first'] = cond_H(Counter(((t[0],), t[-1]) for t in long3)) / H1 if long3 else float('nan')
        F['edge_mi_init_final'] = MI_pairs(Counter((t[0], t[-1]) for t in long3)) / H1 if long3 else float('nan')
        F['edge_mi_init_final_excess'] = F['edge_mi_init_final'] - (MI_pairs(Counter((u[0], u[-1]) for u in shl if len(u) >= 3)) / H1)
    # --- line / edge effects across text boundaries
    cross = Counter((texts[i][-1], texts[i + 1][0]) for i in range(len(texts) - 1))
    within = Counter((a, b) for (a,), b in [(k[0], k[1]) for k in bgp.keys()] for _ in [0])
    within = Counter({(a[0], b): c for (a, b), c in bgp.items()})
    if within and cross:
        mi_w = MI_pairs(within); mi_x = MI_pairs(cross)
        # bias: shuffle pairing
        xs = list(cross.elements()); a_ = [u for u, v in xs]; b_ = [v for u, v in xs]; rng.shuffle(b_)
        mi_x0 = MI_pairs(Counter(zip(a_, b_)))
        ws = list(within.elements()); a2 = [u for u, v in ws]; b2 = [v for u, v in ws]; rng.shuffle(b2)
        mi_w0 = MI_pairs(Counter(zip(a2, b2)))
        F['edge_mi_within_excess'] = (mi_w - mi_w0) / H1
        F['edge_mi_cross_excess'] = (mi_x - mi_x0) / H1
        F['edge_reset_ratio'] = (mi_x - mi_x0) / (mi_w - mi_w0) if (mi_w - mi_w0) > 1e-6 else float('nan')
    # adjacent-text similarity (line / page locality)
    sets = [set(t) for t in texts]
    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0
    adj = [jac(sets[i], sets[i + 1]) for i in range(len(sets) - 1)]
    rnd = [jac(sets[rng.randrange(len(sets))], sets[rng.randrange(len(sets))]) for _ in range(len(adj))]
    F['loc_adj_jaccard'] = float(np.mean(adj)); F['loc_rand_jaccard'] = float(np.mean(rnd))
    F['loc_adj_over_rand'] = float(np.mean(adj)) / max(1e-9, float(np.mean(rnd)))
    F['loc_adj_identical'] = float(np.mean([texts[i] == texts[i + 1] for i in range(len(texts) - 1)]))
    # --- repetition
    tc_ = Counter(tuple(t) for t in texts if len(t) >= 2)
    n2 = sum(tc_.values())
    F['rep_dup_text_frac'] = sum(c for c in tc_.values() if c > 1) / n2 if n2 else float('nan')
    F['rep_distinct_text_ratio'] = len(tc_) / n2 if n2 else float('nan')
    tc3 = Counter(tuple(t) for t in texts if len(t) >= 4)
    n3 = sum(tc3.values())
    F['rep_dup_text4_frac'] = sum(c for c in tc3.values() if c > 1) / n3 if n3 else 0.0
    imm = sum(1 for t in texts for i in range(len(t) - 1) if t[i] == t[i + 1])
    nb = sum(max(0, len(t) - 1) for t in texts)
    exp_imm = float(np.sum(p ** 2))
    F['rep_immediate_rate'] = imm / nb if nb else float('nan')
    F['rep_immediate_ratio'] = (imm / nb) / exp_imm if nb else float('nan')
    gap2 = sum(1 for t in texts for i in range(len(t) - 2) if t[i] == t[i + 2])
    nb2 = sum(max(0, len(t) - 2) for t in texts)
    F['rep_gap2_ratio'] = (gap2 / nb2) / exp_imm if nb2 else float('nan')
    wr = [(len(t) - len(set(t))) / len(t) for t in texts if len(t) >= 3]
    wr_s = [(len(t) - len(set(t))) / len(t) for t in gt if len(t) >= 3]
    F['rep_within_text'] = float(np.mean(wr)) if wr else float('nan')
    F['rep_within_text_vs_glob'] = float(np.mean(wr)) - float(np.mean(wr_s)) if wr else float('nan')
    # longest repeated substring within texts (stream with separators), via n-gram doubling
    best = 1
    for n in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        ng = Counter(tuple(t[i:i + n]) for t in texts for i in range(len(t) - n + 1))
        if any(v > 1 for v in ng.values()):
            best = n
        else:
            break
    F['rep_longest_repeat_log'] = math.log(best)
    F['rep_longest_over_len'] = best / lens.mean()
    # n-gram novelty halves
    half = len(texts) // 2
    A, B = texts[:half], texts[half:]
    for n in (1, 2, 3, 4, 5):
        sa = set(tuple(t[i:i + n]) for t in A for i in range(len(t) - n + 1))
        bt = [tuple(t[i:i + n]) for t in B for i in range(len(t) - n + 1)]
        if bt:
            F[f'nov_ngram{n}_tokens'] = sum(1 for g in bt if g not in sa) / len(bt)
            F[f'nov_ngram{n}_types'] = len(set(bt) - sa) / len(set(bt))
    sa_t = set(tuple(t) for t in A)
    F['nov_text'] = float(np.mean([tuple(t) not in sa_t for t in B])) if B else float('nan')
    # random half baseline -> locality of novelty
    perm = texts[:]; rng.shuffle(perm)
    A2, B2 = perm[:half], perm[half:]
    for n in (1, 2, 3):
        sa = set(tuple(t[i:i + n]) for t in A2 for i in range(len(t) - n + 1))
        bt = [tuple(t[i:i + n]) for t in B2 for i in range(len(t) - n + 1)]
        if bt:
            F[f'nov_ngram{n}_order_excess'] = F[f'nov_ngram{n}_tokens'] - sum(1 for g in bt if g not in sa) / len(bt)

    # --- burstiness
    pos = defaultdict(list)
    for i, x in enumerate(toks):
        pos[x].append(i)
    Bs, Bs_mem = [], []
    for x, _ in cnt.most_common(50):
        ps = pos[x]
        if len(ps) >= 10:
            d = np.diff(ps)
            m, s = d.mean(), d.std()
            Bs.append((s - m) / (s + m))
            if len(d) > 2:
                Bs_mem.append(np.corrcoef(d[:-1], d[1:])[0, 1] if d.std() > 0 else 0)
    if Bs:
        F['burst_B_mean'] = float(np.mean(Bs)); F['burst_B_median'] = float(np.median(Bs))
        F['burst_B_frac_02'] = float(np.mean(np.array(Bs) > 0.2))
        F['burst_memory'] = float(np.nanmean(Bs_mem)) if Bs_mem else float('nan')
    # burstiness for mid-frequency types (rank 11-60)
    Bm = []
    for x, _ in cnt.most_common(60)[10:]:
        ps = pos[x]
        if len(ps) >= 5:
            d = np.diff(ps); m, s = d.mean(), d.std(); Bm.append((s - m) / (s + m))
    if Bm:
        F['burst_B_mid'] = float(np.mean(Bm))
    Br = []
    for x, c in cnt.items():
        if 5 <= c <= 12:
            d = np.diff(pos[x]); m, s_ = d.mean(), d.std(); Br.append((s_ - m) / (s_ + m))
    F['burst_B_rare'] = float(np.mean(Br)) if Br else float('nan')
    F['rep_top_text_share'] = max(tc_.values()) / n2 if n2 else float('nan')
    # dispersion across chunks
    for nch in (10, 50):
        L = N // nch
        if L >= 5:
            disp = []
            for x, _ in cnt.most_common(30):
                arr = np.zeros(nch)
                for i in pos[x]:
                    k = i // L
                    if k < nch:
                        arr[k] += 1
                if arr.mean() > 0:
                    disp.append(arr.var() / arr.mean())
            F[f'burst_disp{nch}'] = float(np.mean(np.log(np.array(disp) + 1e-3))) if disp else float('nan')
    # Church adaptation in texts
    ad, ad0 = [], []
    for x, _ in cnt.most_common(30):
        k1 = sum(1 for t in texts if x in t)
        k2 = sum(1 for t in texts if t.count(x) >= 2)
        k1g = sum(1 for t in gt if x in t); k2g = sum(1 for t in gt if t.count(x) >= 2)
        if k1 and k1g:
            ad.append(k2 / k1); ad0.append(k2g / k1g)
    if ad:
        F['burst_adapt'] = float(np.mean(ad)); F['burst_adapt_excess'] = float(np.mean(ad) - np.mean(ad0))

    # --- long-range correlation (stream including boundaries)
    arr = np.array(ids)
    sh = arr.copy(); nprng.shuffle(sh)

    def mi_lag(a, d):
        if d >= len(a):
            return float('nan')
        x, y = a[:-d], a[d:]
        cap = 200
        x = np.minimum(x, cap); y = np.minimum(y, cap)
        return MI_pairs(Counter(zip(x.tolist(), y.tolist())))
    lags = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64)
    mis = []
    for d in lags:
        v = mi_lag(arr, d) - mi_lag(sh, d)
        F[f'lr_mi_lag{d}'] = v / H1
        mis.append(v)
    mis = np.array(mis)
    good = mis > 1e-4
    lg = np.log(np.array(lags, float))
    if good.sum() >= 3:
        F['lr_mi_decay_slope'] = float(np.polyfit(lg[good], np.log(mis[good]), 1)[0])
    F['lr_mi_far_over_near'] = float(mis[6:].mean() / mis[0]) if mis[0] > 1e-6 else float('nan')
    F['lr_mi_lag2_over_lag1'] = float(mis[1] / mis[0]) if mis[0] > 1e-6 else float('nan')
    lr = np.log(arr.astype(float))
    lr = (lr - lr.mean()) / (lr.std() + 1e-9)
    for d in (1, 2, 5, 10, 50, 100):
        F[f'lr_ac_lag{d}'] = float(np.mean(lr[:-d] * lr[d:])) if d < len(lr) else float('nan')
    F['lr_dfa_logrank'] = dfa_hurst(lr)
    F['lr_dfa_top1'] = dfa_hurst((arr == 1).astype(float))
    F['lr_dfa_rare'] = dfa_hurst((arr > max(1, V // 2)).astype(float))
    F['lr_dfa_textlen'] = dfa_hurst(lens) if len(lens) >= 64 else float('nan')
    # adjacent text-length correlation
    F['lr_textlen_ac1'] = float(np.corrcoef(lens[:-1], lens[1:])[0, 1]) if lens.std() > 0 else 0.0

    # --- compression
    SEP = 0
    stream = []
    for t in texts:
        stream += [rank[x] for x in t] + [SEP]
    b_real = to_bytes(stream)
    # within-text shuffle
    s_w = []
    for t in sh_texts:
        s_w += [rank[x] for x in t] + [SEP]
    s_g = []
    for t in gt:
        s_g += [rank[x] for x in t] + [SEP]
    order = list(range(len(texts))); rng.shuffle(order)
    s_o = []
    for i in order:
        s_o += [rank[x] for x in texts[i]] + [SEP]
    for cname, cf in COMPRESSORS.items():
        cr = cf(b_real)
        F[f'cmp_{cname}_bpt_over_h1'] = 8 * cr / len(stream) / max(H1, 1e-9)
        F[f'cmp_{cname}_vs_within'] = cr / cf(to_bytes(s_w))
        F[f'cmp_{cname}_vs_glob'] = cr / cf(to_bytes(s_g))
        F[f'cmp_{cname}_vs_textorder'] = cr / cf(to_bytes(s_o))
    rs = []
    for t in texts:
        rs += [rank[x] for x in reversed(t)] + [SEP]
    co = [min(x, 16) for x in stream]
    co_g = [min(x, 16) for x in s_g]
    for cname, cf in COMPRESSORS.items():
        F[f'cmp_{cname}_reverse_ratio'] = cf(to_bytes(rs)) / cf(b_real)
        F[f'cmp_{cname}_coarse16_vs_glob'] = cf(to_bytes(co)) / cf(to_bytes(co_g))
    # sorted-text corpus: compressibility of text set (duplication / prefix sharing)
    srt = sorted(texts)
    s_srt = []
    for t in srt:
        s_srt += [rank[x] for x in t] + [SEP]
    F['cmp_lzma_sorted_vs_real'] = COMPRESSORS['lzma'](to_bytes(s_srt)) / COMPRESSORS['lzma'](b_real)
    # LZ78 phrases normalised by shuffled
    sub = ids[:6000]
    sub_s = sub[:]; rng.shuffle(sub_s)
    F['cmp_lz78_vs_shuffle'] = lz78_phrases(sub) / lz78_phrases(sub_s)
    sub2 = ids[:1500]; sub2s = sub2[:]; rng.shuffle(sub2s)
    F['cmp_lz76_vs_shuffle'] = lz76(sub2) / max(1, lz76(sub2s))

    # --- self-similarity across 'pages' (10 contiguous blocks)
    nb_ = 10
    bl = [Counter() for _ in range(nb_)]
    ntx = len(texts)
    for i, t in enumerate(texts):
        bl[min(nb_ - 1, i * nb_ // ntx)].update(t)
    js_adj = [jsd(bl[i], bl[i + 1]) for i in range(nb_ - 1)]
    js_all = [jsd(bl[i], bl[j]) for i in range(nb_) for j in range(i + 1, nb_)]
    F['page_jsd_adj'] = float(np.mean(js_adj)); F['page_jsd_all'] = float(np.mean(js_all))
    F['page_adj_over_all'] = float(np.mean(js_adj) / max(1e-9, np.mean(js_all)))
    blr = [Counter() for _ in range(nb_)]
    for k, i in enumerate(order):
        blr[min(nb_ - 1, k * nb_ // ntx)].update(texts[i])
    js_r = [jsd(blr[i], blr[j]) for i in range(nb_) for j in range(i + 1, nb_)]
    F['page_jsd_over_random'] = float(np.mean(js_all) / max(1e-9, np.mean(js_r)))
    first_seen = {}
    for i, t in enumerate(texts):
        for x in t:
            first_seen.setdefault(x, min(nb_ - 1, i * nb_ // ntx))
    F['page_types_new_last_block'] = sum(1 for v in first_seen.values() if v == nb_ - 1) / V
    ta = set(x for t in A for x in t); tb = set(x for t in B for x in t)
    F['page_half_type_jaccard'] = len(ta & tb) / len(ta | tb)
    # per-block entropy variability
    hb = [H(c) for c in bl if sum(c.values()) > 0]
    F['page_h1_cv'] = float(np.std(hb) / np.mean(hb)) if hb else float('nan')
    # top-type share variability across blocks
    top = cnt.most_common(1)[0][0]
    sh_top = [c[top] / max(1, sum(c.values())) for c in bl]
    F['page_top1_cv'] = float(np.std(sh_top) / max(1e-9, np.mean(sh_top)))

    # --- bigram graph structure
    succ = defaultdict(set); pred = defaultdict(set)
    for (a, b) in within:
        succ[a].add(b); pred[b].add(a)
    topk = [x for x, _ in cnt.most_common(20)]
    F['graph_succ_top20_frac'] = float(np.mean([len(succ[x]) for x in topk])) / V
    F['graph_pred_top20_frac'] = float(np.mean([len(pred[x]) for x in topk])) / V
    F['graph_succ_pred_asym'] = float(np.mean([(len(succ[x]) - len(pred[x])) / max(1, len(succ[x]) + len(pred[x])) for x in topk]))
    rev = sum(1 for (a, b) in within if a != b and (b, a) in within)
    nonself = sum(1 for (a, b) in within if a != b)
    F['graph_reverse_frac'] = rev / nonself if nonself else float('nan')
    tot = sum(within.values())
    pmis = []
    for (a, b), c in within.items():
        if c >= 3:
            pmis.append(math.log2(c * N * N / (tot * cnt[a] * cnt[b])))
    if pmis:
        F['graph_pmi_mean'] = float(np.mean(pmis)); F['graph_pmi_frac3'] = float(np.mean(np.array(pmis) > 3))
        F['graph_pmi_frac_neg'] = float(np.mean(np.array(pmis) < 0))
    # 'forbidden' pairs: expected >= 5 under independence but never seen
    exp_pairs = 0; forb = 0
    tc_top = cnt.most_common(40)
    for a, ca in tc_top:
        for b, cb in tc_top:
            e = ca * cb * tot / (N * N)
            if e >= 5:
                exp_pairs += 1
                if within.get((a, b), 0) == 0:
                    forb += 1
    F['graph_forbidden_frac'] = forb / exp_pairs if exp_pairs else float('nan')
    # deterministic successors: share of bigram tokens where successor prob >= 0.5
    best_s = {}
    for (a, b), c in within.items():
        best_s[a] = max(best_s.get(a, 0), c)
    out_a = Counter()
    for (a, b), c in within.items():
        out_a[a] += c
    F['graph_det_succ_share'] = sum(best_s[a] for a in best_s if best_s[a] / out_a[a] >= 0.5 and out_a[a] >= 5) / max(1, tot)
    # text-level pairwise similarity
    idx = [rng.randrange(len(texts)) for _ in range(800)]
    jj = [jac(sets[idx[i]], sets[idx[i + 1]]) for i in range(0, 798, 2)]
    F['txt_rand_pair_jaccard_zero'] = float(np.mean(np.array(jj) == 0))
    # prefix sharing: share of texts (len>=3) whose first 2 signs open another text
    pre = Counter(tuple(t[:2]) for t in long3)
    F['txt_prefix2_shared'] = float(np.mean([pre[tuple(t[:2])] > 1 for t in long3])) if long3 else float('nan')
    suf = Counter(tuple(t[-2:]) for t in long3)
    F['txt_suffix2_shared'] = float(np.mean([suf[tuple(t[-2:])] > 1 for t in long3])) if long3 else float('nan')
    F['txt_prefix_minus_suffix'] = F['txt_prefix2_shared'] - F['txt_suffix2_shared'] if long3 else float('nan')
    # sign 'productivity': types first seen in second half that are hapax
    return {k: (float(v) if v is not None else float('nan')) for k, v in F.items()}


def family(name):
    return name.split('_')[0]
