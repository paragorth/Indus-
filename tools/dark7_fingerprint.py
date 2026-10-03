"""S-DARK-7: random-statistic fingerprint across scripts.

Builds a pool of random text statistics (compositions of primitives: slice selector x per-token property x
text-level aggregator x corpus-level aggregator, plus ratios and slot entropies), computes each on
length-matched resamples (texts of length 2..8, Indus length weights) of several corpora, bootstraps over
texts, and reports (a) statistics on which Indus, Proto-Elamite and Linear A agree but differ from Linear B
and Ur III seal legends, (b) statistics on which Indus differs from every other corpus, (c) where the Voynich
lines fall. Controls: within-text shuffle (order destroyed), two token granularities for LB/Ur III,
Indus split-half calibration, seq_raw vs seq_all merges.

Usage: python3 tools/dark7_fingerprint.py [B] [N] [seed]
"""
import json, re, sys, os, math, random, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N = int(sys.argv[2]) if len(sys.argv) > 2 else 800
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
LMIN, LMAX = 2, 8
IND_NUM = {f'W{i}' for i in list(range(1, 6)) + [16, 17, 18, 31, 32, 33, 34, 55, 56]}

# ------------------------------------------------------------------------------------------ corpora
# a corpus = list of texts; text = list of (token, is_num)

def load_indus(key):
    d = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    out = []
    for x in d:
        s = x.get(key)
        if s:
            out.append(([(f'W{v}', f'W{v}' in IND_NUM) for v in s], x))
    return out


def load_pe():
    d = json.load(open(os.path.join(ROOT, 'other-scripts/proto-elamite/data/pe_corpus.json')))
    out = []
    for t in d:
        for l in t['lines']:
            if l.get('lacuna') or l.get('header_comment'):
                continue
            toks = []
            for w in l['raw'].split():
                if w == ',':
                    continue
                m = re.match(r'^[\d/]+\((N\d+[A-Z]?)\)', w)
                if m:
                    toks.append((m.group(1), True))
                    continue
                w = re.sub(r'[#?!\[\]]', '', w)
                if w in ('x', '...', '') or '...' in w:
                    continue
                toks.append((w, False))
            if toks:
                out.append((toks, None))
    return out


def load_la():
    d = json.load(open(os.path.join(ROOT, 'other-scripts/linear-a/data/corpus.json')))
    out = []
    for x in d:
        cur = []
        for t in x['tokens']:
            if t['t'] == 'nl':
                if cur:
                    out.append((cur, None))
                cur = []
            elif t['t'] == 'word':
                cur.extend((s, False) for s in t['s'])
            elif t['t'] == 'num':
                cur.append(('NUM', True))
            elif t['t'] == 'logo':
                cur.append((t['v'], False))
        if cur:
            out.append((cur, None))
    return out


def load_lb(syll):
    out = []
    for l in open(os.path.join(ROOT, 'other-scripts/linear-a/data/damos_items.jsonl')):
        it = json.loads(l)
        for line in (it.get('content') or '').split('\n'):
            line = re.sub(r'^\s*\.?\d+[a-z]?\s+', ' ', line)
            if 'vac' in line or 'vest' in line:
                continue
            toks = []
            for w in line.split():
                w = w.strip("[],'/⟦⟧").replace('[', '').replace(']', '')
                if not w or w in ('[', ']', ',', '/', "'"):
                    continue
                if re.fullmatch(r'\d+', w):
                    toks.append(('NUM', True))
                elif syll and re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', w):
                    toks.extend((p, False) for p in w.split('-'))
                else:
                    toks.append((w, False))
            if toks:
                out.append((toks, None))
    return out


def load_ur3(syll):
    out = []
    for l in open(os.path.join(ROOT, 'data/codelib/ur3_legends.jsonl')):
        s = json.loads(l)['seq']
        if syll:
            toks = []
            for w in s:
                toks.extend((p, False) for p in re.split(r'[-{}]', w.strip('_')) if p)
        else:
            toks = [(w, False) for w in s]
        if toks:
            out.append((toks, None))
    return out


def load_voy(chars):
    d = json.load(open(os.path.join(ROOT, 'other-scripts/voynich/data/derived/ZL3b_lines.json')))
    out = []
    for x in d:
        ws = [w for w, u in zip(x['words'], x['uncertain']) if not u and w]
        if chars:
            for w in ws:
                out.append(([(c, False) for c in w], None))
        elif ws:
            out.append(([(w, False) for w in ws], None))
    return out


def shuffle_within(texts, seed):
    r = random.Random(seed)
    out = []
    for toks, meta in texts:
        t = list(toks)
        r.shuffle(t)
        out.append((t, meta))
    return out

# ------------------------------------------------------------------------------------------ properties
PROPS = ['log10rank', 'is_num', 'is_top10', 'is_hapax', 'is_rare(p<1e-3)', 'repeats_prev', 'seen_earlier_in_text',
         'seen_later_in_text', 'next_is_num', 'prev_is_num', 'unigram_surprise_bits', 'bigram_surprise_bits',
         'run_length', 'dist_to_nearest_num(cap5)', 'pmi_with_prev_bits', 'sign_initial_rate', 'sign_final_rate',
         'sign_mean_relpos', 'sign_is_opener_type(init>0.5)', 'sign_doc_freq_frac']
P = len(PROPS)


def featurize(texts):
    """returns dict L -> array (n_L, L, P) for L in LMIN..LMAX; also raw sign-id arrays per L"""
    cnt = collections.Counter(t for toks, _ in texts for t, _ in toks)
    tot = sum(cnt.values())
    rank = {s: i + 1 for i, (s, _) in enumerate(cnt.most_common())}
    big = collections.Counter()
    first = collections.Counter(); last = collections.Counter(); relpos = collections.defaultdict(list)
    docf = collections.Counter()
    for toks, _ in texts:
        ids = [t for t, _ in toks]
        for a, b in zip(ids, ids[1:]):
            big[(a, b)] += 1
        first[ids[0]] += 1; last[ids[-1]] += 1
        n = len(ids)
        for i, t in enumerate(ids):
            relpos[t].append(i / (n - 1) if n > 1 else 0.5)
        for t in set(ids):
            docf[t] += 1
    nd = len(texts)
    V = len(cnt)
    mrel = {t: float(np.mean(v)) for t, v in relpos.items()}
    out = {}
    raw = {}
    for L in range(LMIN, LMAX + 1):
        rows = [toks for toks, _ in texts if len(toks) == L]
        A = np.zeros((len(rows), L, P), dtype=np.float32)
        R = []
        for k, toks in enumerate(rows):
            ids = [t for t, _ in toks]
            nums = [i for i, (_, isn) in enumerate(toks) if isn]
            run = [1] * L
            for i in range(1, L):
                if ids[i] == ids[i - 1]:
                    run[i] = run[i - 1] + 1
            for i in range(L - 2, -1, -1):
                if ids[i] == ids[i + 1]:
                    run[i] = run[i + 1]
            for i, (t, isn) in enumerate(toks):
                p = cnt[t] / tot
                prev = ids[i - 1] if i > 0 else None
                if prev is not None:
                    pb = (big[(prev, t)] + 0.5) / (cnt[prev] + 0.5 * V)
                    bs = -math.log2(pb)
                    pmi = math.log2(pb / p)
                else:
                    bs = -math.log2(p); pmi = 0.0
                dn = min([abs(i - j) for j in nums if j != i] or [5])
                A[k, i] = [math.log10(rank[t]), isn, rank[t] <= 10, cnt[t] == 1, p < 1e-3, i > 0 and ids[i] == ids[i - 1],
                           t in ids[:i], t in ids[i + 1:], i + 1 < L and toks[i + 1][1], i > 0 and toks[i - 1][1],
                           -math.log2(p), bs, run[i], min(dn, 5), pmi, first[t] / cnt[t], last[t] / cnt[t], mrel[t],
                           first[t] / cnt[t] > 0.5, docf[t] / nd]
            R.append(ids)
        out[L] = A
        raw[L] = R
    return out, raw, {'V': V, 'tot': tot, 'n': nd}

# ------------------------------------------------------------------------------------------ statistics
SELECTORS = {'all': lambda L: list(range(L)), 'first': lambda L: [0], 'last': lambda L: [L - 1],
             'first2': lambda L: [0, 1], 'last2': lambda L: [L - 2, L - 1], 'middle': lambda L: list(range(1, L - 1)) or [0],
             'first_half': lambda L: list(range(0, max(1, L // 2))), 'second_half': lambda L: list(range(L // 2, L)),
             'second': lambda L: [min(1, L - 1)], 'penult': lambda L: [max(0, L - 2)], 'slice1-3': lambda L: [i for i in (1, 2) if i < L] or [0],
             'slice2-4': lambda L: [i for i in (2, 3) if i < L] or [L - 1]}
TAGG = {'mean': lambda a: a.mean(1), 'max': lambda a: a.max(1), 'min': lambda a: a.min(1), 'std': lambda a: a.std(1),
        'first-last': lambda a: a[:, 0] - a[:, -1]}
CAGG = {'mean': np.mean, 'median': np.median, 'frac>0': lambda v: np.mean(v > 0), 'std': np.std}


def text_values(F):
    """F: dict L -> (n_L, L, P). returns dict (sel, tagg) -> dict L -> (n_L, P)"""
    out = {}
    for sname, sf in SELECTORS.items():
        for tname, tf in TAGG.items():
            d = {}
            for L, A in F.items():
                idx = sf(L)
                d[L] = tf(A[:, idx, :])
            out[(sname, tname)] = d
    return out


def make_stats(seed, n_base=420, n_ratio=120, n_ent=60):
    r = random.Random(seed)
    base = [(s, p, t, c) for s in SELECTORS for p in range(P) for t in TAGG for c in CAGG]
    r.shuffle(base)
    stats = [('base',) + b for b in base[:n_base]]
    for _ in range(n_ratio):
        a, b = r.sample(base, 2)
        stats.append(('ratio', a, b))
    ents = [(s,) for s in SELECTORS]
    for _ in range(n_ent):
        stats.append(('entropy', r.choice(list(SELECTORS)), r.choice(['bits', 'evenness', 'ntypes'])))
    return stats


def stat_name(st):
    if st[0] == 'base':
        _, s, p, t, c = st
        return f'{c}_over_texts[{t}_{PROPS[p]} @ {s}]'
    if st[0] == 'ratio':
        return stat_name(('base',) + st[1]) + ' / ' + stat_name(('base',) + st[2])
    return f'slot_{st[2]}[signs @ {st[1]}]'


def target_weights(ind):
    c = collections.Counter(len(t) for t, _ in ind if LMIN <= len(t) <= LMAX)
    tot = sum(c.values())
    return {L: c[L] / tot for L in range(LMIN, LMAX + 1)}


def bootstrap(F, TV, raw, stats, W, rng, B, N):
    """returns array (B, n_stats)"""
    nL = {L: max(1, int(round(W[L] * N))) for L in W}
    res = np.zeros((B, len(stats)), dtype=np.float64)
    for b in range(B):
        idx = {L: rng.integers(0, F[L].shape[0], nL[L]) for L in W if F[L].shape[0] > 0}
        cache = {}

        def basev(st):
            _, s, p, t, c = st
            key = (s, p, t)
            if key not in cache:
                cache[key] = np.concatenate([TV[(s, t)][L][idx[L], p] for L in idx])
            return CAGG[c](cache[key])
        for j, st in enumerate(stats):
            if st[0] == 'base':
                res[b, j] = basev(st)
            elif st[0] == 'ratio':
                x = basev(('base',) + st[1]); y = basev(('base',) + st[2])
                res[b, j] = x / y if abs(y) > 1e-9 else np.nan
            else:
                _, s, kind = st
                toks = []
                for L in idx:
                    for i in idx[L]:
                        toks.extend(raw[L][i][k] for k in SELECTORS[s](L))
                cnt = collections.Counter(toks)
                n = len(toks)
                pr = np.array(list(cnt.values())) / n
                H = float(-(pr * np.log2(pr)).sum())
                if kind == 'bits':
                    res[b, j] = H
                elif kind == 'evenness':
                    res[b, j] = H / math.log2(len(cnt)) if len(cnt) > 1 else 0
                else:
                    res[b, j] = len(cnt) / n
    return res


def run_corpus(name, texts, stats, W, B, N, seed):
    F, raw, info = featurize(texts)
    TV = text_values(F)
    rng = np.random.default_rng(seed)
    R = bootstrap(F, TV, raw, stats, W, rng, B, N)
    return R, info


def main():
    out_dir = os.path.join(ROOT, 'data/derived/dark')
    os.makedirs(out_dir, exist_ok=True)
    ind_raw = load_indus('seq_raw')
    W = target_weights(ind_raw)
    stats = make_stats(SEED)
    corp = {
        'IND_raw': ind_raw, 'IND_all': load_indus('seq_all'),
        'PE': load_pe(), 'LA': load_la(),
        'LB_words': load_lb(False), 'LB_syll': load_lb(True),
        'UR3_words': load_ur3(False), 'UR3_syll': load_ur3(True),
        'VOY_words': load_voy(False), 'VOY_chars': load_voy(True),
    }
    rr = random.Random(SEED)
    ind_sh = list(ind_raw); rr.shuffle(ind_sh)
    corp['IND_halfA'] = ind_sh[: len(ind_sh) // 2]
    corp['IND_halfB'] = ind_sh[len(ind_sh) // 2:]
    # held-out sites: texts not from Mohenjo-daro / Harappa
    corp['IND_otherSites'] = [(t, m) for t, m in ind_raw if not str(m.get('cisi', '')).startswith(('M-', 'H-'))]
    corp['IND_MDH'] = [(t, m) for t, m in ind_raw if str(m.get('cisi', '')).startswith(('M-', 'H-'))]
    for k in ['IND_raw', 'PE', 'LA', 'LB_words', 'LB_syll', 'UR3_words', 'UR3_syll', 'VOY_words']:
        corp[k + '_SHUF'] = shuffle_within(corp[k], SEED)
    results = {}
    infos = {}
    for k, texts in corp.items():
        R, info = run_corpus(k, texts, stats, W, B, N, SEED + hash(k) % 1000)
        results[k] = R
        infos[k] = info
        n_used = sum(1 for t, _ in texts if LMIN <= len(t) <= LMAX)
        infos[k]['n_used'] = n_used
        print(f'{k}: texts {info["n"]} (len {LMIN}-{LMAX}: {n_used}), types {info["V"]}, tokens {info["tot"]}', flush=True)
    np.savez_compressed(os.path.join(out_dir, 'loop7_boot.npz'), **{k: v for k, v in results.items()})
    json.dump({'stats': [stat_name(s) for s in stats], 'stats_raw': [list(map(str, s)) for s in stats], 'infos': infos,
               'W': W, 'B': B, 'N': N, 'seed': SEED},
              open(os.path.join(out_dir, 'loop7_meta.json'), 'w'))
    print('saved')


if __name__ == '__main__':
    main()
