"""v53 analysis: language/genre posterior, motifs, information survival,
payload positions, and the inverse test (decode the held-out target half with
the best programs; compare with glyph-shuffled target).

usage: python3 v53_analyze.py TARGET   -> data/v53_ckpt/ana_<TARGET>.json and printed summary
"""
import sys, os, json, math, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v53_lib import *

TARGET = sys.argv[1]
R = json.load(open(os.path.join(CK, f'run_{TARGET}.json')))
CORP = {k: tuple(v) for k, v in load_corpora().items()}
OUT = {}


def P(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ posterior
best = defaultdict(list)
srch = defaultdict(list)
basel = defaultdict(list)
for r in R['results']:
    hf = min(t['heldout']['fit'] for t in r['top'])
    best[r['corpus']].append(hf)
    srch[r['corpus']].append(r['top'][0]['fit'])
    basel[r['corpus']].append(sorted(r['baseline_fits'])[len(r['baseline_fits']) // 2])
sds = [abs(v[0] - v[1]) / math.sqrt(2) for v in best.values() if len(v) > 1]
tau = max(0.02, (sum(s * s for s in sds) / max(1, len(sds))) ** 0.5)
fc = {c: sum(v) / len(v) for c, v in best.items()}
m0 = min(fc.values())
w = {c: math.exp(-(f - m0) / tau) for c, f in fc.items()}
Z = sum(w.values())
post = {c: w[c] / Z for c in sorted(fc, key=fc.get)}
lang_post, genre_post = Counter(), Counter()
for c, p in post.items():
    lang_post[CORP[c][0]] += p
    genre_post[CORP[c][1]] += p
OUT['posterior'] = {'tau': tau, 'heldout_best': {c: best[c] for c in post}, 'search_best': dict(srch),
                    'random_median': dict(basel), 'corpus': post, 'lang': dict(lang_post), 'genre': dict(genre_post)}
P('tau', round(tau, 3))
for c in post:
    P(f'  {c:9s} heldout {[round(x, 3) for x in best[c]]} search {[round(x, 3) for x in srch[c]]} random-median {[round(x, 2) for x in basel[c]]} post {post[c]:.3f}')
P('lang', {k: round(v, 3) for k, v in lang_post.most_common()}, 'genre', {k: round(v, 3) for k, v in genre_post.most_common()})

# ------------------------------------------------------------------ motifs
tops = []
for r in R['results']:
    tt = sorted(r['top'], key=lambda t: t['heldout']['fit'])[:10]
    tops.extend((t['heldout']['fit'], t['prog'], t['res'], r['corpus'], r['seed']) for t in tt)
rng = random.Random(0)
rand = [random_program(rng, corpora=list(CORP)) for _ in range(20000)]


def feats(p):
    f = [f'seg={p["seg"]}', f'abbr={p["abbr"]}', f'reord={p["reord"]}', f'wording={p["wording"]}',
         f'table={p["table"]}', f'bpe={p["bpe"]}', f'nops={min(len(p["ops"]), 3)}', f'width={p["width"] // 10 * 10}']
    f += sorted(set(f'op={o[0]}' for o in p['ops'])) + sorted(set(f'op={o[0]}:{o[2]}' for o in p['ops']))
    return f


rc = Counter(x for p in rand for x in feats(p))
tc = Counter(x for _, p, _, _, _ in tops for x in feats(p))
# replication across seeds: enrichment in each seed separately
bys = defaultdict(Counter)
nby = Counter()
for _, p, _, _, s in tops:
    nby[s] += 1
    for x in feats(p):
        bys[s][x] += 1
mot = []
for x, n in tc.items():
    e = (n / len(tops)) / (rc[x] / len(rand))
    es = [(bys[s][x] / nby[s]) / (rc[x] / len(rand)) for s in sorted(nby)]
    mot.append((e, x, n, es))
mot.sort(reverse=True)
OUT['motifs'] = [{'feat': x, 'enrich': e, 'n': n, 'per_seed': es} for e, x, n, es in mot]
P('motifs (enrichment in top-10 per corpus x seed vs random programs; per seed):')
for e, x, n, es in mot[:22]:
    P(f'  {x:22s} x{e:.2f} n={n} seeds {[round(v, 2) for v in es]}')
OUT['info'] = {'ret_mean': sum(r['ret'] for _, _, r, _, _ in tops) / len(tops),
               'acc_mean': sum(r['acc'] for _, _, r, _, _ in tops) / len(tops),
               'payload_share_mean': sum(r['payload_share'] for _, _, r, _, _ in tops) / len(tops)}
P('info', {k: round(v, 3) for k, v in OUT['info'].items()})

# ------------------------------------------------------------------ inverse test + payload positions
sys.argv = ['v53_run.py', TARGET, '1', '1', '1']
import importlib.util
spec = importlib.util.spec_from_file_location('v53run', os.path.join(HERE, 'v53_run.py'))
RUN = importlib.util.module_from_spec(spec); spec.loader.exec_module(RUN)
RUN.init()
A, B = RUN.TGT
envB = Env(B, CORP, offset=7919)


def lm_train(c, abbr):
    lang, genre, lines = CORP[c]
    ws = words_of(lines)
    keep = ws[:7919] + ws[10319:]
    if c == 'DE_rel':
        keep = ws[:7919] + ws[10319:12000] + ws[24000:]
    V = vowels_for(lang)
    aw = [abbr_word(x, abbr, V) for x in keep]
    tri, bi = Counter(), Counter()
    s = '  ' + ' '.join(aw) + ' '
    for i in range(2, len(s)):
        tri[s[i - 2:i + 1]] += 1; bi[s[i - 2:i]] += 1
    alpha = set(s)
    return {'tri': tri, 'bi': bi, 'A': len(alpha), 'lex': set(aw), 'V': V}


LMC = {}


def lm_score(words, lm):
    s = '  ' + ' '.join(words) + ' '
    lp = 0.0; n = 0
    for i in range(2, len(s)):
        c = lm['tri'].get(s[i - 2:i + 1], 0); b = lm['bi'].get(s[i - 2:i], 0)
        lp += -math.log2((c + 0.1) / (b + 0.1 * lm['A'])); n += 1
    return lp / max(1, n)


def decoded_words(m, lines):
    dec, cov = decode_lines(m, lines)
    out = []
    for u in dec:
        if u is None:
            continue
        u = unreorder(list(u), m['p']['reord'])
        out.append(''.join(u))
    return out, cov


def inverse_test(p, nshuf=10, seed=0):
    m = build(p, envB)
    key = (p['corpus'], p['abbr'])
    if key not in LMC:
        LMC[key] = lm_train(*key)
    lm = LMC[key]
    words, cov = decoded_words(m, B)
    real = {'hit': sum(x in lm['lex'] for x in words) / max(1, len(words)), 'bits': lm_score(words, lm), 'cov': cov,
            'n': len(words), 'sample': ' '.join(words[:25])}
    rng = random.Random(seed)
    sh = []
    for k in range(nshuf):
        Bs = [[''.join(rng.sample(x, len(x))) for x in l] for l in B]
        ws2, cov2 = decoded_words(m, Bs)
        sh.append({'hit': sum(x in lm['lex'] for x in ws2) / max(1, len(ws2)), 'bits': lm_score(ws2, lm), 'cov': cov2})
    # positive: the program's own plaintext output decoded
    own, _, _, _ = encode(m, max_tokens=4000)
    wo, covo = decoded_words(m, own)
    pos = {'hit': sum(x in lm['lex'] for x in wo) / max(1, len(wo)), 'bits': lm_score(wo, lm), 'cov': covo}

    def z(key, sign):
        v = [s[key] for s in sh]; mu = sum(v) / len(v); sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5 or 1e-6
        return sign * (real[key] - mu) / sd, mu
    zh, mh = z('hit', 1); zb, mb = z('bits', -1)
    return {'real': real, 'shuf_hit': mh, 'shuf_bits': mb, 'z_hit': zh, 'z_bits': zb, 'own': pos}


def pad_mask(m, lines):
    """per glyph: True = payload, False = stripped padding, following the program's own strip rules."""
    ops = m['reserved_ops']
    fill = set(x for o in ops if o[0] == 'FILL' for x in o[5])
    masks = []
    for l in lines:
        ml = []
        for i, w in enumerate(l):
            if w in fill:
                ml.append([False] * len(w)); continue
            a, b = 0, len(w)
            if i == len(l) - 1:
                for o in ops[::-1]:
                    if o[0] == 'LINEF' and o[5]:
                        for r in sorted(o[5], key=len, reverse=True):
                            if w[a:b].endswith(r) and b - a > len(r):
                                b -= len(r); break
            for o in ops[::-1]:
                kind, sel = o[0], o[5]
                if not sel:
                    continue
                if kind == 'SFX':
                    for r in sorted(sel, key=len, reverse=True):
                        if w[a:b].endswith(r) and b - a > len(r):
                            b -= len(r); break
                elif kind == 'PFX' or (kind == 'LINEM' and i == 0):
                    for r in sorted(sel, key=len, reverse=True):
                        if w[a:b].startswith(r) and b - a > len(r):
                            a += len(r); break
            ml.append([a <= j < b for j in range(len(w))])
        masks.append(ml)
    return masks


def pos_profile(lines, masks):
    pf = defaultdict(lambda: [0, 0]); gl = defaultdict(lambda: [0, 0]); lp = defaultdict(lambda: [0, 0])
    for l, ml in zip(lines, masks):
        for i, (w, mk) in enumerate(zip(l, ml)):
            for j, (g, k) in enumerate(zip(w, mk)):
                for key in (f'+{j}' if j < 3 else None, f'-{len(w) - j}' if len(w) - j <= 2 else None):
                    if key:
                        pf[key][0] += k; pf[key][1] += 1
                gl[g][0] += k; gl[g][1] += 1
                lk = 'line-first word' if i == 0 else ('line-last word' if i == len(l) - 1 else 'mid-line word')
                lp[lk][0] += k; lp[lk][1] += 1
    f = lambda d: {k: round(a / b, 3) for k, (a, b) in sorted(d.items()) if b >= 50}
    return {'word_pos': f(pf), 'glyph': f(gl), 'line_pos': f(lp)}


alltop = sorted(tops, key=lambda t: t[0])
pick = alltop[:6]
bestper = {}
for t in alltop:
    bestper.setdefault(t[3], t)
pick += [t for c, t in bestper.items() if t not in pick]
inv = []
for f, p, r, c, s in pick:
    try:
        it = inverse_test(p)
    except Exception as e:
        P('inverse fail', c, e); continue
    inv.append({'corpus': c, 'seed': s, 'heldout_fit': f, 'motif': list(map(str, motif_key(p))), **it})
    P(f'  inverse {c:9s} fit {f:.3f} cov {it["real"]["cov"]:.2f} hit {it["real"]["hit"]:.3f} (shuf {it["shuf_hit"]:.3f}, z {it["z_hit"]:+.1f}; own {it["own"]["hit"]:.2f}) bits {it["real"]["bits"]:.2f} (shuf {it["shuf_bits"]:.2f}, z {it["z_bits"]:+.1f}; own {it["own"]["bits"]:.2f}) | {it["real"]["sample"][:90]}')
OUT['inverse'] = inv

# payload positions: consensus over the 10 best programs
cons = []
for f, p, r, c, s in alltop[:10]:
    m = build(p, envB)
    cons.append(pos_profile(B, pad_mask(m, B)))
agg = {}
for part in ('word_pos', 'line_pos', 'glyph'):
    keys = set(k for cp in cons for k in cp[part])
    agg[part] = {k: round(sum(cp[part].get(k, 1.0) for cp in cons) / len(cons), 3) for k in sorted(keys)}
OUT['payload'] = agg
P('payload share by word position', agg['word_pos'])
P('payload share by line position', agg['line_pos'])
P('glyphs most often padding', sorted(agg['glyph'].items(), key=lambda x: x[1])[:8])

if TARGET == 'planted':
    # truth: the hidden program's own strip on the held-out half
    V = load_voynich('ZL3b'); VA, _ = split_halves(V)
    lang, genre, lines = CORP['DE_rel']
    ws = words_of(lines)
    sub = {'DE_rel': (lang, genre, [ws[i:i + 9] for i in range(12000, 24000, 9)])}
    envg = Env(VA, sub, n_plain=12000)
    mh = build(RUN.HIDDEN, envg)
    tm = pad_mask(mh, B)
    truth = pos_profile(B, tm)
    OUT['payload_truth'] = truth
    P('TRUE payload share by word position', truth['word_pos'])
    P('TRUE by line position', truth['line_pos'])
    # per-glyph agreement of the best program's mask with the truth
    accs = []
    for f, p, r, c, s in alltop[:10]:
        pm = pad_mask(build(p, envB), B)
        tot = agr = tp = fp = fn = 0
        for a, b in zip(pm, tm):
            for x, y in zip(a, b):
                for u, v in zip(x, y):
                    tot += 1; agr += (u == v)
                    tp += (not u and not v); fp += (not u and v); fn += (u and not v)
        accs.append({'agree': agr / tot, 'pad_precision': tp / max(1, tp + fp), 'pad_recall': tp / max(1, tp + fn),
                     'base_all_payload': sum(sum(y) for b in tm for y in b) / tot})
    OUT['payload_recovery'] = accs
    P('payload recovery (top10):', [{k: round(v, 2) for k, v in a.items()} for a in accs[:5]])
    # the true program re-scored on the target (is truth inside the search space under this scoring?)
    envA = Env(A, CORP)
    TA = target_profile(A)
    OUT['hidden_rescored'] = evaluate(RUN.HIDDEN, envA, TA)
    P('hidden program scored blind:', {k: round(v, 3) if isinstance(v, float) else v for k, v in OUT['hidden_rescored'].items()})

json.dump(OUT, open(os.path.join(CK, f'ana_{TARGET}.json'), 'w'))
