"""v75 library: streams, chunks and the structural fingerprint.

Every real designation system (v75_texts.py) and every ordinary-language text is written into Voynich-like pages
(v72_lib._pages_from_entries: one entry = one paragraph, lines of <= 8 words, pages of ~160 tokens inside one
section), spelled with a lossy merge code into 11 payload symbols, dressed in the planted v72 surface machinery
and pulled back out with the frozen v72 extraction E1c (sha256 prefix 2803cbeb0deb51bf). The Voynich goes through
E1c directly. Generators (v72 GENS) are fitted to each surface (reference or Voynich) and extracted the same way.
The fingerprint is computed on the extracted stream, in chunks of about N tokens made of random whole pages.
"""
import os, sys, json, math, random, re, collections, pickle
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v72_lib as V
CK = os.path.join(os.path.dirname(HERE), 'data', 'v75_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = V.LOOPS
RULE = V.RULES['E1c_keepd']
assert V.rule_hash('E1c_keepd') == '2803cbeb0deb51bf'
N_TOK = 400


def row(fn, rid, method, result, verdict):
    V.row(fn, rid, method, result, verdict)


def systems():
    return json.load(open(os.path.join(CK, 'systems.json')))


def ref_surface(S, sid, seed):
    ents = [(s, ws) for s, ws in S[sid]['entries']]
    pages = V._pages_from_entries(ents, None, line_w=8, page_tok=160, cap=40000, prefix=sid[:6])
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    code = V.payload_code(words, seed=seed, mode='merge')
    pay = V.encode_payload(pages, code)
    surf = V.surface(pay, seed=seed + 1)
    for p in surf:
        for l in p['lines']: l.pop('orig', None)
    return surf


def voy_surface(name='ZL3b'):
    return V.voynich(name)


def extract(pages):
    return V.extract(pages, RULE)


def gen_streams(surf, seed):
    out = {}
    for g, f in V.GENS.items():
        out[g] = extract(f(surf, seed=seed))
    return out


def gshuffle(pages, seed):
    """global token shuffle across pages and sections (keeps the page/line skeleton)."""
    rng = random.Random(seed)
    ws = [w for p in pages for l in p['lines'] for w in l['w']]; rng.shuffle(ws); it = iter(ws)
    return [dict(p, lines=[dict(l, w=[next(it) for _ in l['w']]) for l in p['lines']]) for p in pages]


def chunks(pages, k, seed, n=N_TOK, page_filter=None):
    rng = random.Random(seed)
    idx = [i for i, p in enumerate(pages) if page_filter is None or page_filter(p)]
    tot = sum(sum(len(l['w']) for l in pages[i]['lines']) for i in idx)
    out = []
    if tot < n: return out
    for _ in range(k):
        rng.shuffle(idx); ch, t = [], 0
        for i in idx:
            ch.append(pages[i]); t += sum(len(l['w']) for l in pages[i]['lines'])
            if t >= n: break
        out.append(ch)
    return out


# ------------------------------------------------------------------ fingerprint
def _H(c):
    t = sum(c.values())
    return -sum(v / t * math.log2(v / t) for v in c.values() if v) if t else 0.0


def _MI(pairs):
    if not pairs: return 0.0
    a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return _H(a) + _H(b) - _H(ab)


FEATS = ['wl_mean', 'wl_sd', 'h1', 'h2', 'ttr', 'hapax', 'zipf', 'rigid', 'h_first', 'h_last', 'reuse_ent',
         'rep_in_ent', 'adj_rep', 'sec_mi', 'page_rec', 'direction', 'bigram_mi', 'junc_mi', 'pos_mi', 'head_mi',
         'lenfreq', 'short', 'lag2_rep', 'fl_mi', 'h_len', 'typelen_gap',
         'line_len', 'entry_len', 'line_cv', 'one_line_ent']
LAYOUT = {'line_len', 'entry_len', 'line_cv', 'one_line_ent'}


def fingerprint(chunk, seed=0, nperm=3):
    rng = random.Random(seed)
    T = []          # (page, line_key, entry_key, pos, word, sec)
    ent = -1; lines = []
    for pi, p in enumerate(chunk):
        for li, l in enumerate(p['lines']):
            if l['ps'] or ent < 0: ent += 1
            lk = (pi, li); lines.append((lk, ent, l['w']))
            for wi, w in enumerate(l['w']): T.append((pi, lk, ent, wi, w, p['sec']))
    words = [t[4] for t in T]
    Wn = words[:N_TOK]
    f = {}
    L = np.array([len(w) for w in words], float)
    f['wl_mean'] = L.mean(); f['wl_sd'] = L.std()
    sc = Counter(c for w in words for c in w); f['h1'] = _H(sc)
    bg = Counter(); ctx = Counter()
    for w in words:
        x = '^' + w + '$'
        for a, b in zip(x, x[1:]): bg[(a, b)] += 1; ctx[a] += 1
    f['h2'] = _H(bg) - _H(ctx)
    cw = Counter(Wn); f['ttr'] = len(cw) / len(Wn); f['hapax'] = sum(1 for v in cw.values() if v == 1) / len(cw)
    fr = np.array(sorted(Counter(words).values(), reverse=True)[:50], float)
    r = np.arange(1, len(fr) + 1)
    f['zipf'] = np.polyfit(np.log(r), np.log(fr), 1)[0] if len(fr) > 3 else 0.0
    pc = Counter()
    for w in set(words):
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if w[i] != w[j]: pc[(w[i], w[j])] += cw.get(w, 1)
    num = den = 0
    for (a, b), v in pc.items():
        if a < b:
            u = pc.get((b, a), 0); num += max(v, u); den += v + u
    for (a, b), v in pc.items():
        if a > b and (b, a) not in pc: num += v; den += v
    f['rigid'] = num / den if den else 1.0
    f['h_first'] = _H(Counter(w[0] for w in words)); f['h_last'] = _H(Counter(w[-1] for w in words))
    seen_ent = defaultdict(set); reuse = 0; rep_in = 0; cur_ent = None; cur_set = set()
    for t in T:
        e, w = t[2], t[4]
        if e != cur_ent: cur_ent = e; cur_set = set()
        if seen_ent[w] - {e}: reuse += 1
        if w in cur_set: rep_in += 1
        cur_set.add(w); seen_ent[w].add(e)
    f['reuse_ent'] = reuse / len(T); f['rep_in_ent'] = rep_in / len(T)
    adj = sum(1 for _, _, ws in lines for a, b in zip(ws, ws[1:]) if a == b)
    nadj = sum(max(0, len(ws) - 1) for _, _, ws in lines)
    f['adj_rep'] = adj / max(1, nadj)
    lag2 = sum(1 for _, _, ws in lines for a, b in zip(ws, ws[2:]) if a == b)
    f['lag2_rep'] = lag2 / max(1, sum(max(0, len(ws) - 2) for _, _, ws in lines))
    # section MI excess (permute section labels across pages)
    secs = [p['sec'] for p in chunk]
    def secmi(sl):
        return _MI([(t[4], sl[t[0]]) for t in T])
    base = secmi(secs); pm = []
    for _ in range(nperm):
        s2 = secs[:]; rng.shuffle(s2); pm.append(secmi(s2))
    f['sec_mi'] = base - np.mean(pm)
    # page recurrence excess (shuffle tokens across pages)
    def prec(assign):
        byp = defaultdict(Counter)
        for pg, w in assign: byp[pg][w] += 1
        return sum(v - 1 for c in byp.values() for v in c.values()) / len(assign)
    pa = [(t[0], t[4]) for t in T]
    pm = []
    for _ in range(nperm):
        ws2 = words[:]; rng.shuffle(ws2); pm.append(prec([(pa[i][0], ws2[i]) for i in range(len(ws2))]))
    f['page_rec'] = prec(pa) - np.mean(pm)
    # within-line direction, bigram MI, junction MI, position MI (within-line permutation null)
    def linestats(LS):
        od = Counter(); big = []; jun = []; pos = []
        for ws in LS:
            for i in range(len(ws)):
                pos.append((ws[i], min(i, 3)))
                for j in range(i + 1, len(ws)):
                    if ws[i] != ws[j]: od[(ws[i], ws[j])] += 1
            for a, b in zip(ws, ws[1:]): big.append((a, b)); jun.append((a[-1], b[0]))
        dd = []
        for (a, b), v in od.items():
            if a < b:
                u = od.get((b, a), 0)
                if v + u >= 2: dd.append(abs(v - u) / (v + u))
            elif (b, a) not in od and v >= 2: dd.append(1.0)
        return (np.mean(dd) if dd else 0.0), _MI(big), _MI(jun), _MI(pos)
    LS = [ws for _, _, ws in lines]
    d0, b0, j0, p0 = linestats(LS); acc = []
    for _ in range(nperm):
        L2 = []
        for ws in LS:
            x = ws[:]; rng.shuffle(x); L2.append(x)
        acc.append(linestats(L2))
    acc = np.array(acc).mean(0)
    f['direction'] = d0 - acc[0]; f['bigram_mi'] = b0 - acc[1]; f['junc_mi'] = j0 - acc[2]; f['pos_mi'] = p0 - acc[3]
    # head word: MI(type, entry-first) vs permutation inside entries
    byent = defaultdict(list)
    for t in T: byent[t[2]].append(t[4])
    def headmi(E):
        return _MI([(w, i == 0) for ws in E for i, w in enumerate(ws)])
    E0 = list(byent.values()); h0 = headmi(E0); pm = []
    for _ in range(nperm):
        E2 = []
        for ws in E0:
            x = ws[:]; rng.shuffle(x); E2.append(x)
        pm.append(headmi(E2))
    f['head_mi'] = h0 - np.mean(pm)
    ty = list(Counter(words).items())
    if len(ty) > 5:
        a = np.array([len(w) for w, _ in ty], float); b = np.log([v for _, v in ty])
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        f['lenfreq'] = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0
    else: f['lenfreq'] = 0.0
    f['short'] = float(np.mean(L <= 2))
    f['fl_mi'] = _MI([(w[0], w[-1]) for w in words if len(w) > 1])
    f['h_len'] = _H(Counter(len(w) for w in words))
    f['typelen_gap'] = float(np.mean([len(w) for w in cw])) - float(np.mean([len(w) for w in Wn]))
    ll = np.array([len(ws) for ws in LS], float)
    f['line_len'] = ll.mean(); f['line_cv'] = ll.std() / ll.mean()
    el = np.array([len(v) for v in byent.values()], float); f['entry_len'] = el.mean()
    nl = Counter(e for _, e, _ in lines); f['one_line_ent'] = np.mean([v == 1 for v in nl.values()])
    return [float(f[k]) for k in FEATS]
