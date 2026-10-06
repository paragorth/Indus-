#!/usr/bin/env python3
"""la71 restoration-aware Linear A parser (default corpus for future loops).

Source: data/LinearAInscriptions.js (lineara.xyz, GORILA-based) plus SigLA's per-sign flags
(data/la22_ckpt/sigla.json, Salgarella & Castellan, CC BY-NC-SA; only sign codes, roles and the
sure/unsure flag are used).

What the editions encode (audited in la71 cycle 1):
  lineara.xyz  U+1076B  lacuna mark (break), glued to the token it touches, standalone on lost lines,
                        or inside a word that the edition bridges across a break
               [[X]]    scribal erasure                    ]X / X[   partial bracket at a break
               A+[?] A+[ ]  ligature with an unread / lost component
               '—' '≈' '?' 'None'  illegible or lost sign (already 'unk' in corpus.json)
               It does NOT carry GORILA's per-sign underdots / half-brackets / restoration brackets.
  SigLA        per-sign 'unsure-reading' flag, 'erasure' role; no integer numerals.

Token schema: identical to data/corpus.json (tools/build_corpus.py), so tokens of version 'all'
equal corpus.json token for token, plus
  'st' : 'read' | 'damaged' | 'restored' | 'erased' | 'illegible'
  'fl' : list of flags  edgeL edgeR (touches a break on that side), bridge (word joined across a
         break), part (bracketed / unread ligature component), unsure (SigLA unsure sign),
         variant (SigLA reads a different sign), nodraw (sign not drawn in SigLA's copy),
         erased, illegible
Status rules (worst wins): erased > restored (bridge) > damaged (edgeL, edgeR, part, unsure, unid,
variant, nodraw, worn) > read.  'cont' (line-start repeat of a break mark) is recorded but is not damage.
'worn': SigLA stipple covers >= 0.20 of the sign box (numbers: of the strip after the preceding sign),
calibrated by hand on 96 boxes (la71 cycle 1).  'illegible' is the edition's own unk tokens.

Versions (load(version)):
  'all'   every token as corpus.json has it (erasures counted as text, as before)
  'rd'    read + damaged: restored and erased tokens become {'t':'unk','v':'#R'} placeholders
  'read'  read only: damaged tokens also become placeholders
Placeholders keep positions, so a dropped number makes its section untestable rather than wrong.
A placeholder carries 'k' (original kind) and 'o' (the original token) for bookkeeping only; tests must
not read 'o' as text.

Output: data/corpus_ra.json (committed, small) and data/la71_ckpt/parse_report.json.
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la71_ckpt')
sys.path.insert(0, HERE)
import build_corpus as bc  # noqa: E402

GAP = '\U0001076b'
RANK = {'read': 0, 'damaged': 1, 'restored': 2, 'erased': 3, 'illegible': 4}
DAMAGE_FLAGS = {'edgeL', 'edgeR', 'part', 'unsure', 'unid', 'variant', 'worn', 'nodraw'}
RESTORE_FLAGS = {'bridge'}


def status_of(flags, kind):
    if kind == 'unk': return 'illegible'
    if 'erased' in flags: return 'erased'
    if flags & RESTORE_FLAGS: return 'restored'
    if flags & DAMAGE_FLAGS: return 'damaged'
    return 'read'


def code_of(ch):
    n = unicodedata.name(ch, '')
    m = re.match(r'LINEAR A SIGN (AB|A)(\d+)', n)
    return (m.group(1) + str(int(m.group(2)))) if m else None


def norm_code(c):
    m = re.match(r'(AB|A)0*(\d+)[a-z]?$', c or '')   # SigLA sub-variants (AB131a, AB21f) -> base code
    return m.group(1) + m.group(2) if m else c


def norm_name(n):
    return re.sub(r'[^A-Za-z0-9]', '', n).upper()


def nw(a, b, ms=2, mm=-1, gp=-1):
    n, m = len(a), len(b)
    S = [[0] * (m + 1) for _ in range(n + 1)]; P = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): S[i][0] = gp * i; P[i][0] = 1
    for j in range(1, m + 1): S[0][j] = gp * j; P[0][j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]; Si = S[i]; Sp = S[i - 1]; Pi = P[i]
        for j in range(1, m + 1):
            d = Sp[j - 1] + (ms if ai == b[j - 1] else mm); u = Sp[j] + gp; l = Si[j - 1] + gp
            if d >= u and d >= l: Si[j] = d; Pi[j] = 0
            elif u >= l: Si[j] = u; Pi[j] = 1
            else: Si[j] = l; Pi[j] = 2
    i, j = n, m; out = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and P[i][j] == 0: out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and (j == 0 or P[i][j] == 1): out.append((i - 1, None)); i -= 1
        else: out.append((None, j - 1)); j -= 1
    return out[::-1]


def pair_flags(W, T, i):
    """flags carried by the i-th (word, translit) pair of a document."""
    w, t = W[i], (T[i] or '')
    f = set()
    ch = list(w)
    if ch and ch[0] == GAP and len(ch) > 1:
        # lineara.xyz repeats the mark at the start of the next editorial line after a line that ends in a
        # break ('X#' newline '#Y'); hand check (HT 2, HT 85a, HT 122b): Y sits at an intact left edge.
        if i >= 2 and W[i - 1] == '\n' and W[i - 2].endswith(GAP): f.add('cont')
        else: f.add('edgeL')
    if ch and ch[-1] == GAP and len(ch) > 1: f.add('edgeR')
    if any(c == GAP for c in ch[1:-1]): f.add('bridge')
    # standalone lacuna pairs next to this one (same editorial line)
    if i > 0 and W[i - 1] == GAP: f.add('edgeL')
    if i > 1 and W[i - 1] == '\n' and W[i - 2] == GAP and 'edgeL' not in f: f.add('cont')
    if i + 1 < len(W) and W[i + 1] == GAP: f.add('edgeR')
    if '[[' in t: f.add('erased')
    if '[?]' in t or '[ ]' in t or re.search(r'(^\]|\[$)', t): f.add('part')
    return f


def parse_doc(v):
    W, T = v.get('words', []), v.get('transliteratedWords', [])
    n = min(len(W), len(T)); W, T = W[:n], T[:n]
    toks = []
    for i, (w, t) in enumerate(zip(W, T)):
        f = pair_flags(W, T, i)
        sub = bc.classify(w, t)
        codes = [c for c in (code_of(ch) for ch in w if ch != GAP) if c]
        cur = 0
        for tk in sub:
            tk = dict(tk)
            if tk['t'] in ('nl', 'div'):
                toks.append(tk); continue
            tk['fl'] = set(f)
            if tk['t'] == 'word':
                tk['_codes'] = codes[cur:cur + len(tk['s'])]; cur += len(tk['s'])
            elif tk['t'] == 'logo':
                tk['_codes'] = codes[cur:cur + 1]; cur += 1
            else:
                tk['_codes'] = []
            toks.append(tk)
    # same fraction merge as build_corpus; merged token takes the union of flags
    merged = []
    for tk in toks:
        if tk['t'] == 'frac' and merged and merged[-1]['t'] == 'num':
            merged[-1]['frac'] = merged[-1]['frac'] + tk['v']; merged[-1]['fl'] |= tk['fl']
        elif tk['t'] == 'frac':
            merged.append({'t': 'num', 'v': 0, 'frac': tk['v'], 'fl': tk['fl'], '_codes': []})
        else:
            merged.append(tk)
    return merged


def split_codes(tok):
    """per-sign codes of a word/logo token, best effort (a pair that split into several tokens shares
    its code list; we give each word token its own sign count)."""
    return tok.get('_codes', [])


def sigla_align(doc_toks, sig, wear=None, thr=None):
    """flag lineara signs against SigLA (la71 re-parse): unsure / unid / variant / nodraw / erased / worn.
    Numbers get 'worn' from the stipple in the strip to the right of the sign box before them."""
    thr = thr or WEAR_THR
    a = []
    for ti, tk in enumerate(doc_toks):
        if tk['t'] == 'word':
            for c in tk['_codes']: a.append((c, ti))
        elif tk['t'] == 'logo':
            for c in tk['_codes'][:1]: a.append((c, ti))
    occ = [o for o in sig['occ'] if o['role'] in ('syllabogram', 'logogram', 'transaction')
           or (o['role'] == 'erasure' and o['code'])]
    b = [norm_code(o['code']) if o['code'] else '?unid' for o in occ]
    box = {}
    if wear and 'boxes' in wear:
        for bx in wear['boxes']: box[bx[0]] = bx
    st = Counter()
    if not a or not b: return st
    key = lambda c: re.sub(r'^(AB|A)', '', c or '')   # editions differ in A/AB prefixes (A100 = AB100 VIR)
    a = [(key(c), ti) for c, ti in a]; b = [key(x) if not x.startswith('?') else x for x in b]
    al = nw([x[0] for x in a], b)
    tokbox = {}
    for i, j in al:
        if i is None: continue
        ti = a[i][1]
        if j is None:
            doc_toks[ti]['fl'].add('nodraw'); st['nodraw'] += 1
            continue
        o = occ[j]; st['aligned'] += 1
        if o['role'] == 'erasure':
            doc_toks[ti]['fl'].add('erased'); st['erased_sigla'] += 1
        elif not o['code']:
            doc_toks[ti]['fl'].add('unid'); st['unid'] += 1
        elif a[i][0] != b[j]:
            doc_toks[ti]['fl'].add('variant'); st['variant'] += 1
        if o['code'] and not o['sure']:
            doc_toks[ti]['fl'].add('unsure'); st['unsure'] += 1
        bx = box.get(o['n'])
        if bx:
            tokbox.setdefault(ti, []).append(bx)
            if bx[7] >= thr:
                doc_toks[ti]['fl'].add('worn'); st['worn'] += 1
    st['sigla_extra'] = sum(1 for i, j in al if i is None)
    # numbers: strip right of the last box of the preceding sign token, same height, up to the next box
    # on that line (or 2.5 box widths); cover = share of strip pixels within R px of a stipple dot
    if wear and 'dots' in wear:
        D = wear['dots']
        for ti, tk in enumerate(doc_toks):
            if tk['t'] != 'num': continue
            prev = next((q for q in range(ti - 1, -1, -1) if q in tokbox), None)
            if prev is None or ti - prev > 2: st['num_nobox'] += 1; continue
            bx = max(tokbox[prev], key=lambda r: r[2])
            x0 = bx[2] + bx[4]; y0, y1 = bx[3], bx[3] + bx[5]
            nxt = next((q for q in range(ti + 1, len(doc_toks)) if q in tokbox), None)
            x1 = x0 + int(2.5 * bx[4])
            if nxt is not None:
                nb = min(tokbox[nxt], key=lambda r: r[2])
                if nb[3] < y1 and nb[3] + nb[5] > y0 and nb[2] > x0: x1 = min(x1, nb[2])
            x1 = min(x1, wear['size'][0])
            cov = strip_cover(D, x0, y0, x1, y1)
            tk['_wear'] = cov; st['num_measured'] += 1
            if cov >= thr:
                tk['fl'].add('worn'); st['num_worn'] += 1
    return st


WEAR_THR = 0.20
WEAR_R = 12


def strip_cover(dots, x0, y0, x1, y1):
    """share of the rectangle within WEAR_R px of a dot (grid estimate, 4 px)."""
    if x1 <= x0 or y1 <= y0: return 0.0
    pts = [(x, y) for x, y in dots if x0 - WEAR_R <= x < x1 + WEAR_R and y0 - WEAR_R <= y < y1 + WEAR_R]
    if not pts: return 0.0
    import numpy as np
    gx = np.arange(x0, x1, 4); gy = np.arange(y0, y1, 4)
    X, Y = np.meshgrid(gx, gy)
    near = np.zeros(X.shape, bool)
    for x, y in pts:
        near |= (X - x) ** 2 + (Y - y) ** 2 <= WEAR_R ** 2
    return float(near.mean())


def build():
    raw = bc.load_raw()
    sig = json.load(open(os.path.join(CK, 'sigla71.json')))
    wp = os.path.join(CK, 'wear.json')
    wear = json.load(open(wp)) if os.path.exists(wp) else {}
    sname = {norm_name(k): k for k in sig}
    out = []; acc = Counter(); with_sigla = 0
    for k, v in raw.items():
        toks = parse_doc(v)
        sk = sname.get(norm_name(k))
        if sk and sig[sk]['occ']:
            with_sigla += 1
            acc.update(sigla_align(toks, sig[sk], wear.get(sk)))
        clean = []
        for tk in toks:
            if tk['t'] in ('nl', 'div'):
                clean.append(tk); continue
            fl = tk.pop('fl'); tk.pop('_codes', None); w = tk.pop('_wear', None)
            tk['st'] = status_of(fl, tk['t'])
            tk['fl'] = sorted(fl) + (['illegible'] if tk['t'] == 'unk' else [])
            clean.append(tk)
        out.append({'id': k, 'site': v.get('site'), 'support': v.get('support'), 'scribe': v.get('scribe'),
                    'context': v.get('context'), 'findspot': v.get('findspot'),
                    'sigla': bool(sk and sig[sk]['occ']), 'tokens': clean})
    return out, acc, with_sigla


PLACE = {'t': 'unk', 'v': '#R'}


def version(corpus, ver='rd'):
    """corpus.json-compatible list for version 'all' | 'rd' | 'read' | 'rnd'.
    'rnd' (control): as many tokens of each kind as 'read' removes, chosen at random (seed 71) among
    tokens of that kind and site group (HT / other), so that a change seen in 'read' can be told apart
    from plain data loss."""
    keep = {'all': None, 'rd': {'read', 'damaged', 'illegible'}, 'read': {'read', 'illegible'},
            'rnd': None}[ver]
    drop = set()
    if ver == 'rnd':
        import random
        rng = random.Random(71); pools = {}; need = Counter()
        for di, d in enumerate(corpus):
            g = 'HT' if d.get('site') == 'Haghia Triada' else 'X'
            for ti, tk in enumerate(d['tokens']):
                if tk['t'] in ('nl', 'div', 'unk'): continue
                pools.setdefault((g, tk['t']), []).append((di, ti))
                if tk.get('st') not in ('read', 'illegible'): need[(g, tk['t'])] += 1
        for k, pool in pools.items():
            drop |= set(rng.sample(pool, need[k]))
    res = []
    for di, d in enumerate(corpus):
        toks = []
        for ti, tk in enumerate(d['tokens']):
            if ver == 'rnd':
                ok = (di, ti) not in drop
            else:
                ok = keep is None or tk['t'] in ('nl', 'div') or tk.get('st', 'read') in keep
            if ok:
                toks.append({x: y for x, y in tk.items() if x not in ('st', 'fl')})
            else:
                toks.append(dict(PLACE, st=tk['st'], k=tk['t'],
                                 o={x: y for x, y in tk.items() if x not in ('st', 'fl')}))
        e = {x: y for x, y in d.items() if x != 'tokens'}
        e['tokens'] = toks
        e['words'] = ['-'.join(t['s']) for t in toks if t['t'] == 'word']
        e['logograms'] = [t['v'] for t in toks if t['t'] == 'logo']
        e['numbers'] = [[t['v'], t['frac']] for t in toks if t['t'] == 'num']
        res.append(e)
    return res


def load(ver='rd'):
    """Default loader for future loops: la71_parse.load('rd')."""
    return version(json.load(open(os.path.join(DATA, 'corpus_ra.json'))), ver)


def main():
    os.makedirs(CK, exist_ok=True)
    corpus, acc, ws = build()
    # sanity: version 'all' equals corpus.json token for token
    old = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus.json')))}
    allv = version(corpus, 'all'); bad = 0
    for d in allv:
        if [{x: y for x, y in t.items()} for t in d['tokens']] != old[d['id']]['tokens']: bad += 1
    json.dump(corpus, open(os.path.join(DATA, 'corpus_ra.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    st = Counter(); bysite = defaultdict(Counter); bykind = defaultdict(Counter); fl = Counter()
    for d in corpus:
        s = d['site'] or '?'
        for t in d['tokens']:
            if t['t'] in ('nl', 'div'): continue
            st[t['st']] += 1; bysite[s][t['st']] += 1; bykind[t['t']][t['st']] += 1
            for f in t['fl']: fl[f] += 1
    rep = {'docs': len(corpus), 'docs_with_sigla': ws, 'mismatch_vs_corpus_json': bad, 'status': st,
           'flags': fl, 'by_kind': bykind, 'by_site': bysite, 'sigla_alignment': acc}
    json.dump(rep, open(os.path.join(CK, 'parse_report.json'), 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('docs', 'docs_with_sigla', 'mismatch_vs_corpus_json', 'status', 'flags',
                                          'sigla_alignment')}))
    print('by kind', {k: dict(v) for k, v in bykind.items()})
    for s, c in sorted(bysite.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        print('%-22s n %5d read %5d damaged %4d restored %3d erased %2d illegible %3d  non-read %.3f'
              % (s, n, c['read'], c['damaged'], c['restored'], c['erased'], c['illegible'], 1 - c['read'] / n))


if __name__ == '__main__':
    main()
