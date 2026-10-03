#!/usr/bin/env python3
"""S-DARK-44: ARE THE HEADS (CLOSERS) MEASURE / COMMODITY CLASSES, AS IN PROTO-ELAMITE?
In Proto-Elamite the entry-final class sign predicts the numeral system (capacity vs count). Arrow: the Indus head
(closer-slot sign, S289/S291) should predict the numeral SERIES (short vs tall, S204), the numeral VALUE, and the counted
ITEM of any count in the same text.
  cycle 1: MD + Harappa; head x series / value / item, three numeral sets (ALL, ADJ = numeral directly before the head,
           FAR = count elsewhere in the text); permutation null = heads shuffled among texts within site x object type x
           length (1,000x); Cramer's V, MI (bits), cells by permutation z. Controls: planted corpora with heads assigned
           independently of the numerals (20x, must give null); Proto-Elamite through the same code (must recover the
           class -> number-system link, within-tablet shuffle).
  cycle 2: replication on held-out Wells sites and on IM77 (bridge M-numbers; all texts and the 210 IM77-only texts).
  cycle 3: head -> measure-class table; reverse prediction head | (series, item) on held-out texts vs modal baseline.
  cycle 4: the quantity-seal reading and the out-of-sample prediction (tall-count heads never take short counts).
Usage: python3 tools/dark_loop44.py <cycle> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json, sys, random, collections, math, csv, os
ROOT = '/home/user/Indus-/'
sys.path.insert(0, ROOT + 'other-scripts/proto-elamite/tools')
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rnd = random.Random(44)
OUT = ROOT + 'data/derived/dark/'
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- sign classes (Wells numbers) ----------------
MARK = {1, 2, 31}                                   # W1, W2 (connective), tall-1 W31 = M86: markers (S234, S-DARK-15.3)
SHORT = (set(range(3, 8)) | set(range(12, 21)) | set(range(25, 30)))
TALL = set(range(32, 40))
FIXED12 = {55, 56}                                  # 12 and 24: named numbers (S273), not counts
VAL = {**{i: i for i in range(3, 8)}, **{i: i - 10 for i in range(12, 21)}, **{i: i - 20 for i in range(25, 30)},
       **{i: i - 30 for i in range(32, 39)}, 39: 9}
NUMS = SHORT | TALL
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700, 595]   # closer paradigm S289/S288 (+ W700 voucher unit)
SUF = {400, 90}
GOODS = {390, 405, 407, 900, 220, 161, 162, 167, 168, 169, 287, 59}   # variably counted goods (S198): trees, bracket, plain fish (W and M numbers)
BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
def Mname(w):
    m = BR.get(str(w)); return f'W{w}/M{"-".join(map(str, m))}' if m else f'W{w}'
# IM77 (Mahadevan) numerals: tall M87-96 (M86 excluded), short M101-120 (M97-100 = W1/W2 markers excluded; M121 = 12)
M_TALL = {87: 2, 88: 2, 89: 3, 90: 3, 91: 3, 92: 3, 93: 4, 94: 4, 95: 5, 96: 5}      # values for 93-96 approximate
M_SHORT = {102: 3, 103: 3, 104: 4, 105: 4, 106: 5, 107: 5, 108: 6, 109: 6, 110: 7, 111: 7, 112: 7, 113: 8, 114: 8,
           115: 9, 116: 9, 117: 10, 118: 10, 119: 8, 120: 9}
M_CL = [342, 211, 12, 15, 254, 60, 328, 252, 245, 66]      # 245 (W617) and 66 (W236) are proposed bridge entries
M_SUF = {176, 1}

def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t, 'other')
def lbin(n): return 2 if n <= 2 else 3 if n == 3 else 4 if n == 4 else 5 if n <= 6 else 7

def make_records(texts, nums, vals, cls, suf, mark, fixed=()):
    """texts: list of dict(id, site, ot, seq). One record per text with a head and >= 1 numeral token.
    numeral token = (series, value, item, adjacent-to-head)."""
    recs = []
    for t in texts:
        s = list(t['seq'])
        while len(s) > 1 and s[-1] in suf: s.pop()
        if len(s) < 2 or s[-1] not in cls: continue
        head = s[-1]; toks = []
        for i, x in enumerate(s[:-1]):
            if x in nums and x not in mark and x not in fixed:
                item = s[i + 1]
                toks.append(dict(series='T' if x in (TALL if nums is NUMS else set(M_TALL)) else 'S', value=vals.get(x, 0),
                                 item=item, adj=(i == len(s) - 2)))
        if toks:
            recs.append(dict(id=t['id'], site=t['site'], ot=t['ot'], head=head, toks=toks,
                             stratum=(t['site'] if t.get('big', True) else 'other', t['ot'], lbin(len(s)))))
    return recs

def load_wells(level, sites):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    T = []
    for r in C:
        s = r[level]
        if not s or r['complete'] != 'Y': continue
        big = r['site'] in ('Mohenjo-daro', 'Harappa')
        if sites == 'big' and not big: continue
        if sites == 'heldout' and big: continue
        T.append(dict(id=r['cisi'], site=r['site'], ot=otype(r['type']), seq=list(s), big=big))
    return T

def load_im77(subset=None):
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[(r['text_no'], r['side'])].append(r)
    keep = None
    if subset:
        sets = json.load(open(OUT + 'loop21_cycle1_sets.json'))
        keep = {tuple(x) for x in sets[subset]}
    T = []
    for k, ls in by.items():
        if keep is not None and k not in keep: continue
        ls.sort(key=lambda r: int(r['line'] or 0))
        seq = []
        bad = False
        for r in ls:
            for x in r['signs_clean'].split():
                if not x.isdigit(): continue
                if x == '0': bad = True
                else: seq.append(int(x))
        if bad or len(seq) < 2: continue
        site = ls[0]['site']; big = site in ('Mohenjodaro', 'Harappa')
        ot = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet',
              'pottery graffito': 'pot'}.get(ls[0]['object_type'], 'other')
        T.append(dict(id=k[0] + '.' + k[1], site=site, ot=ot, seq=seq, big=big))
    return T

# ---------------- association statistics ----------------
def table(pairs):
    n = len(pairs); a = collections.Counter(x for x, _ in pairs); b = collections.Counter(y for _, y in pairs)
    j = collections.Counter(pairs)
    mi = sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in j.items())
    chi = sum((j[(x, y)] - a[x] * b[y] / n) ** 2 / (a[x] * b[y] / n) for x in a for y in b)
    k = min(len(a), len(b)) - 1
    V = math.sqrt(chi / (n * k)) if k > 0 else 0.0
    return mi, V, j, a, b

def pairs_of(recs, key, sel, headmap=None):
    out = []
    for i, r in enumerate(recs):
        h = headmap[i] if headmap else r['head']
        for t in r['toks']:
            if sel == 'ADJ' and not t['adj']: continue
            if sel == 'FAR' and t['adj']: continue
            if sel == 'GOODS' and (t['adj'] or t['item'] not in GOODS): continue
            v = t[key]
            if key == 'value': v = v if v <= 7 else 8   # 8 = '8 or more'
            if key == 'item' and t['adj']: v = 'HEAD'
            out.append((h, v))
    return out

def permute_heads(recs):
    byS = collections.defaultdict(list)
    for i, r in enumerate(recs): byS[r['stratum']].append(i)
    hm = [r['head'] for r in recs]
    for idx in byS.values():
        v = [hm[i] for i in idx]; rnd.shuffle(v)
        for i, x in zip(idx, v): hm[i] = x
    return hm

def pool_rare(pairs, minn=5, side=1):
    c = collections.Counter(p[side] for p in pairs)
    return [(p[0], p[1] if side == 0 or c[p[1]] >= minn else 'OTHER') if side == 1 else
            ((p[0] if c[p[0]] >= minn else 'OTHER'), p[1]) for p in pairs]

def assoc(recs, key, sel, nperm, label, name=lambda h: str(h), topcells=8, minitem=5):
    real = pairs_of(recs, key, sel)
    if key == 'item': real = pool_rare(real, minitem, 1)
    if len(real) < 10 or len(set(h for h, _ in real)) < 2 or len(set(v for _, v in real)) < 2:
        P(f'  [{label}] {key} {sel}: too few ({len(real)} tokens)'); return None
    mi, V, j, a, b = table(real)
    cells = collections.defaultdict(list); nmi = []; nV = []
    rare = None
    if key == 'item':
        cnt = collections.Counter(p[1] for p in pairs_of(recs, key, sel)); rare = {v for v, c in cnt.items() if c < minitem}
    for _ in range(nperm):
        hm = permute_heads(recs)
        pp = pairs_of(recs, key, sel, hm)
        if key == 'item': pp = [(h, 'OTHER' if v in rare else v) for h, v in pp]
        m2, V2, j2, _, _ = table(pp)
        nmi.append(m2); nV.append(V2)
        for c in set(j) | set(j2): cells[c].append(j2.get(c, 0))
    pm = sum(x >= mi for x in nmi) / nperm; pv = sum(x >= V for x in nV) / nperm
    nmi.sort(); nV.sort()
    P(f'  [{label}] head x {key} ({sel}): n = {len(real)} tokens, {len(a)} heads x {len(b)} classes; '
      f'MI {mi:.3f} bits (null mean {sum(nmi)/nperm:.3f}, 95% {nmi[int(0.95*nperm)-1]:.3f}, P = {pm:.4f}); '
      f"Cramer's V {V:.3f} (null {sum(nV)/nperm:.3f}, 95% {nV[int(0.95*nperm)-1]:.3f}, P = {pv:.4f})")
    zs = []
    for c, lst in cells.items():
        mu = sum(lst) / nperm; sd = math.sqrt(sum((x - mu) ** 2 for x in lst) / nperm) or 1e-9
        zs.append((abs(j.get(c, 0) - mu) / sd, c, j.get(c, 0), mu, (j.get(c, 0) - mu) / sd))
    zs.sort(reverse=True)
    P('     driving cells (O vs null mean, z): ' + '; '.join(f'{name(c[0])} x {c[1]} {o} vs {mu:.1f} (z {z:+.1f})'
                                                             for _, c, o, mu, z in zs[:topcells]))
    return dict(label=label, key=key, sel=sel, n=len(real), mi=mi, V=V, p_mi=pm, p_V=pv, null_mi=sum(nmi) / nperm,
                null95_mi=nmi[int(0.95 * nperm) - 1], cells=[(name(c[0]), str(c[1]), o, round(mu, 1), round(z, 1)) for _, c, o, mu, z in zs[:topcells]])

def profile(recs, name=lambda h: str(h), sel='ALL', minn=5):
    """head -> series counts, value distribution, items."""
    prof = collections.defaultdict(lambda: dict(S=0, T=0, vals=collections.Counter(), items=collections.Counter(), texts=0))
    for r in recs:
        used = False
        for t in r['toks']:
            if sel == 'ADJ' and not t['adj']: continue
            if sel == 'FAR' and t['adj']: continue
            if sel == 'GOODS' and (t['adj'] or t['item'] not in GOODS): continue
            p = prof[r['head']]; p[t['series']] += 1; p['vals'][(t['series'], t['value'])] += 1
            p['items']['HEAD' if t['adj'] else t['item']] += 1; used = True
        if used: prof[r['head']]['texts'] += 1
    rows = []
    for h, p in sorted(prof.items(), key=lambda kv: -(kv[1]['S'] + kv[1]['T'])):
        n = p['S'] + p['T']
        if n < minn: continue
        cls = 'TALL' if p['T'] / n >= 0.8 else 'SHORT' if p['S'] / n >= 0.8 else 'MIXED'
        vals = ' '.join(f'{s}{v}:{c}' for (s, v), c in sorted(p['vals'].items(), key=lambda kv: -kv[1])[:6])
        items = ' '.join(f'{name(i) if i != "HEAD" else "HEAD"}:{c}' for i, c in p['items'].most_common(5))
        rows.append((h, n, p['texts'], p['S'], p['T'], cls, vals, items))
        P(f'    {name(h):>14} texts {p["texts"]:4d} tokens {n:4d}  short {p["S"]:4d} tall {p["T"]:4d}  -> {cls:5}  values {vals}  | items {items}')
    return rows

def planted_control(recs, key, sel, nplant, nperm):
    """Heads reassigned independently of the numerals (within stratum); the same test must give null."""
    ps = []
    for k in range(nplant):
        hm = permute_heads(recs)
        fake = [dict(r, head=h) for r, h in zip(recs, hm)]
        real = pairs_of(fake, key, sel)
        if key == 'item': real = pool_rare(real, 5, 1)
        mi = table(real)[0]
        null = []
        for _ in range(nperm):
            hm2 = permute_heads(fake); pp = pairs_of(fake, key, sel, hm2)
            if key == 'item': pp = pool_rare(pp, 5, 1)
            null.append(table(pp)[0])
        ps.append(sum(x >= mi for x in null) / nperm)
    ps.sort()
    P(f'  [planted] {nplant} corpora with heads independent of numerals, head x {key} ({sel}): P values min {ps[0]:.3f} median {ps[len(ps)//2]:.3f}; '
      f'share P < 0.05 = {sum(p < 0.05 for p in ps)/nplant:.2f} (expected 0.05)')
    return ps

def pe_control(nperm):
    """Proto-Elamite: entry-final sign x numeral system; heads shuffled among entries within tablet x entry length."""
    from common import load, entries
    T = load(); E = [e for e in entries(T) if e['signs'][-1] != 'x' and e['system']]
    fc = collections.Counter(e['signs'][-1] for e in E)
    recs = []
    for e in E:
        h = e['signs'][-1] if fc[e['signs'][-1]] >= 10 else 'OTHER'
        sysg = 'CAP' if e['system'] in ('C', 'C*') else 'COUNT' if e['system'] == 'SDB' else 'OTHER'
        recs.append(dict(head=h, toks=[dict(series=sysg, value=0, item='x', adj=False)], stratum=(e['tablet'], lbin(len(e['signs'])))))
    P(f'  [PE positive control] {len(recs)} entries, {len(set(r["head"] for r in recs))} final-sign classes; null = heads shuffled within tablet x entry length')
    res = assoc(recs, 'series', 'ALL', nperm, 'PE within-tablet', topcells=8)
    for r in recs: r['stratum'] = ('all',)
    res2 = assoc(recs, 'series', 'ALL', nperm, 'PE global', topcells=4)
    # class table
    prof = collections.defaultdict(collections.Counter)
    for r in recs: prof[r['head']][r['toks'][0]['series']] += 1
    P('     PE classes (CAP share, n): ' + '; '.join(f'{h} {c["CAP"]/sum(c.values()):.2f} ({sum(c.values())})'
                                                   for h, c in sorted(prof.items(), key=lambda kv: -sum(kv[1].values()))[:16]))
    return res, res2

def run_block(recs, label, nperm, name, keys=('series', 'value', 'item'), sels=('ALL', 'ADJ', 'FAR', 'GOODS')):
    out = []
    P(f'-- {label}: {len(recs)} texts with a head and a counted numeral; heads {dict(collections.Counter(name(r["head"]) for r in recs).most_common())}')
    P(f'   object types {dict(collections.Counter(r["ot"] for r in recs))}; numeral tokens: adjacent-to-head '
      f'{sum(t["adj"] for r in recs for t in r["toks"])}, elsewhere {sum(not t["adj"] for r in recs for t in r["toks"])}')
    for sel in sels:
        for key in keys:
            if key == 'item' and sel == 'ADJ': continue
            res = assoc(recs, key, sel, nperm, label, name)
            if res: out.append(res)
    return out

W = lambda h: Mname(h)
Mn = lambda h: f'M{h}'

if __name__ == '__main__':
    tag = f'loop44_c{CY}_{LV}'
    if CY == 1:
        P(f'== S-DARK-44 cycle 1, level {LV}, nperm {NP}: head x numeral on Mohenjo-daro + Harappa')
        T = load_wells(LV, 'big'); recs = make_records(T, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        res = run_block(recs, 'MD+H all media', NP, W)
        P('  head profiles (all numerals in the text):'); profile(recs, W, 'ALL')
        P('  head profiles (counts of goods: trees, bracket, plain fish, not before the head):'); profile(recs, W, 'GOODS', 3)
        P('  head profiles (numeral directly before the head):'); profile(recs, W, 'ADJ')
        P('  head profiles (count elsewhere in the text):'); profile(recs, W, 'FAR')
        seals = [r for r in recs if r['ot'] == 'seal']
        res += run_block(seals, 'MD+H seals only', NP, W, keys=('series', 'value', 'item'), sels=('ALL', 'FAR'))
        nojar = [r for r in recs if r['head'] not in (740, 700)]
        res += run_block(nojar, 'MD+H heads other than jar and W700', NP, W, keys=('series', 'value'), sels=('ALL', 'FAR'))
        P('-- controls')
        planted_control(recs, 'series', 'ALL', 20, 300); planted_control(recs, 'series', 'FAR', 20, 300)
        planted_control(recs, 'item', 'FAR', 10, 300)
        pe_control(NP)
        json.dump(res, open(OUT + tag + '.json', 'w'), indent=1)
    elif CY == 2:
        P(f'== S-DARK-44 cycle 2, level {LV}, nperm {NP}: replication on held-out Wells sites and IM77')
        T = load_wells(LV, 'heldout'); recs = make_records(T, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        P('   held-out sites: ' + str(dict(collections.Counter(r['site'] for r in recs).most_common())))
        res = run_block(recs, 'Wells held-out sites', NP, W, sels=('ALL', 'FAR', 'GOODS'))
        P('  held-out head profiles (ALL):'); profile(recs, W, 'ALL', 3)
        P('  held-out head profiles (FAR):'); profile(recs, W, 'FAR', 3)
        I = load_im77(); irecs = make_records(I, set(M_TALL) | set(M_SHORT), {**M_TALL, **M_SHORT}, set(M_CL), M_SUF, set(), ())
        res += run_block(irecs, 'IM77 all sites', NP, Mn, sels=('ALL', 'FAR', 'GOODS'))
        P('  IM77 head profiles (ALL):'); profile(irecs, Mn, 'ALL')
        P('  IM77 head profiles (FAR):'); profile(irecs, Mn, 'FAR')
        In = load_im77('new'); nrecs = make_records(In, set(M_TALL) | set(M_SHORT), {**M_TALL, **M_SHORT}, set(M_CL), M_SUF, set(), ())
        res += run_block(nrecs, 'IM77-only texts (never in Wells, S-DARK-21)', NP, Mn, keys=('series', 'value'), sels=('ALL', 'FAR'))
        P('  IM77-only head profiles:'); profile(nrecs, Mn, 'ALL', 1)
        json.dump(res, open(OUT + tag + '.json', 'w'), indent=1)
    elif CY == 3:
        P(f'== S-DARK-44 cycle 3, level {LV}: head -> measure-class table and the reverse prediction')
        T = load_wells(LV, 'big'); recs = make_records(T, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        H = load_wells(LV, 'heldout'); hrecs = make_records(H, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        I = load_im77(); irecs = make_records(I, set(M_TALL) | set(M_SHORT), {**M_TALL, **M_SHORT}, set(M_CL), M_SUF, set(), ())
        In = load_im77('new'); nrecs = make_records(In, set(M_TALL) | set(M_SHORT), {**M_TALL, **M_SHORT}, set(M_CL), M_SUF, set(), ())
        P('-- head -> series class table, MD+H, all numerals (class: >= 80% of tokens one series):')
        rows = profile(recs, W, 'ALL')
        P('-- same, counts elsewhere in the text (FAR):'); rowsF = profile(recs, W, 'FAR')
        # reverse prediction: head | (series, item) and head | series, head | item
        def feat(t, kind):
            return {'series': (t['series'],), 'item': (t['item'] if not t['adj'] else 'HEAD',),
                    'series+item': (t['series'], t['item'] if not t['adj'] else 'HEAD'),
                    'series+value': (t['series'], min(t['value'], 8))}[kind]
        def fit(rs, kind, sel):
            tab = collections.defaultdict(collections.Counter)
            for r in rs:
                for t in r['toks']:
                    if sel == 'FAR' and t['adj']: continue
                    tab[feat(t, kind)][r['head']] += 1
            return tab
        def score(tab, rs, kind, sel, modal, conv=None):
            ok = n = 0
            for r in rs:
                for t in r['toks']:
                    if sel == 'FAR' and t['adj']: continue
                    f = feat(t, kind)
                    if conv: f = conv(f)
                    pred = tab[f].most_common(1)[0][0] if f in tab and tab[f] else modal
                    ok += (pred == r['head']); n += 1
            return ok, n
        for sel in ('ALL', 'FAR'):
            modal = collections.Counter(r['head'] for r in recs for t in r['toks'] if not (sel == 'FAR' and t['adj'])).most_common(1)[0][0]
            # 5-fold within MD+H
            idx = list(range(len(recs))); rnd.shuffle(idx); folds = [idx[k::5] for k in range(5)]
            P(f'-- reverse prediction ({sel} numerals): predict the head from the numeral; modal baseline = {W(modal)}')
            for kind in ('series', 'item', 'series+item', 'series+value'):
                ok = n = 0; okb = 0
                for k in range(5):
                    test = set(folds[k]); tr = [recs[i] for i in idx if i not in test]; te = [recs[i] for i in folds[k]]
                    tab = fit(tr, kind, sel); m = collections.Counter(r['head'] for r in tr for t in r['toks'] if not (sel == 'FAR' and t['adj'])).most_common(1)[0][0]
                    o, nn = score(tab, te, kind, sel, m); ok += o; n += nn
                    okb += sum(1 for r in te for t in r['toks'] if not (sel == 'FAR' and t['adj']) and r['head'] == m)
                tab = fit(recs, kind, sel)
                oh, nh = score(tab, hrecs, kind, sel, modal)
                bh = sum(1 for r in hrecs for t in r['toks'] if not (sel == 'FAR' and t['adj']) and r['head'] == modal)
                hm = collections.Counter(r['head'] for r in hrecs for t in r['toks'] if not (sel == 'FAR' and t['adj'])).most_common(1)[0]
                P(f'    {kind:13}: 5-fold MD+H {ok}/{n} = {ok/max(n,1):.2f} (modal {okb/max(n,1):.2f}); held-out sites {oh}/{nh} = {oh/max(nh,1):.2f} (training modal {bh/max(nh,1):.2f}; held-out own modal {W(hm[0])} {hm[1]/max(nh,1):.2f})')
            # IM77 via bridge, series only (+ value)
            m_modal = 342
            tabS = fit(recs, 'series', sel); conv = lambda f: f
            tabW = {k: collections.Counter({(BR.get(str(h)) or [h])[0]: c for h, c in v.items()}) for k, v in tabS.items()}
            tabW = {k: v for k, v in tabW.items()}
            for lab, rs in (('IM77 all', irecs), ('IM77-only', nrecs)):
                o, nn = score(tabW, rs, 'series', sel, m_modal); b = sum(1 for r in rs for t in r['toks'] if not (sel == 'FAR' and t['adj']) and r['head'] == m_modal)
                P(f'    series -> head on {lab}: {o}/{nn} = {o/max(nn,1):.2f} (modal jar {b/max(nn,1):.2f})')
        # frozen pairs and pot numerals
        P('-- comparison with the frozen pairs (GRAMMAR S197/S229) and pot numerals (S206/S-DARK-23.4):')
        froz = {(33, 520): 'tall 3 + arrow', (32, 226): 'tall 2 + 4-stroke fish', (3, 156): 'short 3 + carrier', (17, 585): '7 + W585',
                (17, 575): '7 + W575', (32, 877): 'tall 2 + W877', (33, 923): 'tall 3 + W923', (33, 520): 'tall 3 + arrow'}
        cnt = collections.Counter()
        for t in T:
            s = t['seq']
            for i in range(len(s) - 1):
                if (s[i], s[i + 1]) in froz: cnt[(s[i], s[i + 1])] += 1
        P('    frozen pairs at MD+H: ' + '; '.join(f'{froz[k]} {v}' for k, v in cnt.most_common()))
        C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
        pot = collections.Counter(); pothead = collections.Counter()
        for r in C:
            if not r['type'].startswith('POT') or not r[LV]: continue
            s = r[LV]
            for i, x in enumerate(s):
                if x in NUMS and x not in MARK:
                    pot[('T' if x in TALL else 'S', VAL.get(x, 0))] += 1
                    pothead[s[-1] if s[-1] in CL else 'no head'] += 1
        P(f'    pot numerals (canonical file, see data caution): {dict(sorted(pot.items()))}; heads on numbered pots {dict(pothead)}')
    elif CY == 4:
        P(f'== S-DARK-44 cycle 4, level {LV}: the quantity-seal reading and one out-of-sample prediction')
        T = load_wells(LV, 'big'); recs = make_records(T, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        H = load_wells(LV, 'heldout'); hrecs = make_records(H, NUMS, VAL, set(CL), SUF, MARK, FIXED12)
        In = load_im77('new'); nrecs = make_records(In, set(M_TALL) | set(M_SHORT), {**M_TALL, **M_SHORT}, set(M_CL), M_SUF, set(), ())
        # classes from MD+H, FAR numerals (counts not part of the head's own frozen pair)
        hseq = {t['id']: t['seq'] for t in H}; nseq = {t['id']: t['seq'] for t in In}; hsite = {t['id']: t['site'] for t in H}
        for sel in ('ADJ', 'ALL', 'FAR'):
            prof = collections.defaultdict(collections.Counter)
            for r in recs:
                for t in r['toks']:
                    if sel == 'FAR' and t['adj']: continue
                    if sel == 'ADJ' and not t['adj']: continue
                    prof[r['head']][t['series']] += 1
            cls = {}
            for h, c in prof.items():
                n = sum(c.values())
                if n >= 10: cls[h] = 'TALL' if c['T'] / n >= 0.8 else 'SHORT' if c['S'] / n >= 0.8 else 'MIXED'
            P(f'-- training classes ({sel} numerals, MD+H, >= 10 tokens): ' + '; '.join(f'{W(h)} {cls[h]} ({prof[h]["S"]}S/{prof[h]["T"]}T)' for h in cls))
            for lab, rs, nm, conv in (('Wells held-out sites', hrecs, W, lambda h: h), ('IM77-only texts', nrecs, Mn, lambda m: next((int(w) for w, ms in BR.items() if m in ms and int(w) in cls), None))):
                tot = viol = 0; det = collections.Counter()
                for r in rs:
                    h = conv(r['head'])
                    if h not in cls or cls[h] == 'MIXED': continue
                    for t in r['toks']:
                        if sel == 'FAR' and t['adj']: continue
                        if sel == 'ADJ' and not t['adj']: continue
                        tot += 1
                        if (cls[h] == 'TALL') != (t['series'] == 'T'):
                            viol += 1; det[(nm(r['head']), t['series'], t['value'], r['id'] + ':' + hsite.get(r['id'], 'IM77') + ':' + '-'.join(map(str, (hseq.get(r['id']) or nseq.get(r['id'])))))] += 1
                P(f'    prediction on {lab} ({sel}): heads of one series keep it: violations {viol} of {tot} '
                  f'(upper bound {"< 3/%d = %.3f" % (tot, 3/tot) if viol == 0 and tot else "%.3f" % (viol/max(tot,1))})' +
                  ('; cases: ' + ', '.join(f'{k[0]} {k[1]}{k[2]} ({k[3]})' for k in list(det)[:12]) if det else ''))
        # what the quantity seals look like
        P('-- quantity seals (MD+H seals with a count not part of the head pair):')
        qs = [r for r in recs if r['ot'] == 'seal' and any(not t['adj'] for t in r['toks'])]
        P(f'    {len(qs)} seals; heads {dict(collections.Counter(W(r["head"]) for r in qs).most_common())}')
        it = collections.Counter(); sv = collections.Counter(); op = 0
        seqs = {t['id']: t['seq'] for t in T}
        for r in qs:
            s = seqs[r['id']]
            if s[0] in (817, 861, 820, 920, 692): op += 1
            for t in r['toks']:
                if t['adj']: continue
                it[(W(t['item']), t['series'])] += 1; sv[(t['series'], t['value'])] += 1
        P(f'    opener present {op}/{len(qs)}; counted items x series: {dict(it.most_common(15))}; values {dict(sorted(sv.items()))}')
        # item -> head
        ih = collections.defaultdict(collections.Counter)
        for r in qs:
            for t in r['toks']:
                if not t['adj']: ih[t['item']][r['head']] += 1
        P('    item -> head (items >= 5): ' + '; '.join(f'{W(i)}: {dict((W(h), c) for h, c in v.most_common(3))}' for i, v in sorted(ih.items(), key=lambda kv: -sum(kv[1].values())) if sum(v.values()) >= 5))
    open(OUT + tag + '.txt', 'w').write('\n'.join(LOG) + '\n')
    print('wrote', OUT + tag + '.txt')
