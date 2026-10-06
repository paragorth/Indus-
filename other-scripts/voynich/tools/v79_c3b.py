"""v79 cycle 3b: repairs and the decisive fingerprint.

(1) Repeat fingerprint: P(mark = mark of the line above) / expectation under within-paragraph shuffles, for every
    channel type: counters and calendar letters (never repeat), class / topic / genus keys (repeat down an entry),
    index letters (content-driven), the v72 anti-repeat marker; vs the Voynich and generators.
(2) Cross-page item keys, cleaned: keys only from margin slots (marks of the line and of lines -2..+2, the line's
    second glyph); body = words 2..n. New control C_TOPIC: Isidore, the line's chapter (CAPUT) as the margin key
    (mod 10, 80% of lines) -- the 'running index of the item a line describes'.
(3) Index letter, honest null: the mark vs the first glyph of the line's rarest body word (and the 27 other
    selector/position pairs) with marks shuffled within paragraph AND line mode; held-out by leaf halves; max over the
    28 pairs compared with the same max on generators.
"""
import sys, os, time, random, re, zlib, unicodedata
import numpy as np
from collections import Counter, defaultdict
import v79_lib as L
import v79_c3 as C3
from v79_c2 import MODE

C3.SLOTS = ['t0', 't-1', 't+1', 't-2', 't+2', 'g2']


def topic_plain():
    t = open(os.path.join(L.ROOT, 'data', 'plain', 'la.txt'), encoding='utf-8').read()
    paras = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
    lines, cap = [], 0
    for p in paras:
        if p.startswith('CAPUT'): cap += 1; continue
        s = unicodedata.normalize('NFD', p.lower()); s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        s = re.sub(r'\(\d+[a-z]?\)', ' ', s)
        ws = [w.replace('j', 'i').replace('v', 'u') for w in re.findall(r'[a-z]+', s)]
        if len(ws) < 12: continue
        for i in range(0, len(ws), 8):
            lines.append(dict(w=ws[i:i + 8], ps=(i == 0), cap=cap, sec='L%d' % min(5, cap // 12)))
    lines = lines[:5000]
    pages = []
    cur = None
    for l in lines:
        if cur is None or len(cur['lines']) >= 24 or cur['sec'] != l['sec']:
            cur = dict(id='tp%03d' % len(pages), sec=l['sec'], lang='-', hand='-', quire='-', lines=[]); pages.append(cur)
        cur['lines'].append(l)
    return pages


def repeat_fp(pages):
    T, al = L.chain_table(pages)
    g, para, ps = T['g'], T['para'], T['ps']
    obs = []; idx = []
    for i in range(1, len(g)):
        if para[i] == para[i - 1] and not ps[i] and not ps[i - 1] and g[i] >= 0 and g[i - 1] >= 0:
            obs.append(g[i] == g[i - 1]); idx.append(i)
    o = np.mean(obs)
    rng = np.random.default_rng(0); nv = []
    for k in range(50):
        Tn = L.shuffle_within_para(T, rng); gn = Tn['g']
        nv.append(np.mean([gn[i] == gn[i - 1] for i in idx if gn[i] >= 0 and gn[i - 1] >= 0]))
    return float(o), float(np.mean(nv)), float(o / np.mean(nv))


def part_B2(R, nnull=40, seed=0):
    wc = Counter(w for r in R for w in r['body'])
    pages_of = defaultdict(set)
    for r in R:
        for w in r['body']: pages_of[w].add(r['page'])
    sels = {
        'rarest': lambda r: [min(r['body'], key=lambda w: wc[w])] if r['body'] else [],
        'recurs_elsewhere': lambda r: [w for w in r['body'] if len(pages_of[w] - {r['page']}) > 0 and wc[w] <= 20],
        'longest': lambda r: [max(r['body'], key=len)] if r['body'] else [],
        'word2': lambda r: r['body'][:1], 'word3': lambda r: r['body'][1:2], 'last': lambda r: r['body'][-1:],
        'any': lambda r: r['body']}
    posf = {'first': lambda w: w[0], 'first_noq': lambda w: w[1] if w[0] == 'q' and len(w) > 1 else w[0],
            'second': lambda w: w[1] if len(w) > 1 else '#', 'last': lambda w: w[-1]}
    body = [i for i, r in enumerate(R) if not r['ps']]
    tags = [R[i]['w'][0][0] for i in body]
    half = np.array([R[i]['half'] for i in body])
    rng = random.Random(seed)
    grp = defaultdict(list)
    for j, i in enumerate(body): grp[(R[i]['para'], R[i]['mode'])].append(j)
    shuf = []
    for k in range(nnull):
        t2 = list(tags)
        for ix in grp.values():
            v = [t2[j] for j in ix]; rng.shuffle(v)
            for j, x in zip(ix, v): t2[j] = x
        shuf.append(np.array(t2))
    tags = np.array(tags)
    out = {}
    for sn, sf in sels.items():
        S = [sf(R[i]) for i in body]
        for pn, pf in posf.items():
            G = [set(pf(w) for w in s) for s in S]
            hit = np.array([t in g for t, g in zip(tags, G)])
            res = []
            for h in (0, 1):
                m = half == h
                nv = np.array([np.mean([t in g for t, g, mm in zip(ts, G, m) if mm]) for ts in shuf])
                o = hit[m].mean()
                res.append((float(o), float(nv.mean()), float((o - nv.mean()) / (nv.std() + 1e-9))))
            out['%s/%s' % (sn, pn)] = res
    return out


def corpora():
    C = {}
    for nm in ('ZL3b', 'IT2a'):
        C[nm] = ('VOY', L.V77.voy(nm))
    TP = topic_plain()
    C['C_TOPIC'] = ('CTL', L.surface_channel(TP, lambda l: l['cap'] % 10, p_write=0.8)[0])
    C['C_GENUS'] = ('CTL', L.surface_channel(C3.genus_pages(), lambda l: zlib.crc32(str(l['gen']).encode()) % 10, p_write=0.8)[0])
    C['C_INDEX'] = ('CTL', C3._index_surface(L.V72.isidore_plain()[:200]))
    B = L.brumati_entries_plain()
    C['C_CLASS'] = ('CTL', L.surface_channel(B, lambda l: l['cls'] % 10, p_write=0.8)[0])
    C['C_PART'] = ('CTL', L.surface_channel(L.brumati_parts_plain(), lambda l: min(l['part'], 4), p_write=0.8)[0])
    C['C_DANTE'] = ('CTL', L.surface_channel(L.dante_plain()[:170], lambda l: l['ch'] % 10)[0])
    CA = L.calendar_plain(years=4)
    C['C_DOM'] = ('CTL', L.surface_channel(CA, lambda l: l['dom'])[0])
    C['C_GOLD'] = ('CTL', L.surface_channel(CA, lambda l: None if l['gold'] is None else l['gold'] % 10, p_write=0.8)[0])
    I = L.V72.isidore_plain()[:200]
    C['N_ISI'] = ('NEG', L.V72.surface(L.V72.encode_payload(I, L.V72.payload_code([w for p in I for l in p['lines'] for w in l['w']])), seed=7))
    for g in ('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'):
        C['G_' + g] = ('GEN', L.V77.generate('ZL3b', g, 794))
    return C


def main():
    t0 = time.time()
    C = corpora()
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(C) if part == 'all' else list(C)[int(part)::2]
    for nm in names:
        kind, pages = C[nm]
        fp = repeat_fp(pages)
        R = C3.records(pages)
        rA = C3.part_A(R, 1500, 2) if kind != 'CTL' or nm in ('C_TOPIC', 'C_GENUS', 'C_CLASS', 'C_INDEX') else None
        rB = part_B2(R) if nm not in ('C_DANTE', 'C_DOM', 'C_GOLD') else None
        L.psave('c3b_%s.pkl' % nm, dict(kind=kind, fp=fp, A=rA, B=rB))
        msg = '%s %.0fs repeat %.3f/%.3f = %.2f' % (nm, time.time() - t0, *fp)
        if rA: msg += ' | A held %.4f z %.1f t0 %.4f z %.1f best %s' % (rA['real']['held'], rA['z_held'], rA['t0'], rA['z_t0'], rA['real']['best0'][:2])
        if rB:
            mx = max(rB.items(), key=lambda x: min(x[1][0][2], x[1][1][2]))
            msg += ' | B rarest/first %s ; best-both-halves %s %s' % ([round(x[2], 1) for x in rB['rarest/first']], mx[0], [round(x[2], 1) for x in mx[1]])
        print(msg, flush=True)


if __name__ == '__main__':
    main()
