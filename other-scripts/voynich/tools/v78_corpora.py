"""v78 corpora: control texts with a known doubling mechanism, written through the v72 surface machinery.

REDUP (real reduplicating languages; full reduplication written with a hyphen is split into two words):
  TL_MED  Tagalog medical book (Gutenberg 17479, Tagalog paragraphs only), TL_NOLI Noli Me Tangere in Tagalog (20228)
  MS_1001 Malay 1001 Nights, 1899 (archive.org Hikajat_1001_malam...), HE Hebrew Bible opening (data/plain/he.txt)
DITTOG (real copying slips): PL_DITT  Petrus Plaoul, five diplomatic witnesses (SCTA), first writing ('as written'),
  real manuscript lines, the paragraphs that contain a struck dittography (58 events)
LIST, TALLY (planted writer conventions on real list bases; positive plants):
  LIST_SYON  Syon catalogue entries; the writer marks 'the same again' by writing an entry's headword twice (p 0.15)
  LIST_SIN   Sinonoma Bartholomei; the same
  TALLY_ING  Apicius + Antidotarium ingredient streams; after an ingredient (p 0.3) a quantity written as a unit token
             repeated n times, n = 1..5 (0.4, 0.3, 0.15, 0.1, 0.05)
NONE (no mechanism): LA Isidore, IT Brumati herbal (v72 loaders)
GEN (generators fitted to the Voynich ZL3b surface): SELFCIT, SC10, JUNC, MK2, STACK (v77)
Output: data/v78_ckpt/corpora.pkl = {name: dict(kind, pages(surface))}
"""
import os, sys, re, json, random, pickle, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L, v72_lib as V

SRC = L.SRC


def words(s):
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.replace('-', ' ')
    return [w for w in re.findall(r'[a-z]+', s) if len(w) >= 2 or w in ('i', 'o', 'y', 'e', 'a')]


def gut_paras(fn):
    t = open(os.path.join(SRC, fn), encoding='utf-8', errors='ignore').read()
    a = t.find('*** START'); b = t.find('*** END')
    t = t[t.find('\n', a) + 1:b] if a >= 0 else t
    return [p for p in re.split(r'\n\s*\n', t) if p.strip()]


TLF = {'ang', 'nang', 'sa', 'na', 'manga', 'mga', 'at', 'ng', 'ay', 'si', 'ni', 'ito', 'iyan', 'kung', 'cung'}
ESF = {'de', 'la', 'el', 'que', 'y', 'en', 'los', 'las', 'por', 'con', 'del', 'se', 'es', 'para'}


def tagalog(fn, cap=40000):
    out = []
    for p in gut_paras(fn):
        ws = words(p)
        if len(ws) < 8: continue
        t = sum(w in TLF for w in ws); e = sum(w in ESF for w in ws)
        if t > 2 * e and t >= 2: out.append(ws)
    return out


def malay(cap=40000):
    t = open(os.path.join(SRC, 'ms_1001.txt'), encoding='utf-8', errors='ignore').read()
    t = re.sub(r'-\s*\n\s*', '-', t)
    out = []
    for p in re.split(r'\n\s*\n', t):
        ws = [w for w in words(p) if len(w) >= 2]
        if len(ws) >= 8: out.append(ws)
    return out[len(out) // 10:]


def hebrew():
    t = open(os.path.join(L.ROOT, 'data', 'plain', 'he.txt'), encoding='utf-8').read()
    out = []; cur = []
    for line in t.split('\n'):
        ws = [w for w in re.findall(r'[א-ת]+', line)]
        cur += ws
        if len(cur) >= 40: out.append(cur); cur = []
    return out


def plaoul_ditt():
    import v63_lib as P
    P.CK = L.CK; P.REPO = os.path.join(SRC, 'plaoulcommentary')
    W = P.plaoul_witnesses(cache=True)
    pages = []; nev = 0
    for k, p in enumerate(W):
        T = [t for t in p['toks'] if t['st'] != 'a' and t['before']]
        hit = False
        for i, t in enumerate(T):
            if t['st'] == 'd':
                nb = [T[j]['before'] for j in (i - 1, i + 1) if 0 <= j < len(T) and T[j]['st'] != 'd']
                if t['before'] in nb: hit = True; nev += 1
        if not hit: continue
        lines = [[]]
        for t in T:
            lines[-1].append(t['before'])
            if t['lb']: lines.append([])
        lines = [l for l in lines if l]
        pages.append(dict(id='pl%04d' % k, sec='g%d' % (k * 6 // len(W)), lang='-', hand='-', quire='-',
                          lines=[dict(w=l, ps=(i == 0)) for i, l in enumerate(lines)]))
    return pages, nev


def v75_entries(sid):
    S = json.load(open(os.path.join(L.ROOT, 'data', 'v75_ckpt', 'systems.json')))
    return [(s, ws) for s, ws in S[sid]['entries'] if ws]


def plant_list(ents, seed, p=0.15):
    rng = random.Random(seed); out = []
    for s, ws in ents:
        ws = list(ws)
        if rng.random() < p: ws = [ws[0]] + ws
        out.append((s, ws))
    return out


def plant_tally(ents, seed, p=0.3, unit='uncia'):
    rng = random.Random(seed); out = []
    for s, ws in ents:
        nw = []
        for w in ws:
            nw.append(w)
            if len(w) > 3 and rng.random() < p:
                n = rng.choices([1, 2, 3, 4, 5], [0.4, 0.3, 0.15, 0.1, 0.05])[0]
                nw += [unit] * n
        out.append((s, nw))
    return out


def build():
    C = {}
    def lang(name, kind, paras, seed):
        pages = L.plain_pages(paras, name[:4].lower(), cap=40000)
        C[name] = dict(kind=kind, pages=L.through_surface(pages, seed))
    lang('TL_MED', 'REDUP', tagalog('tl_17479.txt'), 781)
    lang('TL_NOLI', 'REDUP', tagalog('tl_20228.txt'), 782)
    lang('MS_1001', 'REDUP', malay(), 783)
    lang('HE', 'REDUP', hebrew(), 784)
    pp, nev = plaoul_ditt()
    C['PL_DITT'] = dict(kind='DITTOG', pages=L.through_surface(pp, 785), nev=nev)
    for nm, sid, sd in (('LIST_SYON', 'CAT_SYON', 786), ('LIST_SIN', 'SINONOMA', 787)):
        e = plant_list(v75_entries(sid), sd)
        C[nm] = dict(kind='LIST', pages=L.through_surface(V._pages_from_entries(e, None, cap=40000, prefix=nm[:5]), sd))
    e = plant_tally(v75_entries('ING_APIC') + v75_entries('ING_ANTID'), 788)
    C['TALLY_ING'] = dict(kind='TALLY', pages=L.through_surface(V._pages_from_entries(e, None, cap=40000, prefix='tal'), 788))
    C['LA_ISID'] = dict(kind='NONE', pages=L.through_surface(V.isidore_plain(), 789))
    C['IT_BRUM'] = dict(kind='NONE', pages=L.through_surface(V.brumati_plain(), 790))
    import v77_lib as G
    Z = V.voynich('ZL3b')
    for g, f in G.GENS.items():
        C['GEN_' + g] = dict(kind='GEN', pages=f(Z, seed=7801))
    C['VOY_ZL'] = dict(kind='?', pages=Z); C['VOY_IT'] = dict(kind='?', pages=V.voynich('IT2a'))
    pickle.dump(C, open(os.path.join(L.CK, 'corpora.pkl'), 'wb'))
    for k, v in C.items():
        n = sum(len(l['w']) for p in v['pages'] for l in p['lines'])
        print(k, v['kind'], len(v['pages']), n, v.get('nev', ''))
    return C


if __name__ == '__main__':
    build()
