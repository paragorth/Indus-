"""v30 dialect ladder: build every corpus as a list of pages (each page a list of words).

Sources (all used as raw symbol data only):
  Voynich   data/derived/ZL3b_lines.json, IT2a_lines.json (paragraph lines, words with '?' dropped)
  German    ReF v1.0.2 (Zenodo 5793616), diplomatic tok_dipl, manuscripts 1350-1450
  Italian / Latin  CATMuS medieval (HuggingFace), abbreviation-preserving line transcriptions
  Czech     Dalimil chronicle (cs.wikisource, edition spelling) = data/plain/cs.txt
Large downloads stay in the scratchpad; the output is a small JSON checkpoint.
"""
import json, os, re, glob, random, unicodedata, html, collections
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v30_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('V30_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v30')
KEEP_EXTRA = set('⁊&')


def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    return ''.join(ch for ch in w if unicodedata.category(ch)[0] in 'LM' or ch in KEEP_EXTRA)


def chunk(words, size=200):
    return [words[i:i + size] for i in range(0, len(words), size) if len(words[i:i + size]) >= size // 4]


def voynich(name='ZL3b'):
    L = json.load(open(os.path.join(ROOT, 'data', 'derived', name + '_lines.json')))
    pages = collections.OrderedDict()
    meta = {}
    for l in L:
        if l['ltype'] != 'P' or l['lang'] not in ('A', 'B'):
            continue
        ws = [w for w in l['words'] if '?' not in w and w.islower()]
        pages.setdefault(l['folio'], []).extend(ws)
        meta[l['folio']] = (l['lang'], l['illus'], l['hand'])
    return pages, meta


def ref_text(sig):
    f = glob.glob(os.path.join(SCR, 'ReF-v1.0.2', 'ref-*', sig + '.xml'))[0]
    txt = open(f, encoding='utf-8').read()
    ws = [norm(html.unescape(u)) for u in re.findall(r'<tok_dipl [^>]*utf="([^"]*)"', txt)]
    return [w for w in ws if w]


def catmus(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(SCR, 'cat', pattern))):
        t = pq.read_table(f, columns=['text'])
        ws = []
        for line in t.column('text').to_pylist():
            ws.extend(norm(x) for x in line.split())
        out.append([w for w in ws if w])
    return out


def czech():
    txt = open(os.path.join(ROOT, 'data', 'plain', 'cs.txt'), encoding='utf-8').read()
    return [w for w in (norm(x) for x in txt.split()) if w]


# Pre-reform (digraphic) Czech respelling: the documented 14th-c. habits
# (cz for c-caron, rz for r-caron, ss for s-caron, g for j, w for v, ie for e-caron, no length marks)
OLD_CZ = [('č', 'cz'), ('ř', 'rz'), ('š', 'ss'), ('ž', 'z'), ('ě', 'ie'),
          ('ů', 'o'), ('ň', 'n'), ('ť', 't'), ('ď', 'd'), ('́', ''),
          ('j', 'g'), ('v', 'w'), ('ch', 'ch'), ('c', 'cz')]


def old_czech(w):
    # apply simultaneously, left to right, longest first (c-caron before c)
    out, i = [], 0
    while i < len(w):
        for a, b in OLD_CZ:
            if w.startswith(a, i):
                out.append(b); i += len(a); break
        else:
            out.append(w[i]); i += 1
    return ''.join(out)


def markov_resynth(pages, seed):
    rng = random.Random(seed)
    tri = collections.defaultdict(collections.Counter)
    for p in pages:
        for w in p:
            s = '^^' + w + '$'
            for i in range(2, len(s)):
                tri[s[i - 2:i]][s[i]] += 1
    tab = {k: (list(v.keys()), list(v.values())) for k, v in tri.items()}
    out = []
    for p in pages:
        q = []
        for _ in p:
            ctx, w = '^^', ''
            while True:
                ks, vs = tab[ctx]
                c = rng.choices(ks, vs)[0]
                if c == '$' or len(w) > 25:
                    break
                w += c; ctx = ctx[1] + c
            q.append(w if w else 'o')
        out.append(q)
    return out


def substitute(pages, seed, n_swap=None):
    rng = random.Random(seed)
    alpha = sorted({c for p in pages for w in p for c in w if unicodedata.category(c)[0] == 'L'})
    if n_swap is None:
        perm = alpha[:]; rng.shuffle(perm); key = dict(zip(alpha, perm))
    else:
        freq = collections.Counter(c for p in pages for w in p for c in w)
        top = [c for c, _ in freq.most_common(20) if c in alpha]
        pick = rng.sample(top, n_swap * 2); key = {}
        for a, b in zip(pick[::2], pick[1::2]):
            key[a] = b; key[b] = a
    return [[''.join(key.get(c, c) for c in w) for w in p] for p in pages], key


def build():
    C = {}
    vp, vm = voynich('ZL3b')
    ip, im = voynich('IT2a')
    sel = lambda pred, P=vp, M=vm: [P[f] for f in P if pred(M[f]) and len(P[f]) >= 10]
    C['V_A'] = sel(lambda m: m[0] == 'A')
    C['V_B'] = sel(lambda m: m[0] == 'B')
    C['V_Aherb'] = sel(lambda m: m[0] == 'A' and m[1] == 'H')
    C['V_Bherb'] = sel(lambda m: m[0] == 'B' and m[1] == 'H')
    C['V_B2'] = sel(lambda m: m[0] == 'B' and m[2] == '2')
    C['V_B3'] = sel(lambda m: m[0] == 'B' and m[2] == '3')
    C['V_Apharm'] = sel(lambda m: m[0] == 'A' and m[1] == 'P')
    C['V_Bbio'] = sel(lambda m: m[0] == 'B' and m[1] == 'B')
    C['V_Bstar'] = sel(lambda m: m[0] == 'B' and m[1] == 'S')
    C['V_A_IT'] = [ip[f] for f in ip if im[f][0] == 'A' and len(ip[f]) >= 10]
    # German, ReF manuscripts 1350-1450, prose
    bav1 = ['F022', 'F005', 'F004']          # mittel/nordbairisch 15,1
    bav2 = ['F001', 'F002']                  # mittel/nordbairisch 14,2 (Buch der Natur, Runtingerbuch)
    alem = ['F091', 'F092', 'F089', 'F088']  # hochalemannisch
    rip = ['F154', 'F155']                   # ripuarisch
    for k, sigs in [('G_Bav1', bav1), ('G_Bav2', bav2), ('G_Alem', alem), ('G_Rip', rip)]:
        C[k] = [pg for s in sigs for pg in chunk(ref_text(s))]
    # Italian / Latin, CATMuS
    com1 = catmus('L-Ita.C-14.S-Cur.Milano*') + catmus('L-Ita.C-14.S-Cur.Cologny*')
    com2 = catmus('L-Ita.C-14.S-Cur.Paris__BnF__ita__79*') + catmus('L-Ita.C-14.S-Hyb.Vatican*') + catmus('L-Ita.C-14.S-Cur.Venise*')
    C['I_Com1'] = [pg for d in com1 for pg in chunk(d)]
    C['I_Com2'] = [pg for d in com2 for pg in chunk(d)]
    ita = [d for f in ['L-Ita.C-15.S-Hum.Oxford*', 'L-Ita.C-14.S-Cur.Paris__BnF__ita__434*', 'L-Ita.C-15.S-Hum.Paris__BnF__ita__583*',
                       'L-Ita.C-15.S-Hum.Cambridge*', 'L-Ita.C-15.S-Sem*', 'L-Ita.C-15.S-Hum.Paris__BnF__ita__1019*',
                       'L-Ita.C-15.S-Cur.Oxford*', 'L-Ita.C-15.S-Hyb*', 'L-Ita.C-15.S-Hum.Paris__BnF__ita__70*'] for d in catmus(f)]
    C['I_Ita'] = [pg for d in ita for pg in chunk(d)]
    lat = []
    for f in sorted(glob.glob(os.path.join(SCR, '..', 'catmus', 'L-Lat.C-1[345]*'))) + sorted(glob.glob(os.path.join(SCR, 'cat', 'L-Lat*'))):
        t = pq.read_table(f, columns=['text'])
        ws = [norm(x) for line in t.column('text').to_pylist() for x in line.split()]
        lat.append([w for w in ws if w])
    C['I_Lat'] = [pg for d in lat for pg in chunk(d)]
    # Czech: edition spelling vs pre-reform digraph respelling of the other half of the chronicle
    cz = chunk(czech())
    C['C_Mod'] = cz[0::2]
    C['C_Old'] = [[old_czech(w) for w in p] for p in cz[1::2]]
    C['C_Mod2'] = cz[1::2]
    # cipher-like transforms of Bav2
    C['K_Sub'], k1 = substitute(C['G_Bav2'], 11)
    C['K_Swap3'], k2 = substitute(C['G_Bav2'], 12, n_swap=3)
    # Markov resyntheses (trigram, word-internal)
    C['M_A'] = markov_resynth(C['V_A'], 21)
    C['M_B'] = markov_resynth(C['V_B'], 22)
    json.dump({'corpora': C, 'keys': {'K_Sub': k1, 'K_Swap3': k2}}, open(os.path.join(CK, 'corpora.json'), 'w'), ensure_ascii=False)
    for k, v in C.items():
        n = sum(len(p) for p in v)
        al = collections.Counter(c for p in v for w in p for c in w)
        print(f'{k:10s} pages {len(v):4d} words {n:6d} alphabet {len(al):3d}  e.g. {" ".join(v[0][:8])}')


if __name__ == '__main__':
    build()
