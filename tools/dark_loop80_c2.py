"""Loop 80 cycle 2: do the Meluhhan names look foreign to a Sumerian scribe's name stock?
Ur III personal names from the CDLI ATF dump:
  DOMESTIC = names in administrative slots (ki X-ta, kiszib3 X, giri3 X, X szu ba-ti, dumu X, ugula X) + @seal line 1,
             minus every name that also appears with a foreign gentilic;
  foreign groups = 'NAME (lu2) GN{ki}' and 'NAME mar-tu':
     AMORITE/WEST (mar-tu, Mari, Ebla, Tuttul), HURRIAN/NORTH (Simurrum, Urbilum, Šimanum, Urkiš, Šašrum ...),
     IRAN/ELAM (Anšan, Marhaši, Šimaški, Zidanum, Harši, Hurti, Kimaš ...).
Model: sign-bigram language model with interpolated absolute discounting, trained on DOMESTIC names (10-fold:
domestic test names scored by a model that did not see them). Score = mean surprisal (bits/transition incl. end).
Statistics: AUC foreign-vs-domestic per group (calibration: can the model see foreignness at all?), percentile of
na-na-sa3 and sa6-ma-ar among held-out domestic and among each foreign group; length in signs; head/tail share
(names whose first/last sign is among the 10 commonest domestic initials/finals).
Usage: python3 tools/dark_loop80_c2.py ATF CATALOG OUTJSON"""
import re, csv, sys, json, math, random
from collections import Counter, defaultdict
csv.field_size_limit(10**9)
atf, cat, outj = sys.argv[1:4]
random.seed(80)
per = {}
for r in csv.DictReader(open(cat, errors='replace')):
    if r['id_text'].isdigit():
        per['P%06d' % int(r['id_text'])] = r['period']
GROUP = {}
for g, gns in {
    'AMORITE/WEST': ['mar-tu', 'ma-ri2{ki}', 'eb-la{ki}', 'tu-tu-ul{ki}', 'ur-szu{ki}'],
    'HURRIAN/NORTH': ['si-mu-ru-um{ki}', 'ur-bi2-lum{ki}', 'szi-ma-num2{ki}', 'si-ma-num2{ki}', 'szi-ma-nu-um{ki}', 'szi-ma-ni-um{ki}',
                      'ur-kisz{ki}', 'sza-asz-ru{ki}', 'ti-ki-ti-in-hi{ki}', 'hi-bi2-la-at{ki}', 'ha-bu-ra{ki}', 'szu-ru-ud-hu-um{ki}',
                      'zi-da-ah-ri{ki}', 'ne-gi-ne-hu-um{ki}', 'tal-musz{ki}', 'sze-ti-ir-sza{ki}', 'a-szur5{ki}', 'ha-ar-szi{ki}'],
    'IRAN/ELAM': ['an-sza-an{ki}', 'mar-ha-szi{ki}', 'zi-da-num2{ki}', 'zi-da-nu-um{ki}', 'ba-szim-e{ki}', 'hu-ur5-ti{ki}', 'ki-masz{ki}',
                  'gu-ma-ra-szi{ki}', 'szi-ma-asz-gi4{ki}', '|LU2.SU|{ki}', 'du8-du8-li2{ki}', 'sza-ri2-ip-hu-um{ki}', 'ia3-ab-ra-ad{ki}', 'ma-ah-li{ki}'],
}.items():
    for x in gns:
        GROUP[x] = g
STOP = set('''x u3 mu lu2 dumu dam masz2 gu4 udu niga kas4 kasz4 kin-gi4-a eme-bala aga3-us2 ensi2 engar dub-sar saga zah3 ma2 nu-banda3
dam-gar3 mu-kux(DU) sila3 bad3 sukkal maszkim ugula giri3 kiszib3 szu ba-ti zi-ga ki gurusz geme2 lugal e2 nin igi iti sze ninda kasz i3
siki tug2 ku3-babbar sag nita2 munus u4 a-sza3 erin2 szabra sanga uru gu2-un ba-zi'''.split())
def sgn(name):
    name = re.sub(r'[\[\]#?!<>]', '', name)
    s = [t for t in re.split(r'-', name) if t]
    return s
def okname(tok):
    if not tok or tok in STOP or re.match(r'^[\d(]', tok) or 'x' in tok.split('-') or '...' in tok: return False
    s = sgn(tok)
    return 2 <= len(s) <= 8 and not tok.startswith('{gesz}') and not tok.startswith('{tug2}') and not tok.endswith('{ki}')
dom = Counter(); foreign = defaultdict(Counter); cur = None; ok = False; inseal = False
for l in open(atf, errors='replace'):
    if l.startswith('&P'):
        cur = l[1:8]; ok = per.get(cur, '').startswith('Ur III'); inseal = False; continue
    if not ok: continue
    if l.startswith('@seal'): inseal = True; continue
    if l.startswith('@'): inseal = False
    if not re.match(r"^\s*\d+'?\.", l): continue
    body = re.sub(r'[\[\]#?!<>]', '', l.split('.', 1)[1]).split()
    if inseal and re.match(r'^\s*1\.', l) and len(body) == 1 and okname(body[0]):
        dom[body[0]] += 1
    for i, t in enumerate(body):
        if t in GROUP:
            j = i - 1 if t == 'mar-tu' else (i - 2 if i >= 2 and body[i - 1] == 'lu2' else None)
            if j is not None and j >= 0 and okname(body[j]):
                foreign[GROUP[t]][body[j]] += 1
        nxt = body[i + 1] if i + 1 < len(body) else ''
        prv = body[i - 1] if i > 0 else ''
        if prv in ('kiszib3', 'giri3', 'dumu', 'ugula') and okname(t) and nxt not in ('lu2', 'mar-tu') and nxt not in GROUP:
            dom[t] += 1
        if prv == 'ki' and t.endswith('-ta') and okname(t[:-3]):
            dom[t[:-3]] += 1
        if nxt == 'szu' and i + 2 < len(body) and body[i + 2] == 'ba-ti' and okname(t):
            dom[t] += 1
allforeign = set().union(*[set(c) for c in foreign.values()])
for n in list(dom):
    if n in allforeign or n.startswith('{d}') and len(sgn(n)) < 2:
        del dom[n]
# drop 'names' that are verbs/nouns common outside name slots? keep, as the same filter applies to all groups
D = sorted(dom)
print('domestic distinct names', len(D), {g: len(c) for g, c in foreign.items()})
MEL = {'na-na-sa3': ['na', 'na', 'sa3'], 'sa6-ma-ar': ['sa6', 'ma', 'ar'], 'e-lum-me-luh(?)': ['e', 'lum', 'me', 'luh'],
       'a-li-a-hi (wife, Akkadian)': ['a', 'li', 'a', 'hi']}

class LM:
    def __init__(self, names, d=0.75):
        self.d = d; self.big = defaultdict(Counter); self.uni = Counter()
        for n in names:
            s = ['<s>'] + sgn(n) + ['</s>']
            for a, b in zip(s, s[1:]):
                self.big[a][b] += 1; self.uni[b] += 1
        self.N = sum(self.uni.values()); self.V = len(self.uni) + 1
    def p(self, a, b):
        pu = (self.uni[b] + 0.5) / (self.N + 0.5 * self.V)
        c = self.big.get(a)
        if not c: return pu
        tot = sum(c.values()); lam = self.d * len(c) / tot
        return max(c[b] - self.d, 0) / tot + lam * pu
    def score(self, signs):
        s = ['<s>'] + signs + ['</s>']
        return sum(-math.log2(self.p(a, b)) for a, b in zip(s, s[1:])) / (len(s) - 1)
# 10-fold for domestic
idx = list(range(len(D))); random.shuffle(idx); fold = {D[i]: k % 10 for k, i in enumerate(idx)}
domscore = {}
for f in range(10):
    m = LM([n for n in D if fold[n] != f])
    for n in D:
        if fold[n] == f: domscore[n] = m.score(sgn(n))
full = LM(D)
fscore = {g: {n: full.score(sgn(n)) for n in c} for g, c in foreign.items()}
mscore = {k: full.score(v) for k, v in MEL.items()}
def auc(a, b):
    a = sorted(a); b = sorted(b); wins = 0
    import bisect
    for x in a:
        wins += bisect.bisect_left(b, x) + 0.5 * (bisect.bisect_right(b, x) - bisect.bisect_left(b, x))
    return wins / (len(a) * len(b))
ds = sorted(domscore.values())
def pct(v, arr): arr = sorted(arr); import bisect; return bisect.bisect_left(arr, v) / len(arr)
res = {'n_domestic': len(D), 'n_foreign': {g: len(c) for g, c in foreign.items()}, 'auc': {}, 'meluhha': {}, 'length': {}, 'edge': {}}
for g, sc in fscore.items():
    a = auc(list(sc.values()), ds)
    # permutation null for AUC: shuffle labels
    pool = list(sc.values()) + ds; k = len(sc); null = []
    for _ in range(500):
        random.shuffle(pool); null.append(auc(pool[:k], pool[k:]))
    res['auc'][g] = {'auc': round(a, 3), 'p_perm': round(sum(x >= a for x in null) / 500, 4), 'mean_bits': round(sum(sc.values()) / k, 2)}
res['auc']['DOMESTIC'] = {'mean_bits': round(sum(ds) / len(ds), 2)}
for k, v in mscore.items():
    res['meluhha'][k] = {'bits': round(v, 2), 'pct_domestic': round(pct(v, ds), 3),
                         **{'pct_' + g: round(pct(v, list(sc.values())), 3) for g, sc in fscore.items()}}
# length
def lenstats(names):
    L = Counter(len(sgn(n)) for n in names); n = sum(L.values())
    return {'n': n, 'mean': round(sum(k * v for k, v in L.items()) / n, 2), 'share_3': round(L[3] / n, 3), 'share_2_4': round((L[2] + L[3] + L[4]) / n, 3)}
res['length']['DOMESTIC'] = lenstats(D)
for g, c in foreign.items(): res['length'][g] = lenstats(c)
# edges
ini = Counter(sgn(n)[0] for n in D); fin = Counter(sgn(n)[-1] for n in D)
I10 = {x for x, _ in ini.most_common(10)}; F10 = {x for x, _ in fin.most_common(10)}
def edge(names):
    n = len(names)
    return {'head_top10': round(sum(sgn(x)[0] in I10 for x in names) / n, 3), 'tail_top10': round(sum(sgn(x)[-1] in F10 for x in names) / n, 3)}
res['edge']['top10_initials'] = sorted(I10); res['edge']['top10_finals'] = sorted(F10)
res['edge']['DOMESTIC'] = edge(D)
for g, c in foreign.items(): res['edge'][g] = edge(list(c))
res['edge']['MELUHHA(na-na-sa3, sa6-ma-ar)'] = {'head_top10': sum(v[0] in I10 for v in [MEL['na-na-sa3'], MEL['sa6-ma-ar']]) / 2,
                                               'tail_top10': sum(v[-1] in F10 for v in [MEL['na-na-sa3'], MEL['sa6-ma-ar']]) / 2}
# do the Meluhhan signs/bigrams occur in domestic names at all?
res['meluhha_bigrams_in_domestic'] = {k: [(a, b, full.big[a][b]) for a, b in zip(['<s>'] + v, v + ['</s>'])] for k, v in MEL.items()}
res['examples'] = {g: [n for n, _ in c.most_common(12)] for g, c in foreign.items()}
json.dump(res, open(outj, 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ('auc', 'meluhha', 'length', 'edge')}, indent=1))
print(res['meluhha_bigrams_in_domestic']); print(res['examples'])
