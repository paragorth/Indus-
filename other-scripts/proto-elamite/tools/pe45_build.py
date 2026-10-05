"""pe45 THE NAMES ARE A POPULATION THAT BREEDS.  Build the corpora.

PE   : Susa tablets -> list of name-like strings (pe40 definition: multi-sign entries with numerals,
       final class sign dropped, >= 2 base-form signs left).
UR3  : Ur III (Umma + Girsu) tablets -> personal names taken from fixed name contexts
       (dumu / ugula / giri3 / kiszib3 / ki X-ta / X maszkim / X i3-dab5 / X szu ba-ti).
       Names are cut into syllable signs and the divine determinative {d} is DELETED, so the
       string is opaque.  Truth kept aside: god signs (first syllable after {d}, >= 60% of uses)
       and father-son pairs ('A dumu B' on one line).  Tablets are sampled to copy the PE
       tablet-size template exactly (same number of tablets with 1, 2, 3 ... names).
Output: data/pe45_ckpt/corpora.json
"""
import os, sys, re, csv, json, random, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe40_common import pe_rounds  # noqa
CK = os.path.join(HERE, '..', 'data', 'pe45_ckpt')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
STOP = {'lugal', 'ensi2', 'dumu', 'nu-banda3', 'szabra', 'sanga', 'dub-sar', 'sukkal', 'ugula', 'giri3',
        'kiszib3', 'ki', 'maszkim', 'i3-dab5', 'szu', 'ba-ti', 'lu2', 'nin', 'e2', 'dam', 'gurusz', 'geme2',
        'arad2', 'sza3', 'erin2', 'gu4', 'udu', 'sila4', 'masz2', 'ur', 'nam', 'iti', 'mu', 'u4', 'u3',
        'mu-kux(DU)', 'ba-zi', 'zi-ga', 'kiszib', 'szu-nigin2', 'nigin', 'en', 'nar', 'muhaldim', 'aga3-us2',
        'szesz', 'sa12-du5', 'kurusz-da', 'a-zu', 'lu2-kin-gi4-a', 'ra2-gaba', 'sag-tak4', 'ba-usz2'}


def clean(w):
    return re.sub(r'[#!?*<>\[\]]', '', w)


def bad(w):
    if not w or w in STOP or '{ki}' in w or '...' in w or re.search(r'(^|[-.])x($|[-.])', w):
        return True
    if re.match(r'^\d', w) or '(' in w and re.match(r'^\d', w.split('(')[0] or '0'):
        return True
    if '-' not in w and '{d}' not in w and '.' not in w:
        return True        # one-syllable words are mostly nouns; keep only multi-sign names
    return False


def syl(w):
    w = re.sub(r'\{(?!d\})[^}]*\}', '', w)          # drop other determinatives
    w = w.replace('{d}', '-{d}-')
    t = [x for x in re.split(r'[-.]', w) if x]
    out, god = [], set()
    nxt = False
    for x in t:
        if x == '{d}':
            nxt = True
            continue
        if nxt:
            god.add(len(out))
            nxt = False
        out.append(x)
    return out, god


def ur3_tablets():
    csv.field_size_limit(10 ** 9)
    keep = set()
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and row['id_text'].isdigit():
            pv = (row['provenience'] or '')
            if pv.startswith('Umma') or pv.startswith('Girsu'):
                keep.add('P%06d' % int(row['id_text']))
    T = collections.defaultdict(list)
    PAIRS = collections.defaultdict(list)
    cur = None
    for raw in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            cur = raw[1:8] if raw[1:8] in keep else None
            continue
        if not cur or not re.match(r"^\d+'?\.", raw):
            continue
        tk = [clean(x) for x in raw.split()[1:]]
        tk = [x for x in tk if x]
        got = []
        for i, x in enumerate(tk):
            if x in ('dumu', 'ugula', 'giri3', 'kiszib3') and i + 1 < len(tk):
                got.append(tk[i + 1])
            if x == 'ki' and i + 1 < len(tk) and tk[i + 1].endswith('-ta'):
                got.append(tk[i + 1][:-3])
            if x in ('maszkim', 'i3-dab5') and i > 0:
                got.append(tk[i - 1])
            if x == 'szu' and i + 1 < len(tk) and tk[i + 1] == 'ba-ti' and i > 0:
                got.append(tk[i - 1])
        for i, x in enumerate(tk):
            if x == 'dumu' and 0 < i < len(tk) - 1 and not bad(tk[i - 1]) and not bad(tk[i + 1]):
                got.append(tk[i - 1])
                PAIRS[cur].append((tk[i - 1], tk[i + 1]))
        for w in got:
            if not bad(w):
                T[cur].append(w)
    return T, PAIRS


def main():
    os.makedirs(CK, exist_ok=True)
    R = pe_rounds('str')
    pe = [{'id': t, 'names': [n.split() for n in s]} for t, s in R]
    T, PAIRS = ur3_tablets()
    # god truth on raw names
    gcount, tot = collections.Counter(), collections.Counter()
    for p, ws in T.items():
        for w in ws:
            s, g = syl(w)
            for i, x in enumerate(s):
                tot[x] += 1
                if i in g:
                    gcount[x] += 1
    gods = sorted(x for x in tot if tot[x] >= 3 and gcount[x] / tot[x] >= 0.6 and re.fullmatch(r'[a-z]+[0-9]*', x) and x != 'x')
    # tablet-size template matching
    rng = random.Random(45)
    sizes = [len(t['names']) for t in pe]
    byn = collections.defaultdict(list)
    for p, ws in T.items():
        u = list(dict.fromkeys(ws))  # distinct names in first-seen order
        if len(u) >= 1:
            byn[len(u)].append(p)
    for v in byn.values():
        rng.shuffle(v)
    used = set()
    ur = []
    for n in sorted(sizes, reverse=True):
        cand = None
        for m in list(range(n, n + 40)):
            pool = [p for p in byn.get(m, []) if p not in used]
            if pool:
                cand = pool[0]
                break
        used.add(cand)
        u = list(dict.fromkeys(T[cand]))[:n]
        names = [syl(w)[0] for w in u]
        kin = [[a, b] for a, b in PAIRS.get(cand, []) if a in u and b in u]
        ur.append({'id': cand, 'names': names, 'raw': u, 'kin': kin})
    out = {'PE': pe, 'UR3': ur, 'UR3_gods': gods,
           'UR3_godfreq': {g: tot[g] for g in gods}}
    json.dump(out, open(os.path.join(CK, 'corpora.json'), 'w'))
    print('PE tablets', len(pe), 'names', sum(sizes))
    print('UR3 tablets', len(ur), 'names', sum(len(t['names']) for t in ur), 'kin tablets', sum(1 for t in ur if t['kin']))
    print('gods', len(gods), sorted(gods, key=lambda g: -tot[g])[:30])
    print(collections.Counter(len(n) for t in ur for n in t['names']))
    print(ur[0]['raw'][:10])


if __name__ == '__main__':
    main()
