"""pe15 ('the administration is a food web'): build bipartite consumer x resource
event lists for Proto-Elamite and the controls.

Each network = list of events (consumer, resource, tablet), plus optional truth labels
per tablet (known office / institution) for the controls.

PE_STR  : consumer = entry middle string (entry signs minus final class sign; single-sign
          entries whose sign is a class sign are dropped); resource = class sign (or '-')
          + '|' + number system.  x-free entries only.
PE_SIGN : consumer = each sign of the middle; same resource.
PE_HDR  : consumer = each middle sign; resource = the tablet header (full string).
PE_HRES : consumer = header string; resource = class|system of the entries it governs.
UR3_STR : Ur III (CDLI), name-like string attached to an entry x commodity word.
          Truth = provenience (Umma, Girsu, Puzrish-Dagan, Ur, Nippur, Irisagrig, Garshana).
UR3_SIGN: each syllable of the name x commodity (truth = provenience).
UR3_DAB : Puzrish-Dagan only; truth = the receiving official (PN i3-dab5), a known office.
LB_STR  : Linear B (DAMOS, KN + PY), first syllabic word of a line x first ideogram.
          Truth = series prefix (e.g. KN Da, PY Cn) and site.
LB_SIGN : syllables of the word x ideogram.

usage: python3 pe15_build.py <cdli.atf> <cdli_cat.csv>
"""
import csv, json, os, random, re, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa
from pe4_common import FINAL  # noqa
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CK = os.path.join(HERE, '..', 'data', 'pe15_ckpt')
os.makedirs(CK, exist_ok=True)


def pe():
    S, G, H, HR = [], [], [], []
    for t in load():
        lines = t['lines']
        off = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        hdr = None
        if lines and not lines[0]['numerals']:
            hs = [base(s) for s in lines[0]['signs'] if is_sign(s)]
            if hs and 'x' not in lines[0]['signs']:
                hdr = ' '.join(hs)
        for i, l in enumerate(lines):
            if i == 0 and not l['numerals']:
                continue
            if l['surface'] != 'obverse' and len(off) == 1 and l is off[0]:
                continue
            if not l['numerals'] or 'x' in l['signs'] or l['lacuna']:
                continue
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            sysc = system_of(l['numerals']) or '?'
            cls = sg[-1] if sg[-1] in FINAL else '-'
            mid = sg[:-1] if cls != '-' else sg
            res = cls + '|' + sysc
            if mid:
                S.append((' '.join(mid), res, t['id']))
                for s in mid:
                    G.append((s, res, t['id']))
                    if hdr:
                        H.append((s, hdr, t['id']))
            if hdr:
                HR.append((hdr, res, t['id']))
    return S, G, H, HR


NUM = re.compile(r"^([\d/]+|n)\(([^)]*)\)$")
UNIT = set("sila3 gin2 gur sar ma-na gu2 GAN2 la2 iku bur3 esze3 ban2 gu2-un sila3-ta gin2-ta ma-na-ta gurusz geme2 u4 sze-bi kusz3".split())
FORM = set('''ki giri3 szu mu iti u4 kiszib3 ba-zi zi-ga i3-dab5 sa2-du11 szunigin szu-nigin2 dub-sar
dumu ugula sza3 a-sza3 arad2 ARAD2 u3 ba-usz2 ba-ug7 e2 kiszib nig2-ka9 sa10 lugal mu-kux(DU) ba-ti
la2-ia3 la2-ia3-am3 si-i3-tum zi-ga-am3 ki-ba ba-na-a-ga2-ar szu-ba-ti x ... gurusz geme2 nu-banda3 erin2
diri nig2-ba sag-nig2-gur11-ra-kam sza3-bi-ta kab2-ku5 uri5{ki} nibru{ki} mu-DU bala gaba-ri'''.split())


def clean_tok(w):
    return re.sub(r"[\[\]#?!<>*]", '', w)


def ur3(atf, catf):
    csv.field_size_limit(10 ** 9)
    prov = {}
    for row in csv.DictReader(open(catf, encoding='utf-8')):
        if row['period'].startswith('Ur III') and row['genre'].startswith('Admin'):
            p = row['provenience'].split(' (')[0]
            if p in ('Umma', 'Girsu', 'Puzriš-Dagan', 'Ur', 'Nippur', 'Irisagrig', 'Garšana'):
                prov['P%06d' % int(row['id_text'])] = p
    tabs, cur = {}, None
    for raw in open(atf, encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            cur = raw[1:8] if raw[1:8] in prov else None
            if cur:
                tabs[cur] = []
            continue
        if cur is None:
            continue
        m = re.match(r"^[0-9]+[a-z0-9']*\.\s+(.*)$", raw.rstrip('\n'))
        if m:
            toks = [clean_tok(w) for w in m.group(1).split()]
            tabs[cur].append([w for w in toks if w])
    STR, SIG, DAB = [], [], []
    truth = {}
    for p, L in tabs.items():
        off = None
        for l in L:
            if len(l) == 2 and l[1] == 'i3-dab5':
                off = l[0]
        ev = []
        for i, l in enumerate(L):
            if not l or not NUM.match(l[0]):
                continue
            comm, rest, j = None, [], 0
            while j < len(l):
                if NUM.match(l[j]):
                    j += 1
                    while j < len(l) and l[j] in UNIT:
                        j += 1
                    if j < len(l) and not NUM.match(l[j]) and comm is None:
                        comm = l[j]
                        j += 1
                    continue
                rest.append(l[j])
                j += 1
            if not comm or 'x' in comm or '...' in comm:
                continue
            comm = re.sub(r'\{[^}]*\}', '', comm) or comm
            cons = [w for w in rest if ('-' in w or '{d}' in w) and w not in FORM]
            if not cons and i + 1 < len(L):
                nx = L[i + 1]
                if nx and not NUM.match(nx[0]) and nx[0] not in FORM and len(nx) <= 3 \
                        and not any(w in FORM or 'x' == w or '...' in w for w in nx):
                    cons = nx
            if not cons:
                continue
            name = ' '.join(cons)
            if 'x' in name.split('-') or '...' in name:
                continue
            ev.append((name, comm))
        for name, comm in ev:
            STR.append((name, comm, p))
            for s in re.split(r'[- ]', re.sub(r'\{[^}]*\}', '', name)):
                if s:
                    SIG.append((s, comm, p))
            if prov[p] == 'Puzriš-Dagan' and off:
                DAB.append((name, comm, p))
        truth[p] = {'prov': prov[p], 'office': off}
    return STR, SIG, DAB, truth


def linb():
    STR, SIG, truth = [], [], {}
    for l in open(os.path.join(ROOT, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        it = json.loads(l)
        h = it.get('heading', '')
        m = re.match(r'(KN|PY)\s+([A-Z][a-z]?)', h)
        if not m:
            continue
        truth[h] = {'series': m.group(1) + ' ' + m.group(2), 'site': m.group(1)}
        for ln in (it.get('content') or '').split('\n'):
            ln = re.sub(r'^\s*\.\S+\s+', '', ln)
            words, ideo = [], None
            for w in ln.split():
                w = re.sub(r"[\[\]̣,/'?]", '', w)
                if not w:
                    continue
                if '-' in w and re.fullmatch(r'[a-z0-9*\-]+', w):
                    if ideo is None:
                        words.append(w)
                elif re.match(r'^(\*\d+|[A-Z]{2,})', w) and w not in ('vac.', 'vest.'):
                    if ideo is None:
                        ideo = re.split(r'\+', w)[0]
            if ideo and words:
                STR.append((words[0], ideo, h))
                for s in words[0].split('-'):
                    SIG.append((s, ideo, h))
    return STR, SIG, truth


if __name__ == '__main__':
    S, G, H, HR = pe()
    U, US, UD, ut = ur3(sys.argv[1], sys.argv[2])
    L, LS, lt = linb()
    nets = {'PE_STR': S, 'PE_SIGN': G, 'PE_HDR': H, 'PE_HRES': HR,
            'UR3_STR': U, 'UR3_SIGN': US, 'UR3_DAB': UD, 'LB_STR': L, 'LB_SIGN': LS}
    for k, v in nets.items():
        c = Counter(x[0] for x in v)
        print('%-9s events %6d consumers %5d (deg>=2 %5d) resources %4d tablets %5d' % (
            k, len(v), len(c), sum(1 for x in c.values() if x >= 2), len({x[1] for x in v}), len({x[2] for x in v})))
    json.dump({'nets': nets, 'truth': {'UR3': ut, 'LB': lt}}, open(os.path.join(CK, 'nets.json'), 'w'))
    pc = Counter(ut[p]['prov'] for p in {x[2] for x in U})
    print('UR3 tablets by provenience', pc.most_common())
    oc = Counter(ut[p]['office'] for p in {x[2] for x in UD})
    print('UR3_DAB offices', oc.most_common(12))
    print('UR3 top commodities', Counter(x[1] for x in U).most_common(25))
    print('UR3 sample names', random.Random(1).sample(sorted({x[0] for x in U}), 20))
    print('LB top ideo', Counter(x[1] for x in L).most_common(20))
    print('PE top res', Counter(x[1] for x in S).most_common(20))
    print('PE top hdr', Counter(x[1] for x in H).most_common(10))
