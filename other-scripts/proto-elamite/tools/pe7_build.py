"""pe7 ('names as DNA'): build the name corpora with lineage / tablet / find-group labels.

PE        : distinct multi-sign entry middles (pe4 definition: entry signs minus a
            final class sign), each with its tablets, header sign, class sign,
            numeral system, publication volume (excavation campaign proxy),
            Sb museum-number block (accession batch proxy), herd-office flag.
UR3_PAT   : Ur III seal patronymics 'PN / (title) / dumu PN2' -> (son, father),
            deduplicated, with provenience and tablet (CDLI ATF dump).
OB_PAT    : the same for Old Babylonian seals (second positive).
UR3_SEAL  : all Ur III seal owner names with provenience (find-group positive).
LINB      : Linear B personnel names (DAMOS, same rules as dark_loop56_prep)
            with tablet heading, site and series.
CN_FULL   : Chinese full names with surname label (sanity positive: the inherited
            element is visible in the string).

usage: python3 pe7_build.py <cdli.atf> <cdli_cat.csv>  -> data/pe7_corpora.json
"""
import csv, json, os, re, sys, random, unicodedata, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries, header  # noqa
from pe4_common import FINAL  # noqa

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'data', 'pe7_corpora.json')
HERD = {'P008283', 'P008294', 'P008295', 'P008389', 'P008905', 'P008939', 'P008259', 'P009151'}


def pe_names(cat):
    T = load()
    meta = {t['id']: t for t in T}
    E = entries(T, require_clean=True)
    hdr = {t['id']: (header(t) or [None])[0] for t in T}
    names = collections.OrderedDict()
    for e in E:
        s = e['signs']
        if s[-1] in FINAL and len(s) >= 2:
            cls, mid = s[-1], tuple(s[:-1])
        else:
            cls, mid = '-', tuple(s)
        if len(mid) < 2:
            continue
        t = e['tablet']
        d = names.setdefault(mid, {'seq': list(mid), 'tablets': [], 'cls': [], 'sys': [], 'hdr': [],
                                   'vol': [], 'sb': [], 'herd': 0})
        if t not in d['tablets']:
            d['tablets'].append(t)
            des = meta[t]['designation']
            m = re.match(r'(.+?),', des)
            d['vol'].append(m.group(1) if m else des)
            mus = cat.get(t, '')
            m = re.match(r'Sb 0*(\d+)', mus)
            d['sb'].append(int(m.group(1)) // 100 if m else None)
            d['hdr'].append(hdr[t])
        d['cls'].append(cls)
        d['sys'].append(e['system'])
        if t in HERD or any('M362' in x for x in mid) or cls == 'M362':
            d['herd'] = 1
    return list(names.values())


def clean(w):
    return re.sub(r'[#!?*<>]', '', w)


def tok(word):
    if '[' in word or ']' in word or 'x' in re.split(r'[-.{}]', word) or '...' in word:
        return None
    w = clean(word).replace('_', '')
    w = re.sub(r'\([^)]*\)', lambda m: m.group(0).replace('.', ':').replace('-', ':'), w)
    w = re.sub(r'\{([^}]*)\}', r'-{\1}-', w)
    t = [x for x in re.split(r'[-.]', w) if x]
    return tuple(t) if t else None


STOP = {'dub-sar', 'arad2-zu', 'dumu', 'lugal', 'ensi2', 'sukkal', 'nu-banda3', 'szabra', 'sanga'}


def seals(atf, keep):
    """yield (pid, [line bodies]) for every @seal block of texts in keep."""
    cur, inseal, buf = None, False, []
    with open(atf, errors='replace') as f:
        for line in f:
            if line.startswith('&P'):
                if inseal and buf:
                    yield cur, buf
                cur, inseal, buf = line[1:8], False, []
                continue
            if cur not in keep:
                continue
            if line.startswith('@'):
                if inseal and buf:
                    yield cur, buf
                inseal = line.startswith('@seal')
                buf = []
                continue
            if inseal and re.match(r"^\d+'?\.", line):
                buf.append(line.split('.', 1)[1].strip())
        if inseal and buf:
            yield cur, buf


def patronymics(atf, period_of, prov, per):
    keep = {p for p, v in period_of.items() if v == per}
    pairs, owners = {}, {}
    for pid, lines in seals(atf, keep):
        if not lines:
            continue
        w0 = lines[0].replace('_', '').split()
        if len(w0) != 1:
            continue
        son = tok(w0[0])
        if not son:
            continue
        owners.setdefault(son, set()).add((pid, prov.get(pid, '')))
        for b in lines[1:]:
            w = b.replace('_', '').split()
            w = [x for x in w if x]
            if len(w) == 2 and clean(w[0]) == 'dumu':
                fa = tok(w[1])
                if fa and fa != son:
                    d = pairs.setdefault((son, fa), {'son': list(son), 'father': list(fa), 'tablets': set(), 'prov': set()})
                    d['tablets'].add(pid)
                    d['prov'].add(prov.get(pid, ''))
    P = [{**v, 'tablets': sorted(v['tablets']), 'prov': sorted(v['prov'])} for v in pairs.values()]
    O = [{'seq': list(k), 'tablets': sorted({a for a, _ in v}), 'prov': sorted({b for _, b in v})} for k, v in owners.items()]
    return P, O


def strip_diac(w):
    return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')


def linb():
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    BAD_OCC = None
    src = open(os.path.join(ROOT, 'tools', 'dark_loop56_prep.py')).read()
    ns = {}
    blk = src[src.index('BAD=set'):src.index('def linb_names')]
    exec('import re\n' + blk, ns)
    lb_ok, FIRST_WORD, LIST, OCC = ns['lb_ok'], ns['FIRST_WORD'], ns['LIST'], ns['OCC']
    names = collections.OrderedDict()
    for l in open(os.path.join(ROOT, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        it = json.loads(l)
        h = it.get('heading', '')
        m = re.match(r'(KN|PY|TH|MY|TI)\s+([A-Z][a-z]?)(?:\(\d+\))?\s', h)
        if not m:
            continue
        site, ser = m.group(1), m.group(2)
        if ser not in FIRST_WORD and ser not in LIST:
            continue
        for ln in it['content'].split('\n'):
            ln = strip_diac(ln)
            if not re.match(r'^\.?[0-9AaBb]', ln.strip()) and not ln.startswith(' '):
                continue
            toks = [t for t in ln.split() if not re.match(r'^\.?[0-9AaBbv]+[ab]?$', t) and t not in (',', '/', "'", '[', ']')]
            words = [w for w in (t.strip(",'/") for t in toks) if lb_ok(w) and w not in OCC and not re.fullmatch(r'[0-9]+', w)]
            if ser in FIRST_WORD:
                words = words[:1]
            for w in words:
                if '-' in w and len(w.split('-')) <= 7:
                    seq = tuple(w.split('-'))
                    d = names.setdefault(seq, {'seq': list(seq), 'tablets': [], 'site': [], 'series': []})
                    if h not in d['tablets']:
                        d['tablets'].append(h)
                        d['site'].append(site)
                        d['series'].append(site + ' ' + ser)
    return list(names.values())


def cn_full(n=6000, seed=7):
    L = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'derived', 'dark', 'loop63_corpora', 'cn_given.jsonl'))]
    L = [x for x in L if x.get('sur')]
    rng = random.Random(seed)
    rng.shuffle(L)
    return [{'seq': [x['sur']] + x['seq'], 'sur': x['sur']} for x in L[:n]]


def main(atf, catf):
    csv.field_size_limit(10 ** 9)
    period_of, prov, mus = {}, {}, {}
    with open(catf, newline='') as f:
        for row in csv.DictReader(f):
            p = row['period']
            pid = 'P%06d' % int(row['id_text']) if row['id_text'].isdigit() else row['id_text']
            if p.startswith('Ur III'):
                period_of[pid] = 'UR3'
            elif p.startswith('Old Babylonian') or p.startswith('Early Old Babylonian'):
                period_of[pid] = 'OB'
            elif p.startswith('Proto-Elamite'):
                mus[pid] = row['museum_no']
            else:
                continue
            prov[pid] = (row.get('provenience') or '').split(' (')[0]
    out = {'PE': pe_names(mus)}
    for per in ('UR3', 'OB'):
        P, O = patronymics(atf, period_of, prov, per)
        out[per + '_PAT'] = P
        out[per + '_SEAL'] = O
    out['LINB'] = linb()
    out['CN_FULL'] = cn_full()
    for k, v in out.items():
        print(k, len(v), v[0])
    json.dump(out, open(OUT, 'w'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
