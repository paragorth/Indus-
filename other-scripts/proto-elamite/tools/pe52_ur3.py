"""pe52 control corpus PERS: Ur III personnel / ration lines with personal names and quantities.

From CDLI Ur III administrative tablets (all proveniences), keep obverse/reverse lines of the form
   <numeral group> [unit] <words...>
where at least one word is a personal name (PN). PN list = names found by fixed syntactic frames
(pe50_ur3: ki X-ta, giri3 X, kiszib3 X, X i3-dab5, ... ) plus seal-owner names (loop56 corpora) plus
pe4 'names'. No reading of signs beyond these frames. Each word is split on '-' into opaque tokens;
every token keeps a label NAME (from a PN word) or OTHER (commodity, title, grade ...). Labels are
used only to SCORE the control afterwards, never by the fitting engine.
Quantity: capacity in sila3 (sila3 / ban2 = 10 / barig = 60 / gur = 300) -> system 'C';
plain counts (disz / u / gesz2 / gesz'u) -> system 'K'.
Output: data/pe52_ckpt/pers.json  [{tab, w:[tokens], lab:[0/1], q, sys, prov}]
"""
import csv, json, os, re, collections
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'pe52_ckpt', 'pers.json')
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STOP = {'lugal', 'ensi2', 'dub-sar', 'sukkal', 'nu-banda3', 'szabra', 'sanga', 'ugula', 'dumu', 'lu2', 'ba-zi',
        'ba-usz2', 'i3-dab5', 'szu', 'ba-ti', 'maszkim', 'giri3', 'kiszib3', 'e2', 'iti', 'mu', 'u4', 'sza3',
        'zi-ga', 'gurusz', 'geme2', 'erin2', 'udu', 'gu4', 'sila3', 'gur', 'kasz', 'ninda', 'sze', 'i3', 'zi3',
        'dam', 'lu2-kin-gi4-a', 'ra2-gaba', 'sukkal-mah', 'dumu-lugal', 'arad2', 'ARAD2', 'nin', 'dingir', 'a'}
UNIT = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600}
CAP = {'ban2': 10, 'barig': 60, 'asz': 300}


def clean(w):
    return re.sub(r'[#!?*<>\[\]]', '', w)


def parse_num(toks):
    """leading numeral tokens -> (value, sys, rest) or None"""
    i = 0; v = 0.0; cap = False; cnt = []
    while i < len(toks):
        m = re.match(r"^(\d+)(?:/(\d+))?\(([a-z0-9']+)\)$", toks[i])
        if not m:
            break
        n = float(m.group(1)) / (float(m.group(2)) if m.group(2) else 1.0)
        u = m.group(3)
        if u in CAP:
            cap = True; v += n * CAP[u]
        elif u in UNIT:
            cnt.append(n * UNIT[u])
        else:
            return None
        i += 1
    if i == 0:
        return None
    rest = toks[i:]
    if rest and rest[0] == 'sila3':
        v += sum(cnt); cnt = []; cap = True; rest = rest[1:]
    elif rest and rest[0] == 'gur':
        rest = rest[1:]; cap = True
        if cnt:
            return None
    if cnt and cap:
        return None
    if not cap:
        v = sum(cnt)
    if v <= 0 or not rest:
        return None
    return v, 'C' if cap else 'K', rest


HEADS = {'udu', 'sila4', 'gu4', 'masz2', 'masz2-gal', 'u8', 'ab2', 'ud5', 'amar', 'kir11', 'gukkal', 'masz-da3',
         'udu-nita2', 'dara4', 'munus', 'anszu', 'dusu2', 'szeg9-bar', 'az', 'ugu4-bi2'}


def toks_of(rest, names):
    w2, lab = [], []
    for w, isn in zip(rest, names):
        for tk in re.split(r'[-.]|(?=\{)|(?<=\})', w):
            tk = tk.strip()
            if tk:
                w2.append(tk); lab.append(int(isn))
    return w2, lab


def main():
    herd = []
    per = {}
    for r in csv.DictReader(open(f'{SCR}/cdli_cat.csv', newline='')):
        if r['period'].startswith('Ur III') and r['genre'] == 'Administrative':
            try:
                per['P%06d' % int(r['id_text'])] = r['provenience'].split()[0] if r['provenience'] else '-'
            except ValueError:
                pass
    pn = set()
    caps = json.load(open(os.path.join(DATA, 'pe50_ckpt', 'ur3_caps.json')))
    for a in caps.values():
        for t in a:
            pn.update(t['names'])
    for r in json.load(open(os.path.join(DATA, 'pe4_controls.json')))['names']:
        pn.add('-'.join(r['words']))
    for l in open(os.path.join(REPO, 'data/derived/dark/loop56_corpora/ur3_names_dedup.jsonl')):
        s = json.loads(l)['seq']
        if all(re.match(r'^[a-z0-9{}]+$', x) for x in s):
            pn.add('-'.join(s))
    pn -= STOP
    print('PN list', len(pn))
    out = []; cur = None; inseal = False
    for raw in open(f'{SCR}/cdli.atf', encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        if line.startswith('&P'):
            cur = line[1:8] if line[1:8] in per else None; inseal = False
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            tag = line[1:].strip().split()[0] if line[1:].strip() else ''
            if tag == 'seal':
                inseal = True
            elif tag in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'tablet'):
                inseal = False
            continue
        if inseal:
            continue
        m = re.match(r"^\d+'?\.\s*(.*)$", line)
        if not m:
            continue
        txt = m.group(1)
        if re.search(r'\.\.\.|\bx\b|la2|\$|szu-nigin|\[', txt):
            continue
        toks = [clean(w) for w in txt.split()]
        p = parse_num(toks)
        if not p:
            continue
        v, sy, rest = p
        if any(re.match(r'^\d', w) for w in rest):
            continue
        names = [w in pn or w.startswith('{d}') for w in rest]
        if per[cur].startswith('Puzri') and rest[0] in HEADS and len(rest) <= 8:
            w2, lab = toks_of(rest, names)
            herd.append(dict(tab=cur, w=w2, lab=lab, q=v, sys=sy, prov=per[cur], words=rest))
        if not any(names) or len(rest) > 6:
            continue
        w2, lab = toks_of(rest, names)
        out.append(dict(tab=cur, w=w2, lab=lab, q=v, sys=sy, prov=per[cur], words=rest))
    c = collections.Counter(o['prov'] for o in out)
    print('lines', len(out), 'tablets', len({o['tab'] for o in out}), c.most_common(6))
    print(collections.Counter(o['sys'] for o in out))
    oth = collections.Counter(w for o in out for w, n in zip(o['words'], [x in pn for x in o['words']]) if not n)
    print('top non-name words', oth.most_common(30))
    json.dump(out, open(OUT, 'w'))
    print('herd lines', len(herd), 'tablets', len({o['tab'] for o in herd}), 'with name', sum(any(o['lab']) for o in herd))
    print('herd words', collections.Counter(w for o in herd for w in o['words']).most_common(40))
    json.dump(herd, open(OUT.replace('pers.json', 'herd.json'), 'w'))


if __name__ == '__main__':
    main()
