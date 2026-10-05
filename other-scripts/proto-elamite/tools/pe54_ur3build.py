"""pe54: build Ur III herd 'sightings' (Umma, Girsu) with herdsman and date.

A sighting = one tablet's summed counts in the sheep/goat classes (pe20 term parser),
with the herdsman string (name next to 'sipa', or 'ki PN-ta' / 'kiszib3 PN' as fallback)
and the regnal date from the CDLI catalogue. Output: data/pe54_ckpt/ur3_sightings.json.
"""
import csv, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pe20_common import ur_term, UR_SIGNS
from pe5_common import parse_ur_line
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
CK = os.path.join(HERE, '..', 'data', 'pe54_ckpt')
KINGS = {'Ur-Namma': 0, 'Šulgi': 18, 'Amar-Suen': 66, 'Šū-Suen': 75, 'Ibbi-Suen': 84}

csv.field_size_limit(10 ** 9)
meta = {}
prov_only = {}
for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
    if row['period'].startswith('Ur III') and ('Umma' in row['provenience'] or 'Girsu' in row['provenience']):
        m = re.match(r'^([^.]+)\.(\d\d)', row['dates_referenced'])
        if m and m.group(1) in KINGS:
            meta['P%06d' % int(row['id_text'])] = (KINGS[m.group(1)] + int(m.group(2)),
                                                   'Umma' if 'Umma' in row['provenience'] else 'Girsu')
        else:
            prov_only['P%06d' % int(row['id_text'])] = 'Umma' if 'Umma' in row['provenience'] else 'Girsu'
print('dated', len(meta))
txt = open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace').read()
out = []
STOP = {'ki', 'kiszib3', 'giri3', 'sipa', 'szu', 'ba-ti', 'mu', 'iti', 'u4', 'sza3-bi-ta', 'zi-ga', 'mu-kux(DU)',
        'dumu', 'ugula', 'nu-banda3', 'udu', 'masz2', 'hi-a', 'gub-ba', 'ba-zi', 'i3-dab5', 'szunigin', 'kab2-ku5',
        'e2', 'dub', 'nig2-ka9-ak', 'su-ga', 'la2-ia3', 'si-i3-tum', 'lugal', 'ensi2', 'szabra', 'sag-nig2-gur11-ra-kam'}


def name_of(w):
    if 'sipa' in w:
        i = w.index('sipa')
        if i + 1 < len(w) and w[i + 1] not in STOP:
            return w[i + 1]
        if i > 0 and w[i - 1] not in STOP:
            return w[i - 1]
    for x in w:
        x2 = re.sub(r'-(ta|ra|sze3|ke4|e)$', '', x)
        if x2 not in STOP and not x2.startswith('iti') and len(x2) > 2 and 'x' not in x2.split('-'):
            return x2
    return None


STOPW = {'gub-ba-am3', 'gub-ba', 'ri-ri-ga', 'ri-ri-ga-am3', 'zi-ga', 'zi-ga-am3', 'la2-ia3', 'la2-ia3-am3',
         'sza3-bi-ta', 'mu-kux(DU)', 'libir-am3', 'su-ga', 'siki-bi', 'siki', 'szunigin', 'sza3', 'mu', 'iti', 'ki',
         'kiszib3', 'giri3', 'sipa', 'udu', 'ugula', 'blank', 'diri', 'dumu', 'e2', 'u4', 'ba-zi', 'mu-DU',
         'nig2-ka9-ak', 'kab2-ku5', 'szu', 'masz2', 'ud5', 'u8', 'lugal', 'ensi2'}


def ok_name(x):
    return (x and x not in STOPW and '[' not in x and ']' not in x and ur_term(x) is None
            and x not in ('i3-dab5', 'kuruszda', 'sila4-nita2', 'udu-nita2', 'hi-a') and not re.match(r'^\d', x) and 'x' not in x.split('-') and len(x) > 2
            and not x.endswith('{ki}') and not x.startswith('{gesz}'))


def tablet_owner(seq):
    words = [t.split() for t in seq if not re.match(r'^\d', t)]
    for w in words:
        if 'nig2-ka9-ak' in w:
            i = w.index('nig2-ka9-ak')
            if i + 1 < len(w) and ok_name(w[i + 1]):
                return w[i + 1]
    for w in words:
        if 'sipa' in w:
            i = w.index('sipa')
            if i > 0 and ok_name(w[i - 1]):
                return w[i - 1]
    # standalone name line (one token) after the accounts, before the date
    for w in words[::-1]:
        if len(w) == 1 and ok_name(w[0]):
            return w[0]
    for w in words:
        if 'i3-dab5' in w and w.index('i3-dab5') > 0 and ok_name(w[w.index('i3-dab5') - 1]):
            return w[w.index('i3-dab5') - 1]
    return None


ALL = []
for d in re.split(r'\n(?=&P)', txt):
    pid = d[1:8]
    if pid not in meta and pid not in prov_only:
        continue
    if not re.search(r'\bu8\b|\bud5\b', d):
        continue
    seq = []
    surf = 'o'
    for raw in d.split('\n'):
        if raw.startswith('@seal') or raw.startswith('@envelope'):
            surf = 's'
        if surf == 's':
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.strip())
        if m:
            seq.append(re.sub(r'[#?!<>]', '', m.group(1)))
    v = {}
    for t in seq:
        if not re.match(r'^\d', t):
            if v:
                break
            continue
        if '[' in t:
            v = None
            break
        p = parse_ur_line(t)
        if not p or p[1]:
            continue
        if re.search(r'\b(niga|ba-usz2|ba-ug7|ri-ri-ga|szunigin|bar-su-ga|bar-gal2)\b', p[2]):
            continue
        term = ur_term(p[2])
        if term:
            v[term] = v.get(term, 0) + float(p[0])
    if not v or len(v) < 3 or not ('u8' in v or 'ud5' in v):
        continue
    h = tablet_owner(seq)
    if not h:
        continue
    yr, pv = meta.get(pid, (None, prov_only.get(pid)))
    out.append(dict(id=pid, herd=h, year=yr, prov=pv, v=[v.get(s, 0.0) for s in UR_SIGNS]))
json.dump(out, open(os.path.join(CK, 'ur3_sightings.json'), 'w'))
from collections import Counter
c = Counter(o['herd'] for o in out)
yrs = {}
for o in out:
    yrs.setdefault(o['herd'], set()).add(o['year'])
multi = [h for h in yrs if len(yrs[h] - {None}) >= 2]
print('sightings', len(out), 'herds', len(c), 'herds in >=2 years', len(multi),
      'sightings of those', sum(c[h] for h in multi))
print(Counter(o['prov'] for o in out))
print(c.most_common(25))
