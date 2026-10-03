"""S-DARK-61 prep: Ur III seal legends with NAME and TITLE fields kept apart, for the 'head = title or name-final?' test.
Source: CDLI ATF dump in the session scratchpad (cdli.atf + cdli_cat.csv, as for loops 32/48/56). An @seal section is
read line by line: line 1 = owner name (PN), lines 2+ = title / kin / patronym ('dumu PN', 'arad2 DN', 'dub-sar', ...).
Writes data/derived/dark/loop61_corpora/<name>.jsonl, one JSON per line {"seq": [...], "src": ...}, deduplicated to one
copy per distinct string:
  ur3_legend_fields      one per distinct legend: {"name_signs", "name_elems", "title", "title_kind", "lines"}
  ur3_name_signs         (a) name alone, sign tokens (split on '-'), one per distinct name
  ur3_name_elems         (a) name alone, greedy frequent-chunk elements (loop56 method)
  ur3_name_minus_final   (a') name elements with the LAST element removed (the analogue of stripping the head)
  ur3_name_title         (b) name elements + the first title word of line 2 as FINAL element (legends with a title line)
  ur3_name_proftitle     (b') same, restricted to profession titles (dub-sar, sanga, ...), 'dumu'/'arad2' lines excluded
  ur3_name_theo          (c) names whose final element is theophoric ({d}DN), elements
  ur3_title_only_count   in the log: legends whose line 1 is a bare title word and nothing else (title-only legends)
Linear B and Latin whole names come from loop56_corpora (cleaned of empty / editorial tokens by the analysis script).
Usage: python3 tools/dark_loop61_prep.py
"""
import json, re, os, csv, collections
OUT = 'data/derived/dark/loop61_corpora/'; os.makedirs(OUT, exist_ok=True)
S = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

PROF = {'dub-sar', 'sanga', 'ensi2', 'ugula', 'nu-banda3', 'gudu4', 'dam-gar3', 'sukkal', 'szabra', 'agrig', 'sipa', 'lunga',
        'simug', 'aszgab', 'nagar', 'kuruszda', 'szagina', 'ra2-gab', 'gal5-la2', 'muhaldim', 'kiszib3', 'i3-du8', 'sagi',
        'nar', 'azlag2', 'ma2-lah5', 'szu-i', 'ad-kup4', 'asz-gab', 'bahar2', 'szitim', 'engar', 'lu2', 'gal5-la2-gal',
        'nu-banda3-gu4', 'ka-guru7', 'sza13-dub-ba', 'sa12-du5', 'szusz3', 'gudu4-abzu', 'kuruszda', 'aga3-us2', 'kisal-luh',
        'lugal', 'ensi2-ka', 'nin', 'ugula-e2', 'ugula-gesz2-da', 'dam-gar3', 'szar2-ra-ab-du', 'gal-zu', 'lu2-{d}nanna',
        'gudu4-{d}inanna', 'ugula-usz-bar', 'sipa-ur-ra', 'sipa-udu', 'na-gada', 'gab2-ra', 'kurusz-da', 'ma2-gal-gal',
        'ab-ba-uru', 'sza3-tam', 'gala', 'gala-mah', 'enkud', 'szabra-e2'}
KIN = {'dumu', 'dumu-munus', 'dam', 'szesz', 'ama', 'ab-ba'}
SERV = {'arad2', 'arad2-zu', 'arad', 'arad-zu', 'ir3', 'ir11', 'ir3-zu', 'ir11-zu', 'geme2'}

def clean(line):
    t = line.split('.', 1)[1].strip()
    t = re.sub(r'[#!?\[\]<>]', '', t).replace('_', '')
    if '...' in t or re.search(r'(^|[-\s])x([-\s]|$)', t) or '$' in t or '(' in t and 'blank' in t: return None
    return t
def signs(word): return tuple(x.lower() for x in re.split(r'[-\s]+', word) if x)

def read_legends():
    csv.field_size_limit(10 ** 9)
    cat = {}
    for r in csv.DictReader(open(S + 'cdli_cat.csv', encoding='utf-8', errors='replace')):
        cat[r['id_text'].lstrip('P').lstrip('0')] = r['period']
    legs = []; pid = None; cur = None; lang = None
    for line in open(S + 'cdli.atf', encoding='utf-8', errors='replace'):
        if line.startswith('&P'):
            pid = line[2:8].lstrip('0'); cur = None; lang = None
        elif line.startswith('#atf: lang'):
            lang = line.split()[-1]
        elif line.startswith('@seal'):
            cur = {'pid': pid, 'lines': [], 'lang': lang}; legs.append(cur)
        elif line.startswith('@'):
            cur = None
        elif cur is not None and re.match(r"^\d+'?\.\s", line):
            cur['lines'].append(line)
    out = []
    for lg in legs:
        if not cat.get(lg['pid'], '').startswith('Ur III'): continue
        if lg['lang'] not in (None, 'sux'): continue
        ls = [clean(l) for l in lg['lines']]
        if not ls or any(l is None for l in ls): continue
        out.append(ls)
    return out

def elements_lex(names):
    """greedy frequent-chunk lexicon as in loop56 (chunk of <= 3 signs is an element if it recurs >= 5x and is a whole name,
    a single sign or a theophoric {d} chunk)."""
    whole = collections.Counter(tuple(n) for n in names)
    chunks = collections.Counter()
    for n in names:
        for i in range(len(n)):
            for j in range(i + 1, min(len(n), i + 3) + 1): chunks[n[i:j]] += 1
    return {c for c, v in chunks.items() if v >= 5 and (len(c) == 1 or c in whole or c[0].startswith('{d}'))}
def to_elems(n, lex):
    res = []; i = 0
    while i < len(n):
        best = None
        for j in range(min(len(n), i + 3), i, -1):
            if n[i:j] in lex and not (i == 0 and j == len(n) and len(n) > 1): best = j; break   # never one element = the whole name
        if best is None: best = i + 1
        res.append('-'.join(n[i:best])); i = best
    return tuple(res)

def dump(name, seqs, src):
    seqs = [tuple(s) for s in seqs if s]
    dist = sorted(set(seqs))
    with open(OUT + name + '.jsonl', 'w') as f:
        for s in dist: f.write(json.dumps({'seq': list(s), 'src': src}) + '\n')
    el = collections.Counter(a for s in dist for a in s)
    P(f'{name}: raw {len(seqs)} -> distinct {len(dist)}; element types {len(el)}; mean len {sum(map(len, dist)) / max(1, len(dist)):.2f}; {src}')
    return dist

legs = read_legends()
P(f'Ur III @seal legends read (complete, Sumerian or unmarked): {len(legs)}')
title_only = 0; with_title = 0; kinds = collections.Counter(); recs = []
names_for_lex = []
for ls in legs:
    first = ls[0].split()[0].lower() if ls[0].split() else ''
    if len(ls) == 1 and (first in PROF or first in KIN or first in SERV):
        title_only += 1; continue
    if first in PROF or first in KIN or first in SERV or first in ('dumu',): continue   # legend compressed / no PN
    toks = signs(ls[0])
    if not 1 <= len(toks) <= 8: continue
    if any(t in ('dumu', 'arad2', 'arad', 'ir11', 'ir3') for t in toks): continue
    title = None; kind = None
    if len(ls) >= 2:
        w = ls[1].split()[0].lower() if ls[1].split() else ''
        full2 = ls[1].strip().lower()
        if full2 in PROF or w in PROF: title = full2 if full2 in PROF else w; kind = 'prof'
        elif w in KIN: title = w; kind = 'kin'
        elif w in SERV or full2 in SERV: title = w; kind = 'serv'
        else: title = w; kind = 'other'
        with_title += 1; kinds[kind] += 1
    names_for_lex.append(toks)
    recs.append({'name_signs': list(toks), 'title': title, 'title_kind': kind, 'lines': ls})
P(f'  legends with a PN on line 1: {len(recs)}; with a line 2: {with_title}; line-2 kinds {dict(kinds)}; TITLE-ONLY legends (bare title, no PN): {title_only}')
lex = elements_lex(names_for_lex)
for r in recs: r['name_elems'] = list(to_elems(tuple(r['name_signs']), lex))
# distinct legends (name + title string)
seen = set(); dist = []
for r in recs:
    k = (tuple(r['name_signs']), r['title'])
    if k in seen: continue
    seen.add(k); dist.append(r)
with open(OUT + 'ur3_legend_fields.jsonl', 'w') as f:
    for r in dist: f.write(json.dumps(r) + '\n')
P(f'ur3_legend_fields: {len(dist)} distinct (name, title) legends')
tt = collections.Counter(r['title'] for r in dist if r['title'])
P('  commonest line-2 words:', tt.most_common(25))
dump('ur3_name_signs', [r['name_signs'] for r in recs if len(r['name_signs']) >= 2], 'Ur III PN (line 1), sign tokens, one per distinct name')
NE = dump('ur3_name_elems', [r['name_elems'] for r in recs if len(r['name_elems']) >= 2], 'Ur III PN, greedy-chunk elements, one per distinct name')
dump('ur3_name_minus_final', [r['name_elems'][:-1] for r in recs if len(r['name_elems']) >= 3], 'Ur III PN elements with the final element removed (>= 2 left), one per distinct string')
dump('ur3_name_title', [r['name_elems'] + ['T:' + r['title']] for r in dist if r['title']], 'Ur III PN elements + first word of line 2 (title / kin / servant) as final element, one per distinct legend')
dump('ur3_name_proftitle', [r['name_elems'] + ['T:' + r['title']] for r in dist if r['title_kind'] == 'prof'], 'Ur III PN elements + profession title as final element, one per distinct legend')
dump('ur3_name_theo', [r['name_elems'] for r in recs if len(r['name_elems']) >= 2 and r['name_elems'][-1].startswith('{d}')], 'Ur III PN elements whose final element is theophoric {d}DN, one per distinct name')
fin = collections.Counter(n[-1] for n in NE)
P('  commonest name-final elements:', fin.most_common(20))
P(f'  share of distinct names with a theophoric final: {sum(1 for n in NE if n[-1].startswith("{d}")) / len(NE):.3f}')
open(OUT + 'SOURCES.txt', 'w').write('\n'.join(LOG) + '\n')
