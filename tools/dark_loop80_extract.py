"""Loop 80 cycle 1: every CDLI ATF line mentioning Meluhha (me-luh-ha and variants),
joined to the CDLI catalogue; writes a raw hit dump with +-4 lines of context.
Usage: python3 tools/dark_loop80_extract.py ATF CATALOG OUT.json"""
import re, sys, csv, json
csv.field_size_limit(10**9)
atf, cat, out = sys.argv[1:4]
PAT = re.compile(r'm[ei]-lu[h]?-(?:uh-)?ha')
def clean(l): return re.sub(r'[\[\]#?!<>⸢⸣]', '', l)
texts = {}; cur = None
for l in open(atf, errors='replace'):
    if l.startswith('&P'):
        cur = l[1:8]; texts[cur] = []
    elif cur:
        texts[cur].append(l.rstrip('\n'))
hits = {}
for p, ls in texts.items():
    body = [(i, x) for i, x in enumerate(ls) if re.match(r"^\s*\d+'?\.", x) or x.startswith('@') or x.startswith('$')]
    for k, (i, x) in enumerate(body):
        if re.match(r"^\s*\d+'?\.", x) and PAT.search(clean(x)):
            ctx = [y for _, y in body[max(0, k-6):k+5]]
            hits.setdefault(p, []).append({'line': x.strip(), 'ctx': ctx})
meta = {}
want = set(hits)
for r in csv.DictReader(open(cat, errors='replace')):
    pid = 'P%06d' % int(r['id']) if r.get('id', '').isdigit() else None
    if pid in want:
        meta[pid] = {k: r.get(k, '') for k in ('designation', 'period', 'provenience', 'genre', 'dates_referenced', 'primary_publication', 'language', 'object_type')}
res = [{'p': p, **meta.get(p, {}), 'hits': h, 'n_lines': len([x for x in texts[p] if re.match(r"^\s*\d+'?\.", x)])} for p, h in sorted(hits.items())]
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
from collections import Counter
print(len(res), 'texts', sum(len(r['hits']) for r in res), 'lines')
for k, v in Counter(r.get('period', '?') for r in res).most_common(): print(v, k)
