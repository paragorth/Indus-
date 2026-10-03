"""Loop 4 control data: Ur III seal legends from the CDLI ATF dump with WORD boundaries kept.
Output: data/derived/dark/loop4_ur3_legends.json  = list of legends; legend = list of words; word = list of signs.
Signs = hyphen-separated graphemes; words = space-separated tokens; determinatives in {} dropped."""
import csv, re, json, sys, collections
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
csv.field_size_limit(10**9)
per = {}
for r in csv.DictReader(open(SP + 'cdli_cat.csv', errors='ignore')):
    if 'Ur III' in (r.get('period') or ''):
        per[r['id_text'].zfill(6)] = r.get('object_type', '')
def clean(t): return re.sub(r'[#?!\[\]<>*]|~\w+', '', t)
NUM = re.compile(r'^(\d+|n)\((N|n|disz|u|asz|gesz|barig|ban|szar)', re.I)
out = []; pid = None; in_seal = False; cur = []
def flush():
    global cur
    if pid in per and cur:
        out.append(cur)
    cur = []
for line in open(SP + 'cdli.atf', errors='ignore'):
    if line.startswith('&P'):
        flush(); pid = line[2:8]; in_seal = False; continue
    if pid not in per: continue
    if line.startswith('@seal'): flush(); in_seal = True; continue
    if line.startswith('@'):
        if not line.startswith(('@column', '@surface', '@face')): flush() if not in_seal else None
        if line.startswith(('@obverse', '@reverse', '@envelope', '@tablet', '@left', '@right', '@top', '@bottom')): in_seal = False
        continue
    seal_obj = per[pid] == 'seal'
    if not (in_seal or seal_obj): continue
    m = re.match(r'^\d+\'?\.\s+(.*)', line)
    if not m: continue
    body = re.sub(r'\{[^}]*\}', '', m.group(1))
    words = []
    for w in body.split():
        w = clean(w)
        if not w or NUM.match(w) or re.match(r'^(x|\.\.\.|\d.*|\(.*)$', w): continue
        signs = [s for s in re.split(r'[-.]', w.lower()) if s and s not in ('x', '...')]
        if signs and all(re.match(r'^[a-z0-9]+$', s) for s in signs): words.append(signs)
    if words: cur.extend(words)
flush()
out = [l for l in out if 2 <= sum(len(w) for w in l) <= 14]
json.dump(out, open('data/derived/dark/loop4_ur3_legends.json', 'w'))
c = collections.Counter(tuple(tuple(w) for w in l) for l in out)
print(len(out), 'legends;', len(c), 'distinct; mean signs', sum(sum(len(w) for w in l) for l in out) / len(out))
print(c.most_common(3))
