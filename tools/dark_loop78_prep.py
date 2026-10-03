"""S-DARK-78 prep: one-town / one-institution registers of logographically written personal names from the
China Biographical Database (CBDB SQLite release cbdb_20261003, Hugging Face cbdb/cbdb-sqlite; sha256 of the .sqlite3
checked against latest.json). The 587 MB database is read from a scratch path and NOT kept; only the persons used here are
written to data/derived/dark/loop78_corpora/cbdb_registers.jsonl (one row per person):
  pid, name (full name, CJK only, 2-4 chars), sur, given, addr (index address id), dy (dynasty), entry (first entry code
  class), office (first office title, Chinese), kin (kin person ids that are also in the file),
  reg (list of register keys: 'county:<addr_id>', 'jinshi:<year>').
Registers kept: 16 counties spread over size (by distinct full names), and the 8 largest jinshi cohorts (one exam year).
Usage: python3 tools/dark_loop78_prep.py /path/to/cbdb_20261003.sqlite3
"""
import sys, sqlite3, json, collections, re
DB = sys.argv[1]
c = sqlite3.connect(DB)
CJK = re.compile(r'^[㐀-鿿豈-﫿\U00020000-\U0003134f]+$')
ad = {r[0]: r[1] for r in c.execute('select c_addr_id, c_name_chn from ADDR_CODES')}
dyn = {r[0]: r[1] for r in c.execute('select c_dy, c_dynasty_chn from DYNASTIES')}
P = {}
for pid, nm, sur, giv, addr, dy in c.execute('select c_personid, c_name_chn, c_surname_chn, c_mingzi_chn, c_index_addr_id, c_dy from BIOG_MAIN'):
    if not nm or not CJK.match(nm) or not 2 <= len(nm) <= 4: continue
    P[pid] = dict(pid=pid, name=nm, sur=sur or '', given=giv or '', addr=addr, addr_name=ad.get(addr), dy=dyn.get(dy), reg=[])
# first entry (by sequence) with its code description; jinshi cohorts by year
ec = {r[0]: r[1] for r in c.execute('select c_entry_code, c_entry_desc_chn from ENTRY_CODES')}
first = {}; jin = collections.defaultdict(set)
for pid, code, seq, yr in c.execute('select c_personid, c_entry_code, c_sequence, c_year from ENTRY_DATA order by c_personid, c_sequence'):
    if pid not in P: continue
    if pid not in first: first[pid] = ec.get(code, str(code))
    if code == 36 and yr and yr > 0: jin[yr].add(pid)
oc = {r[0]: r[1] for r in c.execute('select c_office_id, c_office_chn from OFFICE_CODES')}
off = {}
for pid, oid, seq, fy in c.execute('select c_personid, c_office_id, c_sequence, c_firstyear from POSTED_TO_OFFICE_DATA order by c_personid, c_sequence'):
    if pid in P and pid not in off and oid not in (None, 0, -1): off[pid] = oc.get(oid, str(oid))
# county registers: distinct full names per index address (Xian-level codes only)
xian = {r[0] for r in c.execute("select c_addr_id from ADDR_CODES where c_admin_type in ('Xian', '縣')")}
byA = collections.defaultdict(set)
for p in P.values():
    if p['addr'] in xian: byA[p['addr']].add(p['name'])
sizes = sorted(((len(v), a) for a, v in byA.items()), reverse=True)
targets = [4000, 2600, 2000, 1700, 1500, 1300, 1100, 950, 800, 650, 520, 420, 330, 260, 200, 150]
chosen = []
for t in targets:
    best = min((x for x in sizes if x[1] not in chosen), key=lambda x: abs(x[0] - t)); chosen.append(best[1])
for a in chosen:
    for p in P.values():
        if p['addr'] == a: p['reg'].append(f'county:{a}')
coh = sorted(jin, key=lambda y: -len(jin[y]))[:8]
for y in coh:
    for pid in jin[y]: P[pid]['reg'].append(f'jinshi:{y}')
keep = {pid: p for pid, p in P.items() if p['reg']}
kin = collections.defaultdict(set)
for a, b in c.execute('select c_personid, c_kin_id from KIN_DATA'):
    if a in keep and b in keep and a != b: kin[a].add(b); kin[b].add(a)
out = open('data/derived/dark/loop78_corpora/cbdb_registers.jsonl', 'w', encoding='utf-8')
for pid, p in sorted(keep.items()):
    p['entry'] = first.get(pid); p['office'] = off.get(pid); p['kin'] = sorted(kin[pid])
    out.write(json.dumps(p, ensure_ascii=False) + '\n')
out.close()
print('persons kept', len(keep))
for a in chosen: print('county', a, ad.get(a), len(byA[a]), 'distinct full names')
for y in coh: print('jinshi', y, len(jin[y]))
