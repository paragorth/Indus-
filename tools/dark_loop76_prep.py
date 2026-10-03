"""S-DARK-76 prep: comparator corpora with DOCUMENT ids, for the per-object-kind recurrence profile.
Source: CDLI ATF dump + catalogue in the session scratchpad (same files as tools/dark_loop48_prep.py; CC BY).
Writes data/derived/dark/loop76_corpora/
  ur3_seal_impr.jsonl  one row per Ur III @seal impression on a tablet: pid (tablet = archival act), site,
                       legend (words), owner (first word, the seal owner's name unless it is a title / royal formula)
  ur3_admin_names.jsonl  one row per personal-name slot in Ur III administrative tablet bodies (lang sux,
                       period Ur III, @seal excluded): pid, site, slot, name. Slots (closed formula positions,
                       no lemmatisation used): 'giri3 PN', 'kiszib3 PN', 'ki PN-ta' (from), 'PN i3-dab5' (received),
                       'PN szu ba-ti' (received), 'mu-kux(DU) PN' (delivery by). A name = one word, determinatives kept.
"""
import re, csv, json, collections, os, sys
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
OUT = '/home/user/Indus-/data/derived/dark/loop76_corpora/'
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 9)

def cdli_prov():
    r = csv.reader(open(SP + 'cdli_cat.csv', errors='ignore')); hdr = next(r)
    ip = hdr.index('provenience'); ii = hdr.index('id_text'); iper = hdr.index('period'); ig = hdr.index('genre')
    prov = {}; per = {}; gen = {}
    for row in r:
        if len(row) > max(ip, ii, iper, ig):
            p = row[ip].split(' (')[0].strip()
            if p.endswith('?') or not p: p = 'uncertain'
            prov[row[ii]] = p; per[row[ii]] = row[iper]; gen[row[ii]] = row[ig]
    return prov, per, gen

def clean(w): return re.sub(r'[\[\]#?!<>*]', '', w).lower()
TITLE = {'dumu', 'dub-sar', 'lugal', 'arad2', 'arad2-zu', 'lu2', 'ensi2', 'ugula', 'nu-banda3', 'dam-gar3', 'sanga', 'sukkal',
         'szabra', 'agrig', 'sipa', 'kiszib3', 'dam', 'szesz', 'nin', 'ama', 'mu', 'nita', 'kal-ga', 'lugal-kal-ga', 'nita-kal-ga'}
BAD = ('x', '...', '$', '(', ')')
def okname(w):
    return w and not any(b in w for b in BAD) and w not in TITLE and not re.fullmatch(r'[\d/]+.*', w) and len(w) >= 2

def main():
    prov, per, gen = cdli_prov()
    ur3 = lambda pid: 'Ur III' in per.get(pid, '')
    seal = []; admin = []
    cur = None; lang = None; inseal = None; lines = []
    def flush():
        if cur is None or lang != 'sux' or not ur3(cur): return
        site = prov.get(cur, 'uncertain')
        for i, ws in enumerate(lines):
            if not ws: continue
            # giri3 PN / kiszib3 PN / mu-kux(du) PN
            if ws[0] in ('giri3', 'kiszib3', 'mu-kux(du)') and len(ws) >= 2 and okname(ws[1]):
                admin.append(dict(pid=cur, site=site, slot=ws[0], name=ws[1]))
            if ws[0] == 'ki' and len(ws) >= 2 and ws[1].endswith('-ta'):
                n = ws[1][:-3]
                if okname(n): admin.append(dict(pid=cur, site=site, slot='ki-ta', name=n))
            if len(ws) == 2 and ws[1] == 'i3-dab5' and okname(ws[0]):
                admin.append(dict(pid=cur, site=site, slot='i3-dab5', name=ws[0]))
            if len(ws) == 3 and ws[1:] == ['szu', 'ba-ti'] and okname(ws[0]):
                admin.append(dict(pid=cur, site=site, slot='szu-ba-ti', name=ws[0]))
            if ws in (['szu', 'ba-ti'],) and i >= 1 and len(lines[i - 1]) == 1 and okname(lines[i - 1][0]):
                admin.append(dict(pid=cur, site=site, slot='szu-ba-ti', name=lines[i - 1][0]))
    for l in open(SP + 'cdli.atf', errors='ignore'):
        if l.startswith('&'):
            flush(); cur = l[1:8].lstrip('P').lstrip('0'); lang = None; inseal = None; lines = []; continue
        if l.startswith('#atf: lang'): lang = l.split()[-1]; continue
        if l.startswith('@seal'):
            inseal = []; seal.append((cur, lang, inseal)); continue
        if l[:1] in '@$#>':
            if l.startswith('@'): inseal = None
            continue
        if not re.match(r"^\d+'?\.", l): continue
        parts = l.split(None, 1)
        if len(parts) < 2: continue
        toks = [clean(t) for t in parts[1].split()]
        if inseal is not None:
            inseal.extend(t for t in toks if t and t not in ('x', '(x)', '...', '($', 'blank)') and '...' not in t)
        else:
            lines.append([t for t in toks if t])
    flush()
    n = 0
    with open(OUT + 'ur3_seal_impr.jsonl', 'w') as f:
        for pid, lang, toks in seal:
            if lang != 'sux' or not ur3(pid) or not 2 <= len(toks) <= 20: continue
            f.write(json.dumps(dict(pid=pid, site=prov.get(pid, 'uncertain'), legend=toks, owner=toks[0])) + '\n'); n += 1
    with open(OUT + 'ur3_admin_names.jsonl', 'w') as f:
        for r in admin: f.write(json.dumps(r) + '\n')
    print('seal impressions', n, 'admin name slots', len(admin), collections.Counter(r['slot'] for r in admin),
          'tablets', len(set(r['pid'] for r in admin)), collections.Counter(r['site'] for r in admin).most_common(6))
    open(OUT + 'SOURCES.txt', 'w').write(__doc__ + f'\nCounts: seal impressions {n}; admin name slots {len(admin)} on '
                                         f'{len(set(r["pid"] for r in admin))} tablets; slots {dict(collections.Counter(r["slot"] for r in admin))}\n')
main()
