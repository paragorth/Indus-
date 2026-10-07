"""la82: harvest Chronique des fouilles en ligne (EFA/BSA) via OAI-PMH, keep Aegean/Cretan reports dated >= 2014,
fetch each report page and keep those that mention Linear A. Single sequential worker, polite delay.
usage: python3 la82_chronique.py OUTDIR"""
import sys, os, re, json, time, html, urllib.request, urllib.parse
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BASE = 'https://chronique.efa.gr/oai_pmh.php'
def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'la82-research'}), timeout=90) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            time.sleep(5 * (k + 1)); err = e
    print('FAIL', url, err, flush=True); return ''
recs_f = os.path.join(OUT, 'records.jsonl')
if not os.path.exists(recs_f + '.done'):
    url = BASE + '?verb=ListRecords&metadataPrefix=oai_dc&from=2014-01-01'; n = 0
    with open(recs_f, 'w') as f:
        while url:
            t = get(url)
            for rec in re.findall(r'<record>(.*?)</record>', t, re.S):
                g = lambda k: [html.unescape(x.strip()) for x in re.findall(rf'<dc:{k}>(.*?)</dc:{k}>', rec, re.S)]
                f.write(json.dumps(dict(title=(g('title') or [''])[0], date=(g('date') or [''])[0], desc=' '.join(g('description')),
                                        id=(g('identifier') or [''])[0], subj=g('subject'), cov=g('coverage'))) + '\n'); n += 1
            m = re.search(r'<resumptionToken[^>]*>([^<]+)</resumptionToken>', t)
            url = BASE + '?verb=ListRecords&resumptionToken=' + urllib.parse.quote(m.group(1).strip()) if m else None
            print('records', n, flush=True); time.sleep(1)
    open(recs_f + '.done', 'w').write('ok')
recs = [json.loads(l) for l in open(recs_f)]
LA = re.compile(r'lin[ée]aire\s*A\b|linear\s*A\b|Γραμμικ[ήηής]+\s*Α\b|Γραμμική\s*Α', re.I)
direct = [r for r in recs if LA.search(r['title'] + ' ' + r['desc'])]
print('records', len(recs), 'direct LA hits in snippets', len(direct), flush=True)
CRETE = re.compile(r'CR[EÈ]TE|CRETE|ΚΡΗΤ|HÉRAKLION|HERAKLION|LASITHI|LASSITHI|RÉTHYMN|RETHYMN|CHANIA|LA CANÉE|SANTORIN|THÉRA|THERA|KYTH|KÉA|KEA|MÉLOS|MILOS|NAXOS|RHODES|SAMOTHR|KNOSS|ZAKRO|PHAIST|MALIA|PALAIKASTRO|PETRAS|SISSI|ARCHANES|ARKHANES|AKROTIRI|GOURNIA|ZOMINTHOS|KOMMOS|TYLISS|HAGIA TRIADA|AGHIA TRIADA|MYRTOS|PSEIRA|MOCHLOS|KASTELLI|PRINIAS|ANETAKI|SITEIA|SITIA|IERAPETRA|HÉRAKLEION|LASSITHI|AGIOS NIKOLAOS', re.I)
def yr(r):
    m = re.search(r'(19|20)\d\d', r['date'] or r['title']); return int(m.group()) if m else 0
cand = [r for r in recs if yr(r) >= 2014 and (CRETE.search(r['title'] + ' ' + r['desc'] + ' ' + ' '.join(r['cov'] + r['subj'])))]
print('Aegean/Cretan candidates >= 2014', len(cand), flush=True)
hits_f = os.path.join(OUT, 'hits.jsonl'); done = set()
if os.path.exists(hits_f): done = {json.loads(l)['id'] for l in open(hits_f)}
with open(hits_f, 'a') as f:
    for i, r in enumerate(cand):
        if r['id'] in done: continue
        t = get(r['id'].replace('&amp;', '&'))
        t = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', t, flags=re.S); t = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', t)))
        ctx = [t[max(0, m.start() - 600): m.end() + 600] for m in LA.finditer(t)]
        f.write(json.dumps(dict(id=r['id'], title=r['title'], date=r['date'], n_la=len(ctx), ctx=ctx[:6])) + '\n'); f.flush()
        if i % 25 == 0: print('fetched', i, len(cand), flush=True)
        time.sleep(1)
print('DONE', flush=True)
