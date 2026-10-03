#!/usr/bin/env python3
"""V3: fetch open medieval plaintext corpora from Wikisource (raw wikitext), strip markup,
save to data/plain/<lang>.txt and record every source page in data/plain/SOURCES.tsv.
Polite: one request every ~2 s, retries on HTTP 429."""
import sys, os, re, time, json, urllib.request, urllib.parse
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'plain')
os.makedirs(OUT, exist_ok=True)
UA = "VoynichResearch/0.1 (data-only study)"

def get(url, tries=6):
    for i in range(tries):
        try:
            time.sleep(2)
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            return urllib.request.urlopen(req, timeout=60).read().decode('utf-8')
        except Exception as e:
            print('retry', url[:80], e, file=sys.stderr); time.sleep(15 * (i + 1))
    return ''

def api(host, **p):
    p['format'] = 'json'
    return json.loads(get(f"https://{host}/w/api.php?" + urllib.parse.urlencode(p)) or '{}')

def raw(host, title):
    return get(f"https://{host}/w/index.php?" + urllib.parse.urlencode({'title': title, 'action': 'raw'}))

def strip_wiki(t):
    t = re.sub(r'<!--.*?-->', ' ', t, flags=re.S)
    for _ in range(4):
        t = re.sub(r'\{\{[^{}]*\}\}', ' ', t, flags=re.S)
    t = re.sub(r'<ref[^>]*/>', ' ', t); t = re.sub(r'<ref.*?</ref>', ' ', t, flags=re.S)
    t = re.sub(r'\[\[(?:File|Image|Category|Kategorie|Categoria|Soubor|Fichièr)[^\]]*\]\]', ' ', t, flags=re.I)
    t = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', t); t = re.sub(r'\[\[([^\]]*)\]\]', r'\1', t)
    t = re.sub(r'\[https?://\S+\s*([^\]]*)\]', r'\1', t)
    t = re.sub(r'<[^>]+>', ' ', t); t = re.sub(r"'{2,}", '', t)
    t = re.sub(r'^[=|!{}].*$', ' ', t, flags=re.M)  # headings, table rows
    t = re.sub(r'&nbsp;', ' ', t)
    return t

src = open(os.path.join(OUT, 'SOURCES.tsv'), 'a')
def save(lang, host, titles, maxw=30000):
    parts, n = [], 0
    for ti in titles:
        t = strip_wiki(raw(host, ti)); w = len(t.split())
        if w < 30: continue
        parts.append(t); n += w
        src.write(f"{lang}\thttps://{host}/wiki/{urllib.parse.quote(ti.replace(' ', '_'))}\t{w}\n"); src.flush()
        print(lang, ti, w, n, flush=True)
        if n >= maxw: break
    open(os.path.join(OUT, lang + '.txt'), 'w').write('\n'.join(parts))

which = sys.argv[1:] or ['la', 'de', 'it', 'cs', 'oc']
if 'la' in which:   # Isidore, Etymologiae (7th c., copied throughout the 15th); books XVI-XX incl. plants (XVII)
    t = raw('la.wikisource.org', 'Etymologiae (Isidorus)')
    i = t.find('LIBER XVI'); t = strip_wiki(t[i if i > 0 else 0:])
    open(os.path.join(OUT, 'la.txt'), 'w').write(t)
    src.write(f"la\thttps://la.wikisource.org/wiki/Etymologiae_(Isidorus) [from LIBER XVI]\t{len(t.split())}\n")
if 'de' in which:   # Johannes von Tepl, Der Ackermann aus Boehmen (c. 1400, Early New High German)
    # diplomatic transcription of Cod. Pal. germ. 76 (Heidelberg, c. 1470); text sits on transcluded Seite: pages
    t = raw('de.wikisource.org', 'Der Ackermann aus Böhmen (Handschrift 14. Jh.)')
    pages = ['Seite:' + m.replace('_', ' ') for m in re.findall(r'SeitePR\|[^|]*\|([^}]+)\}\}', t)]
    save('de', 'de.wikisource.org', pages)
if 'it' in which:   # Cennino Cennini, Il libro dell'arte (c. 1400, Tuscan recipes)
    d = api('it.wikisource.org', action='query', list='allpages', apprefix="Il libro dell'arte/Capitolo", aplimit=500)
    ts = [x['title'] for x in d['query']['allpages']]
    roman = lambda s: sum({'I':1,'V':5,'X':10,'L':50,'C':100}[c] * (-1 if j + 1 < len(s) and {'I':1,'V':5,'X':10,'L':50,'C':100}[c] < {'I':1,'V':5,'X':10,'L':50,'C':100}[s[j+1]] else 1) for j, c in enumerate(s))
    ts.sort(key=lambda x: roman(x.split()[-1]) if re.fullmatch('[IVXLC]+', x.split()[-1]) else 999)
    save('it', 'it.wikisource.org', ts)
if 'cs' in which:   # Rymovana kronika ceska tak receneho Dalimila (c. 1314, Old Czech; Jirecek 1877 ed.)
    d = api('cs.wikisource.org', action='query', list='allpages', apprefix="Rýmovaná kronika česká tak řečeného Dalimila/", aplimit=500)
    ts = [x['title'] for x in d['query']['allpages'] if 'Úvod' not in x['title']]
    roman = lambda s: sum({'I':1,'V':5,'X':10,'L':50,'C':100}[c] * (-1 if j + 1 < len(s) and {'I':1,'V':5,'X':10,'L':50,'C':100}[c] < {'I':1,'V':5,'X':10,'L':50,'C':100}[s[j+1]] else 1) for j, c in enumerate(s))
    ts.sort(key=lambda x: roman(x.split('/')[-1]) if re.fullmatch('[IVXLC]+', x.split('/')[-1]) else 999)
    save('cs', 'cs.wikisource.org', ts)
if 'oc' in which:   # classical troubadours (12th-13th c.) listed on the Occitan Wikisource portal
    authors = ["Pèire d'Alvèrnhe", 'Raimon Vidal de Besalú', 'Bertran de Bòrn', 'Arnaut Daniel', 'Marcabrun',
               'Monge de Montaudon', 'Peiròl', 'Guiraut Riquièr', 'Jaufre Rudel', 'Bernard de Ventadorn',
               'Guilhèm de Mur', 'Gui d\'Uissèl', 'Castelosa', 'Lo Dalfin', 'Pèire Rotgièr', 'Èble de Sanha']
    ts = []
    for a in authors:
        t = raw('wikisource.org', 'Author:' + a)
        for m in re.findall(r'\[\[([^\]|#:]+)(?:\|[^\]]*)?\]\]', t):
            if m not in ts: ts.append(m)
    print('oc titles', len(ts))
    save('oc', 'wikisource.org', ts)
