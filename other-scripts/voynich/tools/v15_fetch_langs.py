"""v15: fetch letter-model training texts for Occitan, Greek and Hebrew.

Occitan: wikisource.org Category:Occitan (all ns-0 pages, wikitext).
Greek:   Perseus canonical-greekLit, Homer Iliad (TEI XML, GitHub raw).
Hebrew:  Sefaria API v3, Genesis + Exodus (Hebrew, vowel points stripped).
Raw text is written to the scratch dir given as argv[1]; only derived
letter streams (data/derived/v15_lm_*.txt) are kept in the repo.
"""
import json, os, re, sys, time, unicodedata, urllib.request, urllib.parse

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def get(url):
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'v15-research-script/0.1 (corpus statistics)'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            print('retry', url, e); time.sleep(2 + 3 * k)
    return ''


def occitan():
    titles, cont = [], {}
    while True:
        q = {'action': 'query', 'list': 'categorymembers', 'cmtitle': 'Category:Occitan',
             'cmlimit': 500, 'cmnamespace': 0, 'format': 'json'}
        q.update(cont)
        d = json.loads(get('https://wikisource.org/w/api.php?' + urllib.parse.urlencode(q)))
        titles += [m['title'] for m in d['query']['categorymembers']]
        if 'continue' not in d:
            break
        cont = {'cmcontinue': d['continue']['cmcontinue']}
    print('occitan pages', len(titles))
    texts = []
    for i in range(0, len(titles), 40):
        q = {'action': 'query', 'prop': 'revisions', 'rvprop': 'content', 'rvslots': 'main',
             'titles': '|'.join(titles[i:i + 40]), 'format': 'json'}
        d = json.loads(get('https://wikisource.org/w/api.php?' + urllib.parse.urlencode(q)))
        for p in d['query']['pages'].values():
            for r in p.get('revisions', []):
                t = r['slots']['main'].get('*', '')
                t = re.sub(r'\{\{[^{}]*\}\}', ' ', t)
                t = re.sub(r'\{\{[^{}]*\}\}', ' ', t)
                t = re.sub(r'<[^>]+>', ' ', t)
                t = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', t)
                t = re.sub(r'\[\[[^\]]*\]\]', ' ', t)
                texts.append(t)
    open(os.path.join(OUT, 'oc_raw.txt'), 'w').write('\n'.join(texts))


def greek():
    x = get('https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/'
            'tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml')
    body = x.split('<body')[1] if '<body' in x else x
    body = re.sub(r'<note.*?</note>', ' ', body, flags=re.S)
    body = re.sub(r'<[^>]+>', ' ', body)
    open(os.path.join(OUT, 'grc_raw.txt'), 'w').write(body)


def hebrew():
    out = []
    for book, n in (('Genesis', 50), ('Exodus', 40)):
        for c in range(1, n + 1):
            d = json.loads(get(f'https://www.sefaria.org/api/v3/texts/{book}.{c}?version=hebrew') or '{}')
            for v in d.get('versions', [])[:1]:
                t = v.get('text', [])
                out += t if isinstance(t, list) else [t]
            time.sleep(0.2)
    s = '\n'.join(x for x in out if isinstance(x, str))
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'&[a-z]+;', ' ', s)
    open(os.path.join(OUT, 'he_raw.txt'), 'w').write(s)


if __name__ == '__main__':
    which = sys.argv[2:] or ['occitan', 'greek', 'hebrew']
    for f in [globals()[w] for w in which]:
        try:
            f()
        except Exception as e:
            print('FAILED', f.__name__, e)
