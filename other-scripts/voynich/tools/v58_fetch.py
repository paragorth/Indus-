"""v58: download candidate source texts (raw files) into data/v58_ckpt/src/.

Sources: Bibliotheca Augustana (hs-augsburg.de), Perseus / First1KGreek TEI on GitHub raw,
la.wikisource (raw wikitext), The Latin Library, Project Gutenberg, Sefaria API.
Everything downloaded stays in the git-ignored checkpoint folder.
"""
import os, re, sys, time, json, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'data', 'v58_ckpt', 'src')
os.makedirs(SRC, exist_ok=True)
UA = {'User-Agent': 'v58-shape-research/0.1 (academic text-statistics study)'}

def get(url, dest, tries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        return open(dest, 'rb').read()
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = urllib.request.urlopen(req, timeout=60).read()
            open(dest, 'wb').write(data)
            time.sleep(0.4)
            return data
        except Exception as e:
            print('  retry', url, e, file=sys.stderr); time.sleep(3 + 5 * t)
    return b''

AUG = 'https://www.hs-augsburg.de/~harsch/'
AUGUSTANA = {
    # id: (index page, regex for chapter pages in the same directory)
    'hildegard_physica': ('Chronologia/Lspost12/Hildegard/hil_phy0.html', r'hil_phy\d\.html'),
    'celsus_lat_aug': ('Chronologia/Lspost01/Celsus/cel_m000.html', r'cel_m1\d\d\.html'),
    'lorsch_receptarium': ('Chronologia/Lspost09/SummaMedicinae/sum_me00.html', r'sum_me\d\d\.html'),
    'regimen_salernitanum': ('Chronologia/Lspost11/Regimen/reg_intr.html', r'reg_sana\.html'),
    'walahfrid_hortulus': ('Chronologia/Lspost09/Walahfrid/wal_ho00.html', r'wal_ho\d\d\.html'),
    'konrad_buch_der_natur': ('germanica/Chronologie/14Jh/KonradMegenberg/kon_0000.html', r'kon_\w{4}\.html'),
}

def fetch_augustana(wid, idx, pat):
    d = os.path.join(SRC, wid); os.makedirs(d, exist_ok=True)
    base = AUG + idx.rsplit('/', 1)[0] + '/'
    html = get(AUG + idx, os.path.join(d, '_index.html')).decode('utf-8', 'replace')
    links = sorted(set(re.findall(r'href="(%s)"' % pat, html)))
    for l in links:
        if l == idx.rsplit('/', 1)[1]: continue
        get(base + l, os.path.join(d, l))
    print(wid, len(links), 'pages')

RAW = 'https://raw.githubusercontent.com/'
TEI = {
    'pliny_nh_lat': 'PerseusDL/canonical-latinLit/master/data/phi0978/phi001/phi0978.phi001.perseus-lat2.xml',
    'pliny_nh_eng': 'PerseusDL/canonical-latinLit/master/data/phi0978/phi001/phi0978.phi001.perseus-eng1.xml',
    'celsus_lat': 'PerseusDL/canonical-latinLit/master/data/phi0836/phi002/phi0836.phi002.perseus-lat4.xml',
    'celsus_eng': 'PerseusDL/canonical-latinLit/master/data/phi0836/phi002/phi0836.phi002.perseus-eng2.xml',
    'dioscorides_grc': 'OpenGreekAndLatin/First1KGreek/master/data/tlg0656/tlg001/tlg0656.tlg001.1st1K-grc1.xml',
    'theophrastus_hp_grc': 'OpenGreekAndLatin/First1KGreek/master/data/tlg0093/tlg001/tlg0093.tlg001.1st1K-grc1.xml',
}

GUT = {  # Project Gutenberg plain texts
    'apicius_eng': 29728, 'forme_of_cury': 8102, 'culpeper': 49513,
}

LL = {  # Latin Library
    'apicius_lat': ['apicius/apicius%d.html' % i for i in range(1, 11)],
}

WS = {  # la.wikisource raw pages
    'macer_floridus': 'De viribus herbarum',
}

def main():
    which = sys.argv[1:] or ['aug', 'tei', 'gut', 'll', 'ws']
    if 'aug' in which:
        for k, (i, p) in AUGUSTANA.items(): fetch_augustana(k, i, p)
    if 'tei' in which:
        for k, u in TEI.items():
            b = get(RAW + u, os.path.join(SRC, k + '.xml')); print(k, len(b))
    if 'gut' in which:
        for k, n in GUT.items():
            b = get('https://www.gutenberg.org/cache/epub/%d/pg%d.txt' % (n, n), os.path.join(SRC, k + '.txt')); print(k, len(b))
    if 'll' in which:
        for k, pages in LL.items():
            d = os.path.join(SRC, k); os.makedirs(d, exist_ok=True)
            for p in pages: get('https://www.thelatinlibrary.com/' + p, os.path.join(d, p.split('/')[-1]))
            print(k, len(pages))
    if 'ws' in which:
        for k, t in WS.items():
            u = 'https://la.wikisource.org/w/index.php?action=raw&title=' + urllib.parse.quote(t.replace(' ', '_'))
            b = get(u, os.path.join(SRC, k + '.wiki')); print(k, len(b))

if __name__ == '__main__':
    main()
