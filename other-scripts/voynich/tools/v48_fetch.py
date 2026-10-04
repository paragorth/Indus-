"""v48 CLEAN THE SCRIPT, THEN TYPE THE LANGUAGE: fetch the comparison corpora (raw symbol data only).

Sources (all downloads stay in the scratchpad, v48/raw/):
  eBible.org readaloud zips (one txt per chapter): New Testament (Matthew on) unless noted.
  Hugging Face wikimedia/wikipedia 20231101 (datasets-server rows API, spread offsets) for languages without
    an eBible text: Basque, Catalan, Occitan, Georgian, Albanian, Maltese (+ Latin, as a genre check).
  Hugging Face Aasdasd/manchu: Manchu sentences in Moellendorff romanisation.
  Already in the scratchpad: Gaskell & Bowern 2022 texts (pinyin Matthew, Quran, Sanskrit, Nahuatl ...),
    v28 Armenian Wikipedia parquet, v15 Occitan (Wikisource troubadours), v30 medieval MSS (ReF, CATMuS, Old Czech).
"""
import os, sys, json, zipfile, io, re, random, time, urllib.request, urllib.parse

SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v48/raw'
os.makedirs(SCR, exist_ok=True)

EBIBLE = {  # code: (translation id, books ('NT' or 'OT'))
    'lat': ('latVUC', 'NT'), 'ita': ('ita1885', 'NT'), 'deu': ('deu1912', 'NT'), 'ces': ('ces1613', 'NT'),
    'pol': ('polubg', 'NT'), 'hun': ('hun', 'NT'), 'grc': ('grctr', 'NT'), 'heb': ('hebsg', 'NT'),
    'hbo': ('hboWLC', 'OT'), 'arb': ('arb-vd', 'NT'), 'tur': ('turytc', 'NT'), 'rom': ('rmc', 'NT'),
    'spa': ('spaRV1909', 'NT'), 'por': ('porbr2018', 'NT'), 'fra': ('fraLSG', 'NT'), 'ron': ('ron1924', 'NT'),
    'cop': ('copshc', 'NT'), 'rus': ('russyn', 'NT'), 'srp': ('srp1865', 'NT'), 'slk': ('slk', 'NT'),
    'nld': ('nld', 'NT'), 'eng': ('enggnv', 'NT'), 'dan': ('dan1931', 'NT'), 'fin': ('fin', 'NT'),
    'lit': ('lit', 'NT'), 'est': ('ekkpkp', 'NT'), 'bre': ('breBRG', 'NT'), 'pes': ('pesOPV', 'NT'),
    'ydd': ('ydd', 'NT'), 'aii': ('aii', 'NT'), 'hrv': ('hrv', 'NT'), 'rmy': ('rmyGurbet', 'NT'),
}
WIKI = ['eu', 'ca', 'oc', 'ka', 'sq', 'mt', 'la']


def get(url, tries=4):
    import subprocess
    for k in range(tries):
        r = subprocess.run(['curl', '-sS', '-f', '-L', '--max-time', '300', url], capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        print('retry', url[:100], r.stderr[:200], flush=True); time.sleep(3 + 5 * k)
    return None


def _get_urllib(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print('retry', url[:100], e, flush=True); time.sleep(3 + 5 * k)
    return None


def ebible(code, tid, part):
    out = os.path.join(SCR, f'eb_{code}.json')
    if os.path.exists(out): return
    b = get(f'https://ebible.org/Scriptures/{tid}_readaloud.zip')
    if not b: print('FAILED', code); return
    z = zipfile.ZipFile(io.BytesIO(b))
    names = sorted(n for n in z.namelist() if n.endswith('_read.txt') and '_000_' not in n)
    docs = []
    for n in names:
        num = int(n.split('_')[-4]) if n.count('_') >= 4 else 0
        m = re.search(r'_(\d{3})_([A-Z0-9]{3})_(\d+)_read', n)
        if not m: continue
        bk = int(m.group(1))
        if part == 'NT' and bk < 70: continue
        if part == 'OT' and bk >= 70: continue
        t = z.read(n).decode('utf-8-sig', 'replace').split('\n')
        docs.append([l.strip() for l in t[1:] if l.strip() and not re.fullmatch(r'[\d\.\s]+', l.strip())])
        if sum(len(' '.join(d).split()) for d in docs) > 60000: break
    json.dump(docs, open(out, 'w'), ensure_ascii=False)
    print(code, tid, len(docs), 'chapters', flush=True)


def wiki(lang, n_rows=1500, chunk=100):
    out = os.path.join(SCR, f'wk_{lang}.json')
    if os.path.exists(out): return
    s = json.loads(get('https://datasets-server.huggingface.co/size?dataset=wikimedia/wikipedia&config=20231101.' + lang))
    N = s['size']['config']['num_rows']
    rng = random.Random(48)
    docs, nw = [], 0
    while nw < 70000 and len(docs) < n_rows * 3:
        off = rng.randrange(0, max(1, N - chunk))
        q = urllib.parse.urlencode(dict(dataset='wikimedia/wikipedia', config='20231101.' + lang, split='train',
                                        offset=off, length=chunk))
        b = get('https://datasets-server.huggingface.co/rows?' + q)
        if not b: continue
        for r in json.loads(b)['rows']:
            t = r['row']['text']
            if len(t) < 800: continue   # stubs are lists of names and numbers
            ls = [l.strip() for l in t.split('\n') if len(l.split()) >= 6]
            if ls: docs.append(ls); nw += sum(len(l.split()) for l in ls)
    json.dump(docs, open(out, 'w'), ensure_ascii=False)
    print('wiki', lang, len(docs), nw, flush=True)


def manchu():
    out = os.path.join(SCR, 'mnc.json')
    if os.path.exists(out): return
    docs, cur = [], []
    for off in range(0, 34560, 100):
        b = get('https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(
            dict(dataset='Aasdasd/manchu', config='default', split='train', offset=off, length=100)))
        if not b: continue
        for r in json.loads(b)['rows']:
            cur.append(' '.join(r['row']['input']))
            if len(cur) == 30: docs.append(cur); cur = []
        if sum(len(l.split()) for d in docs for l in d) > 60000: break
    json.dump(docs, open(out, 'w'), ensure_ascii=False)
    print('manchu', len(docs), flush=True)


if __name__ == '__main__':
    for c, (t, p) in EBIBLE.items():
        ebible(c, t, p)
    for l in WIKI:
        wiki(l)
    manchu()
