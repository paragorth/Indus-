"""S-DARK-48 prep: provenance-tagged corpora for the centralization index.
Writes data/derived/dark/loop48_corpora/<name>.jsonl, one line per text: {"site", "type", "seq"}.
Sources (all open, see loop32_corpora/SOURCES.txt): CDLI ATF dump + catalogue (Ur III seal legends, proto-cuneiform,
provenience by P-number), DAMOS Linear B (site = heading prefix KN/PY/TH/MY), EDH Italy (modern_region),
CDLI Proto-Elamite (other-scripts/proto-elamite/data/pe_corpus.json, provenience field).
Raw files live in the session scratchpad (SP); the URLs in SOURCES.txt re-create them.
"""
import os, re, csv, json, collections
csv.field_size_limit(10**9)
ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/loop48_corpora/'
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
os.makedirs(OUT, exist_ok=True)
NOTES = []

def dump(name, texts, note):
    with open(OUT + name + '.jsonl', 'w') as f:
        for site, typ, seq in texts:
            f.write(json.dumps({'site': site, 'type': typ, 'seq': list(seq)}) + '\n')
    c = collections.Counter(s for s, _, _ in texts)
    line = f'{name}: {len(texts)} texts; sites {dict(c.most_common(12))}. {note}'
    NOTES.append(line); print(line)

def cdli_prov():
    r = csv.reader(open(SP + 'cdli_cat.csv', errors='ignore')); hdr = next(r)
    ip = hdr.index('provenience'); ii = hdr.index('id_text'); iper = hdr.index('period')
    prov = {}; per = {}
    for row in r:
        if len(row) > max(ip, ii, iper):
            p = row[ip].split(' (')[0].strip()
            if p.endswith('?'): p = 'uncertain'
            prov[row[ii]] = p or 'uncertain'; per[row[ii]] = row[iper]
    return prov, per

def clean_tok(w):
    w = re.sub(r'[\[\]#?!<>*]', '', w).lower()
    return w

def ur3_and_qpc():
    prov, per = cdli_prov()
    legends = []    # (pid, [words])
    qpc = []        # (pid, [toks]) entry lines
    cur = None; lang = None; inseal = None
    for l in open(SP + 'cdli.atf', errors='ignore'):
        if l.startswith('&'):
            cur = l[1:8].lstrip('P').lstrip('0'); lang = None; inseal = None; continue
        if l.startswith('#atf: lang'):
            lang = l.split()[-1]; continue
        if l.startswith('@seal'):
            inseal = []; legends.append((cur, lang, inseal)); continue
        if l[:1] in '@$#>':
            if l.startswith('@'): inseal = None
            continue
        if not re.match(r"^\d+'?\.", l): continue
        parts = l.split(None, 1)
        if len(parts) < 2: continue
        if inseal is not None:
            toks = [clean_tok(t) for t in parts[1].split()]
            inseal.extend(t for t in toks if t and t not in ('x', '(x)', '...', '($', 'blank)') and '...' not in t)
        elif lang == 'qpc':
            toks = []
            for w in parts[1].split():
                if w == ',': continue
                m = re.match(r'^[\d/]+\((N\d+[A-Z]?)\)', w)
                if m: toks.append(m.group(1)); continue
                w = re.sub(r'[#?!\[\]]', '', w)
                if w in ('x', '...', '', 'N', '[N]') or w.startswith('...'): continue
                toks.append(w)
            if toks: qpc.append((cur, toks))
    # Ur III seal legends: words, one per impression (collapse is done by the engine per site)
    words = []; sylls = []
    for pid, lang, toks in legends:
        if lang != 'sux' or not 2 <= len(toks) <= 20: continue
        if per.get(pid, '').startswith('Ur III') is False and 'Ur III' not in per.get(pid, ''): continue
        site = prov.get(pid, 'uncertain')
        words.append((site, 'legend', toks))
        syl = [s for w in toks for s in re.split(r'[-{}]', w) if s]
        if 2 <= len(syl) <= 30: sylls.append((site, 'legend', syl))
    dump('ur3_words', words, 'CDLI @seal legends (lang sux, period Ur III), words, one per impression; site = CDLI provenience')
    dump('ur3_syll', sylls, 'same legends split into syllable signs')
    pc = [(prov.get(pid, 'uncertain'), 'entry', toks) for pid, toks in qpc if len(toks) >= 2]
    pc = [(s if s != 'Susa' else 'Susa(qpc)', t, q) for s, t, q in pc]
    dump('proto_cuneiform', pc, 'CDLI lang qpc entry lines >= 2 signs; site = provenience (Uruk, Jemdet Nasr, Umma?, Ur archaic)')

def linb():
    texts = []
    for l in open(ROOT + 'other-scripts/linear-a/data/damos_items.jsonl'):
        it = json.loads(l)
        h = it.get('heading') or ''
        site = h.split()[0] if h else '?'
        if not re.fullmatch(r'[A-Z]{2}', site): continue
        for line in (it.get('content') or '').split('\n'):
            line = re.sub(r'^\s*\.?\d+[a-z]?\s+', ' ', line)
            if 'vac' in line or 'vest' in line: continue
            toks = []
            for w in line.split():
                w = w.strip("[],'/⟦⟧").replace('[', '').replace(']', '')
                if not w or w in ('[', ']', ',', '/', "'"): continue
                toks.append('NUM' if re.fullmatch(r'\d+', w) else w)
            if len(toks) < 2: continue
            syl = []
            for w in toks:
                if w == 'NUM' or re.fullmatch(r'[A-Z][A-Z0-9*+]*', w) or '-' not in w: syl.append(w)
                else: syl += [x for x in w.split('-') if x]
            texts.append((site, 'line', syl))
    dump('linb_syll', texts, 'DAMOS Linear B tablet lines >= 2 tokens as syllabic signs (ideograms, NUM whole); site = heading prefix')
    words = []
    for l in open(ROOT + 'other-scripts/linear-a/data/damos_items.jsonl'):
        pass
    # word-level version from the same parse
    texts_w = []
    for l in open(ROOT + 'other-scripts/linear-a/data/damos_items.jsonl'):
        it = json.loads(l); h = it.get('heading') or ''; site = h.split()[0] if h else '?'
        if not re.fullmatch(r'[A-Z]{2}', site): continue
        for line in (it.get('content') or '').split('\n'):
            line = re.sub(r'^\s*\.?\d+[a-z]?\s+', ' ', line)
            if 'vac' in line or 'vest' in line: continue
            toks = []
            for w in line.split():
                w = w.strip("[],'/⟦⟧").replace('[', '').replace(']', '')
                if not w or w in ('[', ']', ',', '/', "'"): continue
                toks.append('NUM' if re.fullmatch(r'\d+', w) else w)
            if len(toks) >= 2: texts_w.append((site, 'line', toks))
    dump('linb_words', texts_w, 'same lines as sign groups + ideograms + NUM')

def latin():
    texts = []
    for o in range(0, 6000, 1000):
        for it in json.load(open(SP + f'codelib/edh_{o}.json'))['items']:
            t = it.get('transcription') or ''
            t = re.sub(r'\[-+\]|\[\.\.\.\]|\[---\]', ' ', t)
            t = re.sub(r'[\[\]()<>{}!?]', '', t).replace('/', ' ')
            ws = re.findall(r"[a-zA-Z]+", t.lower())
            reg = (it.get('modern_region') or 'uncertain').strip() or 'uncertain'
            if len(ws) >= 2: texts.append((reg, it.get('type_of_inscription') or 'other', ws))
    dump('latin_edh', texts, 'EDH Italy inscriptions >= 2 words, abbreviations expanded; site = modern_region')

def pe():
    D = json.load(open(ROOT + 'other-scripts/proto-elamite/data/pe_corpus.json'))
    texts = []
    for d in D:
        p = d['provenience'].split(' (')[0].replace(' ?', '').strip() or 'uncertain'
        if p == 'uncertain': p = 'other(' + d['provenience'].split('mod. ')[-1].rstrip(')').replace(' ?', '') + ')'
        for ln in d['lines']:
            toks = list(ln['signs']) + [n[1] for n in ln.get('numerals', [])]
            if len(toks) >= 2: texts.append((p, 'entry', toks))
    dump('proto_elamite', texts, 'CDLI Proto-Elamite entry lines >= 2 tokens (signs + numeral sign types); site = provenience')

if __name__ == '__main__':
    ur3_and_qpc(); linb(); latin(); pe()
    with open(OUT + 'SOURCES.txt', 'w') as f:
        f.write('S-DARK-48 provenance-tagged corpora (tools/dark_loop48_prep.py). One line per text: site, type, seq.\n\n')
        f.write('\n'.join(NOTES) + '\n')
