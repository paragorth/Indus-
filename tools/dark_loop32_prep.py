"""S-DARK-32 prep: build the reference library for the typology classifier.
One JSONL per corpus under data/derived/dark/loop32_corpora/<name>.jsonl, each line {"id", "seq"}; a SOURCES.txt with
provenance, licence and the known type of every corpus (L = language writing, D = designed non-linguistic code,
G = grammar without language, A = accounting / tally record, I = Indus, X = Indus-derived control).

Reference corpora re-used from data/codelib (tools/strat_codelibrary.py prep): ur3_legends (words), linear_b (lines, words),
latin_edh (words), heraldry (blazon words), khipu (cord clusters), proto_elamite (entry lines).
New here: ur3_syll (Ur III legends as syllables, one per impression), ur3_names_syll (line-1 owner names, syllables),
linb_syll (Linear B lines as syllabic signs + ideograms), runes_words (RunesDB.de TEI, CC BY-SA 4.0; small),
proto_cuneiform (CDLI ATF lang qpc entry lines), icd10 (CMS ICD-10-CM 2025 codes, characters), hts (USITC HTS 2025 numbers,
2-digit groups), unicode_names (UnicodeData.txt character names, words), aircraft_reg (OpenSky aircraft database registrations,
characters), chess_eco (lichess chess-openings, CC0, moves), chords (Chordonomicon v2, CC BY-NC 4.0, song sections, chords),
indus_im77 (data/im77/im77_corpus_lines.csv, lines of one text joined in reading order, sign 0 dropped).
Usage: python3 tools/dark_loop32_prep.py
"""
import os, re, json, csv, glob, random, collections, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data/derived/dark/loop32_corpora')
RAW = os.path.join(OUT, 'raw')
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
LIB = os.path.join(ROOT, 'data/codelib')
rnd = random.Random(32)
NOTES = []


def dump(name, texts, typ, note):
    texts = [list(t) for t in texts if t]
    with open(os.path.join(OUT, name + '.jsonl'), 'w') as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({'id': i, 'seq': t}, ensure_ascii=False) + '\n')
    L = sorted(len(t) for t in texts)
    V = len({s for t in texts for s in t})
    line = f'{name} [{typ}]: {len(texts)} texts, {sum(L)} tokens, {V} types, median len {L[len(L)//2]}. {note}'
    print(line); NOTES.append(line)


def copy_codelib(name, newname, typ, note):
    texts = [json.loads(l)['seq'] for l in open(os.path.join(LIB, name + '.jsonl'))]
    dump(newname, texts, typ, note)


def indus():
    C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    for key in ('seq_raw', 'seq_strong', 'seq_all'):
        dump('indus_' + key, [[f'W{s}' for s in r[key]] for r in C if r.get(key)], 'I',
             f'data/derived/merged-corpus-canonical.json {key}, all inscribed objects (S-DARK-23 caution: older inscriptions.csv build)')
    # IM77: one text per text_no, lines joined in (side, line) order, sign 0 (break) dropped
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, 'data/im77/im77_corpus_lines.csv'))):
        if r['line'] == '9' or r['direction_code'] == '9' or not r['signs_clean'].strip():
            continue
        rows[r['text_no']].append((int(r['side']), int(r['line']), r['signs_clean'].split()))
    texts = []
    for tn, ls in rows.items():
        ls.sort()
        s = [f'M{x}' for _, _, sg in ls for x in sg if x != '0']
        if s: texts.append(s)
    dump('indus_im77', texts, 'I', 'data/im77/im77_corpus_lines.csv: Mahadevan 1977 texts, lines joined in reading order, sign 0 dropped')


def ur3():
    copy_codelib('ur3_legends', 'ur3_words', 'L', 'CDLI @seal legends (lang sux), words; one copy per distinct legend (codelib)')
    D = json.load(open(os.path.join(SP, 'ur3_legends.json')))
    texts = []
    for d in D:
        toks = [t for l in d['lines'] for t in l if t]
        if any('x' == t or '...' in t for t in toks) or not 2 <= len(toks) <= 20:
            continue
        texts.append(toks)
    dump('ur3_syll', texts, 'L', 'CDLI @seal legends, syllable signs, one text per impression (not deduplicated), 2-20 signs')
    names = []
    for l in open(os.path.join(SP, 'seal_line1.txt'), errors='ignore'):
        w = re.sub(r'[\[\]#?!<>]', '', l.strip().lower())
        if not w or 'x' in w.split('-') or '...' in w:
            continue
        syl = [s for s in w.replace(' ', '-').split('-') if s]
        if 1 <= len(syl) <= 12:
            names.append(syl)
    dump('ur3_names_syll', names, 'L', 'CDLI seal legends line 1 (owner names), syllable signs, one per impression (tools/strat_namecalib.py data)')


def linb():
    copy_codelib('linear_b', 'linb_words', 'L', 'DAMOS Linear B tablet lines: sign groups, ideograms, NUM (codelib)')
    texts = []
    for l in open(os.path.join(LIB, 'linear_b.jsonl')):
        s = []
        for w in json.loads(l)['seq']:
            if w == 'NUM' or re.fullmatch(r'[A-Z][A-Z0-9*+]*', w) or '-' not in w:
                s.append(w)
            else:
                s += [x for x in w.split('-') if x]
        texts.append(s)
    dump('linb_syll', texts, 'L', 'DAMOS Linear B tablet lines split into syllabic signs; ideograms and NUM kept whole')


def latin():
    copy_codelib('latin_edh', 'latin_edh', 'L', 'EDH (Heidelberg) Italy, Latin inscriptions, words, abbreviations expanded (codelib)')


def runes():
    texts = []
    for f in glob.glob(os.path.join(RAW, 'runes_tei/*.xml')):
        t = open(f, encoding='utf-8', errors='ignore').read()
        segs = [s.strip() for s in re.findall(r'<seg type="transliteration" subtype="main" xml:lang="en">([^<]*)</seg>', t)]
        if not segs:
            continue
        words, cur = [], []
        for s in segs:
            if re.fullmatch(r'[a-zA-Zþðŋʀęøœæåäöü]+', s):
                cur.append(s)
            else:
                if cur: words.append(''.join(cur)); cur = []
        if cur: words.append(''.join(cur))
        words = [w for w in words if w]
        if words: texts.append(words)
    dump('runes_words', texts, 'L', 'RunesDB.de inscription data (Zenodo 21788307, CC BY-SA 4.0), rune words split at separators; SMALL (< 1,000), test-only')


def heraldry_khipu_pe():
    copy_codelib('heraldry', 'heraldry', 'D', 'Rietstap Armorial general, blazon words (codelib)')
    copy_codelib('khipu', 'khipu', 'A', 'Open Khipu Repository 2.1.0 cord clusters: colour:magnitude (codelib)')
    copy_codelib('proto_elamite', 'proto_elamite', 'A', 'CDLI Proto-Elamite entry lines, signs + numeral type (codelib)')


def proto_cuneiform():
    texts = []
    lang = None
    for l in open(os.path.join(SP, 'cdli.atf'), errors='ignore'):
        if l.startswith('&'):
            lang = None; continue
        if l.startswith('#atf: lang'):
            lang = l.split()[-1]; continue
        if lang != 'qpc' or not re.match(r"^\d+'?\.", l):
            continue
        parts = l.split(None, 1)
        if len(parts) < 2: continue
        toks = []
        for w in parts[1].split():
            if w == ',': continue
            m = re.match(r'^[\d/]+\((N\d+[A-Z]?)\)', w)
            if m: toks.append(m.group(1)); continue
            w = re.sub(r'[#?!\[\]]', '', w)
            if w in ('x', '...', '', 'N', '[N]') or w.startswith('...'): continue
            toks.append(w)
        if toks: texts.append(toks)
    dump('proto_cuneiform', texts, 'A', 'CDLI ATF lang qpc (Uruk III/IV proto-cuneiform) entry lines, signs + numeral sign type')


def codes():
    icd = [l.split()[0] for l in open(os.path.join(RAW, 'icd10cm_codes_2025.txt')) if l.strip()]
    dump('icd10', [list(c) for c in icd], 'D', 'CMS ICD-10-CM FY2025 code list (US government, public domain), one code = one text, characters as signs')
    H = json.load(open(os.path.join(RAW, 'hts.json')))
    hts = []
    for x in H:
        d = x['htsno'].replace('.', '')
        if d.isdigit() and len(d) >= 4 and len(d) % 2 == 0:
            hts.append([d[i:i + 2] for i in range(0, len(d), 2)])
    dump('hts', hts, 'D', 'USITC Harmonized Tariff Schedule 2025 numbers (US government), 2-digit groups as signs (chapter, heading, subheading, statistical suffix)')
    rows = list(csv.reader(open(os.path.join(RAW, 'UnicodeData.txt')), delimiter=';'))
    names = [r[1].split() for r in rows if len(r) > 1 and r[1] and not r[1].startswith('<')]
    rnd.shuffle(names)
    dump('unicode_names', names[:20000], 'D', 'UnicodeData.txt character names (Unicode licence), words as signs, random 20,000 of 41,232')
    r = csv.reader(open(os.path.join(RAW, 'opensky_part.csv'), errors='ignore'))
    h = next(r); i = h.index('registration')
    regs = []
    for row in r:
        if len(row) > i and row[i].strip():
            regs.append(list(row[i].strip().upper()))
    rnd.shuffle(regs)
    dump('aircraft_reg', regs[:20000], 'D', 'OpenSky Network aircraft database (free for research use), registration marks, characters as signs (hyphen kept), random 20,000 of the first 20 MB')


def chess_chords():
    texts = []
    for f in 'abcde':
        for row in csv.DictReader(open(os.path.join(RAW, f'chess_{f}.tsv')), delimiter='\t'):
            mv = [m for m in row['pgn'].split() if not re.match(r'\d+\.$', m)]
            if mv: texts.append(mv)
    dump('chess_eco', texts, 'G', 'lichess-org/chess-openings (CC0), ECO opening lines, SAN moves as signs')
    secs = []
    with open(os.path.join(RAW, 'chordonomicon_part.csv'), errors='ignore') as fh:
        rd = csv.reader(fh); next(rd)
        for row in rd:
            if len(row) < 2: continue
            for p in re.split(r'<[^>]+>', row[1]):
                t = p.split()
                if t: secs.append(t)
    rnd.shuffle(secs)
    dump('chords', secs[:20000], 'G', 'Chordonomicon v2 (Hugging Face ailsntua/Chordonomicon, CC BY-NC 4.0), one song section = one text, chord symbols as signs, random 20,000 sections from the first 25 MB')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    indus(); ur3(); linb(); latin(); runes(); heraldry_khipu_pe(); proto_cuneiform(); codes(); chess_chords()
    with open(os.path.join(OUT, 'SOURCES.txt'), 'w') as f:
        f.write('S-DARK-32 reference library (tools/dark_loop32_prep.py). Types: L language writing, D designed code, G grammar without language, A accounting/tally, I Indus.\n\n')
        f.write('\n'.join(NOTES) + '\n\n')
        f.write('SOURCES: CDLI ATF dump (github cdli-gh/data; seal legends lang sux, proto-cuneiform lang qpc; CC BY); DAMOS Linear B (other-scripts/linear-a/data/damos_items.jsonl);\n'
                'EDH API https://edh.ub.uni-heidelberg.de (CC BY-SA); RunesDB.de TEI https://zenodo.org/records/21788307 (CC BY-SA 4.0); Rietstap Armorial general (archive.org, public domain);\n'
                'Open Khipu Repository 2.1.0 https://doi.org/10.5281/zenodo.5037551; CDLI Proto-Elamite ATF; CMS ICD-10-CM 2025 https://www.cms.gov/files/zip/2025-code-descriptions-tabular-order.zip (US gov);\n'
                'USITC HTS https://hts.usitc.gov/reststop/exportList (US gov); Unicode UnicodeData.txt https://www.unicode.org/Public/UCD/latest/ucd/ (Unicode licence);\n'
                'OpenSky aircraftDatabase.csv https://opensky-network.org/datasets/metadata/ (free for research); lichess chess-openings https://github.com/lichess-org/chess-openings (CC0);\n'
                'Chordonomicon https://huggingface.co/datasets/ailsntua/Chordonomicon (CC BY-NC 4.0); IM77 data/im77 (Mahadevan 1977 via indusscript.in).\n'
                'NOT OBTAINED (no open machine-readable source found, skipped): Egyptian royal titularies, Old Persian / Aramaic seal legends, Phoenician / Punic stamp seals,\n'
                'Chinese oracle-bone / bronze short texts, medieval merchant and masons marks, Mesopotamian potters / brick marks, ISBN / EAN real-number lists, vehicle plates, Dewey call numbers;\n'
                'Rundata (Uppsala, Viking-Age runic) could not be downloaded here (GitHub mirror not reachable from this session; FAA registry returned 403), RunesDB used instead.\n'
                'Raw downloads larger than 5 MB were deleted after prep to keep the repository small; the URLs above re-create them.\n')
