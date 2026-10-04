"""pe35: enlarge the seal-motif data beyond Legrain (pe33).

Sources (depicted content only; never anyone's reading of signs):
  Louvre collections JSON records (collections.louvre.fr/ark:/53355/<ark>.json), field 'description'
  ('Decor: ...' and 'Precision sur l'objet: ...') for Suse III tablets, sealings and cylinders; bibliography gives
  Amiet 1972 (MDP 43) catalogue numbers.  Manual codes from Louvre photographs (scratchpad only) are in MANUAL.
Links tablet -> seal picture:
  (L1) the tablet's own Louvre record (SB number = CDLI museum no.) describes the impression;
  (L2) CDLI 'seal 1 = PESnnnn' with nnnn >= 340 = Amiet MDP 43 no. nnnn (checked here: Louvre SB 6387 = MDP 17, 489 cites
       Amiet n. 1013 and CDLI gives PES1013 for the other tablet with that seal, MDP 06, 309); description from any Louvre record
       citing that Amiet number;
  (L3) the tablet's own record cites an Amiet number described in another record.
Keyword rules were frozen in loops/pe35_cycle1.txt before coding.
"""
import json, os, re, sys, unicodedata, collections
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
SCAN = os.path.join(DATA, 'pe35_ckpt', 'louvre_scan.jsonl')
OUT = os.path.join(DATA, 'pe35_seal_motifs.json')
RULES = [('BOVID', r'taureau|bovin|boeuf|bison|vache|veau|zebu'),
         ('CAPRID', r'bouquetin|chevre|caprin|capride|mouflon|belier|mouton|antilope|gazelle|cerf|ovin'),
         ('FELINE', r'lion|felin|panthere|leopard'), ('CANID', r'chien|canide|loup'),
         ('BIRD', r'oiseau|aigle|rapace'), ('MONSTER', r'monstre|griffon|composite|dragon'),
         ('HUMAN', r'personnage|homme|humain'), ('ANTHRO', r'attitude humaine|anthropomorph|genie'),
         ('BUILDING', r'batiment|edifice|grenier|temple|facade'), ('VESSEL', r'vase|recipient|jarre'),
         ('WATER', r'barque|bateau|poisson|\beau\b|filet'), ('GEOM', r'geometrique|rosace|rosette|losange|spirale'),
         ('PREDATION', r'attaqu|devor|poursui|terrass|chasse|chassant|agress')]
# MANUAL[SB] = (description of what the photograph shows, features, photo source) -- filled after viewing images
MANUAL = {}


def norm(s):
    s = unicodedata.normalize('NFD', s or '').encode('ascii', 'ignore').decode().lower()
    return s.replace('’', "'")


def code(text):
    t = norm(text)
    t_h = t.replace('attitude humaine', '')
    f = set()
    for k, rx in RULES:
        if re.search(rx, t_h if k == 'HUMAN' else t):
            f.add(k)
    return f


def scene_text(d):
    """the part of a Louvre description that describes the picture (drop 'inscription', object words)."""
    s = d.get('desc') or ''
    # amendment (before any test): drop the museum's reading of the TEXT ('compte d'equides', 'compte de moutons'...)
    s = re.sub(r"(?i)comptes? (de |d'|d\u2019)[^,.;]*", ' ', s)
    s = re.sub(r"(?i)(liste|enregistrement|livraison|rations?) (de |d'|d\u2019)[^,.;]*", ' ', s)
    s = re.sub(r'(?i)précision sur l.objet\s*:', ' ', s)
    s = re.sub(r'(?i)tablette (proto-élamite|économique)?', ' ', s)
    s = re.sub(r'(?i)signes proto-élamites|inscription|proto-élamite', ' ', s)
    s = re.sub(r'(?i)scellements?|tablettes?|empreintes? de (sceaux?|cylindres?)( cylindre)?|oblitérée?', ' ', s)
    return s


def amiet_nos(d):
    out = []
    for ref, det in d.get('bib', []):
        if ref.startswith('Amiet, Pierre, Glyptique susienne') and det:
            out += [int(x) for x in re.findall(r'n°\s*(\d+)', det)]
    return out


def sbkey(s):
    m = re.match(r'(?i)sb\s*0*(\d+)', (s or '').strip())
    return int(m.group(1)) if m else None


def build():
    recs = [json.loads(l) for l in open(SCAN)]
    recs = [r for r in recs if r.get('keep')]
    meta = json.load(open(os.path.join(DATA, 'pe8_meta.json')))
    raw = open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8').read()
    pes = {}
    for b in re.split(r'\n(?=&P)', raw):
        m = re.match(r'&(P\d+)', b)
        if m:
            pes[m.group(1)] = sorted({int(x) for x in re.findall(r'PES0*(\d+)', b, re.I)})
    old = {t['id'] for t in json.load(open(os.path.join(DATA, 'pe33_seal_motifs.json')))['tablets']}
    by_sb = collections.defaultdict(list)
    by_amiet = collections.defaultdict(list)
    for r in recs:
        k = sbkey(r.get('num'))
        if k:
            by_sb[k].append(r)
        txt = scene_text(r)
        r['_feat'] = code(txt)
        r['_txt'] = re.sub(r'\s+', ' ', txt).strip()
        for a in amiet_nos(r):
            by_amiet[a].append(r)
    rows = []
    for p, m in meta.items():
        sb = sbkey(m.get('museum_no'))
        own = by_sb.get(sb, []) if sb else []
        links, feats, desc, seals = [], set(), [], []
        if sb in MANUAL:
            dsc, ff, src = MANUAL[sb]
            links.append('photograph %s (coded here, depicted content)' % src); feats |= ff; desc.append(dsc)
            seals.append('SB%d' % sb)
        for r in own:
            if r['_feat'] and sb not in MANUAL:
                links.append('Louvre %s (%s) description' % (r['num'], r['ark'])); feats |= r['_feat']; desc.append(r['_txt'][:200])
                seals += ['A%d' % a for a in amiet_nos(r)] or ['SB%d' % sb]
        amiets = set(x for x in pes.get(p, []) if x >= 340)
        for r in own:
            amiets |= set(amiet_nos(r))
        for a in sorted(amiets):
            for r in by_amiet.get(a, []):
                if r['_feat'] and r not in own:
                    links.append('Amiet MDP 43 n. %d: Louvre %s (%s) description' % (a, r['num'], r['ark']))
                    feats |= r['_feat']; desc.append(r['_txt'][:200]); seals.append('A%d' % a)
        if feats - {'GEOM'} or feats:
            if not links:
                continue
            seal = sorted(set(seals))[0] if seals else 'SB%s' % sb
            rows.append({'id': p, 'designation': m['designation'], 'museum_no': m.get('museum_no'), 'seal': seal,
                         'amiet': sorted(amiets), 'features': sorted(feats), 'links': links, 'describes': desc,
                         'in_pe33': p in old})
    doc = {'source': 'Louvre collections online records (collections.louvre.fr, JSON per ark; descriptions of depicted content) '
                     'and Amiet 1972 (MDP 43) catalogue numbers cited there; CDLI ATF seal ids PESnnnn (nnnn >= 340 = Amiet no.). '
                     'Features from frozen keyword rules (loops/pe35_cycle1.txt) or from photographs (MANUAL).',
           'features': [k for k, _ in RULES], 'tablets': sorted(rows, key=lambda r: r['id'])}
    json.dump(doc, open(OUT, 'w'), indent=1, ensure_ascii=False)
    return doc


if __name__ == '__main__':
    d = build()
    T = d['tablets']
    new = [t for t in T if not t['in_pe33']]
    print(len(T), 'coded;', len(new), 'new (not in pe33)')
    print(collections.Counter(f for t in new for f in t['features']).most_common())
    for t in T:
        print(t['id'], t['designation'], t['museum_no'], t['seal'], t['features'], 'OLD' if t['in_pe33'] else 'NEW', '|', t['describes'][0][:90])
