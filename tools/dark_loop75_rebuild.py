"""Loop 75, cycle 1: rebuild the S268 canonical corpus from the current data/raw/inscriptions.csv.

S268 procedure (reconstructed and verified against the old file row by row):
  * one record per csv row whose text parses to >= 1 non-zero sign (Wells numbers, '0' = illegible dropped);
  * text reversed into reading order (the canonical convention);
  * metadata fields cisi, site, type, dir., time, period, phase, area-section, block-house, room-grid,
    symbol, cult, material, shape, complete copied verbatim, except symbol/cult 'None' -> '';
  * seq_raw = parsed text; seq_strong / seq_all = strong / strong+probable merges of
    data/derived/sign_allographs_levels.json (transitively closed); seq = seq_all.
v2 adds two fields only: 'id' (csv row id) and 'v1' (True if the row is in the old canonical file).
Writes data/derived/merged-corpus-canonical.v2.json and data/derived/dark/loop75_rebuild_diff.txt.
"""
import csv, json, collections, sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import parse_text, load_levels

ROOT = '/home/user/Indus-/'
F = ['cisi', 'site', 'type', 'dir.', 'time', 'period', 'phase', 'area-section', 'block-house', 'room-grid',
     'symbol', 'cult', 'material', 'shape', 'complete']


def record(r, strong, allm):
    s = parse_text(r['text'])
    x = {f: ('' if f in ('symbol', 'cult') and r[f] == 'None' else r[f]) for f in F}
    x['seq'] = [allm.get(k, k) for k in s]
    x['seq_raw'] = s
    x['seq_strong'] = [strong.get(k, k) for k in s]
    x['seq_all'] = list(x['seq'])
    return x


def objkey(r):
    return r['id'].split('.')[0]


def main():
    strong, allm = load_levels()
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    old = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    # ordered alignment: old file must be a subsequence of the csv, matched on cisi + reading-order text
    j = 0; inold = []
    for r in rows:
        if j < len(old) and old[j]['cisi'] == r['cisi'] and old[j]['seq_raw'] == parse_text(r['text']):
            inold.append(j); j += 1
        else:
            inold.append(None)
    assert j == len(old), 'old canonical is not an ordered subsequence of the csv'
    V2 = []; mism = []
    for r, k in zip(rows, inold):
        if not parse_text(r['text']):
            assert k is None
            continue
        x = record(r, strong, allm)
        if k is not None:
            o = old[k]
            for f in list(o.keys()):
                if o[f] != x[f]: mism.append((r['id'], f, o[f], x[f]))
            if set(o) != set(x): mism.append((r['id'], 'keys', sorted(o), sorted(x)))
        x['id'] = r['id']; x['v1'] = k is not None
        V2.append(x)
    json.dump(V2, open(ROOT + 'data/derived/merged-corpus-canonical.v2.json', 'w'))

    # classify every csv row not in the old file
    old_objs = {objkey(r) for r, k in zip(rows, inold) if k is not None}
    old_cisi = {o['cisi'] for o in old if o['cisi'] not in ('-', '')}
    byid = {r['id']: r for r in rows}
    out = []; cls = collections.Counter(); bysite = collections.Counter(); bytype = collections.Counter()
    for r, k in zip(rows, inold):
        if k is not None: continue
        s = parse_text(r['text'])
        enters = bool(s)
        if objkey(r) in old_objs: c = 'resplit_line'           # another line/side of an object already in v1
        elif r['cisi'] not in ('-', '') and r['cisi'] in old_cisi: c = 'resplit_cisi'   # same CISI number, new object id
        elif r['cisi'] not in ('-', ''): c = 'new_object_cisi'
        else: c = 'new_object_nocisi'
        pot = r['type'].startswith('POT')
        tag = (c + ('_pot' if pot and c.startswith('new') else ''))
        cls[(enters, tag)] += 1
        if enters: bysite[r['site']] += 1; bytype[r['type'].split(':')[0]] += 1
        ref = (r['sanskrit'] or '')
        ref = ref if ref.startswith('ref:') else ''
        refsame = ''
        if ref:
            t = byid.get(ref[4:].strip())
            refsame = 'ref-same-text' if t is not None and parse_text(t['text']) == s else 'ref-other-text'
        out.append(f"{r['id']}\t{r['cisi']}\t{r['site']}\t{r['type']}\t{r['text']}\t{'ENTERS' if enters else 'illegible(dropped)'}\t{tag}\t{ref} {refsame}")

    newpots = sum(1 for x in V2 if not x['v1'] and x['type'].startswith('POT'))
    oldpots = sum(1 for x in old if x['type'].startswith('POT'))
    csvpots = sum(1 for r in rows if r['type'].startswith('POT'))
    L = []
    L.append('Loop 75 rebuild diff: data/raw/inscriptions.csv (sha256 in data/raw/SOURCES.txt) -> merged-corpus-canonical.v2.json')
    L.append(f'csv rows {len(rows)}; old canonical {len(old)}; v2 {len(V2)} (= csv rows with >=1 legible sign)')
    L.append(f'old canonical is an ORDERED SUBSEQUENCE of the current csv: all {len(old)} old rows matched on cisi + reading-order text.')
    L.append(f'field-by-field mismatches on the {len(old)} shared rows (all 19 fields incl. seq, seq_raw, seq_strong, seq_all): {len(mism)}')
    for m in mism[:50]: L.append('  MISMATCH ' + repr(m))
    L.append(f'csv rows not in old file: {sum(1 for k in inold if k is None)}; of these illegible (all-zero text, dropped by the S268 parser as before): {sum(v for (e, _), v in cls.items() if not e)}; entering v2: {len(V2) - len(old)}')
    L.append('class counts (enters_v2, class):')
    for k, v in sorted(cls.items()): L.append(f'  {k}: {v}')
    L.append('  resplit_line = another line/side of an object id already in v1; resplit_cisi = new object id carrying a CISI number already in v1;')
    L.append('  new_object_cisi = CISI number absent from v1; new_object_nocisi = no CISI number (cisi "-") and no sibling line in v1 (a separate object, e.g. a HARP sherd)')
    L.append(f'entering rows by site: {dict(bysite.most_common())}')
    L.append(f'entering rows by object class: {dict(bytype.most_common())}')
    L.append(f'pot rows: csv {csvpots}; old canonical {oldpots}; v2 {oldpots + newpots} (+{newpots}); illegible pot rows dropped {csvpots - oldpots - newpots}')
    L.append('')
    L.append('id\tcisi\tsite\ttype\ttext\tstatus\tclass\tref')
    L += out
    open(ROOT + 'data/derived/dark/loop75_rebuild_diff.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:20]))


if __name__ == '__main__':
    main()
