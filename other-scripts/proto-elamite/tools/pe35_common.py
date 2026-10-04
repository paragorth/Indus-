"""pe35: rows for the enlarged seal set = pe33 (Legrain) primary rows + pe35 (Louvre/Amiet) rows.
Each row: id, vol, seal, motif (set), content (pe33 content features), size, t, src ('pe33' or 'pe35')."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe33_common import load as load33, content, size_bin, pe18_tablets  # noqa
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe35_ckpt')
os.makedirs(CK, exist_ok=True)


def load(which='all'):
    """which: 'all' (pe33 + new), 'new' (pe35 tablets not in pe33), 'old' (pe33 only)."""
    rows33, T = load33(True)
    for r in rows33:
        r['src'] = 'pe33'
    byid = {t['id']: t for t in T}
    M = json.load(open(os.path.join(DATA, 'pe35_seal_motifs.json')))['tablets']
    have = {r['id'] for r in rows33}
    new = []
    for m in M:
        if m['in_pe33'] or m['id'] in have:
            continue
        t = byid.get(m['id'])
        if t is None:
            continue
        new.append({'id': m['id'], 'vol': t['vol'], 'seal': m['seal'], 'motif': set(m['features']),
                    'content': content(t), 'size': size_bin(t), 't': t, 'src': 'pe35'})
    if which == 'new':
        return new, T
    if which == 'old':
        return rows33, T
    return rows33 + new, T
