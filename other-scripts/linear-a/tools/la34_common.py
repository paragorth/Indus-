#!/usr/bin/env python3
"""la34 shared loading: SigLA occurrence index -> lineara.xyz documents, tablet units (two sides of one
tablet merged), site, support, published scribe attribution (lineara.xyz 'scribe' field, used only as a
held-out label, never inside clustering)."""
import os, re, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la34_ckpt')
IMG = os.environ.get('LA34_IMG', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/la34crop')
GREEK = str.maketrans({'α': '', 'β': '', 'γ': '', 'δ': ''})


def fname(src):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', src.split('document/', 1)[-1])


def corpus():
    return {d['id']: d for d in json.load(open(os.path.join(D, 'corpus.json')))}


def sigla_to_id(name, cid):
    k = re.sub(r'\s', '', name).translate(GREEK)
    for cand in (k, k.lower() if k.startswith('HT154') else k, k.replace('HT123', 'HT123+124'),
                 re.sub(r'a$', 'r', k) if k.startswith('GO') else k, re.sub(r'b$', 'v', k) if k.startswith('GO') else k,
                 k[:4] + k[4:].lower(), k.replace('-latus', '')):
        if cand in cid: return cand
    return None


def unit_of(did, allids):
    m = re.match(r'(.*\d)([ab]|[rv])$', did)
    if m and not did.startswith('HT154'):
        b = m.group(1)
        if (b + 'c') not in allids: return b
    return did


def site_of(name):
    m = re.match(r'([A-Z]+)', name.replace(' ', ''))
    return m.group(1) if m else 'X'


def load():
    """returns occ (list with doc_id, unit, site) and unit meta {unit: {site, scribe, support, docs}}"""
    cid = corpus()
    occ = json.load(open(os.path.join(CK, 'occ_index.json')))
    allids = set(cid)
    meta = {}
    for o in occ:
        did = sigla_to_id(o['doc'], cid)
        o['did'] = did
        u = unit_of(did, allids) if did else o['doc'].replace(' ', '')
        o['unit'] = u
        o['site'] = site_of(o['doc'])
        o['file'] = os.path.join(IMG, fname(o['src']))
        m = meta.setdefault(u, {'site': o['site'], 'scribes': set(), 'support': '', 'docs': set()})
        m['docs'].add(o['doc'])
        if did:
            if cid[did].get('scribe'): m['scribes'].add(cid[did]['scribe'])
            m['support'] = cid[did].get('support', '')
    for u, m in meta.items():
        m['scribe'] = sorted(m['scribes'])[0] if len(m['scribes']) == 1 else ''
        m['docs'] = sorted(m['docs']); m['scribes'] = sorted(m['scribes'])
    return occ, meta, cid


if __name__ == '__main__':
    occ, meta, cid = load()
    print('occ', len(occ), 'matched', sum(o['did'] is not None for o in occ), 'units', len(meta))
    sc = collections.Counter(m['scribe'] for m in meta.values() if m['scribe'])
    print('units with scribe', sum(sc.values()), 'scribes', len(sc), 'scribes with >=2 units', sum(v >= 2 for v in sc.values()),
          'units in those', sum(v for v in sc.values() if v >= 2))
    print(sorted([o['doc'] for o in occ if o['did'] is None and o['n'] == 1])[:60])
