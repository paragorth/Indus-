"""v27 cycle 2 post-analysis: cross-transcription agreement of the whole-book paths (ZL3b vs IT2a, cards matched
by folio + line number), and bifolio enrichment of cross-page links among same-section pairs."""
import sys, os, json, glob, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_lib as L
import v8_lib

HDR = v8_lib.page_headers()
R = {tuple(json.load(open(f))['job']): json.load(open(f)) for f in glob.glob(os.path.join(L.CK, 'c2_*.json'))}


def edges(r):
    c = [tuple(x) for x in r['cids']]
    return c, set(zip(c, c[1:]))


def secmap(name):
    C = L.voynich_pages(name)
    return {p['id']: p['sec'] for p in C}


out = {}
if ('ZL', 'real', 0) in R and ('IT2a', 'real', 0) in R:
    cz, ez = edges(R[('ZL', 'real', 0)]); ci, ei = edges(R[('IT2a', 'real', 0)])
    common = set(cz) & set(ci)
    ez2 = {e for e in ez if e[0] in common and e[1] in common}
    ei2 = {e for e in ei if e[0] in common and e[1] in common}
    sh = len(ez2 & ei2); und = len({frozenset(e) for e in ez2} & {frozenset(e) for e in ei2})
    out['cross_transcription'] = dict(common_cards=len(common), zl_edges=len(ez2), it_edges=len(ei2), shared_directed=sh,
                                      shared_undirected=und, chance=round(len(ez2) * len(ei2) / (len(common) ** 2), 2),
                                      restart_jaccard_ZL=R[('ZL', 'real', 0)]['restart_jaccard'],
                                      jaccard=round(sh / len(ez2 | ei2), 4))

sm = secmap('ZL3b')
for key, r in sorted(R.items()):
    if key[0] not in ('ZL', 'IT2a', 'PL'): continue
    c = [tuple(x) for x in r['cids']]
    pg = lambda x: x[0]
    bif = lambda x: HDR.get(x[0], {}).get('B')
    E = list(zip(c, c[1:]))
    cross_ss = [(a, b) for a, b in E if pg(a) != pg(b) and sm.get(pg(a)) == sm.get(pg(b))]
    hit = sum(1 for a, b in cross_ss if bif(a) and bif(a) == bif(b))
    # expectation: random cross-page same-section pairs
    rng = random.Random(0); cards = c
    bysec = {}
    for x in cards: bysec.setdefault(sm.get(pg(x)), []).append(x)
    tot = ok = 0
    for _ in range(200000):
        s = rng.choice(list(bysec)); L2 = bysec[s]
        if len(L2) < 2: continue
        a, b = rng.choice(L2), rng.choice(L2)
        if pg(a) == pg(b): continue
        tot += 1; ok += (bif(a) is not None and bif(a) == bif(b))
    # adjacent pages (consecutive in binding order) as a second comparison
    order = {f: i for i, f in enumerate(HDR)}
    adj = sum(1 for a, b in cross_ss if abs(order.get(pg(a), -9) - order.get(pg(b), 99)) == 1)
    out[str(key)] = dict(cross_same_sec=len(cross_ss), bifolio=hit, bifolio_rate=round(hit / max(1, len(cross_ss)), 4),
                         bifolio_base=round(ok / max(1, tot), 4), adjacent_page=adj,
                         same_page=r['m']['same_page'], same_sec=r['m']['same_sec'], jac=r['restart_jaccard'],
                         stable=r['stable_edges'], stable_same_page=r['stable_same_page'],
                         written_hits=r['m']['written_hits'], written_total=r['m']['written_total'],
                         stable_hidden=r.get('stable_hidden'), edge_mean=r['m']['edge_mean'])
for k, v in out.items(): print(k, v)
json.dump(out, open(os.path.join(L.CK, 'c2_report.json'), 'w'))
