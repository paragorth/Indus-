"""S-DARK-28 cycle 4: the same derived-vs-base test on Proto-Elamite (Dahl sign list).

Graphic derivations are given by the transliteration itself: variant forms X~a, X~b ... of a
base X (added marks / hatching, the analogue of Indus 'added strokes') and compounds |A+B|
(the analogue of ligatures).  For each (derived, base) pair with >= MINF tokens each we
compare slot profile (alone / first / middle / last in the entry), numeral-system profile
of the entry (capacity / sexagesimal / bisexagesimal / fraction / N23 / none), left and
right neighbours (top 60), by Jensen-Shannon divergence against 1,000 frequency-matched
random signs (freq within x2 of the base, not a form of the same base).
"""
import json, random, collections, re, time
import numpy as np
random.seed(28); np.random.seed(28)
OUT = 'data/derived/dark/'; NPERM = 1000; MINF = 5
D = json.load(open('other-scripts/proto-elamite/data/pe_corpus.json'))
CAP = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39', 'N30'}; BIS = {'N51', 'N54'}; FRAC = {'N08', 'N02'}; SDB = {'N01', 'N14', 'N34', 'N45'}
def numclass(nums):
    codes = {c for _, c in nums}
    if not codes: return 'none'
    if codes & CAP: return 'capacity'
    if codes & FRAC: return 'fraction'
    if codes & BIS: return 'bisex'
    if 'N23' in codes: return 'N23'
    if codes & SDB: return 'sdb'
    return 'other'
NC = ['none', 'capacity', 'fraction', 'bisex', 'N23', 'sdb', 'other']
POS = ['alone', 'first', 'middle', 'last']
ENT = []
for t in D:
    for l in t['lines']:
        s = [x for x in l['signs'] if x and not x.startswith('[')]
        if not s or l.get('lacuna'): continue
        ENT.append(dict(seq=s, nc=numclass(l['numerals']), surf=l['surface']))
freq = collections.Counter(x for e in ENT for x in e['seq'])
top = [w for w, _ in freq.most_common(60)]; ti = {w: i for i, w in enumerate(top)}
P = collections.defaultdict(lambda: dict(pos=np.zeros(4), nc=np.zeros(7), L=np.zeros(62), R=np.zeros(62), n=0, relpos=[], ents=set()))
for k, e in enumerate(ENT):
    s = e['seq']; n = len(s)
    for i, x in enumerate(s):
        p = P[x]; p['n'] += 1; p['ents'].add(k)
        p['pos'][0 if n == 1 else 1 if i == 0 else 3 if i == n - 1 else 2] += 1
        p['nc'][NC.index(e['nc'])] += 1
        l = s[i - 1] if i > 0 else None; r = s[i + 1] if i < n - 1 else None
        p['L'][ti.get(l, 60) if l is not None else 61] += 1; p['R'][ti.get(r, 60) if r is not None else 61] += 1
        if n > 1: p['relpos'].append(i / (n - 1))
for x, p in P.items(): p['relpos'] = float(np.mean(p['relpos'])) if p['relpos'] else np.nan
def base_of(x):
    m = re.match(r'^(M\d+[a-z]?)~', x)
    return m.group(1) if m else None
def parts_of(x):
    m = re.match(r'^\|(.+)\|$', x)
    if not m: return []
    return [p for p in re.split(r'[+x&.]', m.group(1)) if re.match(r'^M\d+', p)]
def jsd(p, q, eps=0.5):
    p = p + eps; q = q + eps; p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
EDGES = []
for x in P:
    b = base_of(x)
    if b and b in P: EDGES.append((x, b, 'variant'))
    for i, pp in enumerate(parts_of(x)):
        if pp in P and pp != x: EDGES.append((x, pp, 'compound-' + ('first' if i == 0 else 'later')))
def stem(x):
    b = base_of(x); return b or x
rep = [f'# LOOP 28 cycle 4: Proto-Elamite variants and compounds vs their bases ({time.strftime("%Y-%m-%dT%H:%M")})']
rep.append(f'entries {len(ENT)}, sign forms {len(P)}, edges {len(EDGES)} ({collections.Counter(t for _,_,t in EDGES)})')
PROFS = ['pos', 'nc', 'L', 'R']
rows = []
for x, b, t in EDGES:
    if P[x]['n'] < MINF or P[b]['n'] < MINF: continue
    fb = P[b]['n']
    pool = [r for r in P if r not in (x, b) and fb / 2 <= P[r]['n'] <= fb * 2 and stem(r) != stem(b) and stem(b) not in parts_of(r) and stem(r) not in parts_of(x)]
    if len(pool) < 5: continue
    obs = {w: jsd(P[b][w], P[x][w]) for w in PROFS}
    nul = {w: np.array([jsd(P[r][w], P[x][w]) for r in random.choices(pool, k=NPERM)]) for w in PROFS}
    pct = {w: float(np.mean(nul[w] <= obs[w])) for w in PROFS}
    co = len(P[x]['ents'] & P[b]['ents']); coexp = len(P[x]['ents']) * len(P[b]['ents']) / len(ENT)
    rows.append(dict(x=x, b=b, t=t, nx=P[x]['n'], nb=fb, pct=pct, drel=P[x]['relpos'] - P[b]['relpos'],
                     drel_null=float(np.nanmean([P[x]['relpos'] - P[r]['relpos'] for r in random.choices(pool, k=200)])),
                     dlast=P[x]['pos'][3] / P[x]['n'] - P[b]['pos'][3] / fb, cooc=co, coexp=coexp,
                     posb=POS[int(np.argmax(P[b]['pos']))], posx=POS[int(np.argmax(P[x]['pos']))],
                     ncb=NC[int(np.argmax(P[b]['nc']))], ncx=NC[int(np.argmax(P[x]['nc']))]))
rep.append(f'rows with both forms >= {MINF} tokens: {len(rows)}')
rep.append(f'{"type":15s} {"n":>3s} | mean pct of JSD(base, derived) in null:  pos   numsys  left  right | drel obs/null  dlast | cooc obs/exp')
for t in ['variant', 'compound-first', 'compound-later', 'ALL']:
    rs = [r for r in rows if t == 'ALL' or r['t'] == t]
    if not rs: continue
    pm = {w: np.mean([r['pct'][w] for r in rs]) for w in PROFS}
    sim = np.mean(np.random.rand(NPERM, len(rs)), axis=1)
    pv = {w: float(np.mean(sim <= pm[w])) for w in PROFS}
    rep.append(f'{t:15s} {len(rs):3d} | ' + ' '.join(f'{pm[w]:.2f}{"*" if pv[w] < 0.05 else " "}' for w in PROFS) +
               f' | {np.nanmean([r["drel"] for r in rs]):+.3f}/{np.nanmean([r["drel_null"] for r in rs]):+.3f} {np.mean([r["dlast"] for r in rs]):+.3f} | {sum(r["cooc"] for r in rs)}/{sum(r["coexp"] for r in rs):.1f}')
    rep.append(f'   share of pairs where the derived form keeps the base\'s modal slot: {np.mean([r["posb"] == r["posx"] for r in rs]):.2f}; keeps the base\'s modal numeral system: {np.mean([r["ncb"] == r["ncx"] for r in rs]):.2f}')
rep.append('detail (derived <- base type n_x n_b | pct pos nc L R | slot base->derived | numsys base->derived | drel):')
for r in sorted(rows, key=lambda r: (r['t'], -r['nx'])):
    rep.append(f"  {r['x']:16s} <- {r['b']:8s} {r['t']:14s} {r['nx']:4d} {r['nb']:4d} | {r['pct']['pos']:.2f} {r['pct']['nc']:.2f} {r['pct']['L']:.2f} {r['pct']['R']:.2f} | {r['posb']}->{r['posx']} | {r['ncb']}->{r['ncx']} | {r['drel']:+.2f}")
open(OUT + 'loop28_cycle4_pe.txt', 'w').write('\n'.join(rep) + '\n')
print('\n'.join(rep[:20]))
