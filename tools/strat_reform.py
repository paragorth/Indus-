"""Step-reform vs drift test of sign inventories across stratigraphic phases.

Hypothesis under test: the sign system was an administrative code reformed in steps, so whole sets of
signs (closers, openers, numerals) appear or vanish together at one phase boundary. A language's
vocabulary turns over gradually, sign by sign.

Series (site x object group x dating scheme), phases oldest first:
  Harappa HARP tablets          3B | 3B/C | 3C          (phase field, period 3; 'B?' -> B)
  Harappa Vats strata tablets   VI+VII | V | IV | III | II | I  (Stratum I is the top, latest)
  Harappa Vats strata seals     same
  Mohenjo-daro seals            Early | Int III | Int II | Int I | Late III | Late II | Late I(+IA/IB)
  Mohenjo-daro seals coarse     Early | Intermediate | Late
  Dholavira seals               4 | 5 | 6
Tests: (a) sign x phase tables and coverage; (b) switches (present->absent or absent->present at a
boundary) for signs with n>=8 in the series, concentration index = share of switches at the biggest
boundary, vs (i) phase labels shuffled within the series, (ii) birth-death drift simulation (log-
frequency random walk, step sd calibrated to the observed number of switches); (c) switching signs by
slot (modal slot in parsed_texts.json) vs slot labels shuffled; (d) co-switching of co-occurring sign
pairs vs profile permutation; (e) comparison with S95 (voucher 4->3), S49 (tablet lengthening), S45/S175
(Dholavira stage 6). Run on seq (strong+probable merges), seq_strong and seq_raw.
Output: data/derived/strat_reform.txt
"""
import json, collections, random, math, sys
import numpy as np

random.seed(7); np.random.seed(7)
C = json.load(open('data/derived/merged-corpus-canonical.json'))
P = json.load(open('data/derived/parsed_texts.json'))
BR = json.load(open('data/derived/bridge_extended.json'))
OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)

# modal slot per sign (Wells numbers)
slotc = collections.defaultdict(collections.Counter)
for o in P:
    for s, l in zip(o['seq'], o['slots']): slotc[s][l] += 1
SLOT = {s: c.most_common(1)[0][0] for s, c in slotc.items()}
def M(w):
    m = BR.get(str(w)); return 'M' + '/'.join(str(x) for x in m) if m else 'M?'

def hp(o):  # Harappa HARP phase
    ph = o['phase'].strip()
    if o['site'] != 'Harappa' or o['period'] != '3': return None
    return {'B': '3B', 'B?': '3B', 'B/C': '3B/C', 'C': '3C'}.get(ph)
def hs(o):  # Harappa Vats strata
    ph = o['phase'].strip()
    if o['site'] != 'Harappa' or not ph.startswith('Stratum'): return None
    r = ph.split()[1]
    return {'VII': 'VI+VII', 'VI': 'VI+VII', 'V': 'V', 'IV': 'IV', 'III': 'III', 'II': 'II', 'I': 'I'}.get(r)
def md(o):
    if o['site'] != 'Mohenjo-daro': return None
    per = o['period'].strip(); ph = o['phase'].strip()
    if per == 'Early': return 'Early'
    if per == 'Intermediate' and ph in ('I', 'II', 'III'): return 'Int ' + ph
    if per == 'Late' and ph in ('I', 'IA', 'IB', 'II', 'III'): return 'Late ' + ('I' if ph.startswith('I') and ph != 'II' and ph != 'III' else ph)
    return None
def mdc(o):
    if o['site'] != 'Mohenjo-daro': return None
    per = o['period'].strip(); return per if per in ('Early', 'Intermediate', 'Late') else None
def dh(o):
    if o['site'] != 'Dholavira': return None
    return o['period'].strip() if o['period'].strip() in ('4', '5', '6') else None

SERIES = [
    ('Harappa HARP tablets', 'TAB', hp, ['3B', '3B/C', '3C']),
    ('Harappa HARP all objects', None, hp, ['3B', '3B/C', '3C']),
    ('Harappa Vats-strata tablets', 'TAB', hs, ['VI+VII', 'V', 'IV', 'III', 'II', 'I']),
    ('Harappa Vats-strata seals', 'SEAL', hs, ['VI+VII', 'V', 'IV', 'III', 'II', 'I']),
    ('Mohenjo-daro seals (7 levels)', 'SEAL', md, ['Early', 'Int III', 'Int II', 'Int I', 'Late III', 'Late II', 'Late I']),
    ('Mohenjo-daro seals (3 periods)', 'SEAL', mdc, ['Early', 'Intermediate', 'Late']),
    ('Mohenjo-daro all objects (3 periods)', None, mdc, ['Early', 'Intermediate', 'Late']),
    ('Dholavira seals', 'SEAL', dh, ['4', '5', '6']),
]
NPERM = 2000; NMIN = 8

def table(texts, phases, key):
    """sign -> counts per phase (list), token totals per phase"""
    tab = collections.defaultdict(lambda: [0] * len(phases)); tot = [0] * len(phases)
    pi = {p: i for i, p in enumerate(phases)}
    for ph, seq in texts:
        for s in seq: tab[s][pi[ph]] += 1; tot[pi[ph]] += 1
    return tab, tot

def switches(tab, signs):
    """per sign: list of (boundary index, direction) where presence flips"""
    sw = {}
    for s in signs:
        pres = [c > 0 for c in tab[s]]
        ev = [(b, '+' if pres[b + 1] else '-') for b in range(len(pres) - 1) if pres[b] != pres[b + 1]]
        if ev: sw[s] = ev
    return sw

def conc(sw, nb):
    per = [0] * nb
    for ev in sw.values():
        for b, d in ev: per[b] += 1
    tot = sum(per)
    return per, (max(per) / tot if tot else float('nan'))

def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float); p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    def kl(a, b):
        mask = a > 0; return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def run_series(name, tgroup, keyf, phases, field):
    texts = [(keyf(o), o[field]) for o in C if keyf(o) and (tgroup is None or o['type'].split(':')[0] == tgroup) and o[field]]
    texts = [(p, s) for p, s in texts if p in phases]
    nb = len(phases) - 1
    tab, tot = table(texts, phases, keyf)
    ntext = collections.Counter(p for p, _ in texts)
    say(f'\n=== {name} [{field}] ===')
    say('  texts per phase: ' + ', '.join(f'{p} {ntext[p]}' for p in phases) + f' (total {len(texts)})')
    say('  tokens per phase: ' + ', '.join(f'{p} {t}' for p, t in zip(phases, tot)))
    signs = [s for s in tab if sum(tab[s]) >= NMIN]
    say(f'  signs with n>={NMIN}: {len(signs)} (distinct signs overall {len(tab)})')
    if len(texts) < 60 or len(signs) < 10:
        say('  too small for the switch test; coverage only'); return None
    sw = switches(tab, signs)
    per, ci = conc(sw, nb)
    nswitch = sum(per)
    say(f'  switches per boundary: ' + ', '.join(f'{phases[b]}->{phases[b+1]} {per[b]}' for b in range(nb)) + f'; total {nswitch}; signs switching {len(sw)}/{len(signs)}')
    say(f'  concentration index (share at biggest boundary): {ci:.3f}')
    # JSD between adjacent phases (signs n>=8 only)
    mat = np.array([tab[s] for s in signs], float)
    js = [jsd(mat[:, b], mat[:, b + 1]) for b in range(nb)]
    say('  JSD adjacent phases: ' + ', '.join(f'{phases[b]}->{phases[b+1]} {js[b]:.3f}' for b in range(nb)))
    # (i) shuffle phase labels within series
    labs = [p for p, _ in texts]; seqs = [s for _, s in texts]
    nullc = []; nulln = []; nulljmax = []; nullper = []
    for _ in range(NPERM):
        random.shuffle(labs)
        t2, _t = table(list(zip(labs, seqs)), phases, None)
        sw2 = switches(t2, signs); per2, c2 = conc(sw2, nb)
        nullc.append(c2); nulln.append(sum(per2)); nullper.append(per2)
        m2 = np.array([t2[s] for s in signs], float)
        nulljmax.append(max(jsd(m2[:, b], m2[:, b + 1]) for b in range(nb)))
    nullc = np.array(nullc); nulln = np.array(nulln); nullper = np.array(nullper)
    p_conc = (np.sum(nullc >= ci) + 1) / (NPERM + 1)
    p_n = (np.sum(nulln >= nswitch) + 1) / (NPERM + 1)
    say(f'  (i) shuffle control: switches {nulln.mean():.1f} (95% {np.percentile(nulln,2.5):.0f}-{np.percentile(nulln,97.5):.0f}), real {nswitch}, p(real>=null) = {p_n:.3f}')
    say(f'      per-boundary expected: ' + ', '.join(f'{phases[b]}->{phases[b+1]} {nullper[:,b].mean():.1f}' for b in range(nb)))
    say(f'      concentration null {np.nanmean(nullc):.3f}, real {ci:.3f}, p = {p_conc:.3f}; max-JSD null {np.mean(nulljmax):.3f}, real {max(js):.3f}, p = {(np.sum(np.array(nulljmax)>=max(js))+1)/(NPERM+1):.3f}')
    # (ii) drift simulation: log-frequency random walk per sign, multinomial draws with real phase token totals
    base = mat.sum(1) / mat.sum()
    def drift(sigma, reps=400):
        cs, ns, pers = [], [], []
        for _ in range(reps):
            w = np.cumsum(np.random.normal(0, sigma, (len(signs), len(phases))), axis=1)
            f = base[:, None] * np.exp(w); f /= f.sum(0, keepdims=True)
            t2 = {}
            draws = [np.random.multinomial(tot[k], f[:, k]) for k in range(len(phases))]
            for i, s in enumerate(signs): t2[s] = [int(d[i]) for d in draws]
            sw2 = switches(t2, signs); per2, c2 = conc(sw2, nb); cs.append(c2); ns.append(sum(per2)); pers.append(per2)
        return np.array(cs), np.array(ns), np.array(pers)
    best = None
    for sigma in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        cs, ns, pers = drift(sigma)
        d = abs(ns.mean() - nswitch)
        if best is None or d < best[0]: best = (d, sigma, cs, ns, pers)
    d, sigma, cs, ns, pers = best
    say(f'  (ii) drift simulation calibrated at sigma={sigma}: switches {ns.mean():.1f} (real {nswitch}); concentration null {np.nanmean(cs):.3f}, p = {(np.sum(cs>=ci)+1)/(len(cs)+1):.3f}')
    say(f'       drift per-boundary expected: ' + ', '.join(f'{phases[b]}->{phases[b+1]} {pers[:,b].mean():.1f}' for b in range(nb)))
    # (c) switching signs at the biggest boundary, by slot
    bmax = int(np.argmax(per))
    swb = {s: d for s, ev in sw.items() for b, d in ev if b == bmax}
    say(f'  biggest boundary {phases[bmax]}->{phases[bmax+1]}: {len(swb)} switching signs')
    for s in sorted(swb, key=lambda s: -sum(tab[s])):
        say(f'    W{s} ({M(s)}, {SLOT.get(s,"?")}) {swb[s]} counts {tab[s]} n={sum(tab[s])}')
    # excess at the biggest boundary over shuffle expectation, by sign
    slots_all = [SLOT.get(s, '?') for s in signs]
    obs_slot = collections.Counter(SLOT.get(s, '?') for s in swb)
    allsw = {s for s, ev in sw.items()}
    say(f'  (c) slot of switching signs (any boundary) vs slot labels shuffled among the {len(signs)} tested signs:')
    obs_any = collections.Counter(SLOT.get(s, '?') for s in allsw)
    base_slot = collections.Counter(slots_all)
    for sl in ['OPENER', 'MARKER', 'TITLE', 'CLOSER', 'SUFFIX', 'COUNT', 'NAME']:
        if base_slot[sl] == 0: continue
        null = []
        for _ in range(NPERM):
            random.shuffle(slots_all)
            idx = {s: slots_all[i] for i, s in enumerate(signs)}
            null.append(sum(1 for s in allsw if idx[s] == sl))
        null = np.array(null)
        say(f'      {sl}: {obs_any[sl]}/{base_slot[sl]} signs switch (expected {null.mean():.1f}; p_high {(np.sum(null>=obs_any[sl])+1)/(NPERM+1):.3f}, p_low {(np.sum(null<=obs_any[sl])+1)/(NPERM+1):.3f})')
    # (d) co-switching: pairs with identical switch profile; co-occurring pairs vs not
    prof = {s: tuple(ev) for s, ev in sw.items()}
    co = collections.Counter()
    for ph, seq in texts:
        u = sorted(set(seq) & set(sw))
        for i in range(len(u)):
            for j in range(i + 1, len(u)): co[(u[i], u[j])] += 1
    ss = sorted(sw)
    pairs = [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]]
    def same(prof):
        n_co = sum(1 for a, b in pairs if co[(a, b)] >= 2 and prof[a] == prof[b])
        n_nc = sum(1 for a, b in pairs if co[(a, b)] < 2 and prof[a] == prof[b])
        return n_co, n_nc
    oc, onc = same(prof)
    nco = sum(1 for a, b in pairs if co[(a, b)] >= 2); nnc = len(pairs) - nco
    vals = list(prof.values()); null = []
    for _ in range(NPERM):
        random.shuffle(vals); pr = dict(zip(ss, vals)); null.append(same(pr)[0])
    null = np.array(null)
    say(f'  (d) co-switching: pairs of switching signs with the same profile: co-occurring (>=2 texts) {oc}/{nco}, not co-occurring {onc}/{nnc}; profile permutation expected {null.mean():.1f}, p = {(np.sum(null>=oc)+1)/(NPERM+1):.3f}')
    return dict(name=name, field=field, phases=phases, per=per, ci=ci, p_conc=p_conc, nswitch=nswitch, exp=nulln.mean(), p_n=p_n, js=js, bmax=bmax, swb=swb, tab=tab, sigma=sigma, p_drift=(np.sum(cs>=ci)+1)/(len(cs)+1))

results = {}
for field in ['seq', 'seq_strong', 'seq_raw']:
    say(f'\n##################### field {field} #####################')
    for name, tg, kf, ph in SERIES:
        r = run_series(name, tg, kf, ph, field)
        if r: results[(name, field)] = r

# (e) known changes
say('\n=== (e) cross-check with known changes ===')
say('S95: voucher value 4->3 across HARP 3A/B -> 3C (moulded tablets mainly; S95b weakened).')
say('S49: tablet texts lengthen 3A/B -> 3C; jar-final rate flat.')
say('S45/S175/S181: Dholavira stage 6 drops the jar closer and long texts; sign inventory not shrunk.')
for (name, field), r in results.items():
    if field != 'seq': continue
    say(f'  {name}: biggest boundary {r["phases"][r["bmax"]]}->{r["phases"][r["bmax"]+1]} with {r["per"][r["bmax"]]} of {r["nswitch"]} switches (expected total under shuffle {r["exp"]:.1f}); concentration p (shuffle) {r["p_conc"]:.3f}, p (drift) {r["p_drift"]:.3f}')

say('\n=== Summary across fields ===')
for (name, field), r in results.items():
    say(f'  {name:40s} {field:11s} switches {r["nswitch"]:3d} (exp {r["exp"]:5.1f}, p {r["p_n"]:.3f}) conc {r["ci"]:.2f} p_shuf {r["p_conc"]:.3f} p_drift {r["p_drift"]:.3f} JSDmax {max(r["js"]):.3f} at {r["phases"][int(np.argmax(r["js"]))]}->{r["phases"][int(np.argmax(r["js"]))+1]}')

open('data/derived/strat_reform.txt', 'w').write('\n'.join(OUT) + '\n')
