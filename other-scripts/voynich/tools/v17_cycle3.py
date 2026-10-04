"""v17 cycle 3: one exemplar or several? and is the 'seam' more than the junction rule?

Part A: junction-preserving resynthesis (composed directly, no exemplar): each next word is drawn from
real mid-line successors of words ending in the same glyph (line-final slot drawn from real
line-final successors), line-initial words from the real line-initial pool. Compare the 'seam'
(FIN of a word x INIT of the next) and the ghost-site rates; use it as an extra null for the grid.
Part B: several exemplars. Per quire and per hand, the best exemplar width (and phase model) is chosen
on one half of the folios and its score is read on the other half (exhaustive search over all widths,
both units, both phase models; held-out retest). Held-out Z summed over groups and folds, plus
the agreement of the two halves' chosen widths, against line-order-shuffle and junction nulls.
Positive control: the fixed-width-32 planted copy and the real-layout copy (cycle 1 B and A).
Part C: replication on the IT2a transcription (ghost channel, glyph units, line-order null).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v17_lib import *
from multiprocessing import Pool
import bisect

NREP = int(os.environ.get('NREP', 80))

class JunctionGen:
    def __init__(self, paras):
        self.pf = defaultdict(Counter); self.li = defaultdict(Counter)
        self.mid = defaultdict(lambda: defaultdict(Counter)); self.end = defaultdict(lambda: defaultdict(Counter))
        for p in paras:
            s = p['sec']
            for li, ws in enumerate(p['lines']):
                (self.pf if li == 0 else self.li)[s][ws[0]] += 1
                for k in range(1, len(ws)):
                    a = G(ws[k - 1]); key = a[-1] if a else ''
                    (self.end if k == len(ws) - 1 else self.mid)[s][key][ws[k]] += 1
        self.cache = {}
    def _draw(self, rng, c, key):
        if key not in self.cache: self.cache[key] = (list(c.keys()), list(np.cumsum(list(c.values()))))
        ks, cs = self.cache[key]
        return ks[bisect.bisect_right(cs, rng.random() * cs[-1])]
    def gen(self, paras, rng):
        out = []
        for p in paras:
            s = p['sec']; Ls = []
            for li, ws in enumerate(p['lines']):
                src = self.pf[s] if li == 0 else self.li[s]
                w = self._draw(rng, src, ('pf' if li == 0 else 'li', s)); line = [w]
                for k in range(1, len(ws)):
                    a = G(w); key = a[-1] if a else ''
                    tab = self.end if k == len(ws) - 1 else self.mid
                    c = tab[s].get(key) or self.mid[s].get(key) or self.li[s]
                    w = self._draw(rng, c, (k == len(ws) - 1, s, key)); line.append(w)
                Ls.append(line)
            q = dict(p); q['lines'] = Ls; out.append(q)
        return out

V = voynich_paras(); TV = crossfit_tables(V); JG = JunctionGen(V)
init_pool = [ws[0] for p in V for ws in p['lines'][1:]]
fin_pool = [ws[-1] for p in V for ws in p['lines'][:-1]]
PCS = {'PC_voy_fixed32': copy_from_exemplar(V, random.Random(2), width_factor=1.4, fixed_width=32, plant_init=init_pool,
                                            plant_fin=fin_pool, p_ditto=0.03, p_skip=0.03)[0],
       'PC_voy_ex_w1.30': copy_from_exemplar(V, random.Random(1), width_factor=1.3, p_ditto=0.05, p_skip=0.05)[0]}

def sites(paras):
    gt = gap_table(paras, TV); mid = ~gt['brk']
    return {'rep_rate': float(gt['rep'][mid].mean()), 'seam_corr': float(np.corrcoef(gt['fin'][mid], gt['init'][mid])[0, 1]),
            'init_hi': float((gt['init'][mid] > 1.0).mean()), 'fin_hi': float((gt['fin'][mid] > 1.0).mean())}

def heldout(paras, groupkey):
    """exhaustive width search per group on one fold, retest on the other."""
    gt = gap_table(paras, TV); W = residual(gt)
    grp = np.array([str(paras[i][groupkey]) for i in gt['para']])
    w = W['ghost']; tot = []; agree = []; picks = {}
    for g in sorted(set(grp)):
        best = {}
        for f in (0, 1):
            m = (grp == g) & (gt['fold'] == f) & (w != 0)
            if m.sum() < 150: break
            for unit in ('g', 'w'):
                grid = LG if unit == 'g' else LW
                x = gt['xg'][m] if unit == 'g' else gt['xw'][m].astype(float)
                A, R = periodogram(x, w[m], grid)
                # standardise R (Exp(1) under null) to a z-like score: sqrt(2R) - 1 not needed; use A and sqrt(R)
                best[(f, unit, 'A')] = (grid, A); best[(f, unit, 'R')] = (grid, np.sqrt(R))
        else:
            # choose on fold f (training), read on fold 1-f
            for f in (0, 1):
                cands = [(best[(f, u, ph)][1].max(), u, ph, int(best[(f, u, ph)][1].argmax())) for u in ('g', 'w') for ph in ('A', 'R')]
                cands.sort(reverse=True); _, u, ph, i = cands[0]
                grid, sc = best[(1 - f, u, ph)]
                v = sc[i] if ph == 'A' else (sc[i] ** 2 - 1.0)   # A ~ N(0,1); R - 1 has mean 0, sd 1
                tot.append(float(v)); picks[(g, f)] = (u, ph, float(grid[i]))
            a, b = picks[(g, 0)], picks[(g, 1)]
            if a[0] == b[0]:
                agree.append(abs(a[2] - b[2]) / (0.5 * (a[2] + b[2])))
    return {'Z': float(np.sum(tot) / math.sqrt(max(1, len(tot)))), 'n': len(tot),
            'agree_med': float(np.median(agree)) if agree else None, 'picks': {'%s|%d' % k: v for k, v in picks.items()}}

def gridstat(paras, tabs):
    gt = gap_table(paras, tabs); W = residual(gt)
    grid, res = scan(gt, W, 'ghost', 'g')
    return summarize(grid, res)

def jobB(args):
    src, kind, seed = args
    base = V if src == 'VOYNICH' else PCS[src]
    rng = random.Random(seed)
    p = base if kind == 'obs' else null_lshuf(base, rng) if kind == 'lshuf' else JG.gen(base, rng)
    out = {gk: heldout(p, gk) for gk in ('quire', 'hand')}
    if src == 'VOYNICH' and kind != 'lshuf': out['grid'] = gridstat(p, TV)
    return src, kind, seed, out

def jobC(args):
    kind, seed = args
    VI = jobC.VI
    p = VI if kind == 'obs' else null_lshuf(VI, random.Random(seed))
    return kind, seed, gridstat(p, jobC.TI)

def main():
    t0 = time.time(); out = {}
    # A
    obs = sites(V)
    nj = [sites(JG.gen(V, random.Random(300 + i))) for i in range(20)]
    out['A'] = {'obs': obs, 'junction_null': {k: (float(np.mean([d[k] for d in nj])), float(np.std([d[k] for d in nj]))) for k in obs}}
    print('A', json.dumps(out['A']), flush=True)
    # B
    jobs = [(s, 'obs', 0) for s in ['VOYNICH'] + list(PCS)]
    jobs += [('VOYNICH', k, 9000 + i) for k in ('lshuf', 'junction') for i in range(NREP)]
    jobs += [(s, 'lshuf', 9500 + i) for s in PCS for i in range(30)]
    with Pool(2) as pool: R = pool.map(jobB, jobs, chunksize=4)
    agg = defaultdict(lambda: defaultdict(list))
    for src, kind, seed, o in R: agg[src][kind].append(o)
    resB = {}
    for src, d in agg.items():
        o = d['obs'][0]; r = {}
        for gk in ('quire', 'hand'):
            e = {'Z': o[gk]['Z'], 'n': o[gk]['n'], 'agree_med': o[gk]['agree_med']}
            for kind in ('lshuf', 'junction'):
                if kind not in d: continue
                zs = [n[gk]['Z'] for n in d[kind]]; ag = [n[gk]['agree_med'] for n in d[kind] if n[gk]['agree_med'] is not None]
                e['p_' + kind] = (1 + sum(z >= o[gk]['Z'] for z in zs)) / (1 + len(zs))
                e['nullZ_' + kind] = (float(np.mean(zs)), float(np.std(zs)))
                if o[gk]['agree_med'] is not None and ag:
                    e['p_agree_' + kind] = (1 + sum(a <= o[gk]['agree_med'] for a in ag)) / (1 + len(ag))
            e['picks'] = o[gk]['picks']; r[gk] = e
        if 'grid' in o and 'junction' in d:
            g = {}
            for sec, v in o['grid'].items():
                na = [n['grid'][sec]['Amax'] for n in d['junction'] if sec in n['grid']]
                g[sec] = {'Amax': v['Amax'], 'LA': v['LA'], 'pA_junction': (1 + sum(a >= v['Amax'] for a in na)) / (1 + len(na))}
                if 'Rmax' in v:
                    nr = [n['grid'][sec]['Rmax'] for n in d['junction'] if sec in n['grid']]
                    g[sec].update({'Rmax': v['Rmax'], 'LR': v['LR'], 'pR_junction': (1 + sum(a >= v['Rmax'] for a in nr)) / (1 + len(nr))})
            r['grid_vs_junction_null'] = g
        resB[src] = r
        print('B', src, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'picks'} for k, v in r.items()}), flush=True)
    out['B'] = resB
    save('cycle3.json', out)
    # C: IT2a replication
    VI = voynich_paras('IT2a'); jobC.VI = VI; jobC.TI = crossfit_tables(VI)
    jobs = [('obs', 0)] + [('lshuf', 9900 + i) for i in range(60)]
    with Pool(2) as pool: R = pool.map(jobC, jobs, chunksize=4)
    o = [s for k, _, s in R if k == 'obs'][0]; N = [s for k, _, s in R if k != 'obs']
    rc = {}
    for sec, v in o.items():
        na = [n[sec]['Amax'] for n in N if sec in n]
        rc[sec] = {'Amax': v['Amax'], 'LA': v['LA'], 'pA': (1 + sum(a >= v['Amax'] for a in na)) / (1 + len(na))}
        if 'Rmax' in v:
            nr = [n[sec]['Rmax'] for n in N if sec in n]
            rc[sec].update({'Rmax': v['Rmax'], 'LR': v['LR'], 'pR': (1 + sum(a >= v['Rmax'] for a in nr)) / (1 + len(nr))})
    out['C'] = rc; print('C', json.dumps(rc), flush=True)
    save('cycle3.json', out)
    print('done %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()
