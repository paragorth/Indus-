"""v17 cycle 1: candidate ghost sites + calibration of the exemplar-grid search.

(1) Candidate sites on the real text: adjacent repeats and a-b-a-b repeats mid-line (dittography
    candidates), mid-line line-initial-like and line-final-like words, and the 'ghost seam'
    (a line-final-like word followed by a line-initial-like word, as at an exemplar break) against
    within-line shuffle and a composed-directly Markov resynthesis.
(2) Positive controls: Voynich paragraphs used as the exemplar (true line breaks known), copied with
    dittography + eye-skip and re-lineated at 0.75x or 1.3x width; Voynich text re-laid at a fixed
    exemplar width 32 with planted line-initial / line-final words; Latin (Caesar) exemplar width 38
    with dittography only (REP channel). Negative controls: Markov resynthesis (no exemplar), Latin
    with dittography at random positions.
Detection statistic: max over L of the paragraph-anchored cosine score A(L) and of the free-phase
power R(L), pooled (ALL) and per section, against line-order-shuffle and within-line-shuffle nulls.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v17_lib import *
from multiprocessing import Pool

NREP = int(os.environ.get('NREP', 60))

def latin_paras():
    lines = vlib.load_ref('Latin-Caesar')
    words = [w for L in lines for w in L['words']]
    paras = []; i = 0; k = 0
    while i < len(words) - 40:
        n = 60 + (k * 37) % 80   # paragraph of 60..140 words
        flat = words[i:i + n]; i += n
        paras.append({'folio': 'L%d' % k, 'sec': 'LA', 'fold': k % 2, 'lines': [flat]}); k += 1
        if i > 34000: break
    return paras

def stat(paras, tabs, unit, chans, secs_true=None):
    gt = gap_table(paras, tabs, unit); W = residual(gt, chans)
    grid, res = scan(gt, W, 'ghost', 'g')
    return summarize(grid, res), (grid, res)

def job(args):
    name, paras, tabs, unit_name, chans, kind, seed = args
    unit = G if unit_name == 'G' else list
    rng = random.Random(seed)
    if kind == 'lshuf': p2 = null_lshuf(paras, rng)
    elif kind == 'wshuf': p2 = null_wshuf(paras, rng)
    else: p2 = paras
    s, _ = stat(p2, tabs, unit, chans)
    return name, kind, seed, s

def main():
    t0 = time.time()
    V = voynich_paras()
    TV = crossfit_tables(V)
    rng = random.Random(17)
    out = {}
    # ---- (1) candidate sites ----
    def sites(paras):
        gt = gap_table(paras, TV); mid = ~gt['brk']
        return {'n_mid': int(mid.sum()), 'rep_rate': float(gt['rep'][mid].mean()),
                'seam_corr': float(np.corrcoef(gt['fin'][mid], gt['init'][mid])[0, 1]),
                'init_hi': float((gt['init'][mid] > 1.0).mean()), 'fin_hi': float((gt['fin'][mid] > 1.0).mean())}
    obs = sites(V); MK = Markov(V)
    nul = {'wshuf': [sites(null_wshuf(V, random.Random(i))) for i in range(20)],
           'markov': [sites(MK.gen(V, random.Random(100 + i))) for i in range(20)]}
    out['sites'] = {'obs': obs, 'null': {k: {m: (float(np.mean([d[m] for d in v])), float(np.std([d[m] for d in v]))) for m in obs} for k, v in nul.items()}}
    print('sites', out['sites'], flush=True)
    # ---- (2) datasets ----
    sets = {}
    for wf in (0.75, 1.3):
        cp, mw = copy_from_exemplar(V, random.Random(1), width_factor=wf, p_ditto=0.05, p_skip=0.05)
        sets['PC_voy_ex_w%.2f' % wf] = (cp, TV, 'G', ('init', 'fin', 'rep'), mw)
    init_pool = [ws[0] for p in V for ws in p['lines'][1:]]
    fin_pool = [ws[-1] for p in V for ws in p['lines'][:-1]]
    cp, mw = copy_from_exemplar(V, random.Random(2), width_factor=1.4, fixed_width=32, plant_init=init_pool,
                                plant_fin=fin_pool, p_ditto=0.03, p_skip=0.03)
    sets['PC_voy_fixed32'] = (cp, TV, 'G', ('init', 'fin', 'rep'), mw)
    sets['NC_markov'] = (MK.gen(V, random.Random(3)), TV, 'G', ('init', 'fin', 'rep'), None)
    sets['VOYNICH'] = (V, TV, 'G', ('init', 'fin', 'rep'), None)
    LA = latin_paras(); TL = crossfit_tables(LA, list)
    for pd in (0.03, 0.10, 0.30):
        cp, mw = copy_from_exemplar(LA, random.Random(4), width_factor=1.4, fixed_width=38, p_ditto=pd, unit=list)
        sets['PC_latin_ditto%.2f' % pd] = (cp, TL, 'L', ('rep',), mw)
    # negative: Latin laid directly at width 53 with dittography at random places
    cp, _ = copy_from_exemplar(LA, random.Random(5), width_factor=1.0, fixed_width=53, unit=list)
    r5 = random.Random(6)
    for p in cp:
        for L in p['lines']:
            if len(L) > 3 and r5.random() < 0.3:
                k = r5.randrange(1, len(L)); L.insert(k, L[k - 1])
    sets['NC_latin_random_ditto'] = (cp, TL, 'L', ('rep',), None)
    jobs = []
    for nm, (ps, tb, un, ch, mw) in sets.items():
        jobs.append((nm, ps, tb, un, ch, 'obs', 0))
        for k in ('lshuf', 'wshuf'):
            for i in range(NREP): jobs.append((nm, ps, tb, un, ch, k, 1000 + i))
    with Pool(2) as pool:
        R = pool.map(job, jobs, chunksize=4)
    agg = defaultdict(lambda: {'obs': None, 'lshuf': [], 'wshuf': []})
    for nm, kind, seed, s in R:
        if kind == 'obs': agg[nm]['obs'] = s
        else: agg[nm][kind].append(s)
    res = {}
    for nm, d in agg.items():
        o = d['obs']; mw = sets[nm][4]; rr = {'true_mean_exemplar_width': mw}
        for sec, v in o.items():
            e = {'Amax': v['Amax'], 'LA': v['LA']}
            for k in ('lshuf', 'wshuf'):
                na = [n[sec]['Amax'] for n in d[k] if sec in n]
                e['pA_' + k] = (1 + sum(a >= v['Amax'] for a in na)) / (1 + len(na))
                if 'Rmax' in v:
                    nr = [n[sec]['Rmax'] for n in d[k] if sec in n]
                    e['pR_' + k] = (1 + sum(a >= v['Rmax'] for a in nr)) / (1 + len(nr))
            if 'Rmax' in v: e.update({'Rmax': v['Rmax'], 'LR': v['LR']})
            rr[sec] = e
        res[nm] = rr
        print(nm, json.dumps(rr), flush=True)
    out['calib'] = res
    save('cycle1.json', out)
    print('done %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()
