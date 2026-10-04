"""v32 cycle 1: period scan 2-40 (+ calendar periods) over ~25 paragraph features x 2 statistics, family-wise max-z
against paragraph-shuffle nulls. Voynich quire 20 (star units, ZL paragraphs, IT2a paragraphs), non-starred sections,
the Ado martyrology (real calendar: dominical letter period 7 + Roman date ~30.4) and planted calendars."""
import os, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import v32_lib as V

R = int(os.environ.get('V32_R', 200))


def corpora():
    q = V.load_q20()
    C = {
        'VOY-star': q,
        'VOY-para': V.load_q20(mode='para'),
        'VOY-IT2a': V.load_q20('IT2a', mode='para'),
        'SEC-bio-B': V.load_section('B', 'B'),
        'SEC-herbal-A': V.load_section('H', 'A'),
        'SEC-herbal-B': V.load_section('H', 'B'),
        'ADO-full': V.ado_units(),
        'ADO-date-noletter': V.ado_units(letter=False),
        'ADO-body': V.ado_units(letter=False, date=False),
        'PLANT-open-29.5-s0.3': V.plant_calendar(q, 29.53, 0.3, 1, 'opening'),
        'PLANT-open-7-s0.3': V.plant_calendar(q, 7, 0.3, 2, 'opening'),
        'PLANT-vocab-28-s0.05': V.plant_calendar(q, 28, 0.05, 3, 'vocab'),
    }
    return C


def run(args):
    name, units, seed = args
    ck = os.path.join(V.CK, f'c1_{name}.pkl')
    if os.path.exists(ck): return pickle.load(open(ck, 'rb'))
    t = time.time()
    num, cat, vec = V.features(units, star=name.startswith('VOY-star'))
    obs = V.scan(num, cat, vec)
    nulls = V.perm_nulls(num, cat, vec, R, seed)
    res = V.family_test(obs, nulls)
    out = {'name': name, 'N': len(units), 'zmax': res['zmax'], 'p_fw': res['p_fw'], 'top': V.fmt_top(res),
           'tops': res['tops'], 'secs': time.time() - t,
           # best z at the calendar periods (any feature)
           'cal': {str(P): float(res['Z'][:, list(V.PERIODS).index(P)].max()) for P in [7.0, 12.0, 27.32, 28.0, 29.53, 30.0, 30.44, 36.0, 36.5]},
           'Z': res['Z'], 'keys': res['keys'], 'nullmax': res['nullmax'], 'O': res['O'], 'Nn': res['Nn'].astype(np.float32)}
    pickle.dump(out, open(ck, 'wb'))
    print(name, len(units), f"zmax {out['zmax']:.2f} p_fw {out['p_fw']:.3f}", out['top'][:200], flush=True)
    return out


if __name__ == '__main__':
    C = corpora()
    jobs = [(k, v, 1000 + i) for i, (k, v) in enumerate(C.items())]
    with Pool(2) as p:
        outs = p.map(run, jobs, chunksize=1)
    rows = []
    for o in outs:
        cal = ', '.join(f'{k}:{v:.1f}' for k, v in o['cal'].items())
        rows.append((o['name'], o['N'], f"{o['zmax']:.2f}", f"{o['p_fw']:.3f}", o['top'], cal))
    with open(os.path.join(V.CK, 'c1_summary.txt'), 'w') as fh:
        for r in rows: fh.write(' | '.join(str(x) for x in r) + '\n')
    print('done')
