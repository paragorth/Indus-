"""summarise v90 checkpoints: content-wheel calls among survivors."""
import json, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L

def is_content(p, zmin=3.0, gap=0.5):
    return p.get('lift_z', 0) >= zmin and p.get('lift', 0) - p.get('clift', 1.0) >= gap and p.get('clift_z', 0) < 2.0

def summarize(pre, name):
    f = os.path.join(L.CK, '%s_%s.json' % (pre, name))
    if not os.path.exists(f): return None
    r = json.load(open(f))
    out = {'name': name}
    nsv = ncw = 0; fields = Counter(); best = None; zs = []
    for s, d in r['schemes'].items():
        for sv in d['surv']:
            nsv += 1
            cw = [p for p in sv['prof'] if is_content(p)]
            if cw: ncw += 1
            for p in cw:
                fields['+'.join(p['cols'])] += 1
            for p in sv['prof']:
                zs.append(p.get('lift_z', 0))
                sp = p.get('lift', 0) - p.get('clift', 1.0)
                if best is None or sp > best[0]:
                    best = (sp, '+'.join(p['cols']), p.get('lift', 0), p.get('lift_z', 0), p.get('clift', 0), p.get('clift_z', 0), p['r_page'] * 100, p['r_para'] * 100)
    d0 = r['schemes']['0']
    out.update({'n_surv': nsv, 'n_content': ncw, 'fields': fields.most_common(4), 'best': best,
                'whole_lift': (d0['whole'][0]['lift'], d0['whole'][0]['lift_z']),
                'bits': (d0['best_bits'], d0['indep_bits'], d0['whole_bits']),
                'top_decomp': ['+'.join(L.FIELDS[c] for c in g) for g in d0['surv'][0]['decomp']],
                'n_scored': sum(dd['n_scored'] for dd in r['schemes'].values())})
    return out

if __name__ == '__main__':
    pre = sys.argv[1]
    for n in sys.argv[2:]:
        o = summarize(pre, n)
        if o: print(json.dumps(o, default=lambda x: round(x, 3) if isinstance(x, float) else str(x)))
