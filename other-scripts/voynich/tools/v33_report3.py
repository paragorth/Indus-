"""v33 cycle 3 report: gap ratio and Q ratio of each text vs its own trigram (T) and slot (S) generators."""
import glob
from v33_lib import *

if __name__ == '__main__':
    old = set(r.name for r in rule_set(20))
    out = {}
    for f in sorted(glob.glob(os.path.join(CK, 'c3_*.json'))):
        n = os.path.basename(f)[3:-5]
        D = json.load(open(f))
        rows = defaultdict(list)
        for key, v in D.items():
            b, r, k = key.split('|')
            if k != 'X': continue
            for g in ('T', 'S'):
                o = D[f'{b}|{r}|{g}']
                grp = 'old' if r in old else 'new'
                rows[('conn', g, grp)].append(v[0] / max(o[0], 1e-9))
                rows[('Q', g, grp)].append(v[1] / max(o[1], 1e-9))
                rows[('NODF', g, grp)].append(v[2] / max(o[2], 1e-9))
        res = {}
        for (m, g, grp), v in rows.items():
            v = np.array(v)
            res[f'{m}/{g}/{grp}'] = (float(np.median(v)), float(np.mean(v < 1)), len(v))
        out[n] = res
        print(f'{n:10s}', ' '.join(f"{k.replace('/', '')}={res[k][0]:.2f}({res[k][1]:.2f})" for k in sorted(res) if k.split('/')[0] in ('conn', 'Q')))
    save('c3_report.json', out)
