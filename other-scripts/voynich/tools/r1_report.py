"""Summarise R1 job files: python3 r1_report.py <cycle> [<cycle> ...]"""
import sys, os, json, glob
from collections import defaultdict
import r1_lib as L


def summarise(cycles):
    rows = defaultdict(lambda: {'real': [], 'shuffle': [], 'planted': []})
    for c in cycles:
        for f in sorted(glob.glob(os.path.join(L.RES, f'c{c}_*.json'))):
            d = json.load(open(f))
            j = d['job']
            rows[(j['fam'], j['script'])][j['mode']].append(d)
    out = []
    for (fam, script), m in sorted(rows.items()):
        r = m['real'][0] if m['real'] else None
        sh, pl = m['shuffle'], m['planted']
        line = {
            'fam': fam, 'script': script,
            'N_real': r['N'] if r else None,
            'real_stage1': r['n_stage1'] if r else None,
            'real_tested2': r.get('n_tested2') if r else None,
            'real_replicated': r['n_replicated'] if r else None,
            'shuffle_runs': len(sh),
            'shuffle_N': sh[0]['N'] if sh else None,
            'shuffle_stage1': [x['n_stage1'] for x in sh],
            'shuffle_replicated': [x['n_replicated'] for x in sh],
            'planted_runs': len(pl),
            'planted_recovered': sum(bool(x.get('recovered')) for x in pl),
            'planted_stage1': [x['n_stage1'] for x in pl],
        }
        out.append(line)
    return out


if __name__ == '__main__':
    cyc = [int(x) for x in sys.argv[1:]]
    for l in summarise(cyc):
        print(json.dumps(l))
