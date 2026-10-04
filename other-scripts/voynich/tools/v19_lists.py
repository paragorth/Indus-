"""v19 cycle 3 add-on: THE MARGIN LISTS. Runs of consecutive one-word label loci (columns such as the f66r word and
character columns, the f49v margin column, ring/label sequences) read in transcription order, each tested for sorting
(engine, keys F/L2/L/R, 1,000 within-list shuffles with the same search). Family-wise: max z over all lists x keys compared
with the same scan on lists whose entries are shuffled once (5 replicate scans)."""
import re, os, json, random, sys
from v19_lib import *

def runs(path):
    out, cur, folio = [], [], None
    for L in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'<(f\w+)\.(\d+),(.)(\w)(\w)>\s+(.*)', L.rstrip('\n'))
        if not m:
            continue
        f, n, mark, typ, sub, txt = m.groups()
        txt = re.sub(r'<[^>]*>', '', txt)
        txt = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', txt)
        txt = re.sub(r'\{([^}]*)\}', r'\1', txt)
        txt = re.sub(r'@\d+;', '?', txt)
        ws = [w for w in re.split(r'[.,<>\-]+', txt) if w]
        one = typ == 'L' and len(ws) == 1 and '?' not in ws[0] and "'" not in ws[0]
        if f != folio or not one:
            if len(cur) >= 8:
                out.append(('%s:%s' % (folio, cur[0][0]), [w for _, w in cur]))
            cur = []
            folio = f
        if one:
            cur.append((n, ws[0]))
    if len(cur) >= 8:
        out.append(('%s:%s' % (folio, cur[0][0]), [w for _, w in cur]))
    return out

if __name__ == '__main__':
    res = {}
    for tr, fn in [('ZL', 'ZL3b-n.txt'), ('IT', 'IT2a-n.txt')]:
        L = runs(os.path.join(DATA, fn))
        print(tr, len(L), 'lists:', ', '.join('%s(%d)' % (i, len(w)) for i, w in L))
        st = [(i, [(0, k, glyphs(w)) for k, w in enumerate(ws)]) for i, ws in L]
        sym = encode(st)
        for key in ['F', 'L2', 'L', 'R']:
            rows = run_engine(st, sym, key, R=1000, restarts=6, ils=25, nrand=20000, seed=5, tag='lists' + tr + key)
            nulls = []
            rng = random.Random(9)
            for rep in range(5):
                sh = []
                for i, ent in st:
                    e = ent[:]; rng.shuffle(e); sh.append((i, e))
                nr = run_engine(sh, sym, key, R=200, restarts=6, ils=25, nrand=0, seed=50 + rep, tag='listsN' + tr + key)
                nulls.append(max(r['z'] for r in nr if r['T'] > 0))
            top = sorted([r for r in rows if r['T'] > 0], key=lambda r: -r['z'])[:4]
            print(' %s %s: null-scan max z %s | top: %s' % (tr, key, ' '.join('%.1f' % z for z in nulls),
                  '; '.join('%s n%d tau%.2f z%.1f p%.3f [%s]' % (r['id'], r['n'], r['tau'], r['z'], r['p'], r['order_s'][:30]) for r in top)))
            res['%s_%s' % (tr, key)] = {'rows': rows, 'null_scan_max': nulls}
    json.dump(res, open(os.path.join(RES, 'c3_lists.json'), 'w'))
