"""print the best rules of a v56 cycle-1 json, with planted / Roman truth when available."""
import sys, json
for f in sys.argv[1:]:
    o = json.load(open(f)); rows = o['rows']
    print('==', o['name'], 'rules', len(rows), 'evals %.2e' % o['n_eval'])
    S = o['summary']; print(' top20-by-train: tr %.2f te mean %.2f max %.2f; all te mean %.2f sd %.2f; n te>3 %d' % (
        S['top_tr_mean'], S['top_te_mean'], S['top_te_max'], S['all_te_mean'], S['all_te_sd'], S['n_te_gt3']))
    if 'digits' in o: print(' planted digits (value=index):', o['digits'])
    if 'roman' in o: print(' roman glyphs:', o['roman'])
    for r in sorted(rows, key=lambda r: -r['z_te'])[:6]:
        print('  te %.2f tr %.2f n %d/%d %s %s %s %s' % (r['z_te'], r['z_tr'], r['ntr'], r['nte'], r['sel'][:30], r['feat'], r['mode'],
              {k: v for k, v in sorted(r['d'].items(), key=lambda x: -x[1])[:9]}))
