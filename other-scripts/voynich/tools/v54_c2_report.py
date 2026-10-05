import sys, os, json, glob, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
for f in sorted(glob.glob(os.path.join(V.CK, 'c2_*.json'))):
    R = json.load(open(f)); nm = R['name']
    vars_ = [k for k in R['folds'][0]['base'] if k != 'types']
    for K in ('4', '8'):
        line = []
        for v in vars_:
            col = sum(F['K'][K]['col'][v] for F in R['folds']) / 5
            nul = sum(F['K'][K]['nul'][v] for F in R['folds']) / 5
            base = sum(F['base'][v] for F in R['folds']) / 5
            rnd = [sum(F['K'][K]['rnd'][r][v] for F in R['folds']) / 5 for r in range(len(R['folds'][0]['K'][K]['rnd']))]
            m, sd = st.mean(rnd), st.pstdev(rnd) or 1e-9
            pr = sum(x >= col for x in rnd) / len(rnd)
            line.append('%s col %.3f base %.3f rnd %.3f z %.1f (rank>=%.2f) shufops %.3f' % (v, col, base, m, (col - m) / sd, pr, nul))
        F0 = R['folds'][0]['K'][K]
        ex = [(round(F['K'][K]['exact_base'], 3), round(F['K'][K]['exact_col'], 3), round(F['K'][K]['exact_rnd'], 3)) for F in R['folds']][:2]
        print(nm, 'K', K, 'ops', F0['ops'], 'exact base/col/rnd', ex)
        for l in line: print('     ', l)
