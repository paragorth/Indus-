"""v43 cycle 2 report."""
import v43_lib as L
R = L.load('cycle2.json'); tau = min(0.5, L.load('cycle1.json')['tau']['6']['p5'])
print('tau6', round(tau, 3))
for ex, d in R.items():
    for ch, res in d.items():
        for nm, v in res.items():
            if nm == 'rand_survivors':
                print(ex, ch, nm, {k: round(x, 3) for k, x in v.items()}); continue
            top = lambda c: sorted(c.items(), key=lambda x: -x[1])[:3]
            print(ex, ch, nm, 'np', v['npages'], 'choose', round(v['choose'], 3), 'test', round(v['test'], 3), 'null', round(v['null_mean'], 3),
                  'z', round(v['z'], 2), 'tau', v['above_tau'], 'rec', v.get('recall'), 'prec', None if 'precision' not in v else round(v['precision'], 2),
                  top(v['secs']), top(v['hands']), top(v['quires']))
