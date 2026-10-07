import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_lib as V
names = sys.argv[1].split(',')
for name in names:
    out_p = os.path.join(V.CK, 'c1_%s.pkl' % name)
    if os.path.exists(out_p): continue
    t = time.time()
    L, meta = V.build(name)
    C = V.Corpus(L, type_salts=V.TYPE_SALTS2)
    items = V.all_items(C)
    res = V.run_items(C, items)
    pickle.dump({'res': res, 'meta': meta, 'n_items': len(items)}, open(out_p, 'wb'))
    print(name, len(items), len(res), round(time.time() - t), flush=True)
