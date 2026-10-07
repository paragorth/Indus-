import json,sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
pre=sys.argv[1]
for n in sys.argv[2:]:
    f=os.path.join(L.CK,'%s_%s.json'%(pre,n))
    if not os.path.exists(f): continue
    r=json.load(open(f))
    for s,d in r['schemes'].items():
        print('==',n,'scheme',s,'scored',d['n_scored'],'whole %.2f indep %.2f best %.2f'%(d['whole_bits'],d['indep_bits'],d['best_bits']))
        print('  single fields page/self/para %:', ' '.join('%s:%.2f/%.2f/%.2f'%(p['cols'][0],p['r_page']*100,p['r_self']*100,p['r_para']*100) for p in d['single']))
        w=d['whole'][0]; print('  whole word page/self/para %.2f %.2f %.2f'%(w['r_page']*100,w['r_self']*100,w['r_para']*100))
        for sv in d['surv'][:3]:
            print('  surv %.3f bits spec_self %.3f'%(sv['bits'],sv['spec_self']*100), ' | '.join('%s %.2f/%.2f/%.2f'%('+'.join(p['cols']),p['r_page']*100,p['r_self']*100,p['r_para']*100) for p in sv['prof']))
