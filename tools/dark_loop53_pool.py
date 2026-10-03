"""S-DARK-53 supplement: full-pool utilisation of the reference code spaces (no subsampling, no Chao1), because a
registry (every code once) has f2 = 0 and Chao1 is undefined / explodes; for a complete code list the utilisation is
simply realised distinct / capacity. Indus is given as Chao1 / capacity (lower bound on the realised stock).
Usage: python3 tools/dark_loop53_pool.py"""
import sys,json,math,collections,random
sys.argv=['x','0']
exec(open('tools/dark_loop53.py').read().split('# ================= cycles')[0])
uw,us=ur3_names()
pools={'D aircraft_reg (chars)':jl('aircraft_reg'),'D hts':jl('hts'),'D icd10':jl('icd10'),'D unicode_names (words)':jl('unicode_names'),
 'G chess_eco':jl('chess_eco'),'L ur3 names 1/legend (syll)':us,'L ur3 names per impression (syll)':jl('ur3_names_syll'),
 'L linear B personnel names':linb_names(),'L latin EDH names':latin_names(),'A proto-elamite middles':pe_middles(),
 'D heraldry':jl('heraldry'),'A khipu':jl('khipu')}
objs=load_indus('seq_raw'); mids=[o['mid'] for o in objs if len(o['mid'])>=1]; lens=[len(m) for m in mids]
r=random.Random(7)
pools['S uniform ID (k=617, Indus lengths) 20000']=synth_uniform(20000,lens,617,r)
pools['S zipf names stock 3000 a=1.0, 20000 draws']=synth_zipf_names(20000,lens,r,3000,400,1.0)
pools['Indus seq_raw middles (dedup)']=mids
out=[]
def P(s): print(s); out.append(s)
P('full-pool capacities (log10): corpus | n | distinct | k | log10 cap naive / pos / ent | log10 util (distinct/cap) naive / pos / ent | Gini_id | Heaps | even')
for lab,pool in pools.items():
    cnt=collections.Counter(pool); D=len(cnt); n=len(pool)
    m=metrics(pool,random.Random(1))
    ld=math.log10(D)
    P(f'  {lab:46s} n={n:6d} D={D:6d} k={m["k"]:5d} cap {m["lcap_naive"]:5.1f}/{m["lcap_pos"]:5.1f}/{m["lcap_ent"]:5.1f}  util {ld-m["lcap_naive"]:6.1f}/{ld-m["lcap_pos"]:6.1f}/{ld-m["lcap_ent"]:6.1f}  Gini {m["gini_id"]:.3f} Heaps {m["heaps"]:.3f} even {m["even_pos"]:.3f} meanL {m["meanL"]:.2f}')
P('  Indus with Chao1 instead of D: util %.1f / %.1f / %.1f (Chao1 %.0f)'%tuple([math.log10(metrics(mids)["chao1"])-metrics(mids)[k] for k in ("lcap_naive","lcap_pos","lcap_ent")]+[metrics(mids)["chao1"]]))
open('data/derived/dark/loop53_pool_log.txt','w').write('\n'.join(out)+'\n')
