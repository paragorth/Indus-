"""pe71 cycle 1b: coarse document-type null (volume x M157 header x has M288, no size band), 5,000 perms; guards against the small-strata over-conditioning of the c1 doc null.  usage: pe71_c1b.py"""
from pe71_lib import *
from pe71_c1 import targets
R=load(); rng=np.random.default_rng(3)
st=np.array(['%s|%d|%d'%(r['vol'],r['hdr']=='M157',r['m288']) for r in R])
P=perm_index(st,rng,5000); s=np.array([r['sealed'] for r in R],int); u=unit_codes(R)
for k,f in targets(R).items(): print(k, fam_stats(f,s,u,P))
