"""Strategy 133: does the anchor threshold scale to an Indus-sized inventory? Planted Tamil with a
400-sign syllabary (all signs decoded, right vocabulary), k = 60 and 100 commonest signs anchored."""
import os, json
os.environ['KSYLL']='400'; os.environ['NDEC']='400'
import strat_anchor150 as A, strat_syllsize_n as M
if __name__=='__main__':
    print('signs',len(M.code),'decoded',M.N,'windows',len(M.W),flush=True)
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(A.run,[int(x) for x in os.environ.get("KLIST","60,100").split(",")]): print(json.dumps(r),flush=True)
