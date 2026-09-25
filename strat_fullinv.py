"""Strategy 127: full-inventory decoding, harder search. 150-sign planted Tamil syllabary, all signs decoded.
(a) right 3,000-word vocabulary, 200,000 iterations; (b) the full 62,868-word Tamil lexicon as objective
(the realistic case: we would not know the actual vocabulary)."""
import os, json, sys
os.environ['KSYLL']='150'; os.environ['NDEC']='150'
import strat_syllsize_n as M, search as S, indus_core as C
def run(which):
    if which=='biglex':
        M.hit=S.string_hit_fn(C.forms(C.load_lexicon('tamil'),'full'),5)
    r=M.run(0,iters=200000); r['objective']=which; return r
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(run,['smallvocab','biglex']): print(json.dumps(r),flush=True)
