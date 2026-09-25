"""Strategy 130: minimum anchors, and the big-lexicon case. 150-sign planted syllabary, all decoded.
(a) right vocabulary, k = 10 and 20 anchors; (b) full 62,868-word Tamil lexicon with k = 40 anchors."""
import os, json
os.environ['KSYLL']='150'; os.environ['NDEC']='150'
import strat_anchor150 as A, strat_syllsize_n as M, search as S, indus_core as C
def run(cfg):
    k,lexmode=cfg
    if lexmode=='big': M.hit=S.string_hit_fn(C.forms(C.load_lexicon('tamil'),'full'),5)
    r=A.run(k); r['lexicon']=lexmode; return r
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(run,[(10,'right'),(20,'right'),(40,'big')]): print(json.dumps(r),flush=True)
