"""Strategy 123: is the failure due to lexicon size? Same clean planted texts as S122, but the objective
uses only the 3,000 words actually used to write them (the 'right small vocabulary')."""
import json, random, math
import strat_easy as E, search as S
small=set(''.join(v) for v in E.vocab)
E.hit=S.string_hit_fn(small,5)
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        for r in p.imap(E.run,[0,1,2]): print(json.dumps(r),flush=True)
