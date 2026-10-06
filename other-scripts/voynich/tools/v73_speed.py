import sys; sys.path.insert(0,'/home/user/Indus-/other-scripts/voynich/tools')
import time, numpy as np, v73_lib as L, v72_lib as V
P=V.voynich('ZL3b'); C=L.Corpus(P,'ZL')
bank,names=L.make_bank(200)
t=time.time(); S=L.score_bank(C,bank[:50]); print('50 rules',time.time()-t)
for i in np.argsort(-S[:,0,0])[:3]: print(names[i], S[i].round(3).tolist())
import cProfile; cProfile.run('L.score_bank(C,bank[:20])',sort='cumtime')
