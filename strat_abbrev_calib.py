"""S164b: calibration for S164 — same perimetric-complexity vs log-frequency Spearman test on cuneiform
(Noto Sans Cuneiform glyphs; sign frequencies from the CDLI ATF dump, readings mapped to Unicode sign names).
Also subsampled to the Indus sign-count (462) and to the Indus token count scale."""
import json,math,sys,random
from fontTools.ttLib import TTFont
from PIL import Image,ImageDraw,ImageFont
import numpy as np
font,freqfile=sys.argv[1],sys.argv[2]
cmap=TTFont(font).getBestCmap(); fnt=ImageFont.truetype(font,160)
def complexity(cp):
    if cp not in cmap: return None
    im=Image.new('L',(700,320),0); ImageDraw.Draw(im).text((20,40),chr(cp),font=fnt,fill=255)
    a=np.array(im)>127; A=a.sum()
    if A<50: return None
    p=np.pad(a,1); inner=p[1:-1,1:-1]&p[:-2,1:-1]&p[2:,1:-1]&p[1:-1,:-2]&p[1:-1,2:]
    P=(a&~inner).sum(); return P*P/(4*math.pi*A)
fq={int(k,16):v for k,v in json.load(open(freqfile)).items()}
pts=[(math.log(v),complexity(k)) for k,v in fq.items() if v>=2]
pts=[p for p in pts if p[1] is not None]
x=np.array([p[0] for p in pts]); y=np.array([p[1] for p in pts])
sp=lambda a,b: np.corrcoef(a.argsort().argsort(),b.argsort().argsort())[0,1]
rho=sp(x,y); null=[sp(x,np.random.permutation(y)) for _ in range(5000)]
print(f"{freqfile}: signs={len(pts)} rho={rho:.3f} P={sum(r<=rho for r in null)/5000:.4f}")
