"""S164: Law of abbreviation in Indus signs. Frequent signs should be graphically simpler if the script is used like
a writing system (Zipf/Kanwal; shown for many scripts). Glyphs rendered from the lipi Indus font (U+E000 + Wells no.).
Complexity = perimetric complexity P^2/(4*pi*A) of the rendered glyph. Stroke-numeral signs are excluded (trivially simple).
Control: permutation null for Spearman rho; slot classes compared."""
import json,math,random,collections as C,sys
from fontTools.ttLib import TTFont
from PIL import Image,ImageDraw,ImageFont
import numpy as np
FONT=sys.argv[1] if len(sys.argv)>1 else 'indus.ttf'
tt=TTFont(FONT); cmap=tt.getBestCmap(); fnt=ImageFont.truetype(FONT,200)
def complexity(n):
    cp=0xE000+n
    if cp not in cmap: return None
    im=Image.new('L',(320,320),0); ImageDraw.Draw(im).text((40,20),chr(cp),font=fnt,fill=255)
    a=np.array(im)>127; A=a.sum()
    if TOFU is not None and a.shape==TOFU.shape and (a==TOFU).mean()>0.995: return None
    if A<50: return None
    # perimeter: count ink pixels with a non-ink 4-neighbour
    p=np.pad(a,1); inner=p[1:-1,1:-1]&p[:-2,1:-1]&p[2:,1:-1]&p[1:-1,:-2]&p[1:-1,2:]
    P=(a&~inner).sum()
    return P*P/(4*math.pi*A)
TOFU=None
_im=Image.new('L',(320,320),0); ImageDraw.Draw(_im).text((40,20),chr(0xE000+8),font=fnt,fill=255); TOFU=np.array(_im)>127
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
freq=C.Counter(s for x in d for s in x['seq'])
# stroke numerals in Wells numbering: 1-19 (short/tall stroke groups) and 31-55 region contain stroke groups; detect by shape instead:
comp={n:complexity(n) for n in freq}
comp={n:c for n,c in comp.items() if c is not None}
NUM=set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56}  # pure stroke groups, checked on rendered glyphs 1-60
signs=[n for n in comp if n not in NUM and freq[n]>=2]
x=np.array([math.log(freq[n]) for n in signs]); y=np.array([comp[n] for n in signs])
def spearman(a,b):
    ra=a.argsort().argsort(); rb=b.argsort().argsort(); return np.corrcoef(ra,rb)[0,1]
rho=spearman(x,y); random.seed(0)
null=[spearman(x,np.random.permutation(y)) for _ in range(5000)]
p=sum(r<=rho for r in null)/5000
print(f"signs={len(signs)} rho(logfreq, complexity)={rho:.3f} one-sided P={p:.4f}")
# frequency bands
order=sorted(signs,key=lambda n:-freq[n])
for lo,hi in ((0,20),(20,60),(60,150),(150,len(order))):
    band=order[lo:hi]; print(f"rank {lo+1}-{hi}: median complexity {np.median([comp[n] for n in band]):.1f} (n={len(band)})")
json.dump({str(n):round(comp[n],2) for n in comp},open('data/derived/sign-complexity.json','w'))
