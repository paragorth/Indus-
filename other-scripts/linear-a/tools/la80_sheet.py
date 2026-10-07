import sys,glob; sys.path.insert(0,'/home/user/Indus-/other-scripts/linear-a/tools')
import la80_common as L
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
pat,out,start=sys.argv[1],sys.argv[2],int(sys.argv[3])
fs=sorted(glob.glob(L.IMG+'/'+pat))[start:start+48]
sheet=Image.new('RGB',(160*8,200*6),'white')
for i,fn in enumerate(fs):
    f,obj=(L.measure_fx(fn) if "Facsimile" in fn else L.measure(fn)); g,_=L.load_gray(fn)
    rgb=np.stack([g,g,g],-1)
    if obj is not None:
        rgb[obj ^ ndi.binary_erosion(obj,iterations=3)]=[255,0,0]
    sheet.paste(Image.fromarray(rgb.astype(np.uint8)).resize((160,200)),((i%8)*160,(i//8)*200))
sheet.save(out)
