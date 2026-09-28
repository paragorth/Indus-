"""Render every Wells sign from the lipi Indus font to a normalised binary bitmap."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont
FONT='tools/indus_font.ttf'
def render(w,size=96,out=48):
    f=ImageFont.truetype(FONT,size)
    im=Image.new('L',(size*2,size*2),0); d=ImageDraw.Draw(im)
    d.text((size//2,size//4),chr(0xE000+w),font=f,fill=255)
    a=np.array(im)>128
    ys,xs=np.nonzero(a)
    if len(xs)==0: return None
    a=a[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h,wd=a.shape; s=max(h,wd)
    pad=np.zeros((s,s),bool); pad[(s-h)//2:(s-h)//2+h,(s-wd)//2:(s-wd)//2+wd]=a
    return np.array(Image.fromarray(pad.astype(np.uint8)*255).resize((out,out),Image.BILINEAR))>60
