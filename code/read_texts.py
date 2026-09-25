"""Segment-first glyph reader for Mahadevan text lines."""
import cv2, numpy as np, json, re, subprocess, os, sys

PDF="/mnt/user-data/uploads/The_Indus_Script__Text__Concordance_and_Tables_-Iravathan_Mahadevan.pdf"
BOX=json.load(open('/home/claude/mh/signlist_boxes.json'))
SLIMG={pg:cv2.imread(f'/home/claude/mh/SL-0{pg}.png',0) for pg in ['43','44','45','46']}
S=56  # canonical size

def canon(mask):
    """binary mask (ink=1) -> SxS float image, aspect preserved, centred; plus aspect w/h"""
    ys,xs=np.nonzero(mask)
    if len(xs)==0: return None,None
    m=mask[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.uint8)*255
    h,w=m.shape; sc=(S-6)/max(h,w); nh,nw=max(1,int(round(h*sc))),max(1,int(round(w*sc)))
    r=cv2.resize(m,(nw,nh),interpolation=cv2.INTER_AREA)
    out=np.zeros((S,S),np.float32); y0=(S-nh)//2; x0=(S-nw)//2; out[y0:y0+nh,x0:x0+nw]=r/255.0
    out=cv2.GaussianBlur(out,(3,3),0)
    return out, w/h

TPL={}
for n,v in BOX.items():
    x0,y0,x1,y1=v['box']; im=SLIMG[v['page']][y0:y1,x0:x1]
    c,a=canon((im<140).astype(np.uint8)); TPL[int(n)]=(c,a,(y1-y0))
TH=np.median([v[2] for v in TPL.values()])   # nominal template height
TPL_H={n:v[2]/TH for n,v in TPL.items()}      # relative height of each sign in the sign list

def score(mask, relh):
    """return sorted list of (score, sign) for a glyph mask; relh = glyph height / line height"""
    c,a=canon(mask)
    if c is None: return []
    cm=c-c.mean(); cn=np.linalg.norm(cm)+1e-6
    out=[]
    for n,(t,ta,_) in TPL.items():
        ar=abs(np.log((a+1e-3)/(ta+1e-3)))
        if ar>0.8: continue
        tm=t-t.mean(); s=float((cm*tm).sum()/(cn*(np.linalg.norm(tm)+1e-6)))
        # penalties: aspect mismatch, relative-height mismatch
        hr=abs(np.log((relh+1e-3)/(TPL_H[n]+1e-3)))
        out.append((s-0.25*ar-0.2*hr, n, s))
    out.sort(reverse=True)
    return out[:5]

def render(page, dpi=300):
    out=f'/home/claude/mh/tx/p{page}'
    os.makedirs('/home/claude/mh/tx',exist_ok=True)
    if not os.path.exists(out+f'-{page:03d}.png'):
        subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-r',str(dpi),'-gray','-png',PDF,out],check=True)
    return cv2.imread(out+f'-{page:03d}.png',0)

def ocr_numbers(img):
    H,W=img.shape; left=img[:, :int(W*0.55)]
    tmp='/home/claude/mh/tx/_left.png'; cv2.imwrite(tmp,left)
    tsv=subprocess.run(['tesseract',tmp,'-','--psm','6','tsv'],capture_output=True,text=True).stdout
    toks=[]
    for line in tsv.splitlines()[1:]:
        p=line.split('\t')
        if len(p)<12: continue
        t=p[11].strip()
        if re.fullmatch(r'\d{1,6}',t): toks.append((t,int(p[6]),int(p[7]),int(p[8]),int(p[9])))
    return toks

def rows_from_tokens(toks):
    toks=sorted(toks,key=lambda t:t[2]); rows=[]
    for t in toks:
        cy=t[2]+t[4]/2
        if rows and abs(cy-rows[-1]['cy'])<18: rows[-1]['toks'].append(t)
        else: rows.append(dict(cy=cy,toks=[t]))
    out=[]
    for r in rows:
        ts=sorted(r['toks'],key=lambda t:t[1])
        out.append(dict(cy=float(np.mean([t[2]+t[4]/2 for t in ts])),h=max(t[4] for t in ts),nums=[t[0] for t in ts]))
    return out

def glyph_groups(band):
    """band: binary ink. returns list of component-groups merged by x-overlap, sorted by x"""
    n,lab,st,cen=cv2.connectedComponentsWithStats(band,8)
    comps=[(st[i][0],st[i][1],st[i][2],st[i][3]) for i in range(1,n) if st[i][4]>=5]
    comps.sort(key=lambda c:c[0])
    groups=[]
    for c in comps:
        x0,x1=c[0],c[0]+c[2]
        if groups and x0<groups[-1]['x1']-1:   # overlaps previous group's x-range
            g=groups[-1]; g['x1']=max(g['x1'],x1); g['comps'].append(c)
        else: groups.append(dict(x0=x0,x1=x1,comps=[c]))
    return groups

def read_line(img, y0, y1, xstart):
    H,W=img.shape
    band=(img[y0:y1, xstart:]<150).astype(np.uint8)
    # remove tiny specks and restrict to ink region
    groups=glyph_groups(band)
    if not groups: return []
    lineh=y1-y0
    # ink line height estimate: from tallest group
    def gmask(gs):
        m=np.zeros_like(band)
        for g in gs:
            for c in g['comps']: 
                x,y,w,h=c; sub=band[y:y+h,x:x+w]; m[y:y+h,x:x+w]|=sub
        return m
    tallest=max(max(c[3] for c in g['comps']) for g in groups)
    LH=max(tallest, lineh*0.45)
    # DP over segmentation: segments = 1..3 consecutive groups, gap constraint
    N=len(groups); best=[(-1e9,None)]*(N+1); best[0]=(0.0,None)
    cache={}
    for j in range(1,N+1):
        for k in (1,2,3):
            i=j-k
            if i<0: continue
            gs=groups[i:j]
            if k>1:
                gaps=[gs[t+1]['x0']-gs[t]['x1'] for t in range(k-1)]
                if max(gaps)>LH*0.35: continue
            key=(i,j)
            if key not in cache:
                m=gmask(gs); ys,xs=np.nonzero(m); relh=(ys.max()-ys.min()+1)/LH
                sc=score(m,relh); cache[key]=sc
            sc=cache[key]
            if not sc: sc=[(-0.5,0,-0.5)]
            wid=gs[-1]['x1']-gs[0]['x0']
            v=best[i][0]+sc[0][0]*wid - 0.02*LH   # width-weighted, small per-segment cost
            if v>best[j][0]: best[j]=(v,(i,sc))
    # backtrack
    segs=[]; j=N
    while j>0:
        i,sc=best[j][1]; segs.append((i,j,sc)); j=i
    segs.reverse()
    out=[]
    for i,j,sc in segs:
        out.append(dict(sign=sc[0][1], score=round(sc[0][2],3), alt=[(n,round(s,3)) for _,n,s in sc[1:3]],
                        x0=int(groups[i]['x0']+xstart), x1=int(groups[j-1]['x1']+xstart), ngroups=j-i))
    return out

def ink_bands(img, xstart):
    H,W=img.shape; right=(img[:, xstart:int(W*0.985)]<150).astype(np.uint8)
    prof=right.sum(1); bands=[]; inb=False
    for y,v in enumerate(prof):
        if v>=2 and not inb: y0=y; inb=True
        elif v<2 and inb:
            if y-y0>=8: bands.append((y0,y))
            inb=False
    # merge bands separated by tiny gaps (<6px) -- dots above signs
    merged=[]
    for b in bands:
        if merged and b[0]-merged[-1][1]<6: merged[-1]=(merged[-1][0],b[1])
        else: merged.append(b)
    return [b for b in merged if b[1]-b[0]>=12]

def parse_page(page):
    img=render(page); H,W=img.shape; xstart=int(W*0.5)
    rows=rows_from_tokens(ocr_numbers(img))
    bands=ink_bands(img,xstart)
    out=[]; used=set()
    for r in rows:
        nums=r['nums']
        if len(nums)>=2 and len(nums[0])==4 and len(nums[1])==6: kind='object'
        elif len(nums)==1 and len(nums[0])==5: kind='side'
        elif len(nums)==1 and len(nums[0])==2: kind='line'
        elif len(nums)==1 and len(nums[0])==6: kind='object?'
        else: kind='other'
        # nearest ink band by centre
        cands=[(abs((b[0]+b[1])/2-r['cy']),i) for i,b in enumerate(bands) if i not in used]
        if not cands: out.append(dict(page=page,kind=kind,nums=nums,cy=int(r['cy']),glyphs=[],band=None)); continue
        d,i=min(cands)
        if d>45: out.append(dict(page=page,kind=kind,nums=nums,cy=int(r['cy']),glyphs=[],band=None)); continue
        used.add(i); y0,y1=bands[i]
        glyphs=read_line(img,max(0,y0-3),min(H,y1+3),xstart)
        out.append(dict(page=page,kind=kind,nums=nums,cy=int(r['cy']),y0=int(y0),y1=int(y1),glyphs=glyphs,band=i))
    return out

if __name__=='__main__':
    page=int(sys.argv[1])
    for r in parse_page(page):
        print(r['kind'], r['nums'], ' '.join(f"{g['sign']}({g['score']})" for g in r['glyphs']))
