import glob,re,sys
for sc in ['proto-elamite','linear-a','voynich']:
    p=f'other-scripts/{sc}/STRATEGIES.md'; s=open(p).read(); new=[]
    for f in sorted(glob.glob(f'other-scripts/{sc}/loops/*.txt')):
        pre=f.split('/')[-1].split('_')[0]
        if not glob.glob(f'other-scripts/{sc}/loops/{pre}_final.txt'): continue
        for line in open(f):
            m=re.match(r'\| (?:[a-z]+\d+:)?([A-Z]+-[\d.]+[\w.]*) \|',line)
            if not m or line.count('|')<5: continue
            rid=f"{pre}:{m.group(1)}"
            row='| '+rid+' |'+line.strip()[len(m.group(0)):]
            if ('| '+rid+' |') not in s and rid not in [r.split('|')[1].strip() for r in new]: new.append(row)
    if new:
        k=s.index('## Summary'); s=s[:k].rstrip('\n')+'\n'+'\n'.join(new)+'\n\n'+s[k:]; open(p,'w').write(s)
    print(sc,[r.split('|')[1].strip() for r in new])
