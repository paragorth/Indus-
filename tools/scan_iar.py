import pymupdf,re,sys,glob
SITES=r'(Rakhigarhi|Farmana|Bhirrana|Kanmer|Khirsara|Juni Kuran|Dholavira|Karanpura|Shikarpur|Bagasra|Gola Dhoro|Kunal|Banawali|Lothal|Kalibangan|Rojdi|Surkotada|Desalpar|Baror|Tarkhanewala|Chak 86|Masudpur|Khanak|Sanauli|Sinauli|Madina|Mitathal|Girawad|Binjor|4MSR|Bhimgoda|Kotada|Navinal|Shirvan|Nagwada|Pabumath|Vadnagar|Loteshwar|Datrana|Moti Pipli|Kerala-no-Dhoro|Padri|Kuntasi)'
for f in sorted(glob.glob('r*.pdf')):
    try: d=pymupdf.open(f)
    except Exception as e: print(f,'ERR',e); continue
    if d.page_count<10: continue
    title=d[0].get_text()[:60].replace('\n',' ')
    hits=[]
    for i,p in enumerate(d):
        t=p.get_text()
        if re.search(r'(?i)\bseals?\b|sealing|inscribed|Harappan (?:signs?|script|characters)',t) and re.search(SITES,t):
            sites=sorted(set(re.findall(SITES,t)))
            snip=re.findall(r'(?i)[^.\n]{0,80}(?:\bseals?\b|sealings?|Harappan signs?|script|characters)[^.\n]{0,80}',t)
            hits.append((i+1,sites,snip[:2]))
    print('=====',f,d.page_count,title)
    for h in hits: print('  p',h[0],h[1],'|',' / '.join(s.strip() for s in h[2])[:220])
