"""Loop 6 (geography as plaintext): build data/derived/dark/site_coords.csv for every corpus site with >= 5 texts.
Coordinates are from published site locations (approximate, decimal degrees, WGS84, +-0.05 deg); 'uncertain' marks
sites where the identification or the location is less secure. Derived variables: distance to the Arabian Sea coast
(modern coastline polyline, and a 'Harappan' coastline that treats the Great and Little Rann of Kutch as marine),
distance to Mohenjo-daro and to Harappa (haversine, km), river system, elevation class.
"""
import csv, math

# name, lat, lon, river, elev_m, uncertain, note
SITES = [
 ("Harappa",             30.63, 72.87, "Indus",   170, 0, "Ravi (Indus system), Punjab plain"),
 ("Mohenjo-daro",        27.33, 68.14, "Indus",    45, 0, "Indus right bank, Sindh"),
 ("Dholavira",           23.89, 70.21, "Gujarat",  15, 0, "Khadir island, Great Rann of Kutch; the Rann was a shallow marine inlet in the 3rd millennium"),
 ("Lothal",              22.52, 72.25, "Gujarat",   5, 0, "Bhogavo/Sabarmati, head of the Gulf of Khambhat"),
 ("Kalibangan",          29.47, 74.13, "Ghaggar", 175, 0, "Ghaggar left bank, Rajasthan"),
 ("Chanhu-daro",         26.17, 68.33, "Indus",    35, 0, "Indus left bank (Nawabshah), Sindh"),
 ("Nausharo",            29.45, 67.63, "Indus",   150, 0, "Kachi plain, Bolan river piedmont (Indus system)"),
 ("Banawali",            29.60, 75.39, "Ghaggar", 210, 0, "Ghaggar/Sarasvati palaeochannel, Haryana"),
 ("Allahdino",           24.93, 67.42, "Coast",    20, 0, "Malir river near Karachi, 15 km from the sea"),
 ("Rakhigarhi",          29.28, 76.11, "Ghaggar", 220, 0, "Drishadvati palaeochannel, Haryana"),
 ("Lakhanjo-daro",       27.72, 68.85, "Indus",    65, 1, "Sukkur, Sindh; location approximate"),
 ("Kot-Diji",            27.34, 68.71, "Indus",    50, 0, "Indus left bank, Khairpur, Sindh"),
 ("Bala-kot",            25.42, 66.73, "Coast",    10, 0, "Las Bela, near Sonmiani Bay (Makran coast)"),
 ("Farmana",             29.04, 76.31, "Ghaggar", 220, 0, "Rohtak district, Haryana (Drishadvati)"),
 ("Kanmer",              23.37, 70.87, "Gujarat",  25, 0, "eastern Kutch, Little Rann margin"),
 ("Surkotada",           23.62, 70.84, "Gujarat",  30, 0, "eastern Kutch"),
 ("Shikarpur",           23.17, 70.43, "Gujarat",  10, 1, "Shikarpur (Kutch, Bhachau taluka), Gulf of Kutch; taken as the Gujarat site, not Shikarpur in Sindh"),
 ("Amri",                26.17, 68.01, "Indus",    40, 0, "Indus right bank, Dadu, Sindh"),
 ("Bhirrana",            29.55, 75.55, "Ghaggar", 200, 0, "Ghaggar palaeochannel, Fatehabad, Haryana"),
 ("Gola Dhoro (Bagasra)",23.07, 70.62, "Gujarat",  10, 0, "Gulf of Kutch coast"),
 ("Rupar",               30.97, 76.53, "Indus",   260, 0, "Sutlej at the Shivalik foothills (Indus system; often grouped with the Ghaggar region)"),
 ("Ur",                  30.96, 46.10, "Abroad",   10, 0, "Mesopotamia; excluded from Indus gradients"),
]

# Modern Arabian Sea coastline (Makran to the Gulf of Khambhat), coarse polyline
COAST_MODERN = [(25.12,62.33),(25.26,63.47),(25.21,64.64),(25.42,66.60),(24.85,67.00),(24.40,67.40),(24.00,67.50),
 (23.70,68.30),(23.20,68.60),(22.83,69.35),(22.47,69.07),(22.24,68.97),(21.64,69.60),(20.90,70.37),(20.70,71.00),
 (21.20,72.10),(21.77,72.15),(22.30,72.60),(21.10,72.70),(22.95,70.30),(22.60,70.00)]
# Harappan-period coastline: add the Great Rann and Little Rann as marine
COAST_HARAPPAN = COAST_MODERN + [(23.90,69.50),(24.05,70.00),(24.00,70.60),(23.80,71.00),(23.40,71.00),(23.20,71.30),(23.60,70.40)]

def hav(a,b,c,d):
    R=6371.0; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(h))

def dcoast(lat,lon,line): return min(hav(lat,lon,a,b) for a,b in line)

MD=(27.33,68.14); HP=(30.63,72.87)
def elev_class(m): return "low" if m<50 else ("mid" if m<150 else "high")

rows=[]
for name,lat,lon,river,elev,unc,note in SITES:
    rows.append(dict(site=name,lat=lat,lon=lon,river=river,elev_m=elev,elev_class=elev_class(elev),
        dist_coast_modern_km=round(dcoast(lat,lon,COAST_MODERN)),
        dist_coast_harappan_km=round(dcoast(lat,lon,COAST_HARAPPAN)),
        dist_md_km=round(hav(lat,lon,*MD)), dist_hp_km=round(hav(lat,lon,*HP)),
        uncertain=unc, note=note))
with open('data/derived/dark/site_coords.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r['site'],r['lat'],r['lon'],r['river'],r['elev_class'],r['dist_coast_modern_km'],r['dist_coast_harappan_km'],r['dist_md_km'],r['dist_hp_km'])
