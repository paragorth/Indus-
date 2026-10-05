"""v55: validate the ephemeris against JPL DE421 (1900-2050) and 15th-c. eclipses."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from v55_sky import positions, build_daily, julian_cal_to_jd, jd_to_julian_cal, easter_julian
from skyfield.api import load
bsp = sys.argv[1]
ts = load.timescale(); eph = load(bsp); earth = eph['earth']
names = {'sun': 'sun', 'moon': 'moon', 'mercury': 'mercury', 'venus': 'venus', 'mars': 'mars',
         'jupiter': 'jupiter barycenter', 'saturn': 'saturn barycenter'}
rng = np.random.default_rng(0)
jd = 2415100 + rng.uniform(0, 54000, 400)
t = ts.tt_jd(jd)
mine = positions(jd - 2451543.5)
for b, n in names.items():
    lat, lon, _ = earth.at(t).observe(eph[n]).apparent().ecliptic_latlon(epoch='date')
    err = (mine[b] - lon.degrees + 180) % 360 - 180
    sgn = (np.floor(mine[b] / 30) == np.floor(lon.degrees / 30)).mean()
    print(f'{b:8s} |err| median {np.median(abs(err)):.3f} deg, 99% {np.percentile(abs(err),99):.3f}, same sign {sgn:.3f}')
# eclipses 1400-1450: syzygies with small lunar latitude
tab = build_daily(1400, 1451, ut_hour=12)
el = (tab['moon'] - tab['sun']) % 360
jdn = tab['jdn']
for kind, target, latmax in (('solar', 0, 1.4), ('lunar', 180, 0.9)):
    x = (el - target + 180) % 360 - 180
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    hits = [i for i in idx if abs(tab['moonlat'][i]) < latmax or abs(tab['moonlat'][i + 1]) < latmax]
    ds = [jd_to_julian_cal(np.array([jdn[i] + 1])) for i in hits]
    s = ['%d-%02d-%02d' % (a[0][0], a[1][0], a[2][0]) for a in ds]
    print(kind, len(s), 'e.g.', [q for q in s if q[:4] in ('1406', '1415', '1433')])
print('Easter 1404..1408', [easter_julian(y) for y in range(1404, 1409)])
