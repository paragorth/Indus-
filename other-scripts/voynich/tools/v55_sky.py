"""v55 THE SKY IS THE ANSWER KEY -- a self-contained low-precision ephemeris.

Geocentric ecliptic longitudes (equinox of date, i.e. tropical zodiac) of the
Sun, Moon and the five classical planets, after P. Schlyter's published
orbital elements and perturbation terms (numbers only, no interpretation).
Validated in v55_validate.py against JPL DE421 (1900-2050) and against
recorded 15th-century eclipses.

Also: Julian-calendar dates, weekdays, Julian computus Easter, and a
'fake sky' variant whose periods are rescaled (null control).
"""
import numpy as np

D2R = np.pi / 180.0


def _kepler(M, e):
    E = M + e * np.sin(M) * (1.0 + e * np.cos(M))
    for _ in range(8):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


# name: (N0,N1, i0,i1, w0,w1, a, e0,e1, M0,M1)
ELEM = {
    'mercury': (48.3313, 3.24587e-5, 7.0047, 5.00e-8, 29.1241, 1.01444e-5, 0.387098, 0.205635, 5.59e-10, 168.6562, 4.0923344368),
    'venus': (76.6799, 2.46590e-5, 3.3946, 2.75e-8, 54.8910, 1.38374e-5, 0.723330, 0.006773, -1.302e-9, 48.0052, 1.6021302244),
    'mars': (49.5574, 2.11081e-5, 1.8497, -1.78e-8, 286.5016, 2.92961e-5, 1.523688, 0.093405, 2.516e-9, 18.6021, 0.5240207766),
    'jupiter': (100.4542, 2.76854e-5, 1.3030, -1.557e-7, 273.8777, 1.64505e-5, 5.20256, 0.048498, 4.469e-9, 19.8950, 0.0830853001),
    'saturn': (113.6634, 2.38980e-5, 2.4886, -1.081e-7, 339.3939, 2.97661e-5, 9.55475, 0.055546, -9.499e-9, 316.9670, 0.0334442282),
}
BODIES = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']


def positions(d, rate=None):
    """d = days since 2000 Jan 0.0 UT (JD - 2451543.5); array.
    rate: optional dict body->multiplier on the mean-anomaly rate (fake sky).
    Returns dict body -> ecliptic longitude deg [0,360), plus 'moonlat'."""
    rate = rate or {}
    d = np.asarray(d, float)
    out = {}
    # Sun
    ws = 282.9404 + 4.70935e-5 * d
    es = 0.016709 - 1.151e-9 * d
    Ms = (356.0470 + 0.9856002585 * d) % 360
    E = _kepler(Ms * D2R, es)
    xv, yv = np.cos(E) - es, np.sqrt(1 - es * es) * np.sin(E)
    vs = np.arctan2(yv, xv) / D2R
    rs = np.hypot(xv, yv)
    lsun = (vs + ws) % 360
    out['sun'] = lsun
    xs, ys = rs * np.cos(lsun * D2R), rs * np.sin(lsun * D2R)
    # Moon
    N = 125.1228 - 0.0529538083 * d
    i = 5.1454
    w = 318.0634 + 0.1643573223 * d
    a, e = 60.2666, 0.054900
    Mm = (115.3654 + 13.0649929509 * rate.get('moon', 1.0) * d) % 360
    E = _kepler(Mm * D2R, e)
    xv, yv = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    v, r = np.arctan2(yv, xv), np.hypot(xv, yv)
    Nr, ir, wr = N * D2R, i * D2R, w * D2R
    xh = r * (np.cos(Nr) * np.cos(v + wr) - np.sin(Nr) * np.sin(v + wr) * np.cos(ir))
    yh = r * (np.sin(Nr) * np.cos(v + wr) + np.cos(Nr) * np.sin(v + wr) * np.cos(ir))
    zh = r * np.sin(v + wr) * np.sin(ir)
    lon = np.arctan2(yh, xh) / D2R
    lat = np.arctan2(zh, np.hypot(xh, yh)) / D2R
    Ls = Ms + ws
    Lm = Mm + w + N
    Dd = (Lm - Ls) * D2R
    F = (Lm - N) * D2R
    Msr, Mmr = Ms * D2R, Mm * D2R
    lon = lon + (-1.274 * np.sin(Mmr - 2 * Dd) + 0.658 * np.sin(2 * Dd) - 0.186 * np.sin(Msr)
                 - 0.059 * np.sin(2 * Mmr - 2 * Dd) - 0.057 * np.sin(Mmr - 2 * Dd + Msr)
                 + 0.053 * np.sin(Mmr + 2 * Dd) + 0.046 * np.sin(2 * Dd - Msr) + 0.041 * np.sin(Mmr - Msr)
                 - 0.035 * np.sin(Dd) - 0.031 * np.sin(Mmr + Msr) - 0.015 * np.sin(2 * F - 2 * Dd)
                 + 0.011 * np.sin(Mmr - 4 * Dd))
    lat = lat + (-0.173 * np.sin(F - 2 * Dd) - 0.055 * np.sin(Mmr - F - 2 * Dd) - 0.046 * np.sin(Mmr + F - 2 * Dd)
                 + 0.033 * np.sin(F + 2 * Dd) + 0.017 * np.sin(2 * Mmr + F))
    out['moon'] = lon % 360
    out['moonlat'] = lat
    # planets
    Mj = (19.8950 + 0.0830853001 * d) % 360
    Msa = (316.9670 + 0.0334442282 * d) % 360
    for name, (N0, N1, i0, i1, w0, w1, a, e0, e1, M0, M1) in ELEM.items():
        N = (N0 + N1 * d) * D2R
        ii = (i0 + i1 * d) * D2R
        w = (w0 + w1 * d) * D2R
        e = e0 + e1 * d
        M = ((M0 + M1 * rate.get(name, 1.0) * d) % 360) * D2R
        E = _kepler(M, e)
        xv, yv = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
        v, r = np.arctan2(yv, xv), np.hypot(xv, yv)
        xh = r * (np.cos(N) * np.cos(v + w) - np.sin(N) * np.sin(v + w) * np.cos(ii))
        yh = r * (np.sin(N) * np.cos(v + w) + np.cos(N) * np.sin(v + w) * np.cos(ii))
        if name in ('jupiter', 'saturn'):
            hl = np.arctan2(yh, xh) / D2R
            mj, ms = Mj * D2R, Msa * D2R
            if name == 'jupiter':
                hl = hl + (-0.332 * np.sin(2 * mj - 5 * ms - 67.6 * D2R) - 0.056 * np.sin(2 * mj - 2 * ms + 21 * D2R)
                           + 0.042 * np.sin(3 * mj - 5 * ms + 21 * D2R) - 0.036 * np.sin(mj - 2 * ms)
                           + 0.022 * np.cos(mj - ms) + 0.023 * np.sin(2 * mj - 3 * ms + 52 * D2R)
                           - 0.016 * np.sin(mj - 5 * ms - 69 * D2R))
            else:
                hl = hl + (0.812 * np.sin(2 * mj - 5 * ms - 67.6 * D2R) - 0.229 * np.cos(2 * mj - 4 * ms - 2 * D2R)
                           + 0.119 * np.sin(mj - 2 * ms - 3 * D2R) + 0.046 * np.sin(2 * mj - 6 * ms - 69 * D2R)
                           + 0.014 * np.sin(mj - 3 * ms + 32 * D2R))
            rr = np.hypot(xh, yh)
            xh, yh = rr * np.cos(hl * D2R), rr * np.sin(hl * D2R)
        out[name] = (np.arctan2(yh + ys, xh + xs) / D2R) % 360
    return out


def jd_to_julian_cal(jd):
    """Julian-calendar (y, m, d) for integer-noon JD numbers (array)."""
    jd = np.asarray(jd, np.int64)
    c = jd + 32082
    dd = (4 * c + 3) // 1461
    e = c - (1461 * dd) // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    mon = m + 3 - 12 * (m // 10)
    yr = dd - 4800 + m // 10
    return yr, mon, day


def julian_cal_to_jd(y, m, d):
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - 32083


def easter_julian(y):
    a, b, c = y % 4, y % 7, y % 19
    dd = (19 * c + 15) % 30
    e = (2 * a + 4 * b - dd + 34) % 7
    mon = (dd + e + 114) // 31
    day = (dd + e + 114) % 31 + 1
    return mon, day


def build_daily(y0=1290, y1=1612, ut_hour=11.25, rate=None):
    """Daily table at about local noon in central Europe / N Italy (UT 11:15)."""
    jd0 = julian_cal_to_jd(y0, 1, 1)
    jd1 = julian_cal_to_jd(y1, 1, 1)
    jdn = np.arange(jd0, jd1)
    d = jdn - 0.5 + ut_hour / 24.0 - 2451543.5
    pos = positions(d, rate)
    yr, mo, dy = jd_to_julian_cal(jdn)
    tab = {'jdn': jdn, 'year': yr, 'month': mo, 'day': dy, 'weekday': (jdn + 1) % 7}  # 0 = Sunday
    tab.update(pos)
    return tab


if __name__ == '__main__':
    import sys, os
    out = sys.argv[1] if len(sys.argv) > 1 else 'sky_real.npz'
    t = build_daily()
    np.savez_compressed(out, **t)
    print('wrote', out, len(t['jdn']))
