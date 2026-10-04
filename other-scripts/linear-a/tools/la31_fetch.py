"""LA-31 'the winds carried the words': fetch terrain and wind (raw data stay in the scratchpad).

DEM: AWS Terrain Tiles (terrarium PNG, zoom 9, ~250 m), mosaicked to a 0.01 deg grid.
Wind: ERA5 daily mean 10 m wind speed and dominant direction via the Open-Meteo archive API,
on a 0.5 deg grid, 2014-2018 (5 years = 1,826 days per point).
Usage: python3 la31_fetch.py dem|wind
"""
import os, sys, io, json, math, time, urllib.request
import numpy as np

SCR = os.environ.get('LA31_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/la31')
LON0, LON1, LAT0, LAT1 = 21.5, 28.5, 34.4, 41.0


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def get(url, path):
    if os.path.exists(path):
        return open(path, 'rb').read()
    for k in range(5):
        try:
            b = urllib.request.urlopen(url, timeout=60).read()
            open(path, 'wb').write(b)
            return b
        except Exception as e:  # noqa
            print('retry', url, e, flush=True); time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def dem():
    from PIL import Image
    z = 9
    x0, y1 = tile_xy(LON0, LAT0, z); x1, y0 = tile_xy(LON1, LAT1, z)
    xs = range(int(x0), int(x1) + 1); ys = range(int(y0), int(y1) + 1)
    os.makedirs(os.path.join(SCR, 'dem'), exist_ok=True)
    tiles = {}
    for x in xs:
        for y in ys:
            b = get(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
                    os.path.join(SCR, 'dem', f'{z}_{x}_{y}.png'))
            a = np.asarray(Image.open(io.BytesIO(b)).convert('RGB')).astype(np.float64)
            tiles[(x, y)] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    # resample to 0.01 deg grid (cell centres)
    res = 0.01
    lons = np.arange(LON0 + res / 2, LON1, res); lats = np.arange(LAT1 - res / 2, LAT0, -res)
    H = np.zeros((len(lats), len(lons)), np.float32)
    n = 2 ** z
    for i, la in enumerate(lats):
        _, ty = tile_xy(0, la, z)
        tx = (lons + 180) / 360 * n
        X = tx.astype(int); Y = int(ty)
        px = ((tx - X) * 256).astype(int); py = int((ty - Y) * 256)
        H[i] = [tiles[(xx, Y)][py, pp] for xx, pp in zip(X, px)]
    np.save(os.path.join(SCR, 'dem_001.npy'), H)
    json.dump(dict(lon0=float(lons[0]), lat0=float(lats[0]), res=res, shape=H.shape),
              open(os.path.join(SCR, 'dem_001.json'), 'w'))
    print('DEM', H.shape, 'land frac', (H > 0).mean())


def wind():
    H = np.load(os.path.join(SCR, 'dem_001.npy'))
    meta = json.load(open(os.path.join(SCR, 'dem_001.json')))
    res = 0.5
    lons = np.arange(LON0, LON1 + 1e-9, res); lats = np.arange(LAT0 + 0.1, LAT1 + 1e-9, res)
    pts = []
    for la in lats:
        for lo in lons:
            # keep points within ~0.6 deg of sea
            i0 = int((meta['lat0'] - la) / 0.01); j0 = int((lo - meta['lon0']) / 0.01)
            sl = H[max(0, i0 - 60):i0 + 60, max(0, j0 - 60):j0 + 60]
            if sl.size and (sl <= 0).any():
                pts.append((round(float(la), 2), round(float(lo), 2)))
    print('wind points', len(pts), flush=True)
    out = os.path.join(SCR, 'wind'); os.makedirs(out, exist_ok=True)
    for k in range(0, len(pts), 10):
        ch = pts[k:k + 10]
        f = os.path.join(out, f'chunk_{k:04d}.json')
        if os.path.exists(f):
            continue
        url = ('https://archive-api.open-meteo.com/v1/archive?latitude=' + ','.join(str(p[0]) for p in ch)
               + '&longitude=' + ','.join(str(p[1]) for p in ch)
               + '&start_date=2014-01-01&end_date=2018-12-31&models=era5'
               + '&daily=wind_speed_10m_mean,wind_direction_10m_dominant&timezone=GMT')
        for t in range(8):
            try:
                b = urllib.request.urlopen(url, timeout=120).read()
                d = json.loads(b)
                if isinstance(d, dict):
                    d = [d]
                for p, r in zip(ch, d):
                    r['req'] = p
                json.dump(d, open(f, 'w'))
                break
            except Exception as e:  # noqa
                print('retry', k, e, flush=True); time.sleep(60 * (t + 1))
        print('chunk', k, flush=True); time.sleep(8)


if __name__ == '__main__':
    {'dem': dem, 'wind': wind}[sys.argv[1]]()
