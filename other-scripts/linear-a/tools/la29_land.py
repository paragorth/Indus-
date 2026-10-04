#!/usr/bin/env python3
"""la29: THE LAND WROTE THE LEDGER. Step 1: per-site land variables (data only).

Sources (open data; rasters stay in the scratchpad, only derived per-site numbers are written):
  - DEM: AWS Terrain Tiles (terrarium PNG, zoom 11, ~63 m at 35 N; SRTM/GMTED/ETOPO blend, has bathymetry)
  - WorldClim 2.1 2.5' monthly precipitation and mean temperature (1970-2000)
  - Corine Land Cover 2018 (EEA ArcGIS REST identify, raster layer), sampled on a point lattice
  - SoilGrids 2.0 REST point query (0-5 cm clay, sand, SOC, pH)
Output: data/la29_land.json  {site: {var_r: value}}, data/la29_ckpt/crete_maps.json (corr. lengths)
"""
import io, json, math, os, sys, time, urllib.request
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la29_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('LA29_SCR', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/geo')
TILES = os.path.join(SCR, 'tiles')
os.makedirs(TILES, exist_ok=True)
Z = 11

# approximate site coordinates (lat, lon); ~0.5-1 km accuracy is enough for 2-10 km catchments
SITES = {
    # Linear A, Crete
    'HT': (35.0592, 24.7922), 'PH': (35.0513, 24.8137), 'KN': (35.2979, 25.1631),
    'KH': (35.5170, 24.0190), 'ZA': (35.0981, 26.2613), 'PK': (35.1937, 26.2607),
    'MA': (35.2925, 25.4926), 'TY': (35.2995, 25.0153), 'AR': (35.1460, 25.2660),
    'PE': (35.2000, 26.1180), 'IO': (35.2283, 25.1190), 'SY': (35.0700, 25.4300),
    'GO': (35.1063, 25.7932), 'PY': (35.0100, 25.5900), 'MK': (35.1855, 25.9050),
    'PS': (35.1626, 25.4450), 'VR': (35.3340, 24.4800), 'KO': (34.9930, 25.0800),
    'AP': (35.1800, 24.6800), 'ZO': (35.2900, 24.8800),
    # Linear A, islands
    'THE': (36.3515, 25.4035), 'KEA': (37.6615, 24.3290), 'MI': (36.7560, 24.5050),
    # Linear B control (KN, KH shared)
    'PYL': (37.0281, 21.6962), 'THB': (38.3220, 23.3190), 'MYC': (37.7308, 22.7564),
    'TIR': (37.5996, 22.8002),
}
LA_NAME = {'Haghia Triada': 'HT', 'Phaistos': 'PH', 'Knossos': 'KN', 'Khania': 'KH', 'Zakros': 'ZA',
           'Palaikastro': 'PK', 'Malia': 'MA', 'Tylissos': 'TY', 'Arkhalkhori': 'AR', 'Petras': 'PE',
           'Iouktas': 'IO', 'Syme': 'SY', 'Gournia': 'GO', 'Pyrgos': 'PY', 'Mokhilos': 'MK',
           'Psykhro': 'PS', 'Vrysinas': 'VR', 'Kophinas': 'KO', 'Apodoulou': 'AP', 'Zominthos': 'ZO',
           'Thera': 'THE', 'Kea': 'KEA', 'Milos': 'MI'}
RADII = [2, 5, 10]


def tile_xy(lat, lon, z=Z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def get_tile(x, y, z=Z):
    p = os.path.join(TILES, f'{z}_{x}_{y}.png')
    if not os.path.exists(p):
        u = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
        for k in range(4):
            try:
                b = urllib.request.urlopen(u, timeout=60).read(); open(p, 'wb').write(b); break
            except Exception as e:
                time.sleep(2 + 3 * k)
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float64)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def mosaic(lat, lon, half_tiles):
    fx, fy = tile_xy(lat, lon)
    x0, y0 = int(fx) - half_tiles, int(fy) - half_tiles
    k = 2 * half_tiles + 1
    M = np.zeros((256 * k, 256 * k))
    for i in range(k):
        for j in range(k):
            M[256 * j:256 * (j + 1), 256 * i:256 * (i + 1)] = get_tile(x0 + i, y0 + j)
    px = (fx - x0) * 256; py = (fy - y0) * 256
    res = 156543.03392 * math.cos(math.radians(lat)) / 2 ** Z   # m per pixel
    return M, px, py, res


def slope_deg(M, res):
    gy, gx = np.gradient(M, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def dem_vars(lat, lon):
    M, px, py, res = mosaic(lat, lon, 4)       # 9x9 tiles ~ 145 km square
    S = slope_deg(M, res)
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    d = np.hypot(xx - px, yy - py) * res / 1000.0     # km
    land = M > 0
    sea = ~land
    out = {}
    sd = d[sea]
    out['coast_km'] = float(sd.min()) if sd.size else 80.0
    out['site_elev'] = float(M[int(py), int(px)])
    for r in RADII:
        disc = d <= r
        L = disc & land
        n = max(L.sum(), 1)
        e = M[L]; s = S[L]
        out[f'land_{r}'] = float(L.sum() / disc.sum())
        out[f'elev_{r}'] = float(e.mean()) if e.size else 0.0
        out[f'elevsd_{r}'] = float(e.std()) if e.size else 0.0
        out[f'slope_{r}'] = float(s.mean()) if s.size else 0.0
        out[f'arable_{r}'] = float(((s < 5) & (e < 600)).sum() / n)
        out[f'upland_{r}'] = float(((s > 15) | (e > 800)).sum() / n)
        out[f'olive_{r}'] = float(((e < 500) & (s < 25)).sum() / n)
        out[f'vine_{r}'] = float(((e > 50) & (e < 700) & (s > 2) & (s < 20)).sum() / n)
        out[f'high_{r}'] = float((e > 1000).sum() / n)
    return out


_wc = {}


def wc(kind, m):
    k = (kind, m)
    if k not in _wc:
        import tifffile
        _wc[k] = tifffile.imread(os.path.join(SCR, kind, f'wc2.1_2.5m_{kind}_{m:02d}.tif')).astype(np.float64)
    return _wc[k]


def wc_vars(lat, lon):
    out = {}
    cell = 2.5 / 60
    for kind in ('prec', 'tavg'):
        for r in (2, 5, 10):
            dr = r / 111.0; dc = r / (111.0 * math.cos(math.radians(lat)))
            vals = []
            for mth in range(1, 13):
                A = wc(kind, mth)
                i0 = int((90 - (lat + dr)) / cell); i1 = int((90 - (lat - dr)) / cell) + 1
                j0 = int((lon - dc + 180) / cell); j1 = int((lon + dc + 180) / cell) + 1
                blk = A[i0:i1, j0:j1]
                blk = blk[blk > -1000]
                vals.append(blk.mean() if blk.size else np.nan)
            vals = np.array(vals)
            if kind == 'prec':
                out[f'rain_{r}'] = float(np.nansum(vals))
                out[f'summerdry_{r}'] = float(np.nansum(vals[5:8]) / max(np.nansum(vals), 1))
            else:
                out[f'temp_{r}'] = float(np.nanmean(vals))
                out[f'tmin_{r}'] = float(np.nanmin(vals))
    return out


CLC_URL = ('https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/identify?'
           'geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all:1&tolerance=0&'
           'mapExtent={a},{b},{c},{d}&imageDisplay=400,400,96&returnGeometry=false&f=json')
CLC_GROUP = {'Non-irrigated arable land': 'cl_arable', 'Permanently irrigated land': 'cl_arable',
             'Olive groves': 'cl_olive', 'Vineyards': 'cl_vine', 'Pastures': 'cl_pasture',
             'Natural grasslands': 'cl_pasture', 'Sclerophyllous vegetation': 'cl_shrub',
             'Transitional woodland-shrub': 'cl_shrub', 'Sparsely vegetated areas': 'cl_bare',
             'Bare rocks': 'cl_bare', 'Complex cultivation patterns': 'cl_mixedag',
             'Land principally occupied by agriculture, with significant areas of natural vegetation': 'cl_mixedag',
             'Fruit trees and berry plantations': 'cl_orchard'}


def clc_point(lat, lon):
    u = CLC_URL.format(lon=lon, lat=lat, a=lon - .01, b=lat - .01, c=lon + .01, d=lat + .01)
    for k in range(3):
        try:
            j = json.loads(urllib.request.urlopen(u, timeout=60).read())
            for r in j.get('results', []):
                lab = r['attributes'].get('Raster.LABEL3') or r['attributes'].get('LABEL3')
                if lab: return lab
            return 'none'
        except Exception:
            time.sleep(2 + 2 * k)
    return 'err'


def clc_vars(lat, lon, r=5.0, step=1.0):
    cache = os.path.join(CK, f'clc_{lat:.4f}_{lon:.4f}.json')
    if os.path.exists(cache):
        labs = json.load(open(cache))
    else:
        labs = []
        n = int(r / step)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if math.hypot(i, j) * step > r: continue
                la = lat + i * step / 111.0; lo = lon + j * step / (111.0 * math.cos(math.radians(lat)))
                labs.append(clc_point(la, lo))
        json.dump(labs, open(cache, 'w'))
    land = [l for l in labs if l not in ('none', 'err', 'Sea and ocean', 'Water bodies', 'Coastal lagoons')]
    out = {g: 0.0 for g in set(CLC_GROUP.values())}
    for l in land:
        g = CLC_GROUP.get(l)
        if g: out[g] += 1
    nl = max(len(land), 1)
    return {k + '_5': v / nl for k, v in out.items()}


def soil_vars(lat, lon):
    cache = os.path.join(CK, f'soil_{lat:.4f}_{lon:.4f}.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    u = (f'https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}'
         '&property=clay&property=sand&property=soc&property=phh2o&depth=0-5cm&value=mean')
    out = {}
    for k in range(3):
        try:
            j = json.loads(urllib.request.urlopen(u, timeout=60).read())
            for L in j['properties']['layers']:
                v = L['depths'][0]['values']['mean']
                out['soil_' + L['name']] = None if v is None else v / L['unit_measure']['d_factor']
            break
        except Exception:
            time.sleep(3 + 3 * k)
    json.dump(out, open(cache, 'w'))
    time.sleep(1)
    return out


def corr_length(lat0=35.25, lon0=25.0):
    """Correlation length (km) of catchment-averaged DEM variables over central/east Crete."""
    res_out = {}
    M, px, py, res = mosaic(lat0, lon0, 6)
    S = slope_deg(M, res)
    land = M > 0
    from scipy.ndimage import uniform_filter
    k = int(5000 / res)
    fl = uniform_filter(land.astype(float), k) + 1e-9
    maps = {'elev_5': uniform_filter(np.where(land, M, 0), k) / fl,
            'slope_5': uniform_filter(np.where(land, S, 0), k) / fl,
            'land_5': fl}
    step = int(1000 / res)
    for name, A in maps.items():
        A = A[::step, ::step]; Lm = land[::step, ::step]
        lags = np.arange(1, 60)
        cor = []
        a = (A - A[Lm].mean()) / A[Lm].std()
        for h in lags:
            x = a[:, :-h][Lm[:, :-h] & Lm[:, h:]]; y = a[:, h:][Lm[:, :-h] & Lm[:, h:]]
            cor.append(np.corrcoef(x, y)[0, 1])
        cor = np.array(cor)
        L = float(lags[np.argmax(cor < math.exp(-1))]) if (cor < math.exp(-1)).any() else 60.0
        res_out[name] = {'efold_km': L, 'cor': [round(c, 3) for c in cor[:30]]}
    return res_out


if __name__ == '__main__':
    out = {}
    for s, (lat, lon) in SITES.items():
        v = {'lat': lat, 'lon': lon}
        v.update(dem_vars(lat, lon))
        v.update(wc_vars(lat, lon))
        v.update(clc_vars(lat, lon))
        v.update(soil_vars(lat, lon))
        out[s] = v
        print(s, {k: round(x, 2) for k, x in v.items() if isinstance(x, float) and k.endswith(('_5', 'km'))}, flush=True)
    json.dump(out, open(os.path.join(DATA, 'la29_land.json'), 'w'), indent=1)
    cl = corr_length()
    json.dump(cl, open(os.path.join(CK, 'crete_corr.json'), 'w'), indent=1)
    print({k: v['efold_km'] for k, v in cl.items()})
