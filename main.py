"""
Geo-Detection-System — prototyp (proof-of-concept).

Pipeline dla JEDNEGO zdjęcia:
  EXIF (GPS) -> NMT z GUGiK (WCS) -> parametry kamery z ODM (IO+EO)
  -> YOLO (LudzieiNamioty.pt) -> środki bboxów -> ray casting na NMT
  -> GeoJSON + opcjonalny podglad w folium.

Uwagi do konwencji:
  - OpenSfM/ODM: rotation w `reconstruction.json` jest wektorem angle-axis (Rodriguesa),
    macierz R przeksztalca *world -> camera*, a translation `t` jest tak dobrane, ze
    pozycja kamery C w lokalnym ukladzie wynosi C = -R^T * t.
  - Lokalny uklad OpenSfM to topocentric ENU wokol `reference_lla.json` (origin geo).
    Dla malego obszaru zakladamy E ~ X(EPSG:2180), N ~ Y(EPSG:2180) — to upraszcza
    transformacje promienia. Roznica (meridian convergence) jest <<1 stopnia.
  - Piksele: origin top-left, os v w dol — `cv2.undistortPoints` tego pilnuje.
  - NMPT (pokrycie terenu) nie jest tu uzywane — dla obiektow wyniesionych nad teren
    (drzewo, dach) bedzie znaczace odchylenie. Do dolozenia w kolejnej iteracji.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import cv2
import exifread
import numpy as np
import rasterio
import requests
from pyproj import Transformer

# ───────────────────────── KONFIGURACJA ─────────────────────────

PROJECT_DIR = Path(__file__).parent
PHOTOS_DIR = PROJECT_DIR / "zdjecia"
MODEL_PATH = PROJECT_DIR / "LudzieiNamioty.pt"

# Zdjecie do prototypu — zmien wedle uznania
IMAGE_NAME = "SING0316.JPG"

# Sciezka do wynikow ODM (po przepuszczeniu zdjec przez OpenDroneMap)
ODM_PROJECT_DIR = PROJECT_DIR / "odm_project"
ODM_RECONSTRUCTION = ODM_PROJECT_DIR / "opensfm" / "reconstruction.json"
ODM_REFERENCE_LLA = ODM_PROJECT_DIR / "opensfm" / "reference_lla.json"
ODM_SHOTS_GEOJSON = ODM_PROJECT_DIR / "odm_report" / "shots.geojson"

# NMT — wynik pobrany z WCS GUGiK lub recznie wskazany lokalny GeoTIFF
NMT_DIR = PROJECT_DIR / "nmt"
LOCAL_NMT_FALLBACK: Optional[Path] = None  # np. Path(r"C:\sciezka\do\nmt.tif")

# Kalibracja Z (offset altitudy). Potensic Atom zapisuje GPS Altitude jako AGL od punktu
# startu drona — nie npm. Lokalny uklad ODM dziedziczy to przesuniecie, wiec bez korekty
# kamera "jest pod terenem" i ray casting nigdy nie trafia w DEM.
#
# Dla teraz: zakladamy, ze srednia altitude lokalna kamer ≈ AGL_LOT_M nad terenem
# w okolicy zdjec (offset = NMT(median_xy) + AGL_LOT_M - mean(alt_local)).
#
# Na przyszlosc: lepiej wykonac pierwsze zdjecie na ziemi w punkcie startu — wtedy
# offset = NMT(start_xy) - alt_local_startowego_zdjecia (bez zalozen o AGL).
AGL_LOT_M = 45.0

# Otwarte API GUGiK do punktowego odpytywania NMT (1 m, EPSG:2180).
# WCS (mapy.geoportal.gov.pl/.../WCS/...) od 2024 wymaga autoryzacji — dlatego
# zamiast pobierac raster, probkujemy NMT punktowo na zadanie (z lokalnym cache).
GUGIK_NMT_URL = "https://services.gugik.gov.pl/nmt/"

# Wynikowe pliki
OUT_GEOJSON = PROJECT_DIR / "detections.geojson"
OUT_MAP_HTML = PROJECT_DIR / "detections_map.html"

# Klasy modelu — dopasuj do tego, co rzeczywiscie zwraca .pt
CLASS_NAMES = {0: "namiot", 1: "people"}


# ───────────────────────── 1. EXIF ─────────────────────────

def _ratio_to_float(v) -> float:
    """exifread zwraca Ratio lub list[Ratio]. Dla Lat/Lon lista = [d,m,s];
    dla Altitude lista bywa jednoelementowa [Ratio]."""
    if isinstance(v, list):
        nums = [float(x.num) / float(x.den) for x in v]
        if len(nums) == 3:
            d, m, s = nums
            return d + m / 60.0 + s / 3600.0
        if len(nums) == 1:
            return nums[0]
        raise ValueError(f"Nieznany ksztalt GPS Ratio: {v}")
    return float(v.num) / float(v.den)


def read_gps_from_exif(image_path: Path) -> tuple[float, float, Optional[float]]:
    """Zwraca (lat, lon, alt_m_above_msl_or_ellipsoid)."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    lat = _ratio_to_float(tags["GPS GPSLatitude"].values)
    lon = _ratio_to_float(tags["GPS GPSLongitude"].values)
    if str(tags.get("GPS GPSLatitudeRef", "N")) == "S":
        lat = -lat
    if str(tags.get("GPS GPSLongitudeRef", "E")) == "W":
        lon = -lon

    alt = None
    if "GPS GPSAltitude" in tags:
        alt = _ratio_to_float(tags["GPS GPSAltitude"].values)
        if str(tags.get("GPS GPSAltitudeRef", "0")) in ("1", "\x01"):
            alt = -alt
    return lat, lon, alt


# ───────────────────────── 2. NMT — abstrakcja DEM ─────────────────────────

class DemSampler(Protocol):
    """Wspolny interfejs dla zrodel NMT. `sample` zwraca wysokosc npm
    w punkcie (x, y) w EPSG:2180 lub None gdy brak danych."""
    def sample(self, x: float, y: float) -> Optional[float]: ...


class GugikDem:
    """NMT z otwartego REST GUGiK (services.gugik.gov.pl/nmt/?request=GetHByXY).
    Wspolrzedne w EPSG:2180. Wynik cache'owany w pamieci po zaokragleniu do 1 m."""

    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 10.0):
        self.session = session or requests.Session()
        self.timeout = timeout
        self._cache: dict[tuple[int, int], Optional[float]] = {}
        self.hits = 0
        self.misses = 0

    def sample(self, x: float, y: float) -> Optional[float]:
        key = (int(round(x)), int(round(y)))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        try:
            r = self.session.get(
                GUGIK_NMT_URL,
                params={"request": "GetHByXY", "x": x, "y": y},
                timeout=self.timeout,
            )
            if not r.ok:
                self._cache[key] = None
                return None
            body = r.text.strip()
            try:
                z = float(body)
            except ValueError:
                self._cache[key] = None
                return None
            # "0" zwraca GUGiK gdy punkt poza zasiegiem lub w niewlasciwym CRS
            if z == 0.0:
                self._cache[key] = None
                return None
            self._cache[key] = z
            return z
        except requests.RequestException:
            self._cache[key] = None
            return None


class RasterDem:
    """NMT z lokalnego GeoTIFF (np. recznie pobrany kafelek)."""

    def __init__(self, path: Path):
        self.path = path
        self._dem = rasterio.open(path)

    def sample(self, x: float, y: float) -> Optional[float]:
        for val in self._dem.sample([(x, y)]):
            z = float(val[0])
            if self._dem.nodata is not None and z == self._dem.nodata:
                return None
            if math.isnan(z):
                return None
            return z
        return None


def resolve_dem() -> DemSampler:
    """Wybiera zrodlo NMT: jesli wskazany lokalny GeoTIFF — uzywa go,
    w przeciwnym razie GugikDem (REST point-query)."""
    if LOCAL_NMT_FALLBACK and LOCAL_NMT_FALLBACK.exists():
        print(f"[NMT] lokalny GeoTIFF: {LOCAL_NMT_FALLBACK}")
        return RasterDem(LOCAL_NMT_FALLBACK)
    print("[NMT] GUGiK REST (services.gugik.gov.pl/nmt, point-query + cache)")
    return GugikDem()


# ───────────────────────── 3. ODM — IO + EO ─────────────────────────

@dataclass
class CameraIO:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # [k1, k2, p1, p2, k3]
    width: int
    height: int


@dataclass
class CameraEO:
    """Pozycja kamery w ukladzie EPSG:2180 + R world->camera w tym samym ukladzie."""
    C_xyz: np.ndarray  # (3,) — X,Y,Z w metrach (X,Y=EPSG:2180, Z=npm)
    R_wc: np.ndarray   # (3,3) — world -> camera


def _opensfm_intrinsics_to_pixels(cam_params: dict, w: int, h: int) -> CameraIO:
    """OpenSfM: focal jest znormalizowany do max(w,h); principal point (cx,cy) — w jednostkach
    znormalizowanych wzgledem max(w,h), z (0,0) w srodku obrazu.

    Model 'perspective' uzywa pola `focal` (jedna ogniskowa), model 'brown' rozdziela
    na `focal_x` i `focal_y`."""
    norm = max(w, h)
    if "focal_x" in cam_params:
        fx = float(cam_params["focal_x"]) * norm
        fy = float(cam_params["focal_y"]) * norm
    else:
        fx = fy = float(cam_params["focal"]) * norm
    cx = w / 2.0 + float(cam_params.get("c_x", 0.0)) * norm
    cy = h / 2.0 + float(cam_params.get("c_y", 0.0)) * norm
    dist = np.array([
        float(cam_params.get("k1", 0.0)),
        float(cam_params.get("k2", 0.0)),
        float(cam_params.get("p1", 0.0)),
        float(cam_params.get("p2", 0.0)),
        float(cam_params.get("k3", 0.0)),
    ], dtype=np.float64)
    return CameraIO(fx, fy, cx, cy, dist, w, h)


def parse_odm_for_image(image_name: str, z_offset: float = 0.0
                        ) -> tuple[CameraIO, CameraEO]:
    """Czyta IO+EO dla podanego zdjecia z wynikow ODM/OpenSfM.

    z_offset: dodatkowa korekta wysokosci (npm) doliczana do C[2]. Sluzy
    skompensowaniu faktu, ze altitudy w ODM dla Potensica sa AGL, nie npm.
    """
    if not ODM_RECONSTRUCTION.exists():
        raise FileNotFoundError(
            f"Brak {ODM_RECONSTRUCTION}. Najpierw przepusc zdjecia przez ODM "
            f"(patrz README — sekcja ODM)."
        )

    rec = json.loads(ODM_RECONSTRUCTION.read_text(encoding="utf-8"))[0]
    if image_name not in rec["shots"]:
        raise KeyError(f"{image_name} nie ma w reconstruction.json")

    shot = rec["shots"][image_name]
    cam_id = shot["camera"]
    cam = rec["cameras"][cam_id]

    # IO
    width, height = int(cam["width"]), int(cam["height"])
    io = _opensfm_intrinsics_to_pixels(cam, width, height)

    # EO — w lokalnym ukladzie topocentric
    rvec = np.array(shot["rotation"], dtype=np.float64)  # angle-axis (world->camera)
    tvec = np.array(shot["translation"], dtype=np.float64)
    R_wc, _ = cv2.Rodrigues(rvec)
    C_local = -R_wc.T @ tvec  # pozycja kamery w lokalnym ENU (origin = reference_lla)

    # Przenies pozycje kamery z lokalnego ENU do EPSG:2180
    ref = json.loads(ODM_REFERENCE_LLA.read_text(encoding="utf-8"))
    ref_lat, ref_lon = float(ref["latitude"]), float(ref["longitude"])
    ref_alt = float(ref.get("altitude", 0.0))

    t_geo = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    ref_x, ref_y = t_geo.transform(ref_lon, ref_lat)

    # ZALOZENIE: lokalne osie ENU pokrywaja sie z osiami EPSG:2180 (X=E, Y=N, Z=Up).
    # Dla malego obszaru blad meridian convergence jest pomijalny.
    C_xyz = np.array([
        ref_x + C_local[0],
        ref_y + C_local[1],
        ref_alt + C_local[2] + z_offset,
    ])

    eo = CameraEO(C_xyz=C_xyz, R_wc=R_wc)
    return io, eo


def compute_z_offset(dem: DemSampler, agl_assumed_m: float = AGL_LOT_M) -> float:
    """Liczy korekte Z (npm), kompensujaca AGL-owy origin Z w ODM/Potensicu.

    Bierze srednia altitude lokalna kamer z shots.geojson, sampluje DEM w pozycji
    medianowej kamery i zaklada, ze srednia kamera leciala AGL_LOT_M nad terenem.
    Zwraca wartosc, ktora nalezy dodac do C[2] w lokalnym ukladzie ODM.
    """
    if not ODM_SHOTS_GEOJSON.exists():
        print(f"[Z] brak {ODM_SHOTS_GEOJSON} — z_offset=0")
        return 0.0
    geo = json.loads(ODM_SHOTS_GEOJSON.read_text(encoding="utf-8"))
    feats = geo["features"]
    lons = np.array([f["geometry"]["coordinates"][0] for f in feats])
    lats = np.array([f["geometry"]["coordinates"][1] for f in feats])
    alts = np.array([f["geometry"]["coordinates"][2] for f in feats])

    med_lon, med_lat = float(np.median(lons)), float(np.median(lats))
    mean_alt_local = float(np.mean(alts))

    t = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x, y = t.transform(med_lon, med_lat)
    nmt_under = dem.sample(x, y)
    if nmt_under is None:
        print(f"[Z] DEM nie zwrocil wartosci dla median_xy=({x:.1f},{y:.1f}); z_offset=0")
        return 0.0

    offset = nmt_under + agl_assumed_m - mean_alt_local
    print(f"[Z] NMT(median_xy)={nmt_under:.2f} m npm, "
          f"mean_alt_local={mean_alt_local:.2f}, AGL={agl_assumed_m} "
          f"-> z_offset={offset:.2f} m")
    return offset


# ───────────────────────── 4. Detekcja YOLO ─────────────────────────

def detect_objects(image_path: Path) -> list[dict]:
    """Zwraca liste detekcji: {cls_id, cls_name, conf, bbox(xyxy), center(u,v)}."""
    from ultralytics import YOLO  # import lazy — zeby skrypt sie ladowal szybko
    model = YOLO(str(MODEL_PATH))
    results = model(str(image_path), verbose=False)[0]

    out = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "cls_id": cls_id,
            "cls_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "conf": float(box.conf.item()),
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        })
    return out


# ───────────────────────── 5. Ray casting ─────────────────────────

def pixel_to_world_ray(uv: tuple[float, float], io: CameraIO, eo: CameraEO
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (origin, direction) promienia w ukladzie swiata (EPSG:2180 + Z).
    `direction` jest jednostkowy."""
    K = np.array([[io.fx, 0, io.cx],
                  [0, io.fy, io.cy],
                  [0, 0, 1.0]], dtype=np.float64)
    pts = np.array([[[uv[0], uv[1]]]], dtype=np.float64)  # shape (1,1,2)
    undist = cv2.undistortPoints(pts, K, io.dist)  # znormalizowane (x,y) w plaszczyznie obrazu
    x, y = undist[0, 0]
    d_cam = np.array([x, y, 1.0])  # promien w ukl. kamery (z patrzy w przod)

    # World -> Camera: x_c = R * x_w  =>  Camera -> World: x_w = R^T * x_c
    d_world = eo.R_wc.T @ d_cam
    d_world /= np.linalg.norm(d_world)
    return eo.C_xyz.copy(), d_world


def raycast_on_dem(origin: np.ndarray, direction: np.ndarray, dem: DemSampler,
                   t_min: float = 0.5, t_max: float = 800.0,
                   step: float = 5.0) -> Optional[np.ndarray]:
    """Marsz po promieniu + bisekcja. Zwraca punkt przeciecia (X,Y,Z) w EPSG:2180 lub None.

    Krok 5 m jest kompromisem dla zrodla NMT typu point-query (kazdy step to potencjalny
    HTTP request); bisekcja na koncu doprecyzowuje do <0.1 m. Dla obiektow lezacych
    na ziemi (namiot, czlowiek) to wystarczy.

    Promien zakladamy "w dol" (jezeli direction[2] >= 0, nie ma sensu szukac przeciecia
    z terenem rozciagajacym sie w dol)."""
    if direction[2] >= 0:
        return None

    prev_t = None
    prev_diff = None
    t = t_min
    while t <= t_max:
        p = origin + t * direction
        z_dem = dem.sample(p[0], p[1])
        if z_dem is not None:
            diff = p[2] - z_dem  # >0 nad terenem, <0 pod
            if prev_diff is not None and (diff <= 0 < prev_diff):
                # Bisekcja miedzy prev_t a t
                lo, hi = prev_t, t
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    pm = origin + mid * direction
                    zm = dem.sample(pm[0], pm[1])
                    if zm is None:
                        break
                    if pm[2] - zm > 0:
                        lo = mid
                    else:
                        hi = mid
                t_hit = (lo + hi) / 2.0
                return origin + t_hit * direction
            prev_t, prev_diff = t, diff
        t += step
    return None


# ───────────────────────── 6. Eksport ─────────────────────────

def export_geojson(features: list[dict], out_path: Path) -> None:
    fc = {"type": "FeatureCollection", "features": []}
    t = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    for f in features:
        x, y, z = f["xyz_2180"]
        lon, lat = t.transform(x, y)
        fc["features"].append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "class": f["cls_name"],
                "conf": round(f["conf"], 3),
                "altitude_m": round(z, 2),
                "pixel_u": round(f["uv"][0], 1),
                "pixel_v": round(f["uv"][1], 1),
            },
        })
    out_path.write_text(json.dumps(fc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] GeoJSON -> {out_path}")


def make_folium_preview(features: list[dict], cam_lat: float, cam_lon: float,
                        out_html: Path) -> None:
    try:
        import folium
    except ImportError:
        return
    m = folium.Map(location=[cam_lat, cam_lon], zoom_start=19, tiles="OpenStreetMap")
    folium.Marker([cam_lat, cam_lon], tooltip="kamera", icon=folium.Icon(color="blue")).add_to(m)
    t = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    colors = {"namiot": "red", "people": "green"}
    for f in features:
        x, y, _ = f["xyz_2180"]
        lon, lat = t.transform(x, y)
        folium.CircleMarker(
            [lat, lon], radius=6,
            color=colors.get(f["cls_name"], "orange"),
            fill=True, fill_opacity=0.8,
            tooltip=f"{f['cls_name']} ({f['conf']:.2f})",
        ).add_to(m)
    m.save(str(out_html))
    print(f"[OUT] mapa -> {out_html}")


# ───────────────────────── MAIN ─────────────────────────

def main() -> None:
    image_path = PHOTOS_DIR / IMAGE_NAME
    print(f"[1] EXIF z {image_path.name}")
    lat, lon, alt = read_gps_from_exif(image_path)
    print(f"    GPS: lat={lat:.6f}, lon={lon:.6f}, alt={alt}")

    print("[2] NMT (sampler)")
    dem = resolve_dem()

    print("[3] Kalibracja Z (offset altitudy)")
    z_offset = compute_z_offset(dem, AGL_LOT_M)

    print("[3b] ODM IO+EO")
    io, eo = parse_odm_for_image(IMAGE_NAME, z_offset=z_offset)
    print(f"    fx={io.fx:.1f}, fy={io.fy:.1f}, cx={io.cx:.1f}, cy={io.cy:.1f}")
    print(f"    C (EPSG:2180) = {eo.C_xyz}")

    print("[4] Detekcja YOLO")
    dets = detect_objects(image_path)
    print(f"    znaleziono {len(dets)} obiektow")

    print("[5] Ray casting")
    features = []
    for d in dets:
        origin, direction = pixel_to_world_ray(d["center"], io, eo)
        hit = raycast_on_dem(origin, direction, dem)
        if hit is None:
            print(f"    [skip] {d['cls_name']} u,v={d['center']} — brak przeciecia z DEM")
            continue
        features.append({
            "cls_name": d["cls_name"],
            "conf": d["conf"],
            "uv": d["center"],
            "xyz_2180": hit,
        })
        print(f"    {d['cls_name']}: XYZ={hit}")

    print("[6] Eksport")
    export_geojson(features, OUT_GEOJSON)
    make_folium_preview(features, lat, lon, OUT_MAP_HTML)
    if isinstance(dem, GugikDem):
        print(f"[NMT cache] hits={dem.hits}, http_requests={dem.misses}")
    print("[OK] gotowe.")


if __name__ == "__main__":
    main()
