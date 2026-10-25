# Geo-Detection-System

Praca inżynierska (WGiK PW): *Wykorzystanie algorytmów widzenia maszynowego w automatycznej
klasyfikacji i georeferencji wybranych obiektów na zdjęciach UAV*.

System wykrywa obiekty (namioty, ludzie) na zdjęciach z drona modelem YOLO i wyznacza ich
współrzędne metodą ray castingu. Promień biegnie z kamery przez piksel obiektu i jest
przecinany z modelem wysokościowym (NMT z GUGiK lub lokalny GeoTIFF NMT/NMPT). Orientacja
(EO) i kalibracja (IO) kamery pochodzą z rekonstrukcji SfM w OpenDroneMap, bo Potensic
Atom zapisuje w EXIF wyłącznie GPS.

## Instalacja (Windows, venv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# PyTorch z CUDA 12.4 — osobno, z indeksu PyTorcha:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python chech_env.py        # sprawdza CUDA i klasy modelu
```

ODM (Docker) uruchamia się raz na lot. Instrukcja jest w [ODM.md](ODM.md). Potrzebne pliki
wynikowe to `odm_project/opensfm/reconstruction.json`, `reference_lla.json`
i `odm_project/odm_report/shots.geojson`.

## Uruchomienie

```powershell
python main.py                                   # config.yaml
python main.py --config config.yaml
python main.py --image SING0304.JPG --image SING0316.JPG
```

Wynik trafia do `output/detections.geojson` (WGS84 + atrybuty, w tym `x_2180`/`y_2180`)
oraz `output/detections_map.html` (podgląd folium). Wszystkie parametry (ścieżki, zdjęcia,
źródło DEM, klasy, confidence, ray casting) są w [config.yaml](config.yaml).

Wywołanie z kodu (np. z przyszłego GUI):

```python
from geodetect import PipelineConfig, build_context, run_single

cfg = PipelineConfig.from_file("config.yaml")
ctx = build_context(cfg)                 # DEM, ODM, model YOLO ładowane raz
res = run_single(cfg, "SING0304.JPG", ctx)
for gd in res.detections:
    print(gd.detection.cls_name, gd.hit.x_2180, gd.hit.y_2180, gd.hit.h)
```

## Testy

```powershell
python -m unittest discover -s tests -t . -v      # testy jednostkowe (offline)
python scripts/compare_legacy.py                  # prototyp vs pakiet (GPU + internet)
```

## Struktura

```
geodetect/
  config.py     PipelineConfig (+ DemConfig, DetectConfig, RaycastConfig), YAML/JSON
  exif.py       odczyt GPS z EXIF
  dem.py        DemSampler: GugikDem (REST GetHByXY), RasterDem (GeoTIFF), resolve_dem
  geo.py        EnuFrame: lokalne ENU (ODM) -> WGS84 -> EPSG:2180 (ściśle)
  odm.py        CameraIO / CameraEO z reconstruction.json, kalibracja Z (z_offset)
  detect.py     Detector (YOLO, model ładowany raz)
  raycast.py    piksel -> promień w ENU, marsz + bisekcja na DEM
  pipeline.py   build_context, run_single -> ImageResult
  export.py     GeoJSON, mapa folium
main.py         CLI
config.yaml     przykładowa konfiguracja
legacy/         zamrożony prototyp z raportu 1 (do porównań)
scripts/        compare_legacy.py — porównanie prototyp vs pakiet
tests/          testy jednostkowe
ODM.md          instrukcja OpenDroneMap
PROGRES/        raporty postępu
```

## Konwencje (ważne)

- **OpenSfM:** `rotation` to angle-axis, R: world→camera, `C = −Rᵀt`. Focal i principal
  point są znormalizowane do max(w, h). „World” to topocentryczne ENU wokół `reference_lla`.
- **ENU ≠ EPSG:2180.** Zbieżność południków PUWG 1992 w rejonie lotu wynosi ok. 1,97°,
  co daje ok. 3,4 cm błędu na każdy metr od originu ODM. Każdy punkt promienia jest więc
  przeliczany ściśle: ENU → (lon, lat, h) → EPSG:2180 (`geodetect/geo.py`).
- **EPSG:2180 w kodzie = (easting, northing).** REST GUGiK `GetHByXY` używa konwencji
  geodezyjnej: `x` = northing, `y` = easting. Zamiana jest tylko w `GugikDem`.
- **Wysokości:** Potensic zapisuje GPS Altitude jako AGL od startu. Obowiązuje
  `H [m npm] = h_local + z_offset`, gdzie `z_offset = NMT(mediana kamer) + AGL − mean(alt_local)`.
