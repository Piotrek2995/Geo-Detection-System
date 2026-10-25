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

Wszystkie parametry (ścieżki, zdjęcia, źródło DEM, klasy, confidence, ray casting,
filtry, agregacja) są w [config.yaml](config.yaml).

### Przetwarzanie wsadowe

```powershell
python main.py --all       # wszystkie zdjęcia z rekonstrukcji ODM
```

Wszystkie zdjęcia korzystają ze wspólnego kontekstu: model YOLO ładuje się raz, a cache
NMT z GUGiK jest jeden. Zdjęcia z folderu, których nie ma w rekonstrukcji ODM (albo
w folderze brakuje pliku), są pomijane z ostrzeżeniem i wymienione w podsumowaniu. Na koniec
detekcje są agregowane DBSCAN-em (osobno dla każdej klasy, w EPSG:2180): ten sam obiekt
widziany na kilku zdjęciach daje jeden punkt z liczbą widoków i rozrzutem.

Wywołanie z kodu (np. z przyszłego GUI):

```python
from geodetect import PipelineConfig, build_context, run_single
from geodetect.batch import run_batch, summary_text
from geodetect.export import export_all

cfg = PipelineConfig.from_file("config.yaml")
ctx = build_context(cfg)                 # DEM, ODM, model YOLO ładowane raz
res = run_single(cfg, "SING0304.JPG", ctx)               # jedno zdjęcie
batch = run_batch(cfg, ctx, images=None,                  # None = wszystkie z ODM
                  progress_cb=lambda i, n, name: print(i, n, name))
print(summary_text(batch))
export_all(batch, cfg.output_dir)
```

### Pliki wyjściowe (`output/`)

| Plik | Zawartość |
|---|---|
| `detections.geojson` | pojedyncze detekcje (WGS84) z polami: `image`, `class`, `conf`, `object_id`, `accepted`, `rejected_reason`, `range_m`, `incidence_deg` (kąt promienia od pionu), `touches_border`, `x_2180`, `y_2180`, `altitude_m`, piksel, bbox |
| `objects.geojson` | obiekty zagregowane: `n_views`, `images`, `rms_m`, `max_m` (rozrzut od środka), `mean_conf`, `mean_incidence_deg`, `mean_range_m`, `duplicate_images` |
| `objects.csv` | to samo co `objects.geojson` + `lon`, `lat` |
| `detections_map.html` | mapa folium z warstwami: kamery, detekcje, obiekty, linie detekcja→obiekt, odrzucone; popupy z atrybutami |
| `spread.png` | odchylenie detekcji od środka obiektu w funkcji `incidence_deg` i `range_m` |

Pole `duplicate_images` (i ostrzeżenie w podsumowaniu) oznacza klaster, w którym to samo
zdjęcie występuje więcej niż raz. To może znaczyć, że scalono dwa blisko położone obiekty.

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
  pipeline.py   build_context, run_single -> ImageResult (+ atrybuty i filtry detekcji)
  batch.py      run_batch -> BatchResult, summary_text
  aggregate.py  DBSCAN, AggregatedObject (pozycja ważona, widoki, rozrzut)
  export.py     GeoJSON, CSV, mapa folium z warstwami, wykres rozrzutu
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
