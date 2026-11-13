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
python main.py --all --surface nmt_local --eps 10
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
| `detections.geojson` | pojedyncze detekcje (WGS84) z polami: `image`, `class`, `class_label`, `conf`, `det_source` (full / tile / merged), `tile_truncated`, `object_id`, `object_status`, `accepted`, `rejected_reason`, `range_m`, `incidence_deg` (kąt promienia od pionu), `touches_border`, `x_2180`, `y_2180`, `altitude_m`, `surface`, `surface_h`, `ground_h`, `above_ground_m`, piksel, bbox |
| `objects.geojson` | obiekty zagregowane: `status` (`confirmed` / `single_view`), `class_label`, `n_views`, `images`, `rms_m`, `max_m` (rozrzut od środka), `mean_conf`, `mean_incidence_deg`, `mean_range_m`, `duplicate_images` |
| `objects.csv` | to samo co `objects.geojson` + `lon`, `lat` |
| `detections_map.html` | mapa folium z warstwami: kamery, detekcje, obiekty potwierdzone, obiekty `single_view` (szare wypełnienie, przerywany obrys), linie detekcja→obiekt, odrzucone; popupy z atrybutami |
| `spread.png` | odchylenie detekcji od środka obiektu w funkcji `incidence_deg` i `range_m` |

Pole `duplicate_images` (i ostrzeżenie w podsumowaniu) oznacza klaster, w którym to samo
zdjęcie występuje więcej niż raz. To może znaczyć, że scalono dwa blisko położone obiekty.

Status obiektu: `confirmed`, gdy obiekt ma co najmniej `aggregate.min_views` widoków
(domyślnie 2), inaczej `single_view`. Obiekty `single_view` zostają w wynikach, ale są
oznaczone. Pomaga to odsiać fałszywe detekcje bez etykiet. Status nie gwarantuje jednak
poprawności: fałszywy alarm na stałym obiekcie (dach, krzaki) widziany z kilku zdjęć też
dostaje `confirmed`.

### Detekcja (`detect`)

- **Modele** leżą w `models/`. W repo są `models/MODEL5.pt` (domyślny) i
  `models/LudzieiNamioty.pt`, pozostałe wagi (`*.pt`, w tym SAM) są w `.gitignore`. `geodetect.detect.list_models("models")`
  zwraca metadane do listy wyboru w GUI: plik, `task` (detect / segment), `names`, `imgsz`
  z treningu, rozmiar i datę. Wagi spoza YOLO (np. SAM) są pomijane. Model
  segmentacyjny (`task=segment`) daje kontury masek: metody `mask_*` z sekcji `point`
  używają ich zamiast SAM.
- **Klasy** są zawsze brane po nazwie z `model.names`, nigdy po ID:
  - `classes` (`all` albo lista);
  - `conf_per_class`;
  - `display_names` (np. `people: człowiek`);
  - `colors` (kolory na mapie);
  - `point.per_class`.

  Nazwa, której nie ma w modelu, daje błąd z listą dostępnych klas.
- **Inferencja**:
  - `conf` (globalny) i `conf_per_class`: model dostaje najniższy próg, a filtr per klasa
    działa potem;
  - `iou` (NMS), `max_det`, `device` (auto / cuda / cpu), `half` (fp16) i `batch`;
  - `imgsz: null` bierze rozmiar z treningu modelu. Dla `LudzieiNamioty.pt` to 1024.
    Efektywna wartość jest logowana. Stary kod też używał 1024, bo ultralytics brał je
    z `overrides` checkpointu.
- **Kafle** (`detect.tiling`, własna implementacja w `geodetect/tiling.py`, nie SAHI):
  - kafle `tile_size` (domyślnie `imgsz`) z zakładką `overlap`, opcjonalnie plus pełny
    obraz; `tiling.classes` ogranicza kafle do wybranych klas;
  - box dotykający **wewnętrznej** krawędzi kafla jest oznaczany jako ucięty. Taki box
    jest odrzucany, gdy ten sam obiekt widać w całości w innym kaflu albo na pełnym
    obrazie, albo gdy inny kafel widzi jego otoczenie i nic tam nie wykrył;
  - ucięte kawałki uzupełniające się są łączone w jeden box, a na końcu działa NMS per
    klasa między kaflami i pełnym obrazem;
  - flaga `touches_border` dotyczy krawędzi **zdjęcia**, nie kafla.

  SAHI odpadło, bo nie odróżnia boxa uciętego od pełnego. Ucięty box przesuwa środek
  obiektu, a więc i georeferencję.

Ocena modeli i wariantów: `experiments/detect.yaml` (docs/experiments.md). Używa 26 zdjęć,
nmt_local i eps 10 m. Precision i recall liczone są na **20 zdjęciach z etykietami**
z eksportu Roboflow `namioty_nasze` v7 (`PYTORCH/28sierpniaMODEL5`, sekcja `eval`
w config.yaml):

- `LudzieiNamioty.pt` trenowano na v4, w którym tych zdjęć nie ma, więc wszystkie
  wyniki są uczciwe.
- `MODEL5.pt` trenowano na v7. Zdjęcia ze splitu train (14) to **wyciek**; uczciwe są
  tylko valid+test (6 zdjęć).

Summary podaje zawsze obie wersje liczb. IoU 0,5, conf ≥ 0,25:

| wariant | namiot P / R (wszystkie) | people P / R (wszystkie) | namiot P / R (valid+test) | people P / R (valid+test) |
|---|---|---|---|---|
| LudzieiNamioty bez kafli | 0,95 / 1,00 | 0,93 / 0,65 | 0,86 / 1,00 | 1,00 / 1,00 |
| LudzieiNamioty kafle + pełny | 0,83 / 1,00 | 0,83 / 0,95 | 0,86 / 1,00 | 1,00 / 1,00 |
| MODEL5 bez kafli ⚠ | 0,95 / 1,00 | 1,00 / 0,95 | 0,86 / 1,00 | 1,00 / 0,83 |
| MODEL5 kafle + pełny ⚠ | 0,91 / 1,00 | 0,95 / 1,00 | 0,75 / 1,00 | 0,86 / 1,00 |

Domyślnie: **MODEL5 bez kafli**. Kafle pomagały ludziom w LudzieiNamioty (recall 0,65 →
0,95), ale MODEL5 widzi ludzi z pełnego obrazu. Na zbiorze uczciwym kafle dokładają mu
tylko fałszywe alarmy (skala obiektu w kaflu inna niż w treningu). Kafle zostają opcją
(`tiling.enabled`, `tiling.classes`).

Progi per klasa wynikają z rozkładu conf TP i FP:

- `people: 0.25` — prawdziwa leżąca osoba miała 0,31;
- `namiot: 0.6` — FP w krzakach mają 0,38–0,58, a TP obu modeli ≥ 0,68.

Jedyny fałszywy namiot bez kafli to dach domu (SING0310 i SING0312, conf 0,74–0,84). Ma
2 widoki, więc jest `confirmed`, i to jest „drugi namiot” z ocen w zadaniach 4–5. Progiem
nie da się go bezpiecznie odciąć. FP z conf > 0,7 trafiają do
`possible_missing_labels/`, bo mogą to być brakujące etykiety.

```powershell
python scripts/run_experiment.py experiments/detect.yaml
```

### Kalibracja wysokości (`z_anchor`)

Potensic zapisuje GPS Altitude jako AGL od punktu startu, więc wysokości w ODM (kamery,
chmura, DEM ODM) nie są wysokościami npm. Obowiązuje `H [m npm] = h_ODM + z_offset`,
a `z_offset` wyznacza się według `z_anchor.mode`:

| Tryb | Skąd z_offset | Kiedy |
|---|---|---|
| `agl_assumption` | `NMT(mediana xy kamer) + agl_m − średnia h kamer` | fallback, wymaga zgadnięcia wysokości lotu |
| `ground_frame` | `NMT(xy anchora) + lens_height_m − h anchora` | gdy pierwsze zdjęcie zrobiono **na ziemi w punkcie startu** (`z_anchor.image`) |
| `dem_alignment` | `mediana(NMT − DEM_ODM)` w ok. 300 punktach otwartego terenu (DSM−DTM < 0,5 m) | **zalecany**, wymaga `odm_dem/dtm.tif` / `dsm.tif` (ODM z `--dtm --dsm`, patrz [ODM.md](ODM.md)) |

`dem_alignment` dopasowuje też płaszczyznę `a·E + b·N + c` do różnic i raportuje jej nachylenie
oraz azymut. Przechył oznacza błąd orientacji modelu ODM; korekta płaszczyzną jest
opcjonalna (`apply_tilt: true`). Próbki NMT trafiają do cache na dysku (`dem.cache_path`).

Gdy ODM nie wygenerował DEM-ów, można zrobić **wstępny** DTM/DSM z chmury punktów ODM
i wskazać go w `z_anchor.odm_dtm` / `odm_dsm`:

```powershell
python scripts/pointcloud_to_dem.py          # -> output/cache/odm_pc_dtm.tif, odm_pc_dsm.tif
python scripts/run_experiment.py experiments/zcal.yaml   # porównanie trybów (docs/experiments.md)
```

Eksperyment `zcal` zawiera też wariant z **diagnostyką `multiview_check`**: **diagnostykę `multiview_check`**. Jest to skan
`z_offset` w zakresie ±15 m, który minimalizuje rozrzut statycznych namiotów. Służy tylko
jako niezależna kontrola, a nie do kalibracji wyników, bo ten sam rozrzut jest potem
miarą jakości. Optimum jest zaniżone o wysokość punktu reprezentatywnego nad gruntem
(środek bboxa).

### Źródła powierzchni ray castingu (`surface`)

Promień z piksela rzutujemy na powierzchnię wybraną w `surface.source` (albo `--surface`
w CLI). Każde źródło deklaruje swój układ wysokości, a pipeline sam decyduje, czy
porównywać wysokości z `z_offset`:

| `source` | Dane | Układ XY | Wysokości | Krok marszu |
|---|---|---|---|---|
| `nmt_gugik_rest` | NMT 1 m z REST GUGiK (`GetHByPointList`, cache na dysku) | EPSG:2180 | npm (PL-EVRF2007-NH) | `raycast.step_m` (5 m) |
| `nmt_local` | arkusze NMT z geoportalu w `data/nmt/` | EPSG:2180 | npm | rozdzielczość rastra |
| `nmpt_local` | arkusze NMPT (pokrycie: drzewa, dachy) w `data/nmpt/` | EPSG:2180 | npm | rozdzielczość rastra |
| `odm_dsm` | `odm_project/odm_dem/dsm.tif` | EPSG:32634 | **ODM lokalne** (bez `z_offset`) | rozdzielczość rastra (0,1 m) |

REST GUGiK udostępnia tylko NMT, więc źródła `nmpt_gugik_rest` nie ma: NMPT trzeba pobrać
jako arkusze. Lokalne arkusze mogą być w formacie GeoTIFF, ASC (ARC/INFO ASCII Grid) albo XYZ.
Wszystkie pliki z folderu są łączone w pamięci w jedną mozaikę, przyciętą do bboxa kamer
z buforem `area_buffer_m`. Arkusze muszą mieć tę samą rozdzielczość i CRS. Plik bez CRS
(ASC bez `.prj`, XYZ) dostaje `default_crs` (EPSG:2180).

Ray casting zwraca **pierwsze** przecięcie. Dla rastrów krok marszu nie przekracza piksela
(`raster_step_factor`), więc cienka przeszkoda (drzewo, krawędź dachu) nie zostanie
przeskoczona. Punkty promienia są próbkowane wsadowo (`raycast.chunk`, dla REST
`rest_chunk` na jedno zapytanie listowe). Interpolacja `bilinear` przy brzegu danych
przechodzi na najbliższy piksel, a nodata daje NaN i jest pomijane. Każde trafienie zapisuje
w `detections.geojson` pola `surface` (źródło) i `surface_h` (wartość powierzchni). Dla
NMPT/DSM dochodzą `ground_h` (NMT w tym samym punkcie) i `above_ground_m` (wysokość
trafienia nad gruntem, czyli wysokość obiektu albo pokrycia).

**Decyzja metodyczna: `z_offset` nie zależy od powierzchni.** `z_offset` jest liczony raz
w `build_context` (`z_anchor`, domyślnie `dem_alignment`) względem NMT z sekcji `dem`.
Zmiana `surface.source` go nie zmienia. Dzięki temu różnice między wynikami na NMT, NMPT
i DSM wynikają wyłącznie z powierzchni, a nie z innej kalibracji. Dla `odm_dsm` porównanie
wysokości odbywa się w układzie ODM (bez `z_offset`), ale wyniki (`altitude_m`) są nadal
w m npm.

#### Pobranie arkuszy NMT / NMPT (geoportal.gov.pl)

1. [geoportal.gov.pl](https://www.geoportal.gov.pl) → *Pobieranie danych* (ikona strzałki)
   → zaznacz warstwę *NMT* (potem osobno *NMPT*) → układ wysokości **PL-EVRF2007-NH**.
2. Kliknij na mapie w obszar lotu i pobierz arkusze z listy (format **ASC** albo GeoTIFF,
   układ PL-1992). Obszar lotu z buforem 100 m:
   - EPSG:2180: `xmin=660111 ymin=664689 xmax=660577 ymax=665065` (ok. 466 × 376 m),
   - WGS84: od 53.822146 N, 21.433191 E do 53.825376 N, 21.440458 E.
   - godła 1:5000: **N-34-79-D-b-1-2** i **N-34-79-D-b-2-1** (1:10000: N-34-79-D-b-1,
     N-34-79-D-b-2). Godła są wyliczone z podziału arkuszowego, więc sprawdź je na
     skorowidzu w geoportalu. Nazwy plików mogą zawierać godło albo identyfikator zgłoszenia.
3. Rozpakuj NMT do `data/nmt/`, a NMPT do `data/nmpt/` (`data/` jest w `.gitignore`).
4. Sprawdź pokrycie, potem uruchom:

```powershell
python scripts/check_surface_coverage.py                  # zasięg, rozdzielczość, dziury
python main.py --all --surface nmt_local --eps 10
python main.py --all --surface nmpt_local --eps 10
python scripts/run_experiment.py experiments/surface.yaml   # wszystkie źródła + analiza drzew
```

**Drzewa i NMPT (lot testowy).** Pokrycie NMPT − NMT > 2 m zajmuje 18% obszaru lotu.
Z 46 promieni 20 przechodzi nad koronami drzew, ale z prześwitem co najmniej 7 m, więc
NMT i NMPT dają to samo trafienie (≤ 0,06 m). Przez korony przechodzą tylko 2 promienie
i oba prowadzą do fałszywego „namiotu” na dachu domu (SING0310, SING0312). Na NMPT
trafienie wypada wtedy 17–31 m bliżej, w koronie drzewa (profil `ray_profile_*.png`).
Na otwartym terenie wybór NMT czy NMPT nie ma znaczenia. Przy obiektach za drzewami NMPT
zatrzyma promień na koronie, a NMT poprowadzi go dalej, do gruntu.

### Punkt reprezentatywny (`point`)

Promień idzie przez jeden piksel detekcji (albo przez wiele pikseli). Piksel wybiera
`point.method`, a `point.per_class` pozwala nadpisać metodę dla wybranej klasy.

| Metoda | Piksel | SAM |
|---|---|---|
| `bbox_center` | środek bboxa YOLO | nie |
| `bbox_bottom_center` | środek dolnej krawędzi bboxa (dla ludzi: stopy) | nie |
| `mask_centroid` | centroid maski | tak |
| `mask_ground_contact` | styk maski z gruntem: skrajny punkt maski w kierunku punktu nadiru (zbiegu pionów) | tak |
| `mask_footprint_median` | `footprint_samples` promieni przez piksele maski, mediana XY trafień | tak |

- **Maski** liczy SAM z ultralytics (`point.sam_model`, domyślnie `sam2.1_b.pt`, lżejszy
  `mobile_sam.pt`, oba pobierane automatycznie). Promptem jest bbox z YOLO. Maski trafiają
  do cache w `output/cache/masks/` (klucz: zdjęcie + bbox + model). SAM ładuje się tylko
  wtedy, gdy któraś metoda wymaga masek.
- **Kontrola jakości** działa na największej spójnej składowej maski. Maskę odrzucają:
  - pole maski / pole bboxa < `min_area_ratio` (`too_small`);
  - maska dotyka wszystkich krawędzi bboxa i wypełnia go w ponad `max_area_ratio`
    (`fills_bbox`);
  - największa składowa to mniej niż `min_largest_frac` całej maski (`fragmented`).

  Samo dotykanie wszystkich krawędzi nie jest błędem, bo bbox YOLO jest ciasny.
- **Fallbacki:**
  - brak maski albo maska odrzucona → `bbox_center`;
  - `mask_ground_contact` dla obiektu bliżej nadiru niż `min_offnadir_deg` → `mask_centroid`
    (pionowe krawędzie są wtedy niewidoczne, a kierunek „w dół” nieokreślony).

  Każda detekcja ma w `detections.geojson` pola `point_method`, `point_used`,
  `point_fallback`, `mask_area_ratio` i `n_rays`.

Geometrię styku z gruntem opisuje docstring `geodetect/points.py`. Ocena metod:

```powershell
python scripts/run_experiment.py experiments/point.yaml   # nmt_local, eps 10 m
```

Wynik trafia do `output/experiments/point/<data>/`: `summary.md`, `multiview.png` oraz
`point_previews/` (bbox, maska, piksele każdej metody, kierunek pionu).

**Wyniki (lot testowy).** Najmniejszy rozrzut namiotu dają metody przez środek bryły:
`bbox_center` (RMS 0,39 m), `mask_footprint_median` (0,44 m) i `mask_centroid` (0,50 m).
Metody „przy gruncie” pogarszają rozrzut do 1,4–1,8 m: `bbox_bottom_center` i
`mask_ground_contact`. Dla rozciągłego obiektu styk z gruntem leży na **bliższej kamerze
krawędzi podstawy**, a ta krawędź zmienia się z kierunkiem widoku. Błąd ma więc rząd
połowy rzutu poziomego namiotu i jest większy niż błąd z wyniesienia środka.
`multiview_check` przesuwa się wtedy z −0,5 m na +1,8…+2,2 m względem `dem_alignment`.
Dlatego domyślnie namioty używają `bbox_center`, a ludzie `bbox_bottom_center`. Postać
stojąca ma mały rzut poziomy, a środek jej bboxa jest ok. 0,9 m nad gruntem.

## Testy

```powershell
python -m unittest discover -s tests -t . -v      # testy jednostkowe (offline)
python scripts/compare_legacy.py                  # prototyp vs pakiet (GPU + internet)
python scripts/run_experiment.py experiments/*.yaml   # eksperymenty (GPU + internet), docs/experiments.md
```

### Znany problem: `PROJ_LIB` z PostgreSQL/PostGIS

Instalator PostgreSQL ustawia globalnie `PROJ_LIB` i `GDAL_DATA` na własny, starszy
`proj.db`. Rasterio (GDAL) zgłasza wtedy ostrzeżenia `proj.db ... comes from another PROJ
installation` i nie rozpoznaje kodów EPSG przy zapisie GeoTIFF. Pakiet przelicza współrzędne
przez pyproj, który ma własną bazę, więc wyniki są poprawne. Żeby pozbyć się problemu,
usuń te zmienne w sesji venv (`Remove-Item Env:PROJ_LIB, Env:GDAL_DATA`) albo ze zmiennych
systemowych, jeśli PostGIS ich nie potrzebuje.

## Struktura

```
geodetect/
  config.py     PipelineConfig (+ DemConfig, DetectConfig, RaycastConfig), YAML/JSON
  exif.py       odczyt GPS z EXIF
  dem.py        DemSampler: GugikDem (REST GetHByXY), RasterDem (GeoTIFF), resolve_dem
  geo.py        EnuFrame: lokalne ENU (ODM) -> WGS84 -> EPSG:2180 (ściśle)
  odm.py        CameraIO / CameraEO z reconstruction(.topocentric).json
  zcal.py       kalibracja Z: agl_assumption / ground_frame / dem_alignment
  detect.py     Detector (YOLO: klasy po nazwie, progi, kafle), list_models / model_info
  tiling.py     siatka kafli, scalanie detekcji z kafli (ucięte boxy, NMS)
  gt_labels.py  etykiety YOLO z eksportu Roboflow jako ground truth (letterbox, obrót)
  points.py     punkt reprezentatywny detekcji (bbox / maska SAM), geometria punktu nadiru
  segment.py    maski SAM (prompt = bbox), cache na dysku, kontrola jakości
  surface.py    powierzchnie ray castingu: REST GUGiK, mozaika arkuszy NMT/NMPT, DSM ODM
  raycast.py    piksel -> promień w ENU, pierwsze przecięcie z powierzchnią (marsz + bisekcja)
  pipeline.py   build_context, run_single -> ImageResult (+ atrybuty i filtry detekcji)
  batch.py      run_batch -> BatchResult, summary_text
  aggregate.py  DBSCAN, AggregatedObject (pozycja ważona, widoki, rozrzut)
  export.py     GeoJSON, CSV, mapa folium z warstwami, wykres rozrzutu
  diagnostics.py residua wzdłuż/w poprzek promienia, multiview_check (skan z_offset)
  experiments.py eksperymenty z YAML: warianty, cache detekcji, metryki, summary.md
  expplots.py   wykresy i podglądy eksperymentów
main.py         CLI
config.yaml     przykładowa konfiguracja
legacy/         zamrożony prototyp z raportu 1 (do porównań)
scripts/        run_experiment.py — eksperymenty z experiments/*.yaml;
                compare_legacy.py — prototyp vs pakiet;
                pointcloud_to_dem.py — wstępny DTM/DSM z chmury punktów ODM;
                check_surface_coverage.py — pokrycie arkuszy NMT/NMPT
experiments/    zcal.yaml, surface.yaml, point.yaml, detect.yaml (docs/experiments.md)
docs/           experiments.md — format eksperymentów i opis metryk
models/         wagi modeli (w repo LudzieiNamioty.pt i MODEL5.pt)
tests/          testy jednostkowe
ODM.md          instrukcja OpenDroneMap
PROGRES/        raporty postępu
```

## Konwencje (ważne)

- **OpenSfM:** czytamy `opensfm/reconstruction.topocentric.json` (topocentryczne ENU wokół `reference_lla`). Nowsze ODM zapisują `reconstruction.json` w układzie UTM z offsetem z `odm_georeferencing/coords.txt`, więc nie jest on topocentryczny (różnica 0,35° obrotu + skala UTM). `rotation` to angle-axis, R: world→camera, `C = −Rᵀt`. Focal i principal
  point są znormalizowane do max(w, h). „World” to topocentryczne ENU wokół `reference_lla`.
- **ENU ≠ EPSG:2180.** Zbieżność południków PUWG 1992 w rejonie lotu wynosi ok. 1,97°,
  co daje ok. 3,4 cm błędu na każdy metr od originu ODM. Każdy punkt promienia jest więc
  przeliczany ściśle: ENU → (lon, lat, h) → EPSG:2180 (`geodetect/geo.py`).
- **EPSG:2180 w kodzie = (easting, northing).** REST GUGiK `GetHByXY` używa konwencji
  geodezyjnej: `x` = northing, `y` = easting. Zamiana jest tylko w `GugikDem`.
- **Wysokości:** Potensic zapisuje GPS Altitude jako AGL od startu. Obowiązuje
  `H [m npm] = h_local + z_offset`, a `z_offset` wyznacza się według `z_anchor.mode` (wyżej).
  Kamery, chmura punktów i DEM ODM mają wspólną oś Z (sprawdzone: wysokość kamery minus
  DEM ODM pod nią daje ok. 42 m AGL).
