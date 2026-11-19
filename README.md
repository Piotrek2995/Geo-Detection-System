# Geo-Detection-System

Praca inżynierska (WGiK PW): *Wykorzystanie algorytmów widzenia maszynowego w automatycznej
klasyfikacji i georeferencji wybranych obiektów na zdjęciach UAV*.

System wykrywa wybrane obiekty na zdjęciach z drona modelem YOLO i wyznacza ich
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

## Interfejs graficzny (GUI)

```powershell
start_gui.bat                     # albo: streamlit run app/main.py
```

Przeglądarka otwiera http://localhost:8501. Strony: Projekt → Detekcja → Georeferencja → Uruchom
→ Wyniki. GUI wywołuje wyłącznie API pakietu `geodetect` (te same obliczenia co CLI). Ustawienia
zapisuje jako zwykły config YAML, a presety detekcji są w `presets/`. Instrukcja dla użytkownika
jest w [docs/instrukcja_uzytkownika.md](docs/instrukcja_uzytkownika.md). Tryb badawczy
(strona Eksperymenty) włącza się w Ustawieniach albo zmienną `GEODETECT_RESEARCH=1`.

## Uruchomienie

```powershell
python main.py                                   # config.yaml
python main.py --config config.yaml
python main.py --image SING0304.JPG --image SING0316.JPG
python main.py --all --surface nmt_local --eps 10
python main.py --all --out output/moj_przebieg   # katalog wyników (domyślnie output/runs/<data_czas>)
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

Wywołanie z kodu (GUI): `run_pipeline` — tej samej funkcji używa CLI. Bez `print`: postęp
przez callback (`Progress`: etap, krok, opis, ułamek 0–1), przerwanie przez `CancelToken`
(sprawdzany przed każdym zdjęciem i między etapami), błędy w `RunResult.error`.

```python
from geodetect.config import PipelineConfig
from geodetect.run import CancelToken, run_pipeline

cfg = PipelineConfig.from_file("config.yaml")
tok = CancelToken()                      # GUI: przycisk "Przerwij" -> tok.cancel()
res = run_pipeline(cfg, progress_cb=lambda p: print(p.stage, f"{p.fraction:.0%}", p.message),
                   cancel=tok, all_images=True)          # out_dir=None -> output/runs/<data_czas>
if res.ok:
    res.objects, res.detections, res.qc                   # AggregatedObject, GeoDetection, QcReport
    res.export["report_html"], res.export["gpkg"]         # pliki wynikowe
```

Etapy: `context` (NMT, ODM, kalibracja Z / 7p, QC, model) → `images` → `aggregate` → `export`
→ `done`. Niżej poziomowo nadal działają `build_context`, `run_single`, `run_batch`.

### Pliki wyjściowe przebiegu (`output/runs/<data_czas>/` albo `--out`)

Współrzędne: WGS84 (`lon`, `lat`) i PUWG 1992 (`x_2180`, `y_2180`), wysokość `altitude_m` /
`h_npm` w m n.p.m.

| Plik | Zawartość |
|---|---|
| `run_report.html` | raport dla odbiorcy spoza projektu: podsumowanie, obiekty z miniaturami (najlepszy widok), mapa, kontrola jakości, parametry (model, klasy, progi, powierzchnia, kalibracja Z / parametry 7p), statystyki, objaśnienia pojęć |
| `run_report.json` | to samo w postaci maszynowej |
| `map.html` | mapa folium: podkłady (ortofotomapa GUGiK, Esri, CARTO/OSM, TOPO, cieniowanie NMT, nakładki EGiB), legenda klas, kamery, detekcje, obiekty `confirmed` / `single_view` / z flagą rozmiaru (czerwony pierścień), dymek z miniaturą obiektu |
| `results.gpkg` | GeoPackage EPSG:2180 (3D): warstwy `objects`, `detections`, `cameras`, `rays` (kamera → trafienie) |
| `objects.geojson`, `detections.geojson` | WGS84 `[lon, lat, h_npm]` |
| `objects.csv`, `detections.csv` | tabele atrybutów (listy rozdzielone `;`) |
| `objects.kml` | obiekty (styl wg klasy, flagowane na czerwono) + ukryty folder detekcji |
| `thumbs/obj_NNN.jpg` | wycinek najlepszego widoku obiektu z ramką detekcji |
| `spread.png` | odchylenie detekcji od środka obiektu w funkcji `incidence_deg` i `range_m` |

Atrybuty obiektu: `class`, `class_label`, `status` (`confirmed` / `single_view`), `n_views`,
`images`, `rms_m`, `max_m` (rozrzut od środka), `mean_conf`, `size_m`, `flags`
(np. `size_out_of_range`), `surface` (powierzchnia ray castingu), `georef_mode`, `qc_status`,
`mean_incidence_deg`, `mean_range_m`, `duplicate_images`. Detekcje dodatkowo: `conf`, `range_m`,
`incidence_deg`, `size_m` / `size_simple_m`, `touches_border`, `accepted` / `rejected_reason`,
`surface_h`, `ground_h`, `above_ground_m`, piksel, bbox. GPKG zapisywany bez GDAL
(`geodetect/gpkg.py`, sqlite3), sprawdzony odczytem GDAL/OGR.

Eksperymenty (`experiments/*.yaml`) zapisują per wariant zestaw podstawowy (`export_all`:
GeoJSON, `objects.csv`, `detections_map.html`, `spread.png`).

### Rozmiar fizyczny obiektów (`size`)

`geodetect/size.py`: promienie przez środki czterech krawędzi bboxa przecinane z poziomą
płaszczyzną na wysokości trafienia. Wymiary lewa–prawa i góra–dół to wymiary poziome obiektu,
z uwzględnieniem kąta padania: wzdłuż pochylenia promienia `bbox_px · R / f / cos θ`, w poprzek
`bbox_px · R / f`. `size_m` to większy z nich. Rozmiar obiektu to mediana po widokach (bez detekcji
uciętych krawędzią kadru). Poza zakresem `size.ranges_m` obiekt dostaje flagę `size_out_of_range`:
nie jest usuwany, tylko oznaczany w eksporcie i na mapie.

Przybliżenie zakłada obiekt płaski. Obiekt wysoki oglądany ukośnie jest zawyżony o ~H·tan θ
(namiot ~2 m przy θ 60°: 7–8 m zamiast ~5 m w pojedynczym widoku). Przy cos θ < `min_cos_incidence`
używane jest `bbox_px · R / f`.

Lot testowy: namiot 5,5 m (mediana z 26 widoków), ludzie 1,9 i 2,5 m, fałszywe „namioty” na dachu
(SING0310, SING0312) ok. 25 m → `size_out_of_range`; piłka 0,41 m (średnica 0,22 m — luźny bbox).

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
| `start_point` | `NMT(start) − (ground_alt_exif_m + mediana(h_ODM − alt_EXIF))` | znany punkt startu (`z_anchor.start.latlon`, np. średnia GPS telefonu); barometr zerowany na ziemi → grunt = EXIF 0 |

W trybie `start_point` zdjęcie `z_anchor.start.image` (zrobione nad punktem startu) służy tylko do
lokalizacji startu i kontroli barometru. Jest automatycznie wykluczane z ODM
(`python scripts/prepare_odm.py --config <cfg>` kopiuje zdjęcia bez niego) i z przetwarzania.
Raport zawiera też diagnostykę skali: regresję `h_ODM = k · alt_EXIF + b`. Wartość k ≠ 1 oznacza
błąd skali rekonstrukcji, przy którym stała różnica nie wystarcza
(patrz [docs/eksperyment_pilka.md](docs/eksperyment_pilka.md)).

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

### Georeferencja rekonstrukcji (`georef`): wyrównanie 7p do NMPT

`z_anchor` poprawia tylko stałą wysokości (ewentualnie płaszczyznę). Gdy ODM georeferuje model
wyłącznie z GPS drona, błędne mogą być też skala, obrót i pochylenie. `georef.mode`:

| tryb | opis |
|---|---|
| `gps_only` (domyślny) | georeferencja ODM + `z_offset` z `z_anchor` |
| `auto` | deformacja w QC (skala pionowa poza [`qc.deform_scale_min`, `qc.deform_scale_max`] albo pochylenie > `qc.tilt_warn_deg`) → `similarity_nmpt`; inaczej geometria w obu wariantach na tych samych detekcjach i 7p tylko, gdy rozrzut obiektów potwierdzonych (≥ `auto_min_views` widoków) spada o > `auto_spread_gain`, a NMAD reszt DSM − NMPT nie rośnie (bez obiektów: decydują reszty) |
| `similarity_nmpt` | transformacja 7p (tx, ty, tz, rx, ry, rz, skala) z dopasowania DSM ODM do NMPT GUGiK (`surface.nmpt_dir`) |

W trybie `similarity_nmpt` obserwacje są dwie. Pierwsza to odległość punktów DSM ODM do NMPT, pionowa
albo wzdłuż normalnej (`residual`). DSM jest wygładzony do oczka NMPT, a zmiany o |r| > `reject_m`
są odrzucane (samochody, roślinność, nowe obiekty). Na drugim etapie działa funkcja straty
Tukeya. Druga obserwacja to priory: XY kamer z GPS EXIF (`sigma_gps_xy_m`, domyślnie 3 m) oraz,
gdy jest `z_anchor.start`, wysokości barometryczne kamer (`sigma_baro_m`, `use_baro_prior`).

Estymacja przebiega tak:
1. Przeszukanie zgrubne (tx, ty, rz, skala) z karą priorów, startując z `gps_only` + `z_offset`.
2. Multi-start Gaussa-Newtona z IRLS i estymacją σ powierzchni.
3. Wybór minimum kosztu MAP.

Raport (`summary.md` eksperymentu, `variants/*/alignment.json`) zawiera:
- parametry z σ, liczonymi z całości oraz z samej powierzchni;
- parametry słabo wyznaczone przez powierzchnię (na płaskim terenie tx, ty i rz wiąże tylko GPS);
- reszty przed i po wyrównaniu oraz reszty priorów;
- `baro_scale_k`, czyli zgodność skali pionowej z barometrem;
- alternatywne minima.

Kamery są przeliczane ściśle: C' = G(C), R_wc' = R_wc·Rjᵀ, gdzie Rj to część obrotowa jakobianu G.
W układzie wyrównanym H = h, więc `z_offset` = 0 (zawiera go tz). DSM ODM jest przeliczany do
EPSG:2180 / m n.p.m. Eksperymenty: `experiments/georef.yaml` (lot „namiot”) i warianty `7p`
w `experiments/pilka.yaml`.

### Kontrola jakości rekonstrukcji (`qc`)

`geodetect/qc.py` działa w `build_context`, po ewentualnym 7p. Skala pionowa, ogniskowa i baza
liczone są na georeferencji `gps_only`. Zgodność z NMPT liczona jest dla **aktywnej** georeferencji
(z_anchor, z pochyleniem jeśli `apply_tilt`, albo 7p), obok wartości „przed” (`gps_only` + z_offset).
Liczona jest tylko w promieniu `qc.core_radius_m` (30 m) od kamer, bo krańce DSM ODM są ekstrapolowane.
Do reszt dopasowywana jest odporna płaszczyzna: jej pochylenie to kontrola `tilt`, a NMAD reszt po
jej usunięciu to kontrola `surface` (zgodność kształtu). Wynik (OK / OSTRZEŻENIE / ZŁY + zalecenia)
trafia do podsumowania batcha, raportu przebiegu i `summary.md` eksperymentów:

| kontrola | miara | domyślne progi (ostrzeżenie / zły) |
|---|---|---|
| skala pionowa | nachylenie k regresji h_ODM kamer vs wysokość barometryczna EXIF | \|k − 1\| > 0,10 / 0,20 (i > 2σ / 3σ nachylenia) |
| ogniskowa | f ODM vs `qc.camera_focal` (albo EXIF f35 — wtedy najwyżej ostrzeżenie) | 10 % / 25 % |
| zgodność kształtu z NMPT (`surface`) | NMAD(DSM ODM − NMPT) po usunięciu płaszczyzny | 0,75 / 1,5 m |
| pochylenie modelu (`tilt`) | nachylenie płaszczyzny reszt DSM ODM − NMPT | 2° / 6° |
| baza kamer | √2·σ_GPS / rozpiętość kamer (oczekiwany błąd orientacji) | 2° / 5° (przy 7p najwyżej ostrzeżenie) |

Wyniki (lot „namiot” / „piłka” stara / „piłka” fixcal):

| tryb | namiot | piłka, stara | piłka, fixcal |
|---|---|---|---|
| gps_only + dem_alignment | OSTRZEŻENIE (pochylenie 3,9°, NMAD kształtu 0,32 m) | ZŁY (k 0,68, 14,7°, baza) | ZŁY (9,1°, NMAD 3,4 m, baza) |
| 7p | OSTRZEŻENIE (1,2°, 0,29 m; ostrzeżenia: EXIF, baza) | ZŁY (k 0,68, NMAD 2,8 m) | ZŁY (NMAD 1,9 m; pochylenie 0,4°) |

Zalecenia QC:
- „ustal kalibrację kamery”: przy złej skali pionowej lub ogniskowej, bo wtedy 7p tylko kompensuje deformację;
- „rozważ 7p”: przy niezgodności z NMPT, pochyleniu lub słabej bazie (nie, gdy 7p już działa);
- „GCP / większa baza”: przy bardzo słabej bazie.

Domyślnie pozostaje `gps_only` + `dem_alignment`, a 7p jest opcją zalecaną przez QC.

Rekonstrukcja z ustaloną kalibracją kamery (z lotu o dobrej geometrii):

```powershell
# cameras.json: wpis "cameras" z opensfm/reconstruction.topocentric.json lotu referencyjnego
docker run --rm -v "${PWD}\odm_pilka_fixcal:/datasets/code" opendronemap/odm --project-path /datasets `
  --cameras /datasets/code/cameras.json --use-fixed-camera-params --dtm --dsm --pc-classify --dem-resolution 2
python scripts/transfer_ortho_points.py      # punkty z ortofotomapy starej rekonstrukcji -> nowa
python scripts/run_experiment.py experiments/pilka_kalibracja.yaml
```

Uwaga: dla Potensic Atom LT EXIF (f35 = 13 mm → f/W 0,345) nie odpowiada geometrii obrazu
2560 × 1440. Szczegóły w [docs/eksperyment_pilka.md](docs/eksperyment_pilka.md), §5.7.

### Eksperyment walidacyjny „piłka na dachu”

Osobny lot i osobny config (`config_pilka.yaml`, `experiments/pilka.yaml`); opis i wyniki
w [docs/eksperyment_pilka.md](docs/eksperyment_pilka.md). Źródło pikseli `detect.source`:
`model` (YOLO), `manual` (punkty z `manual_csv`, narzędzie `scripts/pilka_click.py`) albo `csv`
(zapisane detekcje innego detektora, np. YOLOE z `scripts/pilka_detectors.py`).

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
  zcal.py       kalibracja Z: agl_assumption / ground_frame / dem_alignment / start_point
  align.py      georef similarity_nmpt: wyrównanie 7p rekonstrukcji ODM do NMPT (kamery, DSM)
  qc.py         kontrola jakości rekonstrukcji (skala pionowa, ogniskowa, NMPT przed/po, pochylenie, baza)
  lean.py       przesunięcie dachu na ortofotomapie zależne od wysokości (walidacja GT_transfer)
  size.py       rozmiar fizyczny obiektu z bboxa (kąt padania), flaga size_out_of_range
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
  run.py        run_pipeline(cfg, progress_cb, cancel) -> RunResult (API dla GUI i CLI)
  export.py     eksport przebiegu: GeoJSON, CSV, GPKG, KML, mapa folium, miniatury, run_report.html/json
  gpkg.py       zapis/odczyt GeoPackage przez sqlite3 (bez GDAL)
  basemaps.py   podkłady map folium (GUGiK WMS, Esri, CARTO, EGiB)
  diagnostics.py residua wzdłuż/w poprzek promienia, multiview_check (skan z_offset)
  experiments.py eksperymenty z YAML: warianty, cache detekcji, metryki, summary.md
  expplots.py   wykresy i podglądy eksperymentów
  runs.py       lista przebiegów, porównanie, mapa wyników (API dla GUI)
  photos.py     skan zdjęć (GPS, wysokość), wykrycie zdjęcia startowego
  odmrun.py     uruchamianie ODM w Dockerze (status, postęp, przerwanie)
  coverage.py   dostępność powierzchni: REST GUGiK, pokrycie arkuszy NMT/NMPT
app/            GUI Streamlit: main.py, state.py (stan i walidacja, bez streamlit), ui.py, views/
presets/        presety detekcji GUI (YAML)
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
