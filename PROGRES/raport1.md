# Raport 1 — wstępny prototyp Geo-Detection-System

**Data:** 2026-05-24
**Status:** ✅ Proof-of-concept działa end-to-end na pojedynczym zdjęciu.
**Zdjęcia testowe:** SING0304.JPG, SING0316.JPG (z datasetu 26 zdjęć z lotu).

---

## 1. Co udało się zrobić

Zbudowany został kompletny **prototypowy pipeline georeferencji obiektów** wykrytych
modelem YOLO na pojedynczym zdjęciu z drona Potensic Atom. Pipeline jako wynik
zwraca współrzędne geograficzne (WGS84) każdej detekcji oraz wizualizację na mapie.

### Lista zrealizowanych komponentów

1. **Środowisko (Windows 11 + PyCharm + venv).** Działa: PyTorch 2.6.0+cu124 z CUDA
   12.4 na GPU NVIDIA GeForce RTX 4060, Ultralytics YOLO, OpenCV, rasterio (z wbudowanym
   GDAL), pyproj, requests, exifread, folium.
2. **OpenDroneMap przez Docker.** Zainstalowany Docker Desktop, pobrany obraz
   `opendronemap/odm` (2.31 GB), uruchomiona rekonstrukcja SfM na 26 zdjęciach
   — **26/26 zdjęć zarejestrowane** (rzadkie 100% success rate). Wynik:
   `opensfm/reconstruction.json` z pełną kalibracją kamery i parametrami EO
   każdego ujęcia.
3. **Prototyp `main.py`** podzielony na 6 niezależnych etapów (sekcje 1–6):
   odczyt EXIF → sampler NMT → kalibracja Z → IO/EO z ODM → detekcja YOLO →
   ray casting → eksport.
4. **Dedykowana instrukcja ODM** (`ODM.md`) — krok po kroku od instalacji
   Dockera po weryfikację wyników, z listą typowych problemów.

### Wynik na zdjęciu SING0304.JPG (przykładowy)

| Atrybut | Wartość |
|---|---|
| Detekcja | 1 namiot, confidence **0.884** |
| Środek bboxa (piksele) | (1327, 677) |
| Pozycja kamery (EPSG:2180) | X=660308, Y=664798, **Z=182.2 m npm** |
| Wysokość lotu nad terenem | **~44 m AGL** (Z_kamery − NMT) |
| Współrzędne obiektu (WGS84) | **53.823612°N, 21.436667°E** |
| Wysokość terenu w punkcie trafienia | 139.9 m npm |
| Odległość kamera↔obiekt w terenie | ~66 m |
| Zapytań HTTP do GUGiK na 1 promień | 20 (+ 23 trafienia w cache) |

Na drugim zdjęciu (SING0316.JPG) wynik również wizualnie zgodny z rzeczywistym
położeniem namiotu na mapie OSM.

---

## 2. Jak to działa — pełny pipeline

```
       ┌────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐
JPG ──▶│ 1. EXIF    │───▶│ 2. NMT      │───▶│ 3. Kalibr.  │───▶│ 4. ODM     │
       │   (GPS)    │    │   sampler   │    │   Z offset  │    │   IO + EO  │
       └────────────┘    └──────┬──────┘    └─────────────┘    └─────┬──────┘
                                │                                     │
                                ▼                                     ▼
       ┌────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐
GeoJSON│ 7. Eksport │◀───│ 6. WGS84    │◀───│ 5. Ray cast │◀───│ YOLO       │
+ HTML │            │    │   (pyproj)  │    │  na DEM     │    │ bboxy      │
       └────────────┘    └─────────────┘    └─────────────┘    └────────────┘
```

### 2.1. Odczyt EXIF (`read_gps_from_exif`)
Z EXIF zdjęcia czytamy **wyłącznie GPS** (lat, lon, alt). Potensic Atom **nie zapisuje**
yaw/pitch/roll ani pose gimbala — to zasadnicze ograniczenie, dla którego musimy
korzystać z ODM (sam EXIF nie wystarcza).

### 2.2. NMT z GUGiK — point-query (`GugikDem`)
Pierwotnie zakładaliśmy pobieranie kafelka GeoTIFF przez **WCS GUGiK**, ale od 2024
ten endpoint wymaga uwierzytelnienia (HTTP 401 dla anonimowych zapytań).
Zastosowane rozwiązanie:

- **Otwarty REST GUGiK**: `services.gugik.gov.pl/nmt/?request=GetHByXY&x=...&y=...`
  (parametry x, y w EPSG:2180, zwraca tekst z wysokością npm w metrach).
- Klasa `GugikDem` opakowuje to w interfejs `DemSampler` z **lokalnym cache**
  (klucz: zaokrąglone do 1 m koordynaty, wartość: wysokość).
- W razie potrzeby (offline, eksperymenty, NMPT) dostępna jest alternatywna
  implementacja `RasterDem` na lokalnym GeoTIFF — wystarczy ustawić
  `LOCAL_NMT_FALLBACK` w konfiguracji.

### 2.3. Kalibracja Z (`compute_z_offset`)
**Problem:** Potensic Atom zapisuje GPS Altitude jako **AGL od punktu startu drona**,
nie jako npm. ODM dziedziczy to przesunięcie — w lokalnym układzie kamery są na
~45 m, podczas gdy teren w Mazurach to ~135 m npm. Bez korekty kamera jest "pod
ziemią" i promień nigdy nie trafia w DEM.

**Rozwiązanie tymczasowe (przyjęte na tej iteracji):**

```
offset_z = NMT(median_xy_kamer) + AGL_LOT_M − mean(alt_local_kamer)
```

gdzie `AGL_LOT_M = 45 m` (założenie wysokości lotu). Dla naszego datasetu:
NMT=138.7, AGL=45, mean_alt_local=46.9 → **offset_z ≈ 136.8 m**, kamera ląduje
na ~182 m npm (≈44 m nad terenem 138 m — spójne).

**Lepsze rozwiązanie na przyszłość:** pierwsze zdjęcie wykonywać **na ziemi
w punkcie startu** drona — daje to pewny anchor: `offset_z = NMT(start_xy) −
alt_local_startowego`. Bez założeń o AGL.

### 2.4. Parametry kamery z ODM (`parse_odm_for_image`)
Z `opensfm/reconstruction.json` wyciągamy dla wybranego zdjęcia:

- **Intrinsics (IO):** focal_x, focal_y, principal point (c_x, c_y) i 5
  współczynników dystorsji modelu Browna (k1, k2, p1, p2, k3). OpenSfM normalizuje
  focal do max(w, h) — w kodzie przemnażamy z powrotem na piksele.
- **Extrinsics (EO):** rotation jako wektor angle-axis (Rodriguesa), translation
  jako wektor 3D. Macierz `R = Rodrigues(rotation)` opisuje przekształcenie
  **world→camera**. Pozycja kamery `C = −Rᵀ·t`.
- **Przeniesienie do EPSG:2180:** lokalny układ ODM jest topocentric ENU wokół
  origin z `reference_lla.json`. Przeliczamy origin (lat, lon) → (X, Y) w EPSG:2180
  i dodajemy lokalne X, Y kamery. Założenie upraszczające: dla małego obszaru
  osie ENU ≈ osie EPSG:2180 (zaniedbywalna zbieżność południków, <1°).

### 2.5. Detekcja YOLO (`detect_objects`)
Standardowe wywołanie modelu `LudzieiNamioty.pt` przez ultralytics. Klasy: `0='namiot'`,
`1='people'`. Dla każdej detekcji zapisujemy bbox i jego środek (u, v) w pikselach.

### 2.6. Ray casting (`pixel_to_world_ray` + `raycast_on_dem`)

**Etap A — z piksela do promienia w świecie:**
1. `cv2.undistortPoints` z macierzą K i wektorem dystorsji wylicza znormalizowane
   współrzędne (x, y) w płaszczyźnie obrazu (oryginał piksela top-left, oś v w dół
   — biblioteka to ogarnia).
2. Wektor `d_cam = [x, y, 1]` to kierunek w układzie kamery.
3. `d_world = Rᵀ · d_cam`, normalizowane do długości 1.

**Etap B — przecięcie z terenem (ray marching + bisekcja):**
1. Marsz po promieniu z krokiem **5 m** (wybrany jako kompromis przy
   point-query DEM — każdy step to potencjalny request HTTP).
2. W każdym kroku sample NMT pod punktem; jeśli punkt promienia jest **nad** terenem,
   idziemy dalej. Jeśli **pod** — był to ostatni krok nad ziemią.
3. **Bisekcja** między ostatnim "nad" a pierwszym "pod" (25 iteracji) — uzyskujemy
   precyzję <0.1 m na trafieniu, mimo grubego kroku ray marching.

### 2.7. Eksport (`export_geojson`, `make_folium_preview`)
- Konwersja punktów z EPSG:2180 do WGS84 przez `pyproj.Transformer`.
- **GeoJSON** (`detections.geojson`): punkty z atrybutami `class`, `conf`,
  `altitude_m`, `pixel_u`, `pixel_v`.
- **Mapa folium** (`detections_map.html`): podgląd na OpenStreetMap z markerem
  kamery (niebieski) i detekcji (czerwony=namiot, zielony=people).

---

## 3. Najważniejsze decyzje projektowe i obejścia

| Problem | Rozwiązanie | Dlaczego |
|---|---|---|
| Brak orientacji w EXIF Potensica | Wszystkie kąty z ODM (SfM) | Fundamentalne ograniczenie sprzętu — bez SfM nie ma EO. |
| WCS GUGiK wymaga auth (401) | REST `services.gugik.gov.pl/nmt` + cache | Otwarte API, działa od ręki, bez tokena. |
| GPS Altitude = AGL, nie npm | `compute_z_offset` z założeniem AGL=45m | Bez korekty promień jest pod terenem. |
| Wolne point-query NMT (HTTP) | Krok 5 m + cache w pamięci + bisekcja | Praktyczna prędkość (sekundy na detekcję) przy <0.1 m precyzji końcowej. |
| Lokalny układ ODM ≠ EPSG:2180 | Założenie ENU≈EPSG:2180 (zaniedbujemy convergence) | Dla obszaru rzędu setek metrów błąd <0.1° rotacji. |
| Niektóre obiekty są wyniesione (drzewa, dachy) | Tylko NMT, NMPT odłożone | Na razie liczy się PoC. NMPT wymaga osobnego rastra (BDOT/lidar). |

---

## 4. Co działa, co dalej

### Działa (DONE)
- ✅ Detekcja YOLO i mapowanie klas
- ✅ Pobieranie NMT z GUGiK na zadanie
- ✅ Parsowanie ODM (IO + EO + reference_lla)
- ✅ Kalibracja Z względem terenu
- ✅ Ray casting na DEM z bisekcją
- ✅ Eksport do GeoJSON i mapy folium
- ✅ Sanity check: detekcja na zdjęciu fizycznie blisko prawdziwego namiotu na mapie

### Do zrobienia (NEXT)
- **Pętla po wszystkich zdjęciach** + agregacja detekcji (np. DBSCAN — ten sam namiot
  pojawia się w kilku ujęciach, trzeba je sklastrować w jedno).
- **NMPT (pokrycie terenu)** dla obiektów wyniesionych nad teren. Punkty referencyjne:
  lokalny GeoTIFF z geoportal.gov.pl (ręcznie pobrany kafelek), wymienność `DemSampler`
  już to umożliwia.
- **Lepszy offset Z** — od następnego lotu pierwsze zdjęcie z punktu startu na ziemi
  (mocniejszy anchor altitudy).
- **Walidacja ilościowa** — porównanie wyznaczonych współrzędnych z ground truth (RTK
  lub punkty na ortofoto BDOT/Geoportal); raport błędów [m].
- **Korekta zbieżności południków** — gdy obszar rośnie, dodać obrót lokalnych osi
  ENU o kąt convergence dla danej szerokości.

### Pomysł na poprawę dokładności (na kolejny lot)
> **Pierwsze zdjęcie wykonujemy na ziemi w punkcie startu drona.**
> Wtedy znamy: GPS startu (z EXIF) + altitude_local startu (=0 w ODM albo bardzo niskie).
> Pobieramy NMT(GPS_startu) — i to jest dokładne anchor altitudy npm.
> `offset_z = NMT(start_xy) − alt_local_startowego` — bez żadnego założenia o AGL.

---

## 5. Struktura repo (stan po raporcie 1)

```
Geo-Detection-System/
├── .venv/                              # środowisko Pythona
├── zdjecia/                            # 26 JPG z lotu
├── odm_project/
│   ├── images/                         # kopia zdjęć dla ODM
│   ├── opensfm/
│   │   ├── reconstruction.json         # ← IO + EO każdego zdjęcia
│   │   └── reference_lla.json          # ← origin lokalnego ENU
│   └── odm_report/
│       ├── shots.geojson               # pozycje kamer w WGS84
│       └── report.pdf                  # diagnostyka jakości
├── LudzieiNamioty.pt                   # model YOLO (namiot, people)
├── main.py                             # ← prototyp PoC
├── requirements.txt                    # zależności (bez torcha)
├── ODM.md                              # instrukcja Docker + ODM
├── PROGRES/
│   └── raport1.md                      # ← ten plik
├── detections.geojson                  # wynik (ostatniego uruchomienia)
└── detections_map.html                 # podgląd na OSM
```

---

## 6. Wnioski

Prototyp pokazał, że **cały łańcuch jest wykonalny w czystym Pythonie** bez ciężkich
zależności geoprzestrzennych (jedyną binarną biblioteką jest rasterio, i to opcjonalnie).
Najtrudniejsze nie okazały się elementy oczywiste (ODM, YOLO), lecz drobne **konwencje
i pułapki**:

- AGL vs npm w altitudzie Potensica,
- zamknięty WCS GUGiK i konieczność użycia point-query API,
- różnice modeli kamer w OpenSfM (`focal` vs `focal_x/y`),
- konwencja R, t w OpenSfM (world→camera, `C = −Rᵀ·t`).

Każdy z tych elementów jest dobrze opisany w kodzie (komentarze) i w pamięci projektu —
dzięki temu kolejne iteracje będą mogły bazować na czytelnym proof-of-concept zamiast
odkrywać te same rzeczy od nowa.
