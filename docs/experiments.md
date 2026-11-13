# Eksperymenty porównawcze

Wszystkie porównania (kalibracja Z, powierzchnia, punkt reprezentatywny, detekcja) korzystają
z jednego mechanizmu: `geodetect/experiments.py` + `scripts/run_experiment.py`. Każdy
eksperyment to plik YAML w `experiments/`. Zastępuje on dawne skrypty `scripts/*_compare.py`.

```powershell
.venv\Scripts\python.exe scripts\run_experiment.py experiments\surface.yaml
.venv\Scripts\python.exe scripts\run_experiment.py experiments\zcal.yaml experiments\surface.yaml experiments\point.yaml experiments\detect.yaml
.venv\Scripts\python.exe scripts\run_experiment.py experiments\point.yaml --images SING0304.JPG SING0316.JPG   # szybki test
```

Gdy w jednym wywołaniu podasz kilka plików, detekcje są wspólne także między nimi
(np. zcal, surface i point używają tego samego modelu i tej samej sekcji `detect`).

## Format YAML

```yaml
name: surface                      # nazwa katalogu wyników
description: >                     # trafia do summary.md
  ...
base_config: ../config.yaml        # względem pliku eksperymentu
overrides:                         # wspólne dla wszystkich wariantów
  model: models/LudzieiNamioty.pt  # alias: model / detect.model -> model_path
  aggregate.eps_m: 10.0
axes:                              # iloczyn kartezjański (nazwy wariantów z wartości)
  surface.source: [nmt_gugik_rest, nmt_local, nmpt_local, odm_dsm]
variants:                          # albo / oraz jawna lista
  - name: dem_alignment+tilt
    set: {z_anchor.mode: dem_alignment, z_anchor.apply_tilt: true}
  - name: multiview_check          # z_offset wyliczony z innego wariantu
    derive: {z_offset: multiview_check, from: agl_assumption, span_m: 15, step_m: 0.25}
reference: nmt_gugik_rest          # wariant bazowy dla przesunięć (domyślnie pierwszy)
images: null                       # null = wszystkie zdjęcia z rekonstrukcji ODM
metrics:
  static_classes: [namiot]         # obiekty statyczne: RMS, wzdłuż/w poprzek, przesunięcia
  moving_classes: [people]         # ruchome: tylko liczby i przesunięcia, bez RMS
  min_views_static: 2              # obiekt w tabelach: confirmed i >= tylu widoków
  min_views_scan: 3                # RMS zbiorczy i multiview_check
  multiview: {span_m: 3, step_m: 0.05}   # skan z_offset dla każdego wariantu
  ground_truth: true               # P/R/F1 względem cfg.eval
  missing_label_conf: 0.7          # FP powyżej -> possible_missing_labels/
  tree_profile: {cover_threshold_m: 2.0, step_m: 0.5, n_plots: 3}
  point_previews: 6                # podglądy metod punktu (maski SAM)
  dem_alignment_plot: true
```

Nadpisania używają ścieżek z kropkami do pól `PipelineConfig`. Ścieżki plików są liczone
względem katalogu bazowego configu. Słownik podany dla sekcji (np. `point: {method: ...}`)
nadpisuje tylko podane pola. Nieznane pole, klucz, metryka, `reference` lub `derive.from`
kończą się błędem `ValueError`, zanim cokolwiek zostanie policzone.

**Cache.** Detekcje YOLO są zapamiętywane według klucza (ścieżka modelu + cała sekcja
`detect`). Warianty różniące się tylko geometrią (powierzchnia, kalibracja Z, punkt
reprezentatywny, agregacja) wywołują model raz na zdjęcie. Maski SAM mają osobny cache
dyskowy (`output/cache/masks`). DEM, rekonstrukcja ODM i powierzchnie (mozaiki rastrów)
są budowane raz na eksperyment. Kolumna „czas: detekcja” w summary.md pokazuje 0 dla
wariantów, które dostały detekcje z cache.

## Wyniki

Katalog: `output/experiments/<nazwa>/<RRRR-MM-DD_GGMMSS>/`

| plik | zawartość |
|---|---|
| `summary.md` | tabele gotowe do wklejenia do pracy |
| `summary.csv` | jeden wiersz na wariant, wszystkie metryki (także GT per próg / podzbiór / klasa) |
| `static_objects.csv` | obiekty statyczne każdego wariantu z przesunięciem względem bazowego |
| `variants/<wariant>/` | `config.yaml` wariantu + eksport (`detections.geojson`, `objects.geojson`, `objects.csv`, mapa, `spread.png`) |
| `meta.json` | commit, lista niezacommitowanych plików, wersje Pythona i bibliotek, GPU |
| `experiment.yaml` | kopia definicji |
| `*.png` | wykresy (poniżej) |
| `variants/<wariant>/gt_previews/` | podglądy wszystkich FP i FN |
| `possible_missing_labels/` | FP z conf > `missing_label_conf` |
| `tree_profile.csv`, `ray_profile_*.png` | analiza pokrycia NMPT |
| `point_previews/` | metody punktu na wybranych detekcjach |

## Metryki

**Obiekty statyczne** (`static_classes`, status confirmed, ≥ `min_views_static` widoków):

- liczba widoków;
- RMS i max odległości poziomej trafień od środka klastra;
- RMS składowej **wzdłuż** poziomego kierunku promienia (kamera → trafienie) i **w poprzek**;
- nachylenie regresji `along = a·tan(incidence) + b` z współczynnikiem korelacji r.
  Błąd wysokości Δh przesuwa trafienie wzdłuż promienia o Δh·tan(incidence), więc
  nachylenie a ≈ Δh jest śladem błędu Z. Wartość ≈ 0 oznacza, że Z jest dobrze skalibrowane;
- pozycja (X, Y EPSG:2180, H m n.p.m.);
- przesunięcie względem wariantu bazowego (najbliższy obiekt tej klasy w promieniu 3·eps).

„Główny obiekt” to obiekt statyczny z największą liczbą widoków. **RMS zbiorczy** liczony jest
po wszystkich obiektach statycznych z ≥ `min_views_scan` widokami.

**Klasy ruchome** (`moving_classes`): liczby detekcji i obiektów oraz przesunięcie tych
samych detekcji (zdjęcie + bbox) względem wariantu bazowego. Osoby przemieszczały się między
zdjęciami, więc RMS klastra nie mierzy dokładności i nie jest raportowany.

**Globalne:**

- detekcje i obiekty per klasa i status (confirmed / single_view);
- liczba promieni bez przecięcia;
- czas detekcji i czas całego wariantu.

**multiview_check** (`metrics.multiview` albo `derive`): z_offset minimalizujący rozrzut
obiektów statycznych. To **diagnostyka, nie kalibracja**: minimalizowanie miary, której
potem używamy do oceny, byłoby rozumowaniem kołowym. Wariant `derive` pokazuje tylko, jaki
wynik dałby taki offset (górna granica jakości).

**Ground truth** (`metrics.ground_truth`, sekcja `eval` w config.yaml):

- P / R / F1 per klasa przy IoU `eval.iou` i każdym progu z `eval.thresholds`;
- trzy podzbiory zdjęć:
  - **wszystkie z GT**;
  - **train (v7)** — `eval.train_images`;
  - **valid+test** — zbiór uczciwy.
- Dla modeli z `eval.leaky_models` (trenowanych na tym eksporcie) summary.md ostrzega
  o wycieku danych. Wiersze „wszystkie” i „train” są oznaczone ⚠, ale obie wersje liczb
  są zawsze podane.
- Rozkład conf TP i FP oraz proponowany próg (maksimum F1, przy remisie najniższy) dla
  wszystkich zdjęć i dla valid+test.
- FP z conf > `missing_label_conf` trafiają do `possible_missing_labels/`, bo mogą to
  być brakujące etykiety, a nie błędy modelu.

Etykiety pochodzą z eksportu Roboflow. Zdjęcia z treningu mają tam 3 augmentowane wersje;
`geodetect/gt_labels.py` wybiera tę bez obrotu (rejestracja ORB + RANSAC do oryginału)
i cofa letterbox 1024.

**Pokrycie terenu** (`metrics.tree_profile`):

- Dla każdej detekcji wariantu bazowego promień jest próbkowany co `step_m` od kamery do
  trafienia w NMT (lokalne arkusze). Sprawdzamy, czy przeszedł **nad** pokryciem
  (NMPT − NMT > próg) i czy **przez** nie (promień poniżej NMPT).
- Raport:
  - zdjęcie i klasa;
  - wysokość przeszkody;
  - minimalny prześwit promienia nad NMPT;
  - odległość pozioma między trafieniem w NMT i w NMPT.
- Do tego liczba sprawdzonych promieni i odsetek obszaru lotu (bbox kamer + 100 m)
  z pokryciem powyżej progu.
- Profile (NMT, NMPT, DSM ODM, linia promienia) rysowane są dla najciekawszych promieni:
  najpierw te przechodzące przez pokrycie, potem z najmniejszym prześwitem.

## Eksperymenty w repozytorium

| plik | pytanie | warianty | kluczowe metryki |
|---|---|---|---|
| `zcal.yaml` | jak kalibrować wysokość ODM → npm? | agl_assumption, dem_alignment, dem_alignment+tilt, multiview_check (z batcha agl) | RMS / wzdłuż / nachylenie głównego namiotu, wykres przechyłu ODM |
| `surface.yaml` | na jaką powierzchnię rzutować? | nmt_gugik_rest, nmt_local, nmpt_local, odm_dsm | przesunięcie vs REST, wysokość trafienia nad gruntem, analiza drzew |
| `point.yaml` | przez który piksel puścić promień? | bbox_center, bbox_bottom_center, mask_centroid, mask_ground_contact, mask_footprint_median | RMS, przesunięcie, multiview_check per metoda, podglądy |
| `detect.yaml` | który model i czy kafle? | LudzieiNamioty / MODEL5 × bez kafli / kafle+pełny | P/R/F1 (wszystkie / train / valid+test), conf TP/FP, możliwe brakujące etykiety |

Eksperymenty zcal, surface i point używają **LudzieiNamioty** (trenowany na v4 bez zdjęć
SING03xx), żeby geometria była oceniana na detekcjach niezależnych od etykiet. Wyniki są
wtedy porównywalne z dawnymi skryptami `*_compare.py`.

Dodanie nowego eksperymentu: skopiuj YAML, zmień `name` i warianty. Każde pole configu da
się nadpisać, np. `aggregate.weights: [equal, cos_incidence]` jako oś.
