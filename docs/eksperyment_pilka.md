# Eksperyment walidacyjny: piłka na dachu budynku

## 1. Cel

Celem eksperymentu jest ocena dokładności georeferencji obiektu leżącego **na dachu budynku**.
Obiekt jest lokalizowany przez system: detekcję na zdjęciu z drona, ray casting na model
powierzchni i agregację wielowidokową. Porównano trzy powierzchnie ray castingu: NMT GUGiK,
NMPT GUGiK i DSM z ODM. Porównano też dwie metody kalibracji wysokości, `start_point`
i `dem_alignment`, oraz trzy źródła piksela. Wynik odniesiono do niezależnie wyznaczonego
punktu odniesienia (GT). Sprawdzono ponadto dwie hipotezy:

- **H1 (NMT):** promień przechodzi przez dach do gruntu, więc trafienie jest odsunięte od kamery
  o h_dachu · tan(kąt padania).
- **H2 (NMPT/DSM):** powierzchnia zawiera dach, więc błąd nie zależy od kąta padania.

## 2. Dane

| element | opis |
|---|---|
| obiekt | piłka nożna, średnica ok. 0,22 m, na połaci dachu budynku mieszkalnego |
| lot | 15.11.2026, 14:06–14:12, Potensic Atom, 20 zdjęć 2560×1440 px, wysokość 9,7–25,4 m nad startem, kamera ukośna |
| zdjęcie startowe | SING0380, wykonane ok. 1 m nad punktem startu; służy **wyłącznie** do lokalizacji startu i kontroli barometru, wykluczone z rekonstrukcji i z przetwarzania |
| start | 53,823893° N, 21,435197° E (średnia z 5 pomiarów GNSS telefonu, kontrola na zdjęciu satelitarnym); XY z EXIF zdjęcia startowego leży 2,44 m dalej |
| rekonstrukcja | ODM (`--dtm --dsm --pc-classify --dem-resolution 2 --orthophoto-resolution 1`); zarejestrowano wszystkie 19 zdjęć w jednej rekonstrukcji |
| NMT, NMPT | GUGiK, arkusz N-34-79-D-b-1-2, oczko 1 m, aktualność 2025-05-21, błąd średni położenia 0,09 m, wysokości 0,05 m (PL-EVRF2007-NH) |
| ortofotomapa | GUGiK, 2025-05-21, piksel 0,10 m, RGB (ortofotomapa zwykła, nie prawdziwa) |
| obrysy budynków | EGiB (Krajowa Integracja Ewidencji Gruntów, WMS) |

W punkcie GT wysokość dachu wg NMPT − NMT wynosi **6,65 m** (NMT 143,90 m, NMPT 150,55 m npm).
Budynek jest obecny w NMPT, bo NMPT pochodzi z tego samego nalotu co ortofotomapa (2025).
Środek piłki leży 0,11 m nad połacią. Wobec uzyskanych błędów jest to pomijalne, dlatego
punktem reprezentatywnym detekcji jest środek bboxa.

## 3. Punkt odniesienia (GT) — metodyka

Piłki nie ma na żadnym dostępnym zdjęciu lotniczym ani satelitarnym. Zastosowano więc trzy
warianty GT o różnym stopniu niezależności:

1. **GT Google Maps.** Miejsce na dachu wskazane na zdjęciu satelitarnym Google Maps:
   53,82383712° N, 21,43531527° E (X 660244,30, Y 664881,76 w EPSG:2180).
2. **GT − przechył dachu.** GT 1 skorygowany o przesunięcie dachu względem podstawy budynku
   (building lean), zmierzone na ortofotomapie GUGiK. Lean wyznaczono, dopasowując obrys EGiB
   do maski dachu (maksymalizacja IoU):
   - budynek z GT: **1,39 m, azymut 154°** (dx +0,60, dy −1,25 m), IoU 0,63 → 0,75;
   - wszystkie budynki w kadrze: 1,42 m, azymut 132°.

   Korekta zakłada, że zdjęcie Google ma taki sam przechył jak ortofotomapa GUGiK. Nie musi tak
   być, dlatego wariant ma charakter porównawczy.
3. **GT_transfer.** Piłkę wskazano na ortofotomapie ODM z tego lotu, a następnie przeniesiono ją
   na ortofotomapę GUGiK transformacją Helmerta 2D:
   - punkty dostosowania to 4 pary charakterystycznych punktów **na tym samym dachu**
     (narożniki, końce kalenicy);
   - po transformacji pozycję skorygowano o przechył dachu GUGiK.

   Punkty dachu i piłka leżą na tej samej połaci, więc przechył dotyczy ich jednakowo.
   Parametry Helmerta:

   | skala | obrót | RMS reszt |
   |---|---|---|
   | 0,853 | +19,3° | 0,12 m |

   Pozycja GT_transfer: X 660241,11, Y 664885,20. Leży ona **4,7 m** od GT Google
   (−3,19; +3,44 m).

Dodatkowo raportowany jest błąd względem **położenia piłki na ortofotomapie ODM**. Jest to
odniesienie w układzie samej rekonstrukcji. Mierzy ono spójność wewnętrzną (ray casting,
powierzchnia, kalibracja Z) bez błędu georeferencji rekonstrukcji.

### Budżet niepewności GT (1σ)

| składnik | GT Google | GT − lean | GT_transfer |
|---|---|---|---|
| georeferencja zobrazowania Google | 1,50 | 1,50 | — |
| nieznany przechył dachu na zdjęciu Google | 1,40 | — | — |
| precyzja wskazania miejsca na dachu | 0,30 | — | — |
| wyznaczenie przechyłu (dopasowanie obrysu) | — | 0,30 | 0,30 |
| położenie obrysu EGiB | — | — | 0,30 |
| georeferencja ortofotomapy GUGiK | — | — | 0,20 |
| transformacja Helmerta (RMS reszt) | — | — | 0,12 |
| wskazanie piłki na ortofotomapie ODM | — | — | 0,05 |
| **łącznie (RSS)** | **2,07 m** | **1,53 m** | **0,49 m** |

GT_transfer ma najmniejszą i najlepiej udokumentowaną niepewność, dlatego jest **głównym
odniesieniem**. Rozbieżność GT Google – GT_transfer (4,7 m) przekracza sumaryczną niepewność
obu wariantów (√(2,07² + 0,49²) = 2,1 m) około 2,2-krotnie. Oznacza to, że wskazanie na
Google Maps było obarczone błędem większym niż typowy.

Niepewność położenia startu (GNSS telefonu) oszacowano na 1,5 m. Wpływa ona na z_offset
przez NMT(start). Różnica NMT między punktem z telefonu a XY z EXIF wynosi −0,10 m. Precyzję
barometru orientacyjnie szacuje zdjęcie startowe: EXIF 1,4 m przy deklarowanym ok. 1 m nad
gruntem, różnica +0,4 m.

## 4. Przebieg

1. Lista EXIF i identyfikacja zdjęcia startowego. `scripts/prepare_odm.py --config config_pilka.yaml`
   kopiuje 19 zdjęć do `odm_pilka/images` (bez SING0380).
2. ODM w Dockerze. Wszystkie produkty powstały, a proces zakończył się błędem dopiero
   na etapie raportu PDF (błąd importu numpy w obrazie ODM, bez wpływu na wyniki).
3. Punkty pikselowe piłki: `data/pilka_pixels.csv`. Przygotowano je narzędziem
   `scripts/pilka_click.py`:
   - wstępne wskazania pochodziły z detektorów (yolo26l COCO, YOLOE) i zostały zweryfikowane
     wizualnie;
   - operator zatwierdził lub poprawił każde zdjęcie w powiększeniu 8×;
   - piłka jest widoczna na wszystkich 19 zdjęciach.
4. Pary punktów do GT_transfer: `scripts/pilka_click.py --ortho` → `data/pilka_ortho_pairs.csv`.
5. Dane referencyjne GUGiK: metadane z skorowidzów, wycinek ortofotomapy, obrysy EGiB
   i przechył dachu. Skrypt: `scripts/pilka_reference.py` → `output/pilka/reference/`.
6. Porównanie detektorów: `scripts/pilka_detectors.py` → `output/pilka/detectors/`.
7. Eksperyment: `python scripts/run_experiment.py experiments/pilka.yaml`. Uruchomiono
   15 wariantów: 3 powierzchnie × 2 kalibracje × 3 źródła piksela, bez DSM ODM × dem_alignment,
   bo dla DSM ODM z_offset nie zmienia położenia poziomego trafienia. Wyniki:
   `output/experiments/pilka/2026-11-17_020154/`.

## 5. Wyniki

### 5.1 Kalibracja wysokości

| metoda | z_offset [m] | uwagi |
|---|---|---|
| start_point (grunt = EXIF 0, mediana różnic) | 143,475 | mediana(h_ODM − alt_EXIF) = 0,23 m, MAD 0,89 m, zakres −3,9…+1,8 m |
| dem_alignment | 138,127 | MAD 0,80 m; płaszczyzna różnic NMT − DTM ODM nachylona o **14,2°** (azymut 211°) |
| start_point z regresją (diagnostyka) | 139,044 | h_ODM = 0,682 · alt_EXIF + 4,66 m, r = 0,976 |
| porównawczo: grunt = 1,4 m (odczyt zdjęcia startowego) | 142,075 | wariant odrzucony: zdjęcie startowe nie leżało na ziemi |

Różnica h_ODM − alt_EXIF zmienia się systematycznie z wysokością: ok. +1,1 m na 12 m
i −3,9 m na 25 m. Regresja daje współczynnik **0,68**: rekonstrukcja ODM jest w pionie
ściśnięta względem barometru. Potwierdzają to dwie niezależne obserwacje:

- wysokość dachu w ODM (DSM − DTM = 4,1 m) wobec 6,65 m w NMPT − NMT, czyli stosunek 0,62;
- skala 0,85 i obrót 19° transformacji Helmerta ODM → GUGiK w poziomie.

Przyczyną jest georeferencja rekonstrukcji wyłącznie na podstawie GPS drona. Kamery leżą
w promieniu ok. 10 m, a współrzędne EXIF są skwantowane (zdjęcia SING0380 i SING0381 mają
identyczne XY). Skala, orientacja i pochylenie modelu są przez to słabo wyznaczone.

Przy błędzie skali założenie stałej różnicy (`start_point` z medianą) nie jest spełnione.
Wersja z regresją różni się od `dem_alignment` tylko o 0,9 m, a wersja z medianą o 5,3 m.

### 5.2 Wyniki georeferencji (punkty wskazane ręcznie)

Pozycja zagregowana to średnia trafień z 18–19 widoków.

| powierzchnia | kalibracja | RMS wielowidokowy [m] | błąd vs GT_transfer [m] | vs piłka na ortofotomapie ODM [m] | vs GT Google [m] | vs GT − lean [m] | ΔH vs dach NMPT (mediana) [m] |
|---|---|---|---|---|---|---|---|
| NMT | start_point | 2,59 | 5,86 | 2,73 | 3,24 | 3,21 | −6,88 |
| NMT | dem_alignment | 0,79 | 3,93 | 1,18 | 1,97 | 1,23 | −6,81 |
| NMPT | start_point | **0,44** | 3,44 | **1,05** | 1,94 | 0,86 | −0,38 |
| NMPT | dem_alignment | 1,41 | **2,00** | 1,94 | 2,70 | 1,41 | −1,44 |
| DSM ODM | start_point | 0,67 | 2,87 | 1,27 | 2,07 | 0,70 | +1,44 |

Źródło piksela ma drugorzędne znaczenie. Dla NMPT / start_point błąd względem GT_transfer
wynosi:

| źródło piksela | błąd vs GT_transfer [m] |
|---|---|
| detekcja yolo26l | 3,50 |
| YOLOE (prompt wizualny) | 2,92 |
| wskazanie ręczne | 3,44 |

Różnice są mniejsze niż wpływ georeferencji rekonstrukcji.

### 5.3 Weryfikacja hipotez

- **H1 (NMT) potwierdzona.** Dla NMT / start_point błąd poziomy rośnie liniowo z tangensem kąta
  padania: błąd = 6,72 · tan(inc) + 0,39 m, r = 0,87, n = 19. Współczynnik 6,72 m jest zgodny
  z wysokością dachu 6,65 m. Składowa wzdłuż promienia rośnie z przewidywaniem
  h · tan(inc), nachylenie 1,31, r = 0,96. Trafienia leżą 6,8 m poniżej dachu, czyli na gruncie.
  Przy `dem_alignment` współczynnik jest mniejszy (0,51), co wynika z innego z_offset
  i pochylenia modelu.
- **H2 (NMPT/DSM) potwierdzona co do braku trendu:**
  - NMPT: błąd = 0,18 · tan(inc) + 1,87 m, r = 0,14;
  - DSM ODM: r = 0,30.

  Trafienia leżą na dachu (ΔH −0,4 m dla NMPT). Rozrzut wielowidokowy spada do 0,44 m
  (NMPT) i 0,67 m (DSM ODM). Hipoteza „błąd rzędu dziesiątek centymetrów” jest jednak spełniona
  tylko dla **spójności wewnętrznej**. Bezwzględny błąd poziomy wynosi 2–3,5 m.

### 5.4 Dekompozycja błędu

Położenie piłki na ortofotomapie ODM różni się od GT_transfer o **3,2 m** (+3,13; −0,45 m).
Jest to błąd georeferencji rekonstrukcji w miejscu piłki. Spójna z nim jest analiza obrysów:
na ortofotomapie ODM dach jest przesunięty względem obrysu EGiB o 3,8 m na wschód.
Ponieważ trafienia NMPT i DSM ODM leżą 1,0–1,3 m od piłki na ortofotomapie ODM, błąd
całkowity (2–3,5 m względem GT_transfer) dzieli się następująco:

- **błąd georeferencji ODM (≈ 3 m, dominujący):** obrót, skala i przesunięcie modelu
  wyznaczone z GPS drona;
- **błąd lokalny ray castingu (≈ 1 m):** pozostałe błędy orientacji i kalibracji Z oraz
  rozdzielczość NMPT 1 m na krawędziach dachu.

### 5.5 Porównanie detektorów

Detektory porównano bez trenowania na tych zdjęciach. Dla każdego brano najlepszego
kandydata na zdjęcie (top-1). Trafienie oznacza, że środek boxa leży ≤ 15 px od punktu
ręcznego. Oceniono 18 zdjęć; zdjęcie, z którego wzięto prompt wizualny (SING0381),
wykluczono z oceny. Próg conf wynosił 0,001, imgsz 1280.

| detektor | słownik | prompt | trafienia | conf trafień (mediana) | ms/zdj. (RTX 4060) |
|---|---|---|---|---|---|
| YOLOE-26l | otwarty | wizualny (box piłki z SING0381) | **13/18** | 0,46 | 250 |
| yolo26l | zamknięty (COCO) | klasa „sports ball” | 12/18 | 0,008 | 59 |
| YOLOE-26l | otwarty | tekst „soccer ball” | 12/18 | 0,056 | 73 |
| YOLOE-11l | otwarty | wizualny | 12/18 | 0,58 | 247 |
| YOLOE-11l | otwarty | tekst | 5/18 | 0,004 | 70 |
| YOLO-World v2 (v8l) | otwarty | tekst | 2/18 | 0,019 | 71 |

**Detektor COCO:**
- piłkę z góry rozpoznaje z bardzo niską pewnością (conf 0,001–0,04), choć zwykle trafnie;
- przy conf 0,1 nie wykrył jej na żadnym zdjęciu;
- inferencja kafelkowa pogarszała wyniki: w 8 przypadkach najlepszym kandydatem był inny
  obiekt (kominek, wiadro, liście).

**YOLOE z promptem wizualnym:**
- daje kilkadziesiąt razy wyższą pewność (0,46);
- nieco więcej trafień niż COCO;
- myli się na innych zdjęciach niż detektor COCO, więc oba detektory są komplementarne.

YOLO-World okazał się nieprzydatny. Najlepszy detektor wszedł do eksperymentu jako wariant
`detection_best` (`detect.source: csv`).

## 6. Wnioski

1. Wybór powierzchni ray castingu decyduje o poprawności lokalizacji obiektów na dachach:
   - NMT przenosi trafienie na grunt za budynkiem; błąd jest zgodny z h_dachu · tan(kąta padania)
     i dochodzi do 7 m przy kącie padania 40°;
   - NMPT i DSM ODM usuwają ten efekt: brak trendu z kątem padania, rozrzut wielowidokowy
     0,4–0,7 m.
2. Bezwzględna dokładność (2–3,5 m względem GT_transfer, niepewność odniesienia 0,5 m)
   jest ograniczona przez **georeferencję rekonstrukcji ODM opartą wyłącznie na GPS drona**.
   Przy niskim, krótkim locie (baza ok. 10 m) model ma błąd skali (0,68 w pionie, 0,85 w poziomie),
   obrotu (19°) i pochylenia (14°). Ani `start_point`, ani `dem_alignment` nie korygują
   tych błędów, bo wyznaczają tylko stałą wysokości.
3. Kalibracja `start_point` z medianą jest poprawna tylko przy prawidłowej skali rekonstrukcji.
   Diagnostyka regresyjna (k = 0,68) wykrywa ten przypadek, a jej wynik zgadza się
   z `dem_alignment` z dokładnością do 0,9 m.
4. Wyniki są **zgodne z GT w granicach dokładności odniesienia tylko dla spójności wewnętrznej**
   (≈ 1 m względem piłki na ortofotomapie ODM). Względem GT_transfer błędy przekraczają
   niepewność odniesienia (0,49 m) 4–7-krotnie. Względem GT Google różnice (1,9–2,1 m dla
   NMPT/DSM) mieszczą się w niepewności tego odniesienia (2,1 m), ale GT Google jest niezgodny
   z GT_transfer.
5. Zalecenia dla dalszych lotów:
   - **punkty kontrolne (GCP)** albo wyrównanie modelu do ortofotomapy lub NMPT GUGiK pełną
     transformacją przestrzenną (7 parametrów zamiast samej stałej Z);
   - lot z większą bazą (większy obszar, kilka wysokości);
   - pomiar GT odbiornikiem GNSS RTK.

## 7. Ograniczenia

- Jeden obiekt i jeden lot. Wyniki ilustrują mechanizmy błędów, a nie dają statystycznie
  reprezentatywnej dokładności.
- GT_transfer zależy od poprawności wskazania odpowiadających sobie punktów na zniekształconej
  ortofotomapie ODM oraz od przechyłu dachu zmierzonego względem obrysu EGiB. Dokładność
  obrysu dla tego budynku nie jest znana; przyjęto 0,3 m.
- Część pikseli referencyjnych zaproponowały detektory. Operator każdy z nich zweryfikował
  i zatwierdził lub poprawił, ale ocena detektorów nie jest w pełni niezależna od nich samych.

## 8. Reprodukcja

```powershell
python scripts/prepare_odm.py --config config_pilka.yaml
docker run --rm -v "${PWD}\odm_pilka:/datasets/code" opendronemap/odm --project-path /datasets --dtm --dsm --pc-classify --dem-resolution 2 --orthophoto-resolution 1
python scripts/pilka_detect.py               # wstępne punkty + podglądy (output/pilka/)
python scripts/pilka_click.py                # punkty ręczne -> data/pilka_pixels.csv
python scripts/pilka_reference.py            # GUGiK: metadane, ortofotomapa, EGiB, przechył dachu
python scripts/pilka_click.py --ortho        # pary punktów GUGiK/ODM -> data/pilka_ortho_pairs.csv
python scripts/pilka_detectors.py            # porównanie detektorów -> data/pilka_detections_best.csv
python scripts/run_experiment.py experiments/pilka.yaml
```

Pliki wynikowe w `output/experiments/pilka/<data>/`:

| plik | zawartość |
|---|---|
| `summary.md` | pełny raport |
| `reference_per_image.csv` | wyniki per zdjęcie |
| `reference_variants.csv` | wyniki per wariant |
| `error_vs_incidence.png`, `error_vs_height_*.png` | wykresy błędu |
| `ray_profile_*.png` | profile promieni |
| `reference_map.html` | mapa: GT, start, kamery, trafienia per wariant |
