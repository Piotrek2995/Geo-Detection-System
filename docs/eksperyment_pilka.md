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

### 5.6 Po wyrównaniu rekonstrukcji do NMPT (transformacja 7-parametrowa)

**Metoda** (`georef.mode: similarity_nmpt`, `geodetect/align.py`). Rekonstrukcję ODM wyrównano
do NMPT GUGiK transformacją przestrzennego podobieństwa P' = s·R·(P − c0) + c0 + t. Parametry
to tx, ty, tz, trzy obroty i skala. R = Rz·Ry·Rx, więc pochylenie zależy tylko od rx i ry.

Obserwacje:
- **DSM ODM vs NMPT.** Punkty DSM ODM wygładzono do oczka NMPT 1 m. Jako resztę przyjęto różnicę
  pionową, z funkcją straty Hubera, a w etapie końcowym Tukeya. Punkty z |r| > 1 m odrzucono
  jako zmiany (piłka, samochody, roślinność, krawędzie). σ powierzchni wyznaczono z danych (NMAD
  reszt). Wagę zmniejszono czterokrotnie za korelację sąsiednich próbek (krok 0,5 m przy oczku 1 m).
- **Priory.** XY kamer z GPS EXIF przyjęto z σ = 3 m. Wysokości barometryczne kamer względem
  punktu startu, H = NMT(start) + alt_EXIF, przyjęto z σ = 0,5 m.

Estymacja:
1. Start z georeferencji gps_only + z_offset (start_point).
2. Przeszukanie zgrubne (tx, ty, rz, skala) z karą priorów.
3. Gauss-Newton z IRLS z kilku startów.
4. Wybór minimum kosztu MAP.

Kamery przeliczono ściśle: C' = G(C), R_wc' = R_wc·Rjᵀ, gdzie G = F⁻¹∘T∘F, F: ENU → (x, y EPSG:2180, h),
a Rj to część obrotowa jakobianu G. W układzie wyrównanym wysokość H = h, a tz zawiera całe
przejście h_ODM → H npm, dlatego `z_offset` nie jest potrzebny i wynosi 0. DSM ODM jest przeliczany
do EPSG:2180 / npm. Rasteryzacja używa z-bufora (maksimum w komórce), bo przy dużym pochyleniu
strome ściany przechodzą „za pion”. Interpolacja `griddata` mieszała wtedy dach z dolną warstwą
i obniżała trafienia o 1,5–2 m.

**Problem: model ODM tego lotu nie jest podobny do terenu.** Ogniskowa wyznaczona przez ODM
w samokalibracji wynosi 0,374 (znormalizowana), a w locie „namiot” tą samą kamerą 0,528. Stosunek
1,41 odpowiada odwrotności pionowej skali modelu 1/0,68 = 1,47 (§5.1). Rekonstrukcja jest więc
spłaszczona w pionie przez błędną ogniskową. Taka deformacja jest anizotropowa: skala pozioma
≈ 0,85, pionowa ≈ 1,47. Transformacja podobieństwa jej nie usunie i można ją dopasować tylko
kompromisowo:
- **z priorem barometrycznym** (`7p`): skala 1,35, obrót −19°, pochylenie 5,9°. Prior wymusza
  rozciągnięcie w pionie, przez co poziomo model rośnie zamiast maleć. Reszty kamer względem
  GPS mają RMS 3,9 m;
- **bez priora barometrycznego** (`7p bez baro`, wysokości tylko diagnostycznie): skala 0,703,
  obrót +15,7°, pochylenie 28,7°. Powierzchnia dachu jest dopasowana, a reszty kamer względem
  GPS mają RMS 4,0 m. Wysokości kamer są jednak fizycznie błędne: H_kamer = 0,35·H_baro + b, RMS
  6,9 m. Pochylenie 28,7° nie opisuje rzeczywistej orientacji, tylko kompensuje spłaszczenie.

Reszty DSM − NMPT:

| | przed (gps_only + start_point) | 7p | 7p bez baro |
|---|---|---|---|
| NMAD wszystkich punktów [m] | 3,03 | 2,79 | **1,18** |
| punkty zgodne (\|r\| ≤ 1 m) | — | 420 / 1288 | 757 / 1288 |
| NMAD punktów zgodnych [m] | — | 0,65 | 0,62 |
| H dachu w GT po wyrównaniu (DSM ODM) − NMT [m], NMPT: 6,65 | 8,18 | 5,28 | **6,45** |

Formalne σ parametrów są małe: 0,03–0,05 m dla przesunięć, 0,3–0,6° dla obrotów i 0,5 % dla skali.
Nie obejmują one jednak błędu modelu. Multi-start znalazł minima różniące się kosztem o 1–5 jednostek,
w tym jedno odległe o 6–15 m (inna skala i obrót). Pojedynczy dach dwuspadowy na obszarze ok.
20 × 20 m słabo rozróżnia hipotezy. Pierwsza wersja z samym przeszukaniem zgrubnym trafiała
w fałszywe dopasowanie 8 m od GPS, dlatego wprowadzono karę priorów i wybór minimum MAP.

**Kontrola niezależna od wyrównania: punkty dachu z ortofotomap.** Cztery pary punktów dachu
wskazane na ortofotomapach ODM i GUGiK (§3) nie biorą udziału w wyrównaniu. Odległość punktu ODM
od GUGiK − lean wynosi:
- przed wyrównaniem: 4,5 / 1,5 / 1,8 / 3,8 m;
- po `7p bez baro`: 1,6 / 0,7 / 1,2 / 1,2 m.

Wyniki (piksele ręczne, pozycja zagregowana):

| powierzchnia | georeferencja | RMS wielowidokowy [m] | błąd vs GT_transfer [m] | błąd vs GT Google [m] |
|---|---|---|---|---|
| NMPT | gps_only, start_point | 0,44 | 3,44 | 1,94 |
| NMPT | gps_only, dem_alignment | 1,41 | 2,00 | 2,70 |
| NMPT | 7p | 0,71 | 3,07 | 2,59 |
| NMPT | **7p bez baro** | **0,25** | **1,11** | 3,74 |
| DSM ODM | gps_only, start_point | 0,67 | 2,87 | 2,07 |
| DSM ODM | 7p | 0,91 | 2,93 | 2,75 |
| DSM ODM | **7p bez baro** | 0,71 | **0,81** | 4,94 |
| NMT | gps_only, dem_alignment | 0,79 | 3,93 | 1,97 |
| NMT | 7p bez baro | 1,75 | 4,14 | 0,70 |

**Hipotezy po wyrównaniu.**
- H1: NMT nadal daje największy błąd względem GT_transfer (4,1–4,5 m), bo promień przechodzi
  przez dach.
- H2: dla NMPT po wyrównaniu nadal brak trendu z kątem padania (`7p bez baro`: nachylenie
  +0,01 m, r = 0,02), a rozrzut wielowidokowy spadł do 0,25 m.
- Błąd bezwzględny względem GT_transfer spadł z 2–3,5 m do 0,8–1,1 m. Potwierdza to dekompozycję
  z §5.4: dominującym składnikiem był błąd georeferencji rekonstrukcji.

**Ważność GT_transfer po wyrównaniu.** GT_transfer korzysta tylko z położenia piłki
**względem punktów dachu** na ortofotomapie ODM. Skalę, obrót i przesunięcie tej ortofotomapy
pochłania Helmert 2D wyznaczany z tych samych punktów. Wyrównanie 7p nie zmienia ortofotomapy
ODM, więc GT_transfer pozostaje ważny i jest tym samym punktem dla obu georeferencji. Zastrzeżenie:
ortofotomapa ODM powstała na zdeformowanym DSM ODM. Piłka i punkty dachu leżą jednak na tej samej
połaci, więc deformacja wysokości przenosi się na nie podobnie.

**Zależności danych.**
- Wyrównanie używa NMPT GUGiK. Ray casting w wariantach NMPT używa tego samego NMPT.
- GT_transfer używa ortofotomapy GUGiK i obrysu EGiB (lean) z tej samej kampanii 2025. NMPT
  i ortofotomapa pochodzą z tego samego nalotu i aerotriangulacji, więc mają wspólną
  georeferencję.
- Błąd względem GT_transfer **nie wykryje** wspólnego przesunięcia całej kampanii GUGiK. Jest
  ono jednak małe: deklarowany błąd położenia wynosi 0,1–0,2 m.
- Porównanie 7p vs GT_transfer jest więc niezależne co do obserwacji (GT_transfer nie wchodzi do
  wyrównania), ale nie co do układu odniesienia.

Niezależne od GUGiK są:
- GT Google (słabe, niezgodne z GT_transfer o 4,7 m). Pogorszenie względem niego po wyrównaniu
  wynika z tej niezgodności;
- reszty priorów GPS (RMS ≈ 4 m, w granicach dokładności GPS drona przy skwantowanym EXIF);
- rozrzut wielowidokowy (miara spójności, nie dokładności).

Kontrola punktami dachu jest niezależna od wyrównania, ale również odnosi się do ortofotomapy
GUGiK.

**Drugi lot („namiot”, `experiments/georef.yaml`, bez punktu startu, więc bez priora baro).**
Teren jest otwarty, a obszar większy (ok. 70 × 60 m). Wyrównanie jest dobrze uwarunkowane:
- żaden parametr nie jest słabo wyznaczony, a liczba uwarunkowania wynosi 4,4·10³;
- alternatywne minima są gorsze o > 385 jednostek kosztu;
- parametry: skala 0,950, **pochylenie 2,73°** (dem_alignment: płaszczyzna 2,31°), obrót −0,4°,
  przesunięcie poziome (0,3; −2,1) m;
- reszty DSM − NMPT: NMAD 3,60 → 0,51 m, z czego 40 tys. z 79 tys. punktów zgodnych;
- reszty priora GPS: RMS 4,1 m.

| wariant | RMS głównego namiotu [m] | przesunięcie vs gps_only + dem_alignment (NMT) [m] |
|---|---|---|
| gps_only + dem_alignment, NMT | 0,38 | — |
| gps_only + dem_alignment + pochylenie płaszczyzny, NMT | 1,03 | 1,00 |
| gps_only, DSM ODM | 0,47 | 1,14 |
| 7p, NMT | 1,04 | 1,51 |
| 7p, NMPT | 1,02 | 1,50 |
| 7p, DSM ODM | 0,46 | 1,85 |

Na NMT i NMPT rozrzut wielowidokowy namiotu po wyrównaniu rośnie z 0,38 do 1,0 m, tak samo jak
przy korekcie pochylenia płaszczyzną w `dem_alignment`. Zależność błędu od tan(kąta padania) ma
nachylenie +1,2 m. Pochylenie ~2,5° wyznaczone z całego obszaru nie jest więc zgodne z geometrią
promieni w miejscu namiotu. Prawdopodobnie model ODM jest zdeformowany nieliniowo (typowe
„wygięcie” przy georeferencji z samego GPS), a nie sztywno pochylony. Na DSM ODM rozrzut się nie
zmienia (0,47 → 0,46 m = ·s), bo ray casting na własny model jest niezmienniczy względem
podobieństwa. Dla tego lotu brak niezależnego GT, więc przesunięcia 1,5–1,9 m nie można
rozstrzygnąć co do dokładności.

**Wniosek z wyrównania.**
- Wyrównanie 7p do NMPT usuwa dominujący błąd georeferencji, gdy rekonstrukcja jest
  w przybliżeniu podobna do terenu. W locie „piłka” warunek ten spełnia składowa pozioma, a błąd
  spada do ~1 m.
- Nie naprawi jednak deformacji samej rekonstrukcji (błędna ogniskowa, wygięcie). W locie „piłka”
  prior barometryczny ujawnia tę niezgodność: k = 0,35, RMS 6,9 m. Diagnostyka `baro_scale_k` jest
  w raporcie.
- Właściwa poprawa: stała kalibracja kamery z lotu o dobrej geometrii (ODM `--cameras`, ogniskowa
  0,528), potem ponowna rekonstrukcja i wyrównanie 7p z priorem barometrycznym.
- Opcjonalnej obserwacji 2D (dopasowanie ortofotomap ODM ↔ GUGiK) nie wdrożono: automatyczne
  dopasowanie SIFT zawiodło (§3). Pary wskazane ręcznie posłużyły jako kontrola.

### 5.7 Kalibracja kamery: kompensacja a korekta (rekonstrukcja z ustaloną kalibracją)

**Weryfikacja kamery.** Oba loty mają identyczny EXIF:
- Potensic Atom LT, obiektyw „DS 2.26mm F2.2”;
- obraz 2560 × 1440 (16:9);
- FocalLength 2,26 mm, FocalLengthIn35mmFilm 13 mm, DigitalZoomRatio 0.

Tryb kamery, w zakresie, w jakim EXIF go opisuje, jest więc ten sam.

Ogniskowa oczekiwana z EXIF (f/max(w, h)) zależy od przyjętej definicji:

| definicja | wartość |
|---|---|
| ekwiwalent po przekątnej 43,27 mm | **0,345** |
| przyjęcie ODM (f35/36) | 0,361 |
| matryca 4:3 przycięta do 16:9 na pełnej szerokości | 0,376 |

Samokalibracja ODM dała:
- **0,528** w locie „namiot” (26 zdjęć, ~45 m, obszar 76 m);
- **0,374** w locie „piłka” (19 zdjęć, 10–25 m, baza 9 m).

Wartość z lotu piłki zgadza się z EXIF, ale to ta rekonstrukcja ma zdeformowaną geometrię (k = 0,68, §5.1). Znacznik f35 nie opisuje więc rzeczywistego pola widzenia obrazu 2560 × 1440. Prawdopodobnie EXIF podaje stałą wartość obiektywu i pomija kadrowanie lub stabilizację. Przy słabej geometrii lotu ODM praktycznie nie odchodzi od priora (0,361 → 0,374). Przeniesienie kalibracji z lotu „namiot” uznano za uprawnione: ta sama kamera, rozdzielczość, proporcje i EXIF. Przeniesiono pełny model Browna, z dystorsją i punktem głównym. Ogniskowej z EXIF nie użyto, bo wskazuje wartość zbliżoną do błędnej.

**Rekonstrukcja `odm_pilka_fixcal`:** ODM 3.6 z `--cameras` (kalibracja z lotu „namiot”), `--use-fixed-camera-params`, `--dtm --dsm --pc-classify --dem-resolution 2`. Zarejestrowało się 19 z 19 zdjęć. Punkty dachu i piłkę przeniesiono ze starej ortofotomapy ODM na nową dopasowaniem SIFT (`scripts/transfer_ortho_points.py`): 130 zgodnych dopasowań, transformacja afiniczna, RMS 0,05 m. GT_transfer liczony z nowych par różni się od dotychczasowego o 0,11 m, czyli nie zależy od rekonstrukcji.

**Wyniki** (`experiments/pilka_kalibracja.yaml`, piksele ręczne):

| | stara (f 0,374) | fixcal (f 0,528) |
|---|---|---|
| skala pionowa k (h_ODM kamer vs barometr) | 0,682 ± 0,037 | **0,931 ± 0,056** |
| pochylenie płaszczyzny dem_alignment | 14,2° | **4,4°** |
| z_offset: start_point − dem_alignment | 5,3 m | **1,9 m** |
| Helmert ortofotomapa ODM → GUGiK (4 punkty dachu): skala / obrót | 0,853 / 19,3° | 0,684 / 8,8° |

| rekonstrukcja · georeferencja | skala 7p | pochylenie 7p | k baro po 7p | H modelu w GT Google − NMT (NMPT 6,65) | RMS wielowid. NMPT [m] | błąd vs GT_transfer NMPT / DSM [m] | punkty dachu RMS [m] |
|---|---|---|---|---|---|---|---|
| stara · gps_only + start_point | – | – | – | 8,18 | 0,44 | 3,44 / 2,87 | 3,18 |
| stara · gps_only + dem_alignment | – | – | – | 2,83 | 1,41 | 2,00 / – | 3,18 |
| stara · 7p z baro | 1,350 | 5,9° | 0,935 | 5,28 | 0,71 | 3,07 / 2,93 | 4,10 |
| stara · 7p bez baro | 0,703 | 28,7° | 0,349 | 6,45 | 0,25 | **1,11 / 0,81** | **0,98** |
| fixcal · gps_only + start_point | – | – | – | 3,07* | 0,72 | 3,90 / 5,00 | 5,02 |
| fixcal · gps_only + dem_alignment | – | – | – | 1,14* | 1,10 | 3,30 / – | 5,02 |
| **fixcal · 7p z baro** | **0,950** | **4,6°** | **0,885** | **6,67** | **0,20** | 1,52 / 1,51 | 1,92 |
| fixcal · 7p bez baro | 0,772 | 12,3° | 0,646 | 6,95 | 0,15 | 1,95 / 1,93 | 1,72 |

\* Bez wyrównania punkt GT Google wypada w modelu fixcal poza budynkiem, bo georeferencja GPS jest przesunięta o ~5 m.

NMT dla fixcal (H1): błąd = 4,79·tan(inc) + 1,02 m, r = 0,90; trafienia 6,9 m pod dachem. **H1 potwierdzona także po ustaleniu kalibracji.**

**Odpowiedź na pytanie kluczowe.** Po ustaleniu kalibracji wyrównanie 7p z priorem barometrycznym daje wynik **fizycznie spójny**:
- skala 0,95, pochylenie 4,6° (zgodne niezależnie z płaszczyzną dem_alignment 4,4°);
- wysokości kamer zgodne z barometrem (k = 0,885, RMS 1,0 m);
- powierzchnia modelu w punkcie GT zgodna z NMPT (6,67 vs 6,65 m);
- rozrzut wielowidokowy 0,20 m.

Jest to **korekta**, a nie kompensacja. Dla porównania stara rekonstrukcja bez baro dawała pochylenie 28,7° i wysokości kamer k = 0,35.

Błąd bezwzględny wynosi 1,5 m względem GT_transfer i 1,9 m dla punktów dachu. Jest to mniej niż w każdym wariancie gps_only (2,0–5,0 m), ale nie ~1 m, i nieco więcej niż „kompensujące” 7p bez baro na starej rekonstrukcji (0,8–1,1 m). Pozostała niezgodność ma charakter anizotropii:
- powierzchnia bez priora baro woli skalę 0,77;
- Helmert ortofotomapy daje 0,68;
- barometr wymaga ~1,07.

Model fixcal jest nadal poziomo rozciągnięty względem pionu, mniej więcej o czynnik 1,2–1,4. Najprawdopodobniej sama ogniskowa z lotu „namiot” jest zaniżona: nad płaskim terenem w nadirze f koreluje z wysokością lotu, a tam k = 1,14 ± 0,12. Pełna korekta wymaga kalibracji kamery niezależnej od lotu, np. szachownica OpenCV w trybie 2560 × 1440, albo lotu kalibracyjnego z rzeźbą terenu i kilkoma wysokościami.

**Kompensacja a korekta — wniosek metodyczny.** Lepszy wynik liczbowy (stara rekonstrukcja, 7p bez baro: 0,8–1,1 m) pochodzi z transformacji, która dopasowuje dach kosztem fizycznie błędnej orientacji (28,7°) i wysokości kamer. Działa ona tylko dla punktów na dopasowanej powierzchni i nie przenosi się na inne obiekty ani wysokości. Wynik fixcal + 7p z baro jest liczbowo gorszy o ~0,5 m, ale każdy parametr ma sens fizyczny i jest potwierdzony niezależną obserwacją (barometr, płaszczyzna terenu, wysokość dachu). Za wynik pracy przyjmuje się ten drugi.

**Zastrzeżenie dotyczące GT_transfer.** W NMPT punkt GT_transfer leży na zachodniej krawędzi budynku: okap ~3,3 m nad NMT, dach rośnie na wschód do 6,5–7,5 m. GT Google leży przy kalenicy (6,65 m). Trend H1 (§5.3, 6,72·tan(inc)) wskazuje wysokość piłki ~6,7 m nad gruntem, czyli bliżej kalenicy. GT_transfer może być zatem przesunięty o 1–2 m w stronę okapu, np. przez korektę lean, która dotyczy całego dachu, a nie konkretnej wysokości na połaci. Błędy względem GT_transfer rzędu 1–1,5 m są więc na granicy rozdzielczości tego odniesienia. Rozstrzygnięcie wymaga pomiaru RTK.

**Kontrola jakości (QC, `geodetect/qc.py`)** na trzech rekonstrukcjach:

| kontrola | namiot | piłka — stara | piłka — fixcal |
|---|---|---|---|
| skala pionowa k | 1,144 ± 0,123 (OK — w granicy niepewności, zakres 6 m) | 0,682 (ZŁY) | 0,931 (OK) |
| ogniskowa vs EXIF (0,345) | +53 % (OSTRZEŻENIE*) | +9 % (OK) | +53 % (OSTRZEŻENIE*) |
| NMAD DSM − NMPT | 3,61 m (ZŁY) | 2,91 m (ZŁY) | 3,58 m (ZŁY) |
| baza kamer → błąd orientacji | 76 m → 3,2° (OSTRZEŻENIE) | 9 m → 27° (ZŁY) | 12 m → 20° (ZŁY) |
| zalecenie | 7p | kalibracja kamery, potem 7p; GCP / większa baza | 7p; GCP / większa baza |

\* EXIF nie uwzględnia kadrowania, więc porównanie z EXIF daje najwyżej ostrzeżenie. Z `qc.camera_focal` (kalibracja referencyjna) rozbieżność staje się błędem.

QC poprawnie rozróżnia przypadek „najpierw kalibracja” (stara) od „wystarczy 7p” (fixcal, namiot).

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
   Wyrównanie 7p do NMPT (§5.6) zmniejsza błąd względem GT_transfer do 0,8–1,1 m (NMPT/DSM).
   Wymaga jednak wyłączenia priora barometrycznego, bo rekonstrukcja jest spłaszczona w pionie
   przez błędną samokalibrację ogniskowej (0,374 vs 0,528 w locie „namiot”). Jest to kompensacja.
   Po ustaleniu kalibracji (§5.7) 7p z priorem barometrycznym daje wynik fizycznie spójny:
   skala 0,95, pochylenie 4,6°, dach 6,67 m. Błąd wynosi 1,5 m względem GT_transfer i jest to
   korekta. Pozostała anizotropia ~1,2–1,4 wskazuje, że do pełnej korekty potrzebna jest
   kalibracja kamery niezależna od lotu.
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
