# Instrukcja użytkownika: Geo-Detection-System (GUI)

Program wykrywa wybrane obiekty na zdjęciach z drona i wyznacza ich położenie w terenie
(współrzędne PUWG 1992 i WGS84, wysokość n.p.m.). Wynikiem jest mapa, tabela obiektów i pliki
do programów GIS (QGIS, Google Earth).

Do pracy potrzebne są:

- zdjęcia z lotu z pozycją GPS w EXIF,
- rekonstrukcja zdjęć w OpenDroneMap (program może ją uruchomić sam, w Dockerze),
- model detekcji YOLO w folderze `models/` (dowolne klasy; modele z opisem tekstowym, np. YOLOE,
  pozwalają szukać obiektów bez trenowania),
- opcjonalnie arkusze NMT / NMPT z geoportal.gov.pl dla obszaru lotu (bez nich program korzysta
  z usługi NMT GUGiK online).

## 1. Instalacja

Wymagany jest Windows 10/11, Python 3.12 i (zalecana) karta graficzna NVIDIA.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Do uruchamiania rekonstrukcji z poziomu programu potrzebny jest
[Docker Desktop](https://www.docker.com/products/docker-desktop/). Pierwsze uruchomienie pobiera
obraz OpenDroneMap (kilka GB).

## 2. Uruchomienie

Kliknij dwukrotnie `start_gui.bat` w folderze programu. Otworzy się przeglądarka z adresem
http://localhost:8501. Można też w terminalu:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/main.py
```

Program pamięta ostatnio otwarty projekt. Pasek boczny pokazuje jego stan (liczba zdjęć, czy jest
rekonstrukcja, wybrany model). W sekcji **Ustawienia** na pasku bocznym są presety detekcji i
przycisk przywrócenia ustawień domyślnych.

Strony prowadzą krok po kroku: **Projekt → Detekcja → Georeferencja → Uruchom → Wyniki**.
Każda strona na dole pokazuje, czego jeszcze brakuje (czerwone komunikaty blokują uruchomienie).

## 3. Krok po kroku

### 3.1. Projekt

![Projekt](gui/01_projekt.png)

1. **Folder ze zdjęciami**: wpisz ścieżkę albo kliknij **Wybierz…**. Program pokazuje liczbę zdjęć,
   ile ma pozycję GPS i zakres wysokości lotu, a na mapie pozycje zdjęć.
2. **Punkt startu drona** poprawia dokładność wysokości. Jeśli na początku lotu zrobiono zdjęcie
   tuż nad ziemią w miejscu startu, program wykryje je sam i zaznaczy „zdjęcie startowe”. To zdjęcie
   jest pomijane w rekonstrukcji i detekcji. Zamiast zdjęcia można wpisać współrzędne startu albo
   kliknąć na mapie.
3. **Rekonstrukcja**: wskaż folder z gotową rekonstrukcją OpenDroneMap albo wybierz
   **uruchom teraz**. Program sprawdzi Dockera, skopiuje zdjęcia (bez startowego) i pokaże postęp
   z logiem. Rekonstrukcja trwa od kilkunastu do kilkudziesięciu minut i można ją przerwać.
   Przy krótkich, niskich lotach zaznacz **Użyj kalibracji kamery z innej rekonstrukcji** i wskaż
   rekonstrukcję lotu o dobrej geometrii (większy obszar, kilka wysokości).

Status „Rekonstrukcja gotowa” oznacza, że są wszystkie potrzebne pliki. Szczegóły są w sekcji
**Szczegóły rekonstrukcji**.

### 3.2. Detekcja

![Detekcja](gui/02_detekcja.png)

1. **Model**: wybierz plik z folderu `models/`. Pod listą jest krótki opis (zadanie i klasy).
2. **Wykrywane klasy**: zostaw wszystkie albo wybierz potrzebne.
3. Dla każdej klasy ustaw:
   - **nazwę wyświetlaną** i **kolor** (na mapie i w raportach),
   - **próg pewności**: niższy daje więcej wykrytych obiektów, ale też więcej pomyłek,
   - **punkt lokalizacji**: środek obiektu albo podstawa (dla ludzi: stopy),
   - w sekcji **Zaawansowane**: punkt z maski obiektu (dokładniejszy, wolniejszy) i oznaczanie
     obiektów o nietypowym rozmiarze (dostają flagę, nie są usuwane).
4. **Małe obiekty**: wykrywanie na powiększonych fragmentach zdjęcia pomaga, gdy obiekty są
   małe na zdjęciu (np. ludzie z dużej wysokości), ale wydłuża przetwarzanie.
5. **Podgląd**: wybierz zdjęcie i kliknij **Pokaż detekcje**, żeby sprawdzić ustawienia przed
   pełnym przebiegiem.

**Modele z opisem tekstowym (YOLOE, YOLO-World)** nie mają stałej listy klas. Wpisz po angielsku,
czego szukać (np. `person, car`), albo wybierz **zaznaczenie na zdjęciu** i obrysuj prostokątem
przykładowy obiekt. Model będzie szukał obiektów podobnych do wzorca.

**Presety** (pasek boczny, Ustawienia) zapisują i wczytują ustawienia detekcji: model, klasy,
progi i punkty lokalizacji. Wczytanie presetu nie zmienia projektu (zdjęć ani rekonstrukcji).
Dostępne są przykładowe presety: „Ludzie i namioty (MODEL5)”, „Dowolny obiekt z opisu
tekstowego (YOLOE)”, „Ludzie (model COCO)”.

### 3.3. Georeferencja

![Georeferencja](gui/03_georeferencja.png)

1. **Powierzchnia terenu**: z nią przecinany jest promień od kamery przez obiekt. Wybór
   **Automatycznie** bierze NMPT z pobranych arkuszy (teren z budynkami i drzewami), gdy pokrywają
   obszar lotu, a w przeciwnym razie NMT online. Sekcja **Dostępność danych** pokazuje, co jest
   dostępne.
2. **Georeferencja**: **Automatycznie (zalecane)** sprawdza jakość rekonstrukcji i sam decyduje,
   czy wystarczy GPS drona, czy model trzeba wyrównać do NMPT GUGiK (gdy jest pochylony albo
   zdeformowany). Decyzja jest pokazywana jednym zdaniem. Można też wybrać tryb ręcznie.
3. **Kalibracja wysokości**: domyślne dopasowanie modelu do NMT działa bez dodatkowych pomiarów.
4. **Zaawansowane**: odległość łączenia widoków w jeden obiekt, liczba widoków do potwierdzenia
   obiektu, filtry i foldery arkuszy NMT / NMPT.
5. **Sprawdź jakość rekonstrukcji**: kontrola trwa do minuty, patrz rozdział 4.

### 3.4. Uruchom

![Uruchom](gui/04_uruchom.png)

Tabela pokazuje najważniejsze wybory, a **Pozostałe ustawienia** resztę. Kliknij
**Uruchom przetwarzanie** (wszystkie zdjęcia albo wybrane). W trakcie widać pasek postępu,
przetworzone zdjęcia i log. **Przerwij** zatrzymuje przebieg bez zapisu wyników. Po zakończeniu
**Pokaż wyniki** przechodzi do strony Wyniki.

### 3.5. Wyniki

![Wyniki](gui/05_wyniki.png)

- **Przebieg**: lista zapisanych przebiegów (data, liczba obiektów, model) z oceną jakości obok.
- **Mapa i tabela** z filtrami klasy, statusu i flag. Kliknięcie wiersza albo punktu na mapie
  pokazuje obiekt: jego widoki z poszczególnych zdjęć (wycinki z ramką), parametry i współrzędne.
- **Jakość**: ocena rekonstrukcji z tego przebiegu i zalecenia (rozdział 4).
- **Wykresy**: rozrzut położeń obiektów i pewność wykrycia według klas.
- **Porównaj z innym przebiegiem**: przesunięcia tych samych obiektów między przebiegami i
  różnice ustawień (np. po zmianie powierzchni lub georeferencji).
- **Pobierz**: GeoPackage (QGIS), GeoJSON, CSV, KML (Google Earth), raport HTML. Wszystkie pliki
  są też w folderze `output/runs/<data_godzina>/`.

## 4. Jak czytać wyniki i kontrolę jakości

**Status obiektu**:

- **potwierdzony**: obiekt widoczny na co najmniej dwóch zdjęciach (liczbę można zmienić
  w Georeferencja → Zaawansowane), położenie uśrednione z widoków;
- **jeden widok**: obiekt z jednego zdjęcia, położenie mniej pewne (na mapie przerywany obrys);
- **flaga „nietypowy rozmiar”**: rozmiar obiektu poza zakresem ustawionym dla klasy; możliwa
  pomyłka detekcji.

**Rozrzut** to średnia odległość położeń z poszczególnych zdjęć od położenia obiektu. Mały rozrzut
(poniżej 1 m) oznacza spójne wyniki.

**Kontrola jakości** porównuje rekonstrukcję z niezależnymi danymi:

| Kontrola | Co sprawdza | Co zrobić przy ostrzeżeniu |
|---|---|---|
| Skala pionowa | wysokości modelu a barometr drona | rekonstrukcja z kalibracją kamery z innego lotu |
| Kalibracja kamery | ogniskowa modelu; różnica względem EXIF to tylko informacja | jak wyżej, gdy jest kalibracja referencyjna |
| Zgodność z NMPT | kształt modelu a NMPT GUGiK | sprawdź, czy teren się nie zmienił; wyrównanie do NMPT |
| Pochylenie modelu | przechył modelu względem NMPT | wyrównanie do NMPT (tryb automatyczny robi to sam) |
| Zasięg lotu | czy obszar lotu wystarcza do ustalenia orientacji z GPS | większy obszar, kilka wysokości, punkty kontrolne |

Kolory: zielony **OK**, żółty **OSTRZEŻENIE** (wyniki użyteczne, ze słabszymi stronami), czerwony
**ZŁY** (położenia mogą być błędne o kilka metrów), szary **informacja**. Od razu widać
najważniejsze punkty, reszta jest w sekcji **Szczegóły**. Przy zaleceniu wyrównania do NMPT
przycisk **Zastosuj** (lub **Zastosuj i uruchom ponownie** na stronie Wyniki) zmienia ustawienie
za użytkownika.

## 5. Rozwiązywanie problemów

**Docker nie działa** („Docker Desktop nie jest uruchomiony”): uruchom Docker Desktop, poczekaj,
aż ikona wieloryba przestanie się animować, i odśwież stronę. Gotową rekonstrukcję można zawsze
wskazać bez Dockera.

**Brak arkuszy NMT / NMPT** („Arkusze nie pokrywają obszaru lotu”): pobierz arkusze z
[geoportal.gov.pl](https://www.geoportal.gov.pl) (Dane do pobrania → NMT / NMPT, format ASC lub
GeoTIFF) i zapisz w folderach `data/nmt` i `data/nmpt` (albo wskaż własne w Georeferencja →
Zaawansowane). Komunikat podaje, z której strony obszaru brakuje danych. Bez arkuszy działa NMT
online, a tryb automatyczny pozostaje przy GPS drona.

**Usługa NMT GUGiK nie odpowiada**: sprawdź internet; usługa bywa chwilowo niedostępna. Program
korzysta z zapisanych wcześniej danych, jeśli są. Alternatywa: pobrane arkusze NMT.

**Ostrzeżenia `proj.db ... comes from another PROJ installation`**: zmienne `PROJ_LIB` i
`GDAL_DATA` ustawione przez instalator PostgreSQL/PostGIS. Wyniki są poprawne. Żeby usunąć
ostrzeżenia, usuń te zmienne ze zmiennych systemowych (jeśli PostGIS ich nie potrzebuje) albo
uruchamiaj program z `start_gui.bat` po `Remove-Item Env:PROJ_LIB, Env:GDAL_DATA`.

**Brak modeli na liście**: skopiuj plik `.pt` do folderu `models/` i odśwież stronę.

**Zdjęcia bez GPS**: zdjęcia bez pozycji w EXIF nie trafią do rekonstrukcji. Sprawdź ustawienia
drona (zapis lokalizacji).

**Przetwarzanie jest wolne**: bez karty NVIDIA detekcja działa na procesorze. Wyłącz
„Małe obiekty” i punkt z maski albo przetwórz wybrane zdjęcia.

## 6. Dla zaawansowanych

- **Tryb badawczy** (Ustawienia → Tryb badawczy albo zmienna środowiskowa
  `GEODETECT_RESEARCH=1`) pokazuje stronę **Eksperymenty**: uruchamianie porównań wariantów
  z `experiments/*.yaml` i podgląd ich podsumowań (opis w [experiments.md](experiments.md)).
- Wszystkie ustawienia GUI to plik konfiguracji programu (`config.yaml`, opis w README). Ten sam
  przebieg można uruchomić z wiersza poleceń: `python main.py --all`.
- Zrzuty ekranu tej instrukcji: `python scripts/gui_screenshots.py` przy działającym GUI
  (`streamlit run app/main.py --server.port 8599`).
