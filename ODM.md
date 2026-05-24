# ODM (OpenDroneMap) — instrukcja dla tego projektu

Krótka, dedykowana ściąga do uruchamiania ODM na zdjęciach z drona. Zakłada Windows
+ Docker Desktop + WSL2 (sprawdzone na: Docker 29.4.3, Windows 11, WSL2/Ubuntu).

---

## 1. Jednorazowo: instalacja Docker Desktop

1. Pobierz instalator: <https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe>
2. Uruchom — w wizardzie zaznacz **"Use WSL 2"** (powinno być domyślnie).
3. Wyloguj się z Windows i zaloguj ponownie (instalator o to poprosi).
4. Uruchom Docker Desktop → poczekaj aż ikona w trayu pokaże **"Engine running"**.

### Częsty problem: `docker-credential-desktop not found in PATH`

Jeśli widzisz ten błąd przy `docker pull`, oznacza to, że bieżąca sesja shell-a została
otwarta przed instalacją Dockera i nie ma świeżego PATH. Dwa rozwiązania:

- **Najprościej:** zamknij i otwórz na nowo PowerShell / PyCharm Terminal.
- **Bez restartu:** przedrostek do komendy:
  ```powershell
  $env:PATH = "$env:ProgramFiles\Docker\Docker\resources\bin;" + $env:PATH
  ```
  i potem normalnie `docker ...`.

---

## 2. Jednorazowo: pobranie obrazu ODM (~2.3 GB)

```powershell
docker pull opendronemap/odm
```

Sprawdź, że pobrał się prawidłowo:

```powershell
docker images opendronemap/odm
```

---

## 3. Każdy nowy lot: struktura katalogów

ODM **wymaga**, by zdjęcia siedziały w podkatalogu `images/` wewnątrz katalogu projektu.
Dla tego repo używamy `./odm_project/` jako katalogu projektu. Strukturę robi się raz:

```powershell
New-Item -ItemType Directory -Force -Path .\odm_project\images | Out-Null
Copy-Item .\zdjecia\*.JPG .\odm_project\images\ -Force
```

Po nowym locie wystarczy ponownie skopiować zawartość `./zdjecia/` do
`./odm_project/images/` (i ew. wyczyścić poprzednie wyniki z `./odm_project/`).

> **Konwencja na przyszłość:** Pierwsze zdjęcie wykonuj **na ziemi w miejscu startu
> drona**, przed wzbiciem się w powietrze. Da to "anchor GPS" potrzebny do
> skalibrowania wysokości npm — patrz `main.py / compute_z_offset()`.

---

## 4. Uruchomienie ODM

```powershell
docker run --rm -v "${PWD}\odm_project:/datasets/code" opendronemap/odm --project-path /datasets
```

Wyjaśnienie flag:
- `--rm` — usuń kontener po zakończeniu (nic nie zaśmieca).
- `-v "${PWD}\odm_project:/datasets/code"` — montuje lokalny `odm_project` jako
  `/datasets/code` w kontenerze. **Nazwa `code` jest obowiązkowa** w kontekście
  `--project-path /datasets`.
- `--project-path /datasets` — root w kontenerze, w którym ODM szuka katalogu projektu.

**Czas:** dla ~25 zdjęć i 12 rdzeni CPU: ~15–45 min (bez GPU).

**Postęp** podglądasz w Docker Desktop → **Containers** → klik kontenera → **Logs**.

### Co znaczą etapy w logach

| Etap | Co produkuje | Czy nam potrzebne |
|---|---|---|
| `dataset` | walidacja zdjęć i EXIF | — |
| `split` | (przy dużych zestawach) podział na sub-modele | — |
| `run_opensfm` | **`opensfm/reconstruction.json`** — R, t, kalibracja | **TAK** (IO+EO) |
| `mve / openmvs` | gęsta chmura punktów | — (orto/mesh) |
| `mvs_texturing` | tekstura meshu | — |
| `georeferencing` | **`opensfm/reference_lla.json`** + `odm_report/shots.geojson` | **TAK** (geo origin) |
| `dem` | własny DEM ODM (z lotu) | opcjonalnie |
| `orthophoto` | ortofoto (`odm_orthophoto/`) | opcjonalnie do podglądu |
| `report` | `odm_report/report.pdf` | TAK — diagnostyka |

---

## 5. Weryfikacja wyników

Po skończonym `docker run` powinieneś mieć:

```
odm_project/
├── cameras.json                          # kalibracja IO (rzadko czytamy)
├── opensfm/
│   ├── reconstruction.json               # ← R, t, IO każdego zdjęcia (główne źródło)
│   └── reference_lla.json                # ← origin lokalnego ENU
├── odm_report/
│   ├── shots.geojson                     # ← pozycje kamer w WGS84
│   └── report.pdf                        # diagnostyka jakości rekonstrukcji
└── (... ortofoto, mesh, dem — opcjonalnie)
```

Sanity check przed odpaleniem `main.py`:

```powershell
python -c "import json; r=json.load(open('odm_project/opensfm/reconstruction.json',encoding='utf-8'))[0]; print('shots:', len(r['shots']), '/ kamery:', list(r['cameras'].keys()))"
```

Spodziewane: liczba `shots` ≈ liczba zdjęć z lotu. Jeśli jest dużo mniej (np. 10/26),
to znaczy, że ODM nie był w stanie powiązać większości zdjęć — typowo z powodu zbyt
małego pokrycia (overlap) między klatkami albo słabej jakości obrazu.

W `report.pdf` szukaj:
- **Reconstructed cameras** ≈ liczba wejściowych zdjęć
- **Mean reprojection error** < 1 px (idealnie < 0.5)
- **GSD** (ground sampling distance) — powinno być spójne z wysokością lotu

---

## 6. Czyszczenie / nowe podejście

Jeśli chcesz puścić ODM od zera (np. po zmianie zdjęć):

```powershell
Remove-Item .\odm_project\opensfm,.\odm_project\odm_report,.\odm_project\odm_meshing,.\odm_project\odm_texturing,.\odm_project\odm_georeferencing,.\odm_project\odm_orthophoto,.\odm_project\odm_dem,.\odm_project\mve,.\odm_project\openmvs,.\odm_project\cameras.json,.\odm_project\images.json -Recurse -Force -ErrorAction SilentlyContinue
```

`./odm_project/images/` zostaw — to wejście, nie wynik.

---

## 7. Częste problemy

| Objaw | Powód | Co zrobić |
|---|---|---|
| `error getting credentials ... docker-credential-desktop` | Stary PATH w sesji | Patrz sekcja 1. |
| `Cannot connect to the Docker daemon` | Docker Desktop nie startuje | Otwórz Docker Desktop ręcznie, poczekaj na "Engine running". |
| `Reconstructed cameras: 5/26` | Słaby overlap zdjęć / motion blur / jednolite tło (woda, śnieg) | Lataj z większym overlapem (>70% forward, >60% side). |
| `MemoryError` w MVS | Za mało RAM dla gęstej rekonstrukcji | Dodaj flagę `--feature-quality medium --pc-quality medium` do `docker run`. |
| Kamera "pod terenem" w ray cast | Potensic GPS = AGL od startu, nie npm | `compute_z_offset()` w `main.py` to kompensuje. |
