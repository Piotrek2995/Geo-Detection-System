"""Stan i walidacja GUI — bez importow streamlit (testowalne w tests/test_gui_state.py).

Strony GUI (app/views/*.py) tylko renderuja: odczytuja i ustawiaja PipelineConfig przez funkcje
z tego modulu, uruchamiaja zadania w tle (Job) i wyswietlaja Issue. Cala logika obliczen jest
w bibliotece geodetect (run_pipeline, odmrun, coverage, qc, runs).

Teksty widoczne dla uzytkownika: bez nazw z kodu/configu, bez myslnikow w etykietach.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from geodetect.config import GEOREF_MODES, POINT_METHODS, SURFACE_SOURCES, Z_ANCHOR_MODES, PipelineConfig

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "presets"
STATE_DIR = ROOT / ".gui_state"
LAST_PROJECT = "last_project.yaml"
PREFS = "prefs.json"
RESEARCH_ENV = "GEODETECT_RESEARCH"

ERROR, WARNING, INFO = "error", "warning", "info"

SURFACE_LABELS = {
    "nmt_gugik_rest": "Teren bez budynków, NMT GUGiK online",
    "nmt_local": "Teren bez budynków, NMT z pobranych arkuszy",
    "nmpt_local": "Teren z budynkami i drzewami, NMPT z pobranych arkuszy",
    "odm_dsm": "Model powierzchni z rekonstrukcji (DSM)",
}
SURFACE_SHORT = {"nmt_gugik_rest": "NMT online", "nmt_local": "NMT (arkusze)", "nmpt_local": "NMPT (arkusze)",
                 "odm_dsm": "DSM z rekonstrukcji"}
Z_MODE_LABELS = {
    "dem_alignment": "Dopasowanie modelu do NMT (zalecane)",
    "start_point": "Punkt startu drona",
    "agl_assumption": "Stała wysokość lotu nad terenem",
    "ground_frame": "Zdjęcie z ziemi w rekonstrukcji",
    "fixed": "Wartość podana ręcznie",
}
GEOREF_LABELS = {"auto": "Automatycznie (zalecane)",
                 "gps_only": "Standard: GPS drona",
                 "similarity_nmpt": "Wyrównanie do NMPT GUGiK"}
POINT_LABELS = {"bbox_center": "środek obiektu", "bbox_bottom_center": "podstawa obiektu (np. stopy)",
                "mask_centroid": "środek maski obiektu", "mask_ground_contact": "styk z gruntem (maska)",
                "mask_footprint_median": "środek podstawy (maska)"}
POINT_SIMPLE = ("bbox_center", "bbox_bottom_center")
STATUS_PL = {"confirmed": "potwierdzony", "single_view": "jeden widok"}
ODM_PARTS = {"rekonstrukcja": "rekonstrukcja kamer", "reference_lla": "układ odniesienia",
             "DSM": "model powierzchni (DSM)", "DTM": "model terenu (DTM)", "ortofotomapa": "ortofotomapa"}
ODM_REQUIRED = ("rekonstrukcja", "reference_lla")

assert set(SURFACE_LABELS) == set(SURFACE_SOURCES) and set(Z_MODE_LABELS) == set(Z_ANCHOR_MODES)
assert set(POINT_LABELS) == set(POINT_METHODS) and set(GEOREF_LABELS) == set(GEOREF_MODES)

# sekcje configu zapisywane w presecie (ustawienia detekcji, nie projekt)
PRESET_KEYS = ("model_path", "detect", "point", "size")


@dataclass
class Issue:
    level: str               # error | warning | info
    message: str
    where: str = ""          # strona GUI

    @property
    def blocking(self) -> bool:
        return self.level == ERROR


def has_errors(issues: list[Issue]) -> bool:
    return any(i.blocking for i in issues)


# ───────────── preferencje GUI (lokalnie) ─────────────

def load_prefs(folder: Path = STATE_DIR) -> dict:
    p = Path(folder) / PREFS
    base = {"surface_auto": True, "research_mode": False}
    try:
        base.update(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return base


def save_prefs(prefs: dict, folder: Path = STATE_DIR) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / PREFS).write_text(json.dumps(prefs, indent=1), encoding="utf-8")


def research_mode(prefs: Optional[dict] = None) -> bool:
    """Strona Eksperymenty: zmienna srodowiskowa GEODETECT_RESEARCH=1 albo przelacznik w ustawieniach."""
    if os.environ.get(RESEARCH_ENV, "").strip().lower() in ("1", "true", "tak", "yes"):
        return True
    return bool((prefs or {}).get("research_mode"))


# ───────────── konfiguracja, presety, ostatni projekt ─────────────

def gui_defaults(cfg: PipelineConfig) -> PipelineConfig:
    """Domyslne GUI: georeferencja automatyczna (CLI zachowuje ustawienie z configu)."""
    cfg.georef.mode = "auto"
    return cfg


def base_config(path: Optional[Path] = None) -> PipelineConfig:
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = PipelineConfig.from_file(p) if p.exists() else PipelineConfig.from_dict({}, base_dir=ROOT)
    return gui_defaults(cfg)


def clone(cfg: PipelineConfig) -> PipelineConfig:
    return copy.deepcopy(cfg)


def preset_slug(name: str) -> str:
    """Nazwa presetu = nazwa pliku (bez znakow zabronionych w Windows)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name).strip().strip(".")
    s = re.sub(r"\s+", " ", s)
    if not s:
        raise ValueError("Nazwa presetu nie może być pusta.")
    return s


def list_presets(folder: Path = PRESETS_DIR) -> list[str]:
    folder = Path(folder)
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else []


def preset_dict(cfg: PipelineConfig) -> dict:
    """Czesc configu zapisywana w presecie; model_path wzgledem katalogu repo, jesli to mozliwe."""
    d = cfg.to_dict()

    def rel(v):
        if isinstance(v, dict):
            return {k: rel(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rel(x) for x in v]
        if isinstance(v, str) and Path(v).is_absolute():
            try:
                return Path(v).resolve().relative_to(ROOT).as_posix()
            except ValueError:
                return v
        return v
    out = rel({k: d[k] for k in PRESET_KEYS})
    if out["detect"].get("visual_prompt_image"):      # prompt wizualny dotyczy konkretnego lotu
        out["detect"]["visual_prompt_image"] = None
        out["detect"]["visual_prompt_boxes"] = out["detect"]["visual_prompt_classes"] = None
    return out


def save_preset(cfg: PipelineConfig, name: str, folder: Path = PRESETS_DIR) -> Path:
    p = Path(folder) / f"{preset_slug(name)}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# Preset GUI: {preset_slug(name)} (ustawienia detekcji)\n"
                 + yaml.safe_dump(preset_dict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def load_preset(name: str, current: PipelineConfig, folder: Path = PRESETS_DIR) -> PipelineConfig:
    """Preset nakladany na biezacy projekt: zmienia tylko model i ustawienia detekcji."""
    p = Path(folder) / f"{preset_slug(name)}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Nie ma presetu „{name}”.")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    bad = [k for k in data if k not in PRESET_KEYS]
    if bad:
        raise ValueError(f"Preset „{name}” zawiera nieobsługiwane sekcje: {', '.join(bad)}.")
    merged = current.to_dict()
    merged.update(data)
    return PipelineConfig.from_dict(merged, base_dir=ROOT)


def remember_project(cfg: PipelineConfig, folder: Path = STATE_DIR) -> Path:
    return cfg.save(Path(folder) / LAST_PROJECT)


def last_project(folder: Path = STATE_DIR) -> Optional[PipelineConfig]:
    p = Path(folder) / LAST_PROJECT
    if not p.exists():
        return None
    try:
        return PipelineConfig.from_file(p)
    except Exception as exc:          # stary / uszkodzony plik — zaczynamy od domyslnego
        log.warning("Nie wczytano ostatniego projektu (%s): %s", p, exc)
        return None


# ───────────── projekt ─────────────

def set_start(cfg: PipelineConfig, image: Optional[str] = None, latlon: Optional[tuple] = None) -> None:
    """Punkt startu: zdjecie z ziemi (wykluczane z ODM i przetwarzania) i/lub wspolrzedne."""
    st = cfg.z_anchor.start
    st.image = image or None
    st.latlon = [float(latlon[0]), float(latlon[1])] if latlon else None


def odm_status(ok: dict) -> tuple[bool, str]:
    """(gotowe, jedno zdanie) z odmrun.odm_outputs_ok."""
    missing = [ODM_PARTS.get(k, k) for k, v in ok.items() if not v]
    if not all(ok.get(k) for k in ODM_REQUIRED):
        return False, "Brak rekonstrukcji w tym folderze."
    if missing:
        return True, "Rekonstrukcja gotowa, brakuje: " + ", ".join(missing) + "."
    return True, "Rekonstrukcja gotowa."


def project_status(cfg: PipelineConfig, n_photos: int, ok: dict) -> str:
    """Krotki status do paska bocznego."""
    ready, _ = odm_status(ok)
    return (f"Zdjęcia: {n_photos} · Rekonstrukcja: {'gotowa' if ready else 'brak'} · "
            f"Model: {Path(cfg.model_path).stem}")


def validate_project(cfg: PipelineConfig, photos: Optional[list] = None,
                     odm_ok: Optional[dict] = None) -> list[Issue]:
    w = "Projekt"
    out: list[Issue] = []
    if not Path(cfg.photos_dir).is_dir():
        return [Issue(ERROR, f"Folder zdjęć nie istnieje: {cfg.photos_dir}", w)]
    if photos is not None:
        if not photos:
            out.append(Issue(ERROR, "W folderze nie ma zdjęć (JPG, PNG, TIF).", w))
        no_gps = [p.name for p in photos if not p.has_gps]
        if no_gps:
            out.append(Issue(WARNING, f"{len(no_gps)} zdjęć bez pozycji GPS (np. {', '.join(no_gps[:3])}), "
                                      "rekonstrukcja może ich nie użyć.", w))
        st = cfg.z_anchor.start.image
        if st and st not in {p.name for p in photos}:
            out.append(Issue(ERROR, f"Zdjęcie startowe {st} nie istnieje w folderze zdjęć.", w))
    st = cfg.z_anchor.start
    if cfg.z_anchor.mode == "start_point" and not (st.image or st.latlon):
        out.append(Issue(ERROR, "Wybrana kalibracja wysokości wymaga punktu startu.", w))
    if st.latlon and not (49.0 <= st.latlon[0] <= 55.0 and 14.0 <= st.latlon[1] <= 24.5):
        out.append(Issue(WARNING, "Punkt startu leży poza Polską, sprawdź kolejność szerokości i długości.", w))
    if odm_ok is not None:
        ready, text = odm_status(odm_ok)
        if not ready:
            out.append(Issue(ERROR, f"{text} Wskaż folder z wynikami albo uruchom rekonstrukcję.", w))
    return out


# ───────────── detekcja ─────────────

@dataclass
class DetectionSettings:
    """Ustawienia detekcji edytowane w GUI (mapowane na cfg.detect / point / size)."""
    model_path: Path
    classes: Optional[list[str]]                     # None = wszystkie
    display_names: dict[str, str] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    conf: float = 0.25                               # dla klas bez wlasnego progu
    conf_per_class: dict[str, float] = field(default_factory=dict)
    tiling: bool = False
    tiling_classes: Optional[list[str]] = None
    point_method: str = "bbox_center"
    point_per_class: dict[str, str] = field(default_factory=dict)
    size_ranges: dict[str, list] = field(default_factory=dict)
    prompt_text: Optional[list[str]] = None
    visual_prompt_image: Optional[Path] = None
    visual_prompt_boxes: Optional[list[list[float]]] = None
    visual_prompt_classes: Optional[list[str]] = None

    def class_conf(self, cls: str) -> float:
        return float(self.conf_per_class.get(cls, self.conf))

    def class_point(self, cls: str) -> str:
        return self.point_per_class.get(cls, self.point_method)


def detection_settings(cfg: PipelineConfig) -> DetectionSettings:
    d = cfg.detect
    return DetectionSettings(
        Path(cfg.model_path), d.class_list(), dict(d.display_names), dict(d.colors), d.conf,
        dict(d.conf_per_class), d.tiling.enabled, list(d.tiling.classes) if d.tiling.classes else None,
        cfg.point.method, dict(cfg.point.per_class), {k: list(v) for k, v in cfg.size.ranges_m.items()},
        list(d.prompt_text) if d.prompt_text else None, d.visual_prompt_image,
        [list(b) for b in d.visual_prompt_boxes] if d.visual_prompt_boxes else None,
        list(d.visual_prompt_classes) if d.visual_prompt_classes else None)


def apply_detection(cfg: PipelineConfig, s: DetectionSettings, available: Optional[list[str]] = None) -> None:
    """Ustawienia GUI -> config. Wpisy dla klas, ktorych model nie ma (available), sa pomijane
    z wpisem w logu; wpisy dla niewybranych klas modelu zostaja (wroca po ponownym wyborze)."""
    keep = (lambda n: True) if available is None else (lambda n: n in available)  # noqa: E731

    def filt(dct: dict, what: str) -> dict:
        drop = [k for k in dct if not keep(k)]
        if drop:
            log.info("Pomijam ustawienie %s dla klas spoza modelu: %s", what, drop)
        return {k: v for k, v in dct.items() if keep(k)}
    d = cfg.detect
    cfg.model_path = Path(s.model_path)
    d.source = "model"
    d.classes = "all" if s.classes is None else [c for c in s.classes if keep(c)]
    d.display_names = {k: v for k, v in filt(s.display_names, "nazwa").items() if v and v != k}
    d.colors = filt(s.colors, "kolor")
    d.conf = float(s.conf)
    d.conf_per_class = {k: float(v) for k, v in filt(s.conf_per_class, "prog").items()}
    d.tiling.enabled = bool(s.tiling)
    d.tiling.classes = [c for c in (s.tiling_classes or []) if keep(c)] or None
    cfg.point.method = s.point_method
    cfg.point.per_class = {k: v for k, v in filt(s.point_per_class, "punkt").items() if v != s.point_method}
    cfg.size.ranges_m = {k: [float(v[0]), float(v[1])] for k, v in filt(s.size_ranges, "rozmiar").items() if v}
    d.prompt_text = [t.strip() for t in (s.prompt_text or []) if t.strip()] or None
    d.visual_prompt_image = s.visual_prompt_image
    d.visual_prompt_boxes = s.visual_prompt_boxes or None
    d.visual_prompt_classes = s.visual_prompt_classes or None


def model_class_names(info, cfg: PipelineConfig) -> list[str]:
    """Klasy do wyboru: z promptu dla modeli otwartego slownika, inaczej model.names."""
    from geodetect.detect import prompt_class_names
    if info is not None and getattr(info, "open_vocab", False):
        return prompt_class_names(cfg.detect) or []
    return list(info.names.values()) if info is not None else []


def _plural_klasa(n: int) -> str:
    if n == 1:
        return "klasa"
    return "klasy" if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else "klas"


def model_summary(info, cfg: PipelineConfig) -> str:
    """Jedna linijka opisu modelu, np. "Detekcja, 2 klasy: człowiek, namiot"."""
    task = {"detect": "Detekcja", "segment": "Segmentacja"}.get(info.task, str(info.task).capitalize())
    if getattr(info, "open_vocab", False):
        return f"{task}, szuka obiektów według opisu lub wzorca"
    names = sorted(cfg.detect.display(n) for n in info.names.values())
    shown = ", ".join(names[:8]) + (", …" if len(names) > 8 else "")
    return f"{task}, {len(names)} {_plural_klasa(len(names))}: {shown}"


def parse_prompt(text: str) -> Optional[list[str]]:
    """'ball, tent; person' -> ['ball', 'tent', 'person']."""
    items = [t.strip() for t in re.split(r"[,;\n]", text or "") if t.strip()]
    return items or None


def validate_detection(cfg: PipelineConfig, info=None) -> list[Issue]:
    w = "Detekcja"
    out: list[Issue] = []
    d = cfg.detect
    if d.source == "model" and not Path(cfg.model_path).exists():
        return [Issue(ERROR, f"Nie ma pliku modelu {Path(cfg.model_path).name}.", w)]
    if info is not None and getattr(info, "open_vocab", False):
        if d.uses_visual_prompt():
            if not Path(d.visual_prompt_image).exists():
                out.append(Issue(ERROR, "Zdjęcie wzorcowe nie istnieje.", w))
        elif not d.prompt_text:
            out.append(Issue(ERROR, "Wpisz, czego szukać, albo zaznacz obiekt na zdjęciu.", w))
    if info is not None:
        available = model_class_names(info, cfg)
        chosen = d.class_list()
        bad = [c for c in (chosen or []) if c not in available]
        if bad:
            out.append(Issue(ERROR, f"Model nie zna klas: {', '.join(bad)}.", w))
        if chosen == []:
            out.append(Issue(ERROR, "Nie wybrano żadnej klasy do wykrywania.", w))
    for k, v in d.conf_per_class.items():
        if not 0.0 < float(v) <= 1.0:
            out.append(Issue(ERROR, f"Próg pewności dla „{d.display(k)}” musi być większy od 0 i nie większy od 1.", w))
    for k, v in cfg.size.ranges_m.items():
        if len(v) != 2 or float(v[0]) >= float(v[1]):
            out.append(Issue(ERROR, f"Rozmiar dla „{d.display(k)}”: minimum musi być mniejsze od maksimum.", w))
    return out


# ───────────── georeferencja ─────────────

def surface_availability(rest=None, coverage: Optional[dict] = None,
                         odm_ok: Optional[dict] = None) -> dict[str, tuple[bool, str]]:
    """{zrodlo: (dostepne, krotki opis)} z RestStatus, surface_coverage i odm_outputs_ok."""
    cov = coverage or {}

    def local(key):
        c = cov.get(key)
        if c is None:
            return False, "brak rekonstrukcji do sprawdzenia pokrycia"
        return bool(c.ok), c.message
    dsm = bool((odm_ok or {}).get("DSM"))
    return {
        "nmt_gugik_rest": (bool(rest is not None and rest.ok),
                           "usługa działa" if rest is not None and rest.ok else "usługa niedostępna"),
        "nmt_local": local("nmt_local"),
        "nmpt_local": local("nmpt_local"),
        "odm_dsm": (dsm, "jest w rekonstrukcji" if dsm else "brak w rekonstrukcji"),
    }


def resolve_surface(avail: dict[str, tuple[bool, str]]) -> tuple[str, str]:
    """Wybor automatyczny: NMPT z arkuszy, gdy pokrywa obszar; inaczej NMT online (i dalej)."""
    for src, why in (("nmpt_local", "są arkusze NMPT dla tego obszaru"),
                     ("nmt_gugik_rest", "brak arkuszy NMPT, używam NMT online"),
                     ("nmt_local", "brak NMPT i usługi online, używam arkuszy NMT"),
                     ("odm_dsm", "brak danych GUGiK, używam modelu z rekonstrukcji")):
        if avail.get(src, (False, ""))[0]:
            return src, why
    return "nmt_gugik_rest", "żadna powierzchnia nie jest dostępna"


def validate_georef(cfg: PipelineConfig, coverage: Optional[dict] = None, rest=None,
                    odm_ok: Optional[dict] = None) -> list[Issue]:
    """coverage: {"nmt_local"/"nmpt_local": FolderCoverage}, rest: RestStatus, odm_ok: odm_outputs_ok."""
    w = "Georeferencja"
    out: list[Issue] = []
    src = cfg.surface.source
    if src == "nmt_gugik_rest" and rest is not None and not rest.ok:
        out.append(Issue(ERROR, "Usługa NMT GUGiK nie odpowiada. Wybierz pobrane arkusze albo sprawdź internet.", w))
    if cfg.dem.source == "gugik" and rest is not None and not rest.ok:
        out.append(Issue(WARNING, "Usługa NMT GUGiK nie odpowiada; kalibracja wysokości użyje zapisanych "
                                  "wcześniej danych albo się nie powiedzie.", w))
    for key, label in (("nmt_local", "NMT"), ("nmpt_local", "NMPT")):
        cov = (coverage or {}).get(key)
        if cov is not None and src == key and not cov.ok:
            out.append(Issue(ERROR, f"Arkusze {label} nie pokrywają obszaru lotu ({cov.message}). Pobierz je "
                                    f"z geoportal.gov.pl do folderu {cov.folder}.", w))
    cov = (coverage or {}).get("nmpt_local")
    if cfg.georef.mode == "similarity_nmpt" and cov is not None and not cov.ok:
        out.append(Issue(ERROR, f"Wyrównanie do NMPT wymaga arkuszy NMPT dla obszaru lotu ({cov.message}).", w))
    if cfg.georef.mode == "auto" and cov is not None and not cov.ok:
        out.append(Issue(INFO, "Brak arkuszy NMPT: tryb automatyczny użyje standardu (GPS drona).", w))
    need_dsm = src == "odm_dsm" or cfg.z_anchor.mode == "dem_alignment" or cfg.georef.mode == "similarity_nmpt"
    if need_dsm and odm_ok is not None and not odm_ok.get("DSM"):
        out.append(Issue(ERROR, "Wybrane ustawienia wymagają modelu powierzchni (DSM) z rekonstrukcji.", w))
    st = cfg.z_anchor.start
    if cfg.z_anchor.mode == "start_point" and not (st.image or st.latlon):
        out.append(Issue(ERROR, "Kalibracja z punktu startu wymaga punktu startu (strona Projekt).", w))
    if cfg.z_anchor.mode == "fixed" and cfg.z_anchor.offset_m is None:
        out.append(Issue(ERROR, "Podaj wartość przesunięcia wysokości.", w))
    return out


def apply_action(cfg: PipelineConfig, action: str) -> str:
    """Akcja z zalecenia QC (qc.ACTIONS). Zwraca opis zmiany."""
    if action == "georef_nmpt":
        cfg.georef.mode = "similarity_nmpt"
        return "Ustawiono wyrównanie do NMPT."
    if action == "georef_auto":
        cfg.georef.mode = "auto"
        return "Ustawiono automatyczny wybór georeferencji."
    raise ValueError(f"Nieznana akcja {action!r}")


# ───────────── podsumowania ─────────────

def settings_summary(cfg: PipelineConfig, n_images: Optional[int] = None) -> list[tuple[str, str]]:
    """Najwazniejsze wybory uzytkownika (strona Uruchom)."""
    d = cfg.detect
    cls = d.class_list()
    model = Path(cfg.model_path).stem
    if d.prompt_text:
        model += f" (szuka: {', '.join(d.prompt_text)})"
    elif d.uses_visual_prompt():
        model += " (wzorzec z zaznaczenia)"
    return [
        ("Zdjęcia", (f"{n_images}, " if n_images is not None else "") + f"folder {Path(cfg.photos_dir).name}"),
        ("Model", model),
        ("Klasy", "wszystkie" if cls is None else ", ".join(d.display(c) for c in cls)),
        ("Powierzchnia", SURFACE_LABELS.get(cfg.surface.source, cfg.surface.source)),
        ("Georeferencja", GEOREF_LABELS.get(cfg.georef.mode, cfg.georef.mode)),
    ]


def settings_details(cfg: PipelineConfig) -> list[tuple[str, str]]:
    d = cfg.detect
    cls = d.class_list() or list(d.conf_per_class)
    st = cfg.z_anchor.start
    return [
        ("Rekonstrukcja", str(cfg.odm_dir)),
        ("Punkt startu", st.image or (", ".join(f"{x:.6f}" for x in st.latlon) if st.latlon else "brak")),
        ("Kalibracja wysokości", Z_MODE_LABELS.get(cfg.z_anchor.mode, cfg.z_anchor.mode)),
        ("Progi pewności", ", ".join(f"{d.display(c)} {d.conf_for(c):g}" for c in cls) or f"{d.conf:g}"),
        ("Detekcja na fragmentach zdjęcia", "tak" if d.tiling.enabled else "nie"),
        ("Łączenie widoków", f"do {cfg.aggregate.eps_m:g} m, potwierdzenie od {cfg.aggregate.min_views} widoków"),
    ]


def run_label(r) -> str:
    """Etykieta przebiegu (runs.RunSummary): data, liczba obiektow, model."""
    try:
        when = datetime.strptime(r.name[:17], "%Y-%m-%d_%H%M%S").strftime("%d.%m.%Y %H:%M")
    except ValueError:
        when = r.name
    return f"{when} · {r.n_objects} obiektów · {Path(str(r.model or '')).stem or 'model nieznany'}"


# ───────────── zadania w tle ─────────────

class _ThreadLogHandler(logging.Handler):
    """Zbiera logi z jednego watku (zadania) do bufora linii."""

    def __init__(self, thread_id: int, sink: deque):
        super().__init__(logging.INFO)
        self.thread_id, self.sink = thread_id, sink
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread_id:
            try:
                self.sink.append(self.format(record))
            except Exception:            # noqa: BLE001 — log nie moze przerwac zadania
                pass


class Job:
    """Zadanie w watku tla: postep, log na zywo, przerwanie (GUI odpytuje stan co ~1 s).

    fn(job) wykonuje prace; moze ustawiac job.progress / job.message / job.items i sprawdzac
    job.cancelled. Wynik -> job.result, wyjatek -> job.error (status "error")."""

    def __init__(self, name: str, fn: Callable[["Job"], Any], max_log: int = 2000):
        from geodetect.run import CancelToken
        self.name, self.fn = name, fn
        self.status = "pending"           # pending | running | done | cancelled | error
        self.progress = 0.0
        self.message = ""
        self.logs: deque[str] = deque(maxlen=max_log)
        self.items: list = []             # np. przetworzone zdjecia (miniatury)
        self.result: Any = None
        self.error: Optional[str] = None
        self.token = CancelToken()
        self.started = self.finished = None
        self._thread: Optional[threading.Thread] = None

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    @property
    def running(self) -> bool:
        return self.status in ("pending", "running")

    @property
    def elapsed_s(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def cancel(self) -> None:
        self.token.cancel()
        self.message = "Przerywanie…"

    def start(self) -> "Job":
        self._thread = threading.Thread(target=self._run, name=f"job-{self.name}", daemon=True)
        self.status = "running"
        self.started = time.time()
        self._thread.start()
        return self

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        handler = _ThreadLogHandler(threading.get_ident(), self.logs)
        root = logging.getLogger()
        root.addHandler(handler)
        old = root.level
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        try:
            self.result = self.fn(self)
            self.status = "cancelled" if self.cancelled else "done"
        except Exception as exc:          # noqa: BLE001 — blad zadania pokazywany w GUI
            self.error = f"{type(exc).__name__}: {exc}"
            self.logs.append(traceback.format_exc())
            self.status = "error"
        finally:
            self.finished = time.time()
            root.removeHandler(handler)
            root.setLevel(old)


def start_run_job(cfg: PipelineConfig, all_images: bool = True, images: Optional[list[str]] = None,
                  out_dir: Optional[Path] = None, run=None) -> Job:
    """Przebieg run_pipeline w tle. job.items: nazwy przetworzonych zdjec; wynik: RunResult."""
    from geodetect.run import run_pipeline
    run = run or run_pipeline
    cfg = clone(cfg)

    def work(job: Job):
        def progress(p):
            job.progress, job.message = p.fraction, p.message
            if p.stage == "images" and p.i > 0:
                job.items.append(p.message)
        res = run(cfg, progress_cb=progress, cancel=job.token, all_images=all_images, images=images,
                  out_dir=out_dir)
        if res.error:
            raise RuntimeError(res.error)
        job.message = "Przerwano" if res.cancelled else f"Gotowe: {len(res.objects)} obiektów"
        return res
    return Job("przebieg", work).start()


def start_odm_job(cfg: PipelineConfig, fixed_cameras_from: Optional[Path] = None, runner=None) -> Job:
    """prepare_odm (bez zdjecia startowego) + opcjonalnie cameras.json z innego projektu + ODM w Dockerze."""
    from geodetect import odmrun
    runner = runner or odmrun.run_odm
    cfg = clone(cfg)

    def work(job: Job):
        job.message = "Kopiowanie zdjęć do projektu rekonstrukcji"
        prep = odmrun.prepare_odm(cfg)
        job.logs.append(f"Zdjęcia do rekonstrukcji: {prep.n_images} (pominięte: {', '.join(prep.excluded) or 'brak'})")
        if fixed_cameras_from:
            odmrun.write_cameras_json(Path(fixed_cameras_from), cfg.odm_dir)
        cmd = odmrun.odm_command(cfg.odm_dir, fixed_cameras=bool(fixed_cameras_from))
        job.logs.append("$ " + " ".join(cmd))
        job.message = "Rekonstrukcja trwa (kilkanaście do kilkudziesięciu minut)"

        def line(s: str):
            job.logs.append(s)
            low = s.lower()
            for stage, frac in (("running dataset stage", 0.05), ("running opensfm", 0.1), ("running openmvs", 0.5),
                                ("running odm_filterpoints", 0.6), ("running odm_meshing", 0.65),
                                ("running odm_dem", 0.85), ("running odm_orthophoto", 0.9)):
                if stage in low:
                    job.progress = max(job.progress, frac)
                    job.message = s.strip()[-90:]
        res = runner(cmd, line_cb=line, cancel=lambda: job.cancelled)
        if not res.cancelled and res.returncode != 0:
            raise RuntimeError(f"Rekonstrukcja zakończyła się błędem (kod {res.returncode}), szczegóły w logu.")
        job.progress = 1.0
        return res
    return Job("odm", work).start()


def list_experiments(folder: Path = ROOT / "experiments") -> list[Path]:
    return sorted(Path(folder).glob("*.yaml"))


def start_experiment_job(path: Path, images: Optional[list[str]] = None, out_dir: Optional[Path] = None,
                         runner_cls=None) -> Job:
    """Eksperyment z experiments/*.yaml w tle; wynik: katalog z summary.md."""
    from geodetect import experiments as E
    runner_cls = runner_cls or E.ExperimentRunner

    def work(job: Job):
        exp = E.load_experiment(path)
        job.message = f"{exp.name}: {len(exp.variants)} wariantów"
        return runner_cls(exp, out_dir, images, store=E.DetectionStore()).run()
    return Job("eksperyment", work).start()


def latest_experiment_output(name: str, root: Path = ROOT / "output" / "experiments") -> Optional[Path]:
    d = Path(root) / name
    runs = sorted((p for p in d.iterdir() if (p / "summary.md").exists()), reverse=True) if d.is_dir() else []
    return runs[0] if runs else None


# ───────────── porownanie przebiegow ─────────────

PARAM_LABELS = {
    "model": "model", "classes": "klasy", "conf": "próg pewności", "imgsz": "rozdzielczość detekcji",
    "tiling": "małe obiekty (fragmenty)", "point_method": "punkt lokalizacji", "surface": "powierzchnia",
    "georef_mode": "georeferencja", "georef_decision": "wybór georeferencji", "z_calibration.method":
    "kalibracja wysokości", "z_calibration.offset_m": "przesunięcie wysokości [m]",
    "aggregate.eps_m": "odległość łączenia [m]", "aggregate.min_views": "widoki do potwierdzenia",
    "aggregate.weights": "wagi widoków", "photos_dir": "folder zdjęć", "odm_dir": "rekonstrukcja",
}


def param_label(key: str) -> str:
    if key in PARAM_LABELS:
        return PARAM_LABELS[key]
    head, _, tail = key.partition(".")
    prefix = {"conf_per_class": "próg pewności", "point_per_class": "punkt lokalizacji",
              "display_names": "nazwa", "class_colors": "kolor", "size_ranges_m": "zakres rozmiaru",
              "filters": "filtr"}.get(head)
    return f"{prefix}: {tail}" if prefix and tail else key


def comparison_rows(matches, a, b, labels: Optional[dict] = None) -> list[dict]:
    """Wiersze tabeli porownania (runs.compare_runs): klasa, obiekt A, obiekt B, przesuniecie."""
    labels = labels or {}
    rows = []
    for m in sorted(matches, key=lambda m: (m.shift_m is None, -(m.shift_m or 0))):
        rows.append({"klasa": labels.get(m.cls, m.cls),
                     "obiekt A": None if m.a_id is None else m.a_id + 1,     # numeracja od 1
                     "obiekt B": None if m.b_id is None else m.b_id + 1,
                     "przesunięcie [m]": None if m.shift_m is None else round(m.shift_m, 2),
                     "różnica wysokości [m]": None if m.dh_m is None else round(m.dh_m, 2),
                     "uwagi": "" if m.a_id is not None and m.b_id is not None else
                     ("tylko w A" if m.b_id is None else "tylko w B")})
    return rows
