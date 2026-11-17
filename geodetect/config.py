"""Konfiguracja pipeline'u: dataclassy + wczytywanie z YAML/JSON.

Sciezki wzgledne w pliku konfiguracyjnym sa rozwiazywane wzgledem katalogu,
w ktorym lezy ten plik (dzieki temu config.yaml dziala niezaleznie od cwd).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_GUGIK_NMT_URL = "https://services.gugik.gov.pl/nmt/"


@dataclass
class DemConfig:
    """Zrodlo modelu wysokosciowego.

    source: "gugik"  — otwarty REST GUGiK (GetHByXY, punktowo, NMT 1 m),
            "raster" — lokalny GeoTIFF (NMT lub NMPT) w EPSG:2180, sciezka w `raster_path`.
    """
    source: str = "gugik"
    raster_path: Optional[Path] = None
    gugik_url: str = DEFAULT_GUGIK_NMT_URL
    timeout_s: float = 10.0
    cache_path: Optional[Path] = None  # JSON z probkami GUGiK (np. output/cache/gugik_nmt.json)


@dataclass
class TilingConfig:
    """Inferencja kafelkowa: kafle tile_size (None = imgsz) z zakladka `overlap`,
    opcjonalnie + pelny obraz przeskalowany do imgsz (duze obiekty). Scalanie w
    geodetect/tiling.py."""
    enabled: bool = False
    classes: Optional[list[str]] = None   # klasy brane z kafli (None = wszystkie); pozostale
                                          # tylko z pelnego obrazu (wymusza pelny obraz)
    tile_size: Optional[int] = None
    overlap: float = 0.2
    include_full: bool = True
    edge_margin_px: float = 3.0     # box blizej wewnetrznej krawedzi kafla -> uciety
    nms_iou: float = 0.5            # NMS miedzy kaflami (per klasa)
    contain_ios: float = 0.6        # przeciecie / pole mniejszego: duplikat zawarty w innym boxie


@dataclass
class DetectConfig:
    """Parametry detekcji YOLO — klasy zawsze po NAZWIE z model.names (nie po ID).

    classes: lista nazw albo "all"/None; conf_per_class nadpisuje conf; display_names
    i colors (hex) — nazwy i kolory klas na mapie / w eksporcie. imgsz None = z treningu
    modelu. device: auto | cuda | cuda:0 | cpu. half = fp16 (tylko GPU)."""
    conf: float = 0.25
    conf_per_class: dict[str, float] = field(default_factory=dict)
    classes: Optional[Any] = None
    display_names: dict[str, str] = field(default_factory=dict)
    colors: dict[str, str] = field(default_factory=dict)
    iou: float = 0.7
    imgsz: Optional[int] = None
    device: Optional[str] = "auto"
    half: bool = False
    max_det: int = 300
    batch: int = 8                  # obrazy/kafle na jedno wywolanie modelu
    top_k_per_class: Optional[int] = None   # po filtrze klas: zostaw k najpewniejszych na
                                            # zdjecie (max_det dziala przed filtrem klas)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    # Zrodlo detekcji: "model" (YOLO) albo "manual" — punkty wskazane recznie w CSV
    # (image,u,v,visible), np. punkt odniesienia w eksperymencie walidacyjnym. Punkt manualny
    # staje sie detekcja klasy manual_class z bboxem manual_box_px wokol (u, v).
    # "csv" — zapisane detekcje innego detektora (detections_csv: image,u,v,visible,conf,x1..y2),
    # np. detektor otwartego slownika spoza pipeline'u (scripts/pilka_detectors.py).
    source: str = "model"
    manual_csv: Optional[Path] = None
    detections_csv: Optional[Path] = None
    manual_class: str = "manual"
    manual_box_px: float = 6.0

    def class_list(self) -> Optional[list[str]]:
        """None = wszystkie klasy."""
        if self.classes is None or (isinstance(self.classes, str) and self.classes.lower() == "all"):
            return None
        return [self.classes] if isinstance(self.classes, str) else list(self.classes)

    def conf_for(self, cls_name: str) -> float:
        return float(self.conf_per_class.get(cls_name, self.conf))

    def display(self, cls_name: str) -> str:
        return self.display_names.get(cls_name, cls_name)

    def torch_device(self) -> Optional[str]:
        return None if self.device in (None, "", "auto") else self.device


@dataclass
class RaycastConfig:
    """Marsz po promieniu od `t_min_m` do `t_max_m` + bisekcja na pierwszym przecieciu.

    Krok: `step_m` dla zrodel bez rozdzielczosci (REST, punktowe), a dla rastrow
    rozdzielczosc * `raster_step_factor` (<= 1, zeby nie przeskoczyc cienkiej przeszkody
    na NMPT/DSM). Punkty promienia probkowane wsadowo po `chunk` krokow."""
    step_m: float = 5.0
    raster_step_factor: float = 1.0
    t_min_m: float = 0.5
    t_max_m: float = 800.0
    bisect_iters: int = 25
    chunk: int = 64
    rest_chunk: int = 8           # dla REST mniejsze paczki (mniej zapytan za trafieniem)

    def step_for(self, resolution_m: Optional[float]) -> float:
        if resolution_m is None:
            return self.step_m
        return min(self.step_m, resolution_m * self.raster_step_factor)


SURFACE_SOURCES = ("nmt_gugik_rest", "nmt_local", "nmpt_local", "odm_dsm")


@dataclass
class SurfaceConfig:
    """Powierzchnia, na ktora rzutujemy promienie (ray casting).

    source: nmt_gugik_rest | nmt_local | nmpt_local | odm_dsm (opis w geodetect/surface.py).
    Kalibracja Z (z_anchor) NIE zalezy od tej sekcji — zawsze wzgledem NMT z sekcji `dem`."""
    source: str = "nmt_gugik_rest"
    nmt_dir: Path = Path("data/nmt")
    nmpt_dir: Path = Path("data/nmpt")
    odm_dsm: Optional[Path] = None           # None = odm_dir/odm_dem/dsm.tif
    interpolation: str = "bilinear"          # nearest | bilinear
    default_crs: str = "EPSG:2180"           # dla plikow bez CRS (ASC bez .prj, XYZ)
    area_buffer_m: float = 300.0             # mozaika: bbox kamer + bufor


POINT_METHODS = ("bbox_center", "bbox_bottom_center", "mask_centroid",
                 "mask_ground_contact", "mask_footprint_median")
MASK_METHODS = ("mask_centroid", "mask_ground_contact", "mask_footprint_median")


@dataclass
class PointConfig:
    """Punkt reprezentatywny detekcji — piksel (lub piksele), przez ktory idzie promien.

    method: bbox_center | bbox_bottom_center | mask_centroid | mask_ground_contact |
            mask_footprint_median (opis w geodetect/points.py); per_class nadpisuje method
            dla wybranych klas, np. {people: bbox_bottom_center}.
    Metody mask_* uzywaja SAM (ultralytics, prompt = bbox YOLO); maski w cache na dysku.
    Maska odrzucona przez kontrole jakosci -> bbox_center (z adnotacja w wynikach)."""
    method: str = "bbox_center"
    per_class: dict[str, str] = field(default_factory=dict)
    # SAM
    sam_model: Path = Path("models/sam2.1_b.pt")      # lzejsza opcja: mobile_sam.pt
    sam_device: Optional[str] = None           # None = GPU jesli dostepne
    mask_cache_dir: Path = Path("output/cache/masks")
    # kontrola jakosci maski (najwieksza spojna skladowa)
    min_area_ratio: float = 0.10       # pole maski / pole bboxa ponizej -> "too_small"
    max_area_ratio: float = 0.97       # powyzej i dotyka wszystkich krawedzi -> "fills_bbox"
    min_largest_frac: float = 0.60     # najwieksza skladowa / cala maska ponizej -> "fragmented"
    edge_tol_px: float = 2.0
    # mask_ground_contact
    min_offnadir_deg: float = 5.0      # obiekt blizej nadiru -> kierunek pionu nieokreslony
    contact_band: float = 0.05         # pas przy krawedzi styku (ulamek rozciaglosci maski)
    # mask_footprint_median
    footprint_samples: int = 200
    seed: int = 0

    def method_for(self, cls_name: str) -> str:
        return self.per_class.get(cls_name, self.method)

    def needs_masks(self) -> bool:
        return any(m in MASK_METHODS for m in [self.method, *self.per_class.values()])

    def validate(self) -> None:
        for m in [self.method, *self.per_class.values()]:
            if m not in POINT_METHODS:
                raise ValueError(f"Nieznana metoda punktu {m!r} (dozwolone: {POINT_METHODS})")


@dataclass
class FilterConfig:
    """Filtry pojedynczych detekcji. Odrzucone zostaja w wynikach z `rejected_reason`,
    ale nie biora udzialu w agregacji. None / False = filtr wylaczony."""
    border_margin_px: float = 2.0      # bbox blizej krawedzi niz tyle px -> touches_border
    exclude_border: bool = False
    max_range_m: Optional[float] = None
    max_incidence_deg: Optional[float] = None


@dataclass
class AggregateConfig:
    """DBSCAN w EPSG:2180, osobno dla kazdej klasy.
    weights: "equal" | "conf" | "cos_incidence" — wagi sredniej pozycji obiektu."""
    eps_m: float = 2.5
    min_samples: int = 1
    weights: str = "equal"
    min_views: int = 2          # obiekt z >= min_views widokami -> confirmed, inaczej single_view


Z_ANCHOR_MODES = ("agl_assumption", "ground_frame", "dem_alignment", "fixed", "start_point")


@dataclass
class StartConfig:
    """Punkt startu drona (tryb z_anchor.mode=start_point).

    Barometr Potensica jest zerowany przy starcie na ziemi, wiec grunt w punkcie startu ma
    wysokosc EXIF ground_alt_exif_m (domyslnie 0).

    image:  zdjecie zrobione w miejscu startu tuz nad ziemia (np. dron trzymany ok. 1 m nad
            punktem startu). Sluzy wylacznie do lokalizacji startu (XY z EXIF) i do kontroli
            barometru; jest automatycznie wykluczane z rekonstrukcji ODM
            (scripts/prepare_odm.py) i z przetwarzania. Jego wysokosc EXIF to wysokosc TEJ
            klatki nad gruntem, nie wysokosc gruntu.
    latlon: [lat, lon] WGS84 startu zmierzone niezaleznie (np. srednia GPS telefonu);
            ma priorytet przed XY z EXIF zdjecia `image`.
    ground_alt_exif_m: wysokosc EXIF odpowiadajaca gruntowi w punkcie startu (0 = barometr
            zerowany na ziemi).
    image_height_m: deklarowana wysokosc zdjecia startowego nad gruntem (np. 1.0); roznica
            wzgledem jego wysokosci EXIF jest orientacyjna precyzja barometru (raport)."""
    image: Optional[str] = None
    latlon: Optional[list[float]] = None
    ground_alt_exif_m: float = 0.0
    image_height_m: Optional[float] = None


@dataclass
class ZAnchorConfig:
    """Kalibracja wysokosci: H [m npm] = h_ODM + z_offset. Potensic zapisuje GPS Altitude
    jako AGL od punktu startu, wiec h z ODM nie jest wysokoscia npm.

    mode:
      agl_assumption — z_offset = NMT(mediana xy kamer) + agl_m - srednia h kamer (fallback),
      ground_frame   — klatka zrobiona na ziemi w punkcie startu (`image`):
                       z_offset = NMT(xy) + lens_height_m - h kamery anchora,
      dem_alignment  — dopasowanie DEM z ODM (odm_dtm / odm_dsm) do NMT:
                       z_offset = mediana(NMT - DEM_ODM) w punktach otwartego terenu.
      fixed          — z_offset = offset_m podany wprost (np. wynik zewnetrznego pomiaru
                       albo diagnostyki multiview_check w eksperymentach).
      start_point    — punkt startu (`start`): z_offset = NMT(start_xy) - z_grunt_ODM,
                       gdzie z_grunt_ODM = ground_alt_exif_m (0) + mediana
                       (h_ODM_kamery - alt_EXIF_kamery) po zarejestrowanych kamerach.
    Diagnostyka multiview_check (geodetect.zcal.multiview_z_scan) nie jest trybem kalibracji.
    """
    mode: str = "agl_assumption"
    agl_m: float = 45.0
    # ground_frame
    image: Optional[str] = None
    lens_height_m: float = 0.15
    # dem_alignment (None = odm_dir/odm_dem/dtm.tif i dsm.tif)
    odm_dtm: Optional[Path] = None
    odm_dsm: Optional[Path] = None
    n_samples: int = 300
    buffer_m: float = 10.0          # odsuniecie probek od krawedzi DEM ODM
    max_dsm_dtm_m: float = 0.5      # gdy sa oba DEM-y: tylko otwarty teren
    apply_tilt: bool = False        # stosowac takze dopasowana plaszczyzne (domyslnie tylko offset)
    seed: int = 0
    # fixed
    offset_m: Optional[float] = None
    fixed_label: str = "podany w configu"
    # start_point
    start: StartConfig = field(default_factory=StartConfig)


@dataclass
class EvalConfig:
    """Ocena detekcji wzgledem etykiet (ground truth).

    labels_dir: eksport Roboflow/YOLO (train/valid/test z images/ i labels/), None = brak GT.
    train_images: zdjecia, ktore byly w zbiorze TRENINGOWYM ocenianego modelu (wyciek
    danych) — raportowane osobno; leaky_models: modele (nazwy plikow), dla ktorych ta
    lista obowiazuje (trenowane na tym eksporcie). Dla pozostalych modeli wszystkie
    zdjecia z etykietami sa zbiorem uczciwym."""
    labels_dir: Optional[Path] = None
    train_images: list[str] = field(default_factory=list)
    leaky_models: list[str] = field(default_factory=list)
    iou: float = 0.5
    thresholds: list[float] = field(default_factory=lambda: [0.25, 0.5])


@dataclass
class PipelineConfig:
    photos_dir: Path = Path("zdjecia")
    odm_dir: Path = Path("odm_project")
    model_path: Path = Path("models/LudzieiNamioty.pt")
    output_dir: Path = Path("output")
    images: list[str] = field(default_factory=lambda: ["SING0316.JPG"])
    dem: DemConfig = field(default_factory=DemConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    raycast: RaycastConfig = field(default_factory=RaycastConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    z_anchor: ZAnchorConfig = field(default_factory=ZAnchorConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    point: PointConfig = field(default_factory=PointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ── pochodne sciezki ODM ──
    @property
    def reconstruction_path(self) -> Path:
        """Rekonstrukcja w topocentrycznym ENU wokol reference_lla. Nowsze ODM eksportuja
        reconstruction.json do ukladu UTM-z-offsetem (odm_georeferencing/coords.txt),
        a wersje topocentryczna zostawiaja w reconstruction.topocentric.json."""
        opensfm = self.odm_dir / "opensfm"
        topo = opensfm / "reconstruction.topocentric.json"
        return topo if topo.exists() else opensfm / "reconstruction.json"

    @property
    def odm_dtm_path(self) -> Path:
        return self.z_anchor.odm_dtm or self.odm_dir / "odm_dem" / "dtm.tif"

    @property
    def odm_dsm_path(self) -> Path:
        return self.z_anchor.odm_dsm or self.odm_dir / "odm_dem" / "dsm.tif"

    @property
    def reference_lla_path(self) -> Path:
        return self.odm_dir / "opensfm" / "reference_lla.json"

    @property
    def shots_geojson_path(self) -> Path:
        return self.odm_dir / "odm_report" / "shots.geojson"

    # ── wczytywanie / zapis ──
    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Optional[Path] = None) -> "PipelineConfig":
        cfg = _build(cls, data)
        if base_dir is not None:
            cfg._resolve_paths(Path(base_dir))
        return cfg

    @classmethod
    def from_file(cls, path: Path | str) -> "PipelineConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data, base_dir=path.resolve().parent)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def excluded_images(self) -> set[str]:
        """Zdjecia wylaczone z rekonstrukcji i przetwarzania (zdjecie startowe z ziemi)."""
        return {self.z_anchor.start.image} if self.z_anchor.start.image else set()

    def _resolve_paths(self, base: Path) -> None:
        def res(p: Optional[Path]) -> Optional[Path]:
            if p is None or p.is_absolute():
                return p
            return base / p
        self.photos_dir = res(self.photos_dir)
        self.odm_dir = res(self.odm_dir)
        self.model_path = res(self.model_path)
        self.output_dir = res(self.output_dir)
        self.dem.raster_path = res(self.dem.raster_path)
        self.dem.cache_path = res(self.dem.cache_path)
        self.z_anchor.odm_dtm = res(self.z_anchor.odm_dtm)
        self.z_anchor.odm_dsm = res(self.z_anchor.odm_dsm)
        self.surface.nmt_dir = res(self.surface.nmt_dir)
        self.surface.nmpt_dir = res(self.surface.nmpt_dir)
        self.surface.odm_dsm = res(self.surface.odm_dsm)
        self.point.mask_cache_dir = res(self.point.mask_cache_dir)
        self.point.sam_model = res(self.point.sam_model)
        self.eval.labels_dir = res(self.eval.labels_dir)
        self.detect.manual_csv = res(self.detect.manual_csv)
        self.detect.detections_csv = res(self.detect.detections_csv)


# Pola typu Path (takze Optional[Path]) — konwertowane ze stringow przy wczytywaniu.
_PATH_FIELDS = {"photos_dir", "odm_dir", "model_path", "output_dir", "raster_path", "cache_path",
                "odm_dtm", "odm_dsm", "nmt_dir", "nmpt_dir", "sam_model", "mask_cache_dir", "labels_dir",
                "manual_csv", "detections_csv"}
_NESTED = {"tiling": TilingConfig, "dem": DemConfig, "detect": DetectConfig, "raycast": RaycastConfig,
           "filters": FilterConfig, "aggregate": AggregateConfig, "z_anchor": ZAnchorConfig,
           "surface": SurfaceConfig, "point": PointConfig, "eval": EvalConfig,
           "start": StartConfig}


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Nieznane klucze w konfiguracji {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _NESTED and isinstance(value, dict):
            value = _build(_NESTED[key], value)
        elif key in _PATH_FIELDS and value is not None:
            value = Path(value)
        elif key == "images" and isinstance(value, str):
            value = [value]
        kwargs[key] = value
    return cls(**kwargs)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
