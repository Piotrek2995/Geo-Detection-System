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
class DetectConfig:
    """Parametry detekcji YOLO. `classes` = lista nazw klas z model.names
    (None = wszystkie klasy)."""
    conf: float = 0.25
    classes: Optional[list[str]] = None
    device: Optional[str] = None  # None = automatycznie (GPU jesli dostepne)


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


Z_ANCHOR_MODES = ("agl_assumption", "ground_frame", "dem_alignment")


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


@dataclass
class PipelineConfig:
    photos_dir: Path = Path("zdjecia")
    odm_dir: Path = Path("odm_project")
    model_path: Path = Path("LudzieiNamioty.pt")
    output_dir: Path = Path("output")
    images: list[str] = field(default_factory=lambda: ["SING0316.JPG"])
    dem: DemConfig = field(default_factory=DemConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    raycast: RaycastConfig = field(default_factory=RaycastConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    z_anchor: ZAnchorConfig = field(default_factory=ZAnchorConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)

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


# Pola typu Path (takze Optional[Path]) — konwertowane ze stringow przy wczytywaniu.
_PATH_FIELDS = {"photos_dir", "odm_dir", "model_path", "output_dir", "raster_path", "cache_path",
                "odm_dtm", "odm_dsm", "nmt_dir", "nmpt_dir"}
_NESTED = {"dem": DemConfig, "detect": DetectConfig, "raycast": RaycastConfig,
           "filters": FilterConfig, "aggregate": AggregateConfig, "z_anchor": ZAnchorConfig,
           "surface": SurfaceConfig}


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
