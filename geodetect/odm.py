"""Parametry kamer (IO + EO) z rekonstrukcji OpenSfM/ODM i kalibracja Z.

Konwencje OpenSfM:
  - `rotation` w reconstruction.json to wektor angle-axis (Rodriguesa); macierz R
    przeksztalca *world -> camera*, pozycja kamery C = -R^T * t.
  - focal i principal point znormalizowane do max(w, h); (c_x, c_y) wzgledem srodka obrazu.
  - "world" = lokalny topocentric ENU wokol reference_lla.json. EO trzymamy w ENU,
    przeliczenie do EPSG:2180 robi geo.EnuFrame (scisle, z uwzglednieniem zbieznosci).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dem import DemSampler
from .geo import EnuFrame, GeoRef

log = logging.getLogger(__name__)


@dataclass
class CameraIO:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # [k1, k2, p1, p2, k3]
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]], dtype=np.float64)


@dataclass
class CameraEO:
    """Pozycja kamery i R world->camera w lokalnym ENU ODM (bez z_offset)."""
    C_enu: np.ndarray  # (3,) — E, N, U [m] wzgledem reference_lla
    R_wc: np.ndarray   # (3,3) — world(ENU) -> camera


@dataclass
class ZCalibration:
    """Korekta Z: H[m npm] = h_local + offset. `method` opisuje sposob wyznaczenia."""
    offset: float
    method: str
    dem_under: Optional[float] = None
    mean_alt_local: Optional[float] = None
    agl_assumed_m: Optional[float] = None


def intrinsics_to_pixels(cam_params: dict, w: int, h: int) -> CameraIO:
    """Model 'perspective' ma jedno pole `focal`, model 'brown' — `focal_x` i `focal_y`."""
    norm = max(w, h)
    if "focal_x" in cam_params:
        fx = float(cam_params["focal_x"]) * norm
        fy = float(cam_params["focal_y"]) * norm
    else:
        fx = fy = float(cam_params["focal"]) * norm
    cx = w / 2.0 + float(cam_params.get("c_x", 0.0)) * norm
    cy = h / 2.0 + float(cam_params.get("c_y", 0.0)) * norm
    dist = np.array([float(cam_params.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2", "k3")],
                    dtype=np.float64)
    return CameraIO(fx, fy, cx, cy, dist, w, h)


class OdmReconstruction:
    """Wczytana rekonstrukcja OpenSfM (pierwszy model z reconstruction.json)."""

    def __init__(self, reconstruction_path: Path, reference_lla_path: Path):
        if not reconstruction_path.exists():
            raise FileNotFoundError(
                f"Brak {reconstruction_path}. Najpierw przepusc zdjecia przez ODM (ODM.md).")
        if "topocentric" not in reconstruction_path.name:
            log.warning("Brak reconstruction.topocentric.json — zakladam, ze %s jest w ukladzie "
                        "topocentrycznym (starsze ODM). W nowszych ODM jest w UTM z offsetem!",
                        reconstruction_path.name)
        log.info("ODM: %s", reconstruction_path)
        self.path = reconstruction_path
        rec = json.loads(reconstruction_path.read_text(encoding="utf-8"))[0]
        self.cameras: dict = rec["cameras"]
        self.shots: dict = rec["shots"]

        ref = json.loads(reference_lla_path.read_text(encoding="utf-8"))
        self.ref = GeoRef(float(ref["latitude"]), float(ref["longitude"]),
                          float(ref.get("altitude", 0.0)))

    @property
    def image_names(self) -> list[str]:
        return sorted(self.shots)

    def camera(self, image_name: str) -> tuple[CameraIO, CameraEO]:
        if image_name not in self.shots:
            raise KeyError(f"{image_name} nie ma w reconstruction.json")
        shot = self.shots[image_name]
        cam = self.cameras[shot["camera"]]
        io = intrinsics_to_pixels(cam, int(cam["width"]), int(cam["height"]))

        rvec = np.array(shot["rotation"], dtype=np.float64)
        tvec = np.array(shot["translation"], dtype=np.float64)
        R_wc, _ = cv2.Rodrigues(rvec)
        C_enu = -R_wc.T @ tvec
        return io, CameraEO(C_enu=C_enu, R_wc=R_wc)


def compute_z_offset(shots_geojson: Path, dem: DemSampler, frame: EnuFrame,
                     agl_assumed_m: float) -> ZCalibration:
    """Korekta Z kompensujaca AGL-owy origin wysokosci Potensica/ODM.

    Zaklada, ze srednia kamera leciala `agl_assumed_m` nad terenem pod mediana
    pozycji kamer: offset = NMT(median_xy) + AGL - mean(alt_local).
    """
    if not shots_geojson.exists():
        log.warning("Brak %s — z_offset=0", shots_geojson)
        return ZCalibration(0.0, "none")
    feats = json.loads(shots_geojson.read_text(encoding="utf-8"))["features"]
    coords = np.array([f["geometry"]["coordinates"] for f in feats], dtype=np.float64)
    med_lon, med_lat = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))
    mean_alt_local = float(np.mean(coords[:, 2]))

    x, y = frame.lonlat_to_puwg92(med_lon, med_lat)
    nmt_under = dem.sample(x, y)
    if nmt_under is None:
        log.warning("DEM nie zwrocil wartosci dla median_xy=(%.1f, %.1f) — z_offset=0", x, y)
        return ZCalibration(0.0, "none", mean_alt_local=mean_alt_local)

    offset = nmt_under + agl_assumed_m - mean_alt_local
    log.info("Z: NMT(median_xy)=%.2f m npm, mean_alt_local=%.2f, AGL=%.1f -> z_offset=%.2f m",
             nmt_under, mean_alt_local, agl_assumed_m, offset)
    return ZCalibration(offset, "agl_assumed", nmt_under, mean_alt_local, agl_assumed_m)
