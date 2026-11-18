"""OpenDroneMap z poziomu programu (GUI / skrypty): przygotowanie projektu, Docker, uruchomienie.

prepare_odm      — kopiuje zdjecia do <odm_dir>/images z pominieciem zdjecia startowego z ziemi
                   (z_anchor.start.image); usuwa je, jesli zostalo z wczesniejszego kopiowania.
docker_status    — czy Docker dziala (klient + demon) i czy jest obraz ODM.
camera_calibration / write_cameras_json — kalibracja kamery z rekonstrukcji innego projektu
                   (ustalona kalibracja: --cameras --use-fixed-camera-params, docs §5.7).
odm_command      — lista argumentow `docker run ... opendronemap/odm`.
run_odm          — uruchomienie z logiem linia po linii (callback) i przerwaniem (cancel).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import PipelineConfig

log = logging.getLogger(__name__)

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ODM_IMAGE = "opendronemap/odm"
ODM_FLAGS = ("--dtm", "--dsm", "--pc-classify", "--dem-resolution", "2", "--orthophoto-resolution", "1")
CAMERAS_FILE = "cameras.json"


@dataclass
class PrepareResult:
    images_dir: Path
    copied: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.copied) + len(self.unchanged)


def prepare_odm(cfg: PipelineConfig, dry_run: bool = False) -> PrepareResult:
    excluded = cfg.excluded_images()
    dst = cfg.odm_dir / "images"
    res = PrepareResult(dst)
    if not cfg.photos_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu zdjec {cfg.photos_dir}")
    photos = sorted(p for p in cfg.photos_dir.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    for p in photos:
        if p.name in excluded:
            res.excluded.append(p.name)
            continue
        target = dst / p.name
        if target.exists() and target.stat().st_size == p.stat().st_size:
            res.unchanged.append(p.name)
            continue
        if not dry_run:
            shutil.copy2(p, target)
        res.copied.append(p.name)
    for name in sorted(excluded):
        if (dst / name).exists():
            if not dry_run:
                (dst / name).unlink()
            res.removed.append(name)
    missing = excluded - {p.name for p in photos}
    if missing:
        log.warning("Zdjecie startowe %s nie istnieje w %s", sorted(missing), cfg.photos_dir)
    log.info("ODM images: %s — skopiowane %d, bez zmian %d, wykluczone %s, usuniete %s",
             dst, len(res.copied), len(res.unchanged), res.excluded, res.removed)
    return res


# ───────────── Docker ─────────────

@dataclass
class DockerStatus:
    ok: bool                 # klient i demon odpowiadaja
    version: Optional[str]
    odm_image: bool          # obraz ODM jest lokalnie (inaczej pierwsze uruchomienie go pobierze)
    message: str


def docker_status(timeout_s: float = 15.0, runner=subprocess.run) -> DockerStatus:
    if shutil.which("docker") is None and runner is subprocess.run:
        return DockerStatus(False, None, False, "Nie znaleziono programu Docker, zainstaluj Docker Desktop.")
    try:
        r = runner(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True,
                   timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DockerStatus(False, None, False, f"Docker nie odpowiada ({exc}). Uruchom Docker Desktop.")
    if r.returncode != 0 or not r.stdout.strip():
        msg = (r.stderr or "").strip().splitlines()
        return DockerStatus(False, None, False, "Docker Desktop nie jest uruchomiony"
                            + (f" ({msg[-1]})" if msg else "") + ".")
    ver = r.stdout.strip()
    try:
        img = runner(["docker", "image", "inspect", ODM_IMAGE], capture_output=True, text=True, timeout=timeout_s)
        has = img.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        has = False
    return DockerStatus(True, ver, has, f"Docker {ver} działa" + ("" if has else
                        f"; obraz {ODM_IMAGE} zostanie pobrany przy pierwszym uruchomieniu (kilka GB)"))


# ───────────── kalibracja kamery z innego projektu ─────────────

def camera_calibration(odm_dir: Path) -> dict:
    """Slownik kamer (OpenSfM) z rekonstrukcji projektu ODM — do ustalonej kalibracji."""
    odm_dir = Path(odm_dir)
    for name in (CAMERAS_FILE,):                            # projekt, ktory sam mial ustalona kalibracje
        p = odm_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    for rel in ("opensfm/reconstruction.topocentric.json", "opensfm/reconstruction.json"):
        p = odm_dir / rel
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            cams = data[0]["cameras"] if isinstance(data, list) else data["cameras"]
            if cams:
                return cams
    raise FileNotFoundError(f"Brak rekonstrukcji z kalibracja kamery w {odm_dir}")


def write_cameras_json(src_odm_dir: Path, dst_odm_dir: Path) -> Path:
    cams = camera_calibration(src_odm_dir)
    dst = Path(dst_odm_dir) / CAMERAS_FILE
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cams, indent=2), encoding="utf-8")
    log.info("Kalibracja kamery z %s -> %s (%s)", src_odm_dir, dst, ", ".join(cams))
    return dst


# ───────────── komenda i uruchomienie ─────────────

def odm_command(odm_dir: Path, fixed_cameras: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    """docker run ... opendronemap/odm; odm_dir montowany jako /datasets/code
    (fixed_cameras: <odm_dir>/cameras.json + --use-fixed-camera-params)."""
    cmd = ["docker", "run", "--rm", "-v", f"{Path(odm_dir).resolve()}:/datasets/code", ODM_IMAGE,
           "--project-path", "/datasets", *ODM_FLAGS]
    if fixed_cameras:
        cmd += ["--cameras", f"/datasets/code/{CAMERAS_FILE}", "--use-fixed-camera-params"]
    return cmd + list(extra)


@dataclass
class OdmRunResult:
    returncode: Optional[int]
    cancelled: bool
    duration_s: float
    command: list[str]
    log_tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.cancelled


def run_odm(cmd: list[str], line_cb: Optional[Callable[[str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None, popen=subprocess.Popen,
            tail: int = 200) -> OdmRunResult:
    """Uruchamia komende (odm_command), przekazuje wyjscie linia po linii; cancel() -> kill."""
    t0 = time.perf_counter()
    lines: list[str] = []
    log.info("ODM: %s", " ".join(cmd))
    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                 errors="replace", bufsize=1)
    cancelled = False
    try:
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            del lines[:-tail]
            if line_cb is not None:
                line_cb(line)
            if cancel is not None and cancel():
                cancelled = True
                proc.kill()
                break
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if cancelled:
        log.warning("ODM przerwany na zadanie (kontener moze dzialac jeszcze chwile: docker ps)")
    return OdmRunResult(None if cancelled else rc, cancelled, time.perf_counter() - t0, list(cmd), lines)


def odm_outputs_ok(odm_dir: Path) -> dict[str, bool]:
    """Ktore wyniki ODM potrzebne pipeline'owi istnieja."""
    d = Path(odm_dir)
    rec = d / "opensfm" / "reconstruction.topocentric.json"
    return {"rekonstrukcja": rec.exists() or (d / "opensfm" / "reconstruction.json").exists(),
            "reference_lla": (d / "opensfm" / "reference_lla.json").exists(),
            "DSM": (d / "odm_dem" / "dsm.tif").exists(), "DTM": (d / "odm_dem" / "dtm.tif").exists(),
            "ortofotomapa": (d / "odm_orthophoto" / "odm_orthophoto.tif").exists()}
