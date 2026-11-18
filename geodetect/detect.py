"""Detekcja obiektow modelem YOLO (ultralytics): model ladowany raz, klasy po nazwie.

- Nazwy klas zawsze z model.names; konfiguracja (classes, conf_per_class, display_names,
  colors, point.per_class) odwoluje sie do nazw — nieznana nazwa -> ValueError z lista
  dostepnych klas.
- imgsz domyslnie z treningu modelu (train_args.imgsz w checkpoincie); efektywna wartosc
  jest logowana.
- Model segmentacyjny (task=segment): kontury masek trafiaja do Detection.polygons
  i zastepuja SAM w metodach punktu mask_* (geodetect.points).
- Inferencja kafelkowa: cfg.tiling (geodetect/tiling.py).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import DetectConfig

log = logging.getLogger(__name__)

MODEL_SUFFIXES = (".pt",)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 [px] w ukladzie CALEGO zdjecia
    polygons: Optional[list[np.ndarray]] = None   # kontury maski (model segmentacyjny)
    source: str = "full"             # full | tile | merged (inferencja kafelkowa)
    tile_truncated: bool = False     # box uciety na wewnetrznej krawedzi kafla (bez pary)
    tile_index: Optional[int] = None  # indeks kafla (None = pelny obraz / scalony)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ───────────── modele ─────────────

@dataclass
class ModelInfo:
    path: Path
    name: str
    task: str                        # detect | segment | ...
    names: dict[int, str]
    imgsz: Optional[int]             # z treningu (train_args.imgsz)
    size_mb: float
    date: Optional[str] = None
    train_data: Optional[str] = None
    open_vocab: bool = False          # YOLOE / YOLO-World: klasy z promptu (detect.prompt_text)

    @property
    def label(self) -> str:
        """Etykieta do listy rozwijanej w GUI."""
        cls = ", ".join(self.names.values())
        return f"{self.name} ({self.task}, imgsz {self.imgsz}, klasy: {cls})"


def _imgsz_value(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return int(max(v))
    return int(v)


def model_info(path: Path | str) -> Optional[ModelInfo]:
    """Metadane checkpointu ultralytics bez ladowania na GPU; None, gdy to nie jest
    model YOLO (np. wagi SAM)."""
    import torch
    path = Path(path)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("Nie moge wczytac %s: %s", path, exc)
        return None
    if not isinstance(ck, dict) or "train_args" not in ck:
        return None
    model = ck.get("ema") or ck.get("model")
    names = getattr(model, "names", None)
    if not names:
        return None
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    args = ck.get("train_args") or {}
    task = args.get("task") or getattr(model, "task", None) or "detect"
    return ModelInfo(path, path.name, str(task), {int(k): str(v) for k, v in names.items()},
                     _imgsz_value(args.get("imgsz")), path.stat().st_size / 2 ** 20,
                     ck.get("date"), args.get("data"), is_open_vocab_model(model))


def is_open_vocab_model(model) -> bool:
    """Model otwartego slownika (YOLOE, YOLO-World) — po nazwie klasy modelu ultralytics."""
    name = type(model).__name__
    return "YOLOE" in name or "World" in name


def prompt_class_names(cfg: DetectConfig) -> Optional[list[str]]:
    """Klasy modelu otwartego slownika wg promptu (wizualny ma pierwszenstwo), None = bez promptu."""
    if cfg.uses_visual_prompt():
        names = list(cfg.visual_prompt_classes or [])
        n = len(cfg.visual_prompt_boxes)
        if len(names) != n:
            names = (names or ["obiekt"])[:1] * n
        return list(dict.fromkeys(names))
    if cfg.prompt_text:
        return list(cfg.prompt_text)
    return None


def list_models(folder: Path | str) -> list[ModelInfo]:
    """Modele YOLO w folderze (pliki .pt), posortowane po nazwie; inne wagi pomijane."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in MODEL_SUFFIXES:
            info = model_info(p)
            if info is not None:
                out.append(info)
    return out


def validate_class_names(names: Iterable[str], available: Iterable[str], what: str) -> None:
    available = list(available)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"{what}: klas {missing} nie ma w modelu (dostepne klasy: {available})")


def top_k_per_class(dets: list[Detection], k: Optional[int]) -> list[Detection]:
    """k najpewniejszych detekcji kazdej klasy (None = bez ograniczenia)."""
    if k is None:
        return dets
    out, seen = [], {}
    for d in sorted(dets, key=lambda d: -d.conf):
        if seen.get(d.cls_name, 0) < k:
            out.append(d)
            seen[d.cls_name] = seen.get(d.cls_name, 0) + 1
    return out


# ───────────── detektor ─────────────

class Detector:
    def __init__(self, model_path: Path, cfg: Optional[DetectConfig] = None):
        from ultralytics import YOLO  # import lazy — ciezki (torch)
        self.cfg = cfg or DetectConfig()
        self.model_path = Path(model_path)
        self.model = YOLO(str(model_path))
        self.open_vocab = is_open_vocab_model(getattr(self.model, "model", None))
        self._vp = None
        if self.open_vocab:
            self._setup_prompt()
        self.names: dict[int, str] = {int(k): v for k, v in dict(self.model.names).items()}
        self.task = self.model.task
        c = self.cfg
        validate_class_names(c.class_list() or [], self.names.values(), "detect.classes")
        validate_class_names(c.conf_per_class, self.names.values(), "detect.conf_per_class")
        validate_class_names(c.display_names, self.names.values(), "detect.display_names")
        validate_class_names(c.colors, self.names.values(), "detect.colors")
        validate_class_names(c.tiling.classes or [], self.names.values(), "detect.tiling.classes")
        by_name = {v: k for k, v in self.names.items()}
        self.class_ids: Optional[list[int]] = (
            None if c.class_list() is None else [by_name[n] for n in c.class_list()])
        train_imgsz = _imgsz_value((getattr(self.model, "ckpt", None) or {}).get("train_args", {})
                                   .get("imgsz")) or _imgsz_value(self.model.overrides.get("imgsz"))
        self.imgsz = int(c.imgsz or train_imgsz or 640)
        self.infer_s = 0.0
        self.n_images = 0
        log.info("YOLO: %s (task=%s), klasy=%s, imgsz=%d (%s), conf=%.2f%s, iou=%.2f, "
                 "device=%s, half=%s, kafle=%s", self.model_path.name, self.task, self.names,
                 self.imgsz, "z configu" if c.imgsz else "z treningu modelu", c.conf,
                 f" per klasa {c.conf_per_class}" if c.conf_per_class else "", c.iou,
                 c.device or "auto", c.half, c.tiling.enabled)

    def _setup_prompt(self) -> None:
        """YOLOE / YOLO-World: klasy z promptu tekstowego albo wizualnego (cfg.detect)."""
        c = self.cfg
        names = prompt_class_names(c)
        if names is None:
            log.warning("%s: model otwartego slownika bez promptu — uzywa domyslnych klas",
                        self.model_path.name)
            return
        if c.uses_visual_prompt():
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor, YOLOEVPSegPredictor
            self.model = YOLOE(str(self.model_path))
            per_box = list(c.visual_prompt_classes or [])
            if len(per_box) != len(c.visual_prompt_boxes):
                per_box = [names[0]] * len(c.visual_prompt_boxes)
            seg = self.model.task == "segment"
            self._vp = {"refer_image": str(c.visual_prompt_image),
                        "visual_prompts": {"bboxes": np.asarray(c.visual_prompt_boxes, float),
                                           "cls": np.array([names.index(n) for n in per_box])},
                        "predictor": YOLOEVPSegPredictor if seg else YOLOEVPDetectPredictor}
            log.info("YOLOE prompt wizualny: %s, %d box(y), klasy %s", Path(c.visual_prompt_image).name,
                     len(per_box), names)
            # nazwy klas z promptu wizualnego ustawiane przez predyktor; ustaw je jawnie dla walidacji
            self.model.model.names = dict(enumerate(names))
            return
        try:
            self.model.set_classes(names, self.model.get_text_pe(names))
        except (TypeError, AttributeError):
            self.model.set_classes(names)
        log.info("Model otwartego slownika: prompt tekstowy %s", names)

    # -- surowe wywolanie modelu --
    def _predict(self, sources: list, conf: float) -> list:
        c = self.cfg
        out = []
        t0 = time.perf_counter()
        step = 1 if self._vp else max(1, c.batch)       # prompt wizualny: zdjecie po zdjeciu
        for i in range(0, len(sources), step):
            out.extend(self.model.predict(
                sources[i:i + step], imgsz=self.imgsz, conf=conf, iou=c.iou,
                classes=self.class_ids, device=c.torch_device(), half=c.half,
                max_det=c.max_det, verbose=False, **(self._vp or {})))
        self.infer_s += time.perf_counter() - t0
        return out

    def _min_conf(self) -> float:
        return min([self.cfg.conf, *self.cfg.conf_per_class.values()])

    def _to_detections(self, result, dx: float = 0.0, dy: float = 0.0) -> list[Detection]:
        from .tiling import offset_polygons
        out = []
        polys = None
        if getattr(result, "masks", None) is not None:
            polys = [np.asarray(p, dtype=np.float64) for p in result.masks.xy]
        for k, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            name = self.names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            if conf < self.cfg.conf_for(name):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            p = [polys[k]] if polys is not None and k < len(polys) and len(polys[k]) >= 3 else None
            out.append(Detection(cls_id, name, conf, (x1 + dx, y1 + dy, x2 + dx, y2 + dy),
                                 offset_polygons(p, dx, dy)))
        return out

    # -- API --
    def detect(self, image_path: Path) -> list[Detection]:
        return self.detect_many([image_path])[0]

    def detect_many(self, image_paths: list[Path]) -> list[list[Detection]]:
        """Detekcje dla wielu zdjec; bez kafli zdjecia ida do modelu paczkami po cfg.batch."""
        self.n_images += len(image_paths)
        if self.cfg.tiling.enabled:
            out = [self._detect_tiled(Path(p)) for p in image_paths]
        else:
            res = self._predict([str(p) for p in image_paths], self._min_conf())
            out = [self._to_detections(r) for r in res]
        return [top_k_per_class(d, self.cfg.top_k_per_class) for d in out]

    def _detect_tiled(self, image_path: Path) -> list[Detection]:
        import cv2
        from dataclasses import replace
        from .tiling import merge_detections, tile_grid, touches_interior_edge
        tc = self.cfg.tiling
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(image_path)
        H, W = img.shape[:2]
        tiles = tile_grid(W, H, int(tc.tile_size or self.imgsz), tc.overlap)
        crops = [np.ascontiguousarray(img[t.y0:t.y1, t.x0:t.x1]) for t in tiles]
        dets = []
        tile_cls = set(tc.classes) if tc.classes else None
        for i, (t, r) in enumerate(zip(tiles, self._predict(crops, self._min_conf()))):
            for d in self._to_detections(r, t.x0, t.y0):
                if tile_cls is not None and d.cls_name not in tile_cls:
                    continue
                dets.append(replace(d, source="tile", tile_index=i,
                                    tile_truncated=touches_interior_edge(d.bbox, t, W, H,
                                                                         tc.edge_margin_px)))
        if tc.include_full or tile_cls is not None:
            full = self._to_detections(self._predict([img], self._min_conf())[0])
            if not tc.include_full:      # pelny obraz tylko dla klas spoza tiling.classes
                full = [d for d in full if d.cls_name not in tile_cls]
            dets += full
        merged = merge_detections(dets, tc.nms_iou, tc.contain_ios, tiles)
        log.debug("%s: %d kafli, %d detekcji surowych -> %d po scaleniu", image_path.name,
                  len(tiles), len(dets), len(merged))
        return merged


# ───────────── punkty reczne ─────────────

@dataclass
class ManualPoint:
    image: str
    u: Optional[float]
    v: Optional[float]
    visible: bool
    conf: Optional[float] = None
    box: Optional[tuple[float, float, float, float]] = None   # x1,y1,x2,y2 (CSV detekcji)


def read_manual_points(path: Path) -> dict[str, ManualPoint]:
    """CSV image,u,v,visible[,conf][,x1,y1,x2,y2]. Puste u/v albo visible=0 -> brak punktu."""
    import csv

    def num(row, k):
        t = (row.get(k) or "").strip()
        return float(t) if t else None

    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("image") or "").strip()
            if not name:
                continue
            u, v = num(row, "u"), num(row, "v")
            vis = (row.get("visible") or "").strip().lower() in ("1", "true", "tak", "yes", "t")
            box = tuple(num(row, k) for k in ("x1", "y1", "x2", "y2"))
            out[name] = ManualPoint(name, u, v, vis and u is not None and v is not None,
                                    num(row, "conf"), box if None not in box else None)
    return out


class ManualDetector:
    """Interfejs Detector: "detekcje" z pliku CSV zamiast inferencji.

    source=manual: recznie wskazane piksele (cfg.manual_csv), conf 1, bbox manual_box_px;
    source=csv:    zapisane wczesniej detekcje innego detektora (cfg.detections_csv, np. YOLOE
                   z promptem wizualnym z scripts/pilka_detectors.py), conf i bbox z pliku."""

    task = "manual"

    def __init__(self, cfg: DetectConfig, source: str = "manual"):
        path = cfg.manual_csv if source == "manual" else cfg.detections_csv
        if path is None:
            raise ValueError(f"detect.source={source} wymaga detect."
                             f"{'manual_csv' if source == 'manual' else 'detections_csv'}")
        self.cfg, self.source = cfg, source
        self.points = read_manual_points(Path(path))
        self.names = {0: cfg.manual_class}
        self.infer_s, self.n_images = 0.0, 0
        log.info("Detekcje z CSV (%s): %s (%d widocznych z %d)", source, path,
                 sum(p.visible for p in self.points.values()), len(self.points))

    def detect(self, image_path) -> list[Detection]:
        self.n_images += 1
        p = self.points.get(Path(image_path).name)
        if p is None or not p.visible:
            return []
        h = self.cfg.manual_box_px / 2
        box = p.box if self.source == "csv" and p.box else (p.u - h, p.v - h, p.u + h, p.v + h)
        conf = p.conf if self.source == "csv" and p.conf is not None else 1.0
        return [Detection(0, self.cfg.manual_class, conf, box, source=self.source)]

    def detect_many(self, image_paths) -> list[list[Detection]]:
        return [self.detect(p) for p in image_paths]


def make_detector(model_path: Path, cfg: DetectConfig):
    """Detector (YOLO) albo ManualDetector wg cfg.source."""
    if cfg.source in ("manual", "csv"):
        return ManualDetector(cfg, cfg.source)
    if cfg.source != "model":
        raise ValueError(f"Nieznane detect.source={cfg.source!r} (model | manual | csv)")
    return Detector(model_path, cfg)
