"""Etykiety YOLO z eksportu Roboflow jako ground truth dla zdjec z lotu.

Eksport Roboflow (np. namioty_nasze v7) zawiera zdjecia przeskalowane do 1024x1024
z czarnymi pasami (letterbox "Fit (black edges)"). W splicie train kazde zdjecie ma
3 wersje z augmentacja (m.in. obrot +-6 deg), valid/test sa bez augmentacji.
Dla wersji z augmentacja odtwarzamy przeksztalcenie wzgledem oryginalu (ORB + RANSAC,
podobienstwo) i wybieramy wersje o najmniejszym obrocie. Box po obrocie Roboflow to
prostokat opisany na obroconym boxie: W' = w|c| + h|s|, H' = w|s| + h|c| — z tego
odzyskujemy w, h (dokladnie, o ile Roboflow nie przycial boxa do kadru).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


@dataclass
class GtBox:
    cls_name: str
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 w pikselach oryginalu


@dataclass
class GtImage:
    image_name: str
    boxes: list[GtBox]
    split: str
    source_label: Path
    rotation_deg: float          # obrot wybranej wersji (0 dla valid/test)


def letterbox_params(width: int, height: int, size: int) -> tuple[float, float, float]:
    """(skala, pad_x, pad_y) dla dopasowania width x height do size x size z pasami."""
    s = size / max(width, height)
    return s, (size - width * s) / 2.0, (size - height * s) / 2.0


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s, px, py = letterbox_params(w, h, size)
    out = np.zeros((size, size, 3), img.dtype)
    r = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = int(round(py)), int(round(px))
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """Macierz 2x3 (podobienstwo) mapujaca src -> dst; None, gdy za malo dopasowan."""
    orb = cv2.ORB_create(4000)
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 20:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl is None or inl.sum() < 15:
        return None
    return M


def _unrotate_box(cx, cy, W, H, M_inv: np.ndarray, angle_rad: float):
    """Box opisany po obrocie -> box w ukladzie przed obrotem (srodek + rozmiar)."""
    c = np.array([cx, cy, 1.0])
    x, y = M_inv @ c
    co, si = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    scale = math.hypot(M_inv[0, 0], M_inv[1, 0])
    det = co * co - si * si
    if abs(det) < 1e-3:
        w, h = W, H
    else:
        w = (co * W - si * H) / det
        h = (co * H - si * W) / det
        if w <= 0 or h <= 0:        # niespojne (np. box przyciety) — zostaw opisany
            w, h = W, H
    w, h = w * scale, h * scale
    return x, y, w, h


def load_ground_truth(dataset_dir: Path | str, image_names: list[str], photos_dir: Path,
                      class_names: list[str]) -> dict[str, GtImage]:
    """Ground truth dla zdjec z listy (po nazwie pliku, np. SING0316.JPG)."""
    dataset_dir = Path(dataset_dir)
    out: dict[str, GtImage] = {}
    for name in image_names:
        stem = re.escape(Path(name).stem + "_" + Path(name).suffix.lstrip("."))
        cands = []
        for split in SPLITS:
            lab_dir = dataset_dir / split / "labels"
            if lab_dir.is_dir():
                cands += [(split, p) for p in sorted(lab_dir.iterdir())
                          if re.match(stem + r"\.rf\.", p.name)]
        if not cands:
            continue
        orig = cv2.imread(str(photos_dir / name))
        if orig is None:
            continue
        H0, W0 = orig.shape[:2]
        best = None
        for split, lab in cands:
            img_path = next((lab.parent.parent / "images" / (lab.stem + ext)
                             for ext in (".jpg", ".jpeg", ".png")
                             if (lab.parent.parent / "images" / (lab.stem + ext)).exists()), None)
            if img_path is None:
                continue
            rf = cv2.imread(str(img_path))
            size = rf.shape[0]
            if split == "train":
                M = estimate_similarity(rf, _letterbox(orig, size))
                if M is None:
                    continue
                angle = math.atan2(M[1, 0], M[0, 0])
            else:
                M, angle = np.array([[1.0, 0, 0], [0, 1.0, 0]]), 0.0
            if best is None or abs(angle) < abs(best[3]):
                best = (split, lab, size, angle, M)
        if best is None:
            log.warning("GT %s: nie udalo sie dopasowac zadnej wersji", name)
            continue
        split, lab, size, angle, M = best
        s, px, py = letterbox_params(W0, H0, size)
        boxes = []
        for line in lab.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue      # pomijamy poligony / puste linie
            k, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
            cx, cy, w, h = cx * size, cy * size, w * size, h * size
            x, y, w, h = _unrotate_box(cx, cy, w, h, cv2.invertAffineTransform(M), angle)
            x, y, w, h = (x - px) / s, (y - py) / s, w / s, h / s
            boxes.append(GtBox(class_names[k], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        out[name] = GtImage(name, boxes, split, lab, math.degrees(angle))
    return out


def match_detections(dets: list, gts: list[GtBox], iou_thr: float = 0.5):
    """Zachlanne dopasowanie (po malejacym conf) w obrebie klasy.
    Zwraca (lista (det, gt|None, iou), lista niedopasowanych gt)."""
    from .tiling import iou
    used = set()
    pairs = []
    for d in sorted(dets, key=lambda d: -d.conf):
        best, best_iou = None, iou_thr
        for j, g in enumerate(gts):
            if j in used or g.cls_name != d.cls_name:
                continue
            v = iou(d.bbox, g.bbox)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            pairs.append((d, gts[best], best_iou))
        else:
            pairs.append((d, None, 0.0))
    return pairs, [g for j, g in enumerate(gts) if j not in used]
