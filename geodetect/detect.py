"""Detekcja obiektow modelem YOLO (ultralytics). Model ladowany raz w __init__."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 [px]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Detector:
    def __init__(self, model_path: Path, conf: float = 0.25,
                 classes: Optional[list[str]] = None, device: Optional[str] = None):
        from ultralytics import YOLO  # import lazy — ciezki (torch)
        self.model = YOLO(str(model_path))
        self.names: dict[int, str] = dict(self.model.names)
        self.conf = conf
        self.device = device
        self.class_ids: Optional[list[int]] = None
        if classes:
            by_name = {v: k for k, v in self.names.items()}
            missing = [c for c in classes if c not in by_name]
            if missing:
                raise ValueError(f"Klas {missing} nie ma w modelu (dostepne: {list(by_name)})")
            self.class_ids = [by_name[c] for c in classes]
        log.info("YOLO: %s, klasy=%s", model_path, self.names)

    def detect(self, image_path: Path) -> list[Detection]:
        result = self.model(str(image_path), conf=self.conf, classes=self.class_ids,
                            device=self.device, verbose=False)[0]
        out = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append(Detection(cls_id, self.names.get(cls_id, str(cls_id)),
                                 float(box.conf.item()), (x1, y1, x2, y2)))
        return out
