"""Inferencja kafelkowa: siatka kafli i scalanie detekcji z kafli (+ pelnego obrazu).

Wlasna implementacja zamiast SAHI: SAHI scala boxy z kafli zwyklym NMS/NMM i nie
rozroznia boxa ucietego na granicy kafla od pelnego, a to jest kluczowe dla
georeferencji (srodek ucietego boxa jest przesuniety). Tu:

1. Box dotykajacy WEWNETRZNEJ krawedzi kafla (krawedzi, ktora nie jest krawedzia
   zdjecia; tolerancja edge_margin_px) jest oznaczany jako uciety (tile_truncated).
2. Uciety box, ktory w >= contain_ios zawiera sie w nieucietym boxie tej samej klasy
   (ten sam obiekt widziany w calosci w sasiednim kaflu albo na pelnym obrazie), jest
   odrzucany.
3. Uciety box, ktorego otoczenie (box + jego rozmiar w kazda strone) inny kafel widzi
   w calosci, a ten kafel nie wykryl tam obiektu tej klasy, jest odrzucany jako
   niepotwierdzony (typowo waski "pasek" na krawedzi kafla).
4. Identyczne ciete kawalki (ten sam obiekt uciety ta sama krawedzia w kaflach
   sasiadujacych w drugiej osi) sa deduplikowane NMS-em; dopiero kawalki uzupelniajace
   sie (IoU < nms_iou, nakladaja sie, wyrownane w osi prostopadlej) sa laczone w jeden
   box (suma zasiegow) — obiekt wiekszy niz zakladka, uciety w obu kaflach.
5. Na koncu NMS per klasa: tlumienie przy IoU >= nms_iou albo gdy mniejszy box zawiera
   sie w wiekszym (IoS >= contain_ios) — duplikaty z kafla i pelnego obrazu.
Box, ktory zostal ucietym kawalkiem bez pary, zostaje z flaga tile_truncated (log).
Flaga touches_border (pipeline) dotyczy krawedzi ZDJECIA i jest liczona z bboxa
w ukladzie calego obrazu — kafle jej nie dotycza.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int


def _starts(size: int, tile: int, overlap: float) -> list[int]:
    if tile >= size:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, size - tile, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[Tile]:
    """Kafle tile x tile (przyciete do obrazu) z zakladka >= overlap; ostatni kafel
    w wierszu/kolumnie dosuniety do krawedzi zdjecia."""
    tw, th = min(tile, width), min(tile, height)
    return [Tile(x, y, x + tw, y + th)
            for y in _starts(height, th, overlap) for x in _starts(width, tw, overlap)]


def touches_interior_edge(bbox, tile: Tile, width: int, height: int, margin: float) -> bool:
    """Box (w ukladzie zdjecia) dotyka krawedzi kafla, ktora nie jest krawedzia zdjecia."""
    x1, y1, x2, y2 = bbox
    return ((tile.x0 > 0 and x1 <= tile.x0 + margin)
            or (tile.x1 < width and x2 >= tile.x1 - margin)
            or (tile.y0 > 0 and y1 <= tile.y0 + margin)
            or (tile.y1 < height and y2 >= tile.y1 - margin))


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def ios(a, b) -> float:
    """Przeciecie / pole mniejszego boxa (1 = mniejszy w calosci w wiekszym)."""
    s = min(_area(a), _area(b))
    return _inter(a, b) / s if s > 0 else 0.0


def _iou_1d(a0, a1, b0, b1) -> float:
    i = max(0.0, min(a1, b1) - max(a0, b0))
    u = max(a1, b1) - min(a0, b0)
    return i / u if u > 0 else 0.0


def _pieces_of_same_object(a, b) -> bool:
    """Dwa ciete kawalki: nakladaja sie i sa wyrownane w osi prostopadlej do ciecia."""
    if _inter(a, b) <= 0:
        return False
    return _iou_1d(a[0], a[2], b[0], b[2]) > 0.5 or _iou_1d(a[1], a[3], b[1], b[3]) > 0.5


def _seen_whole_by(b, t: Tile) -> bool:
    """Kafel t widzi box b razem z otoczeniem (rozmiar boxa w kazda strone)."""
    w, h = b[2] - b[0], b[3] - b[1]
    return (t.x0 <= b[0] - w and b[2] + w <= t.x1 and t.y0 <= b[1] - h and b[3] + h <= t.y1)


def _nms(dets: list, nms_iou: float) -> list:
    keep = []
    for d in sorted(dets, key=lambda d: -d.conf):
        if all(iou(d.bbox, k.bbox) < nms_iou for k in keep):
            keep.append(d)
    return keep


def merge_detections(dets: list, nms_iou: float = 0.5, contain_ios: float = 0.6,
                     tiles: Optional[list[Tile]] = None) -> list:
    """Scalanie detekcji z kafli i pelnego obrazu (lista Detection, bbox w ukladzie zdjecia,
    pola tile_truncated i tile_index ustawione przez wywolujacego). `tiles` (lista kafli
    wg tile_index) wlacza regule 3. Zwraca nowa liste Detection."""
    out = []
    for cls in sorted({d.cls_name for d in dets}):
        group = [d for d in dets if d.cls_name == cls]
        whole = [d for d in group if not d.tile_truncated]
        cut = [d for d in group
               if d.tile_truncated and not any(ios(d.bbox, w.bbox) >= contain_ios for w in whole)]
        if tiles is not None:
            kept = []
            for d in cut:
                witness = [i for i, t in enumerate(tiles)
                           if i != d.tile_index and _seen_whole_by(d.bbox, t)]
                confirmed = any(o.tile_index in witness and _inter(o.bbox, d.bbox) > 0
                                for o in group if o is not d)
                if witness and not confirmed:
                    log.debug("Kafle: odrzucam uciety %s %s — kafel %s widzi go w calosci i "
                              "nic tam nie wykryl", cls, [round(v) for v in d.bbox], witness)
                    continue
                kept.append(d)
            cut = kept
        cut = _nms(cut, nms_iou)
        # laczenie cietych kawalkow (union-find)
        parent = list(range(len(cut)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(cut)):
            for j in range(i + 1, len(cut)):
                if _pieces_of_same_object(cut[i].bbox, cut[j].bbox):
                    parent[find(i)] = find(j)
        groups: dict[int, list] = {}
        for i, d in enumerate(cut):
            groups.setdefault(find(i), []).append(d)
        merged = []
        for members in groups.values():
            if len(members) == 1:
                log.info("Kafle: %s %s uciety na granicy kafla bez pary — zostaje z flaga",
                         cls, [round(v) for v in members[0].bbox])
                merged.append(members[0])
                continue
            b = np.array([m.bbox for m in members])
            best = max(members, key=lambda m: m.conf)
            polys = [p for m in members for p in (m.polygons or [])] or None
            merged.append(replace(best, bbox=(float(b[:, 0].min()), float(b[:, 1].min()),
                                              float(b[:, 2].max()), float(b[:, 3].max())),
                                  polygons=polys, source="merged", tile_truncated=False,
                                  tile_index=None))
        # NMS per klasa (IoU albo zawieranie), najwyzszy conf wygrywa
        cand = sorted(whole + merged, key=lambda d: -d.conf)
        keep = []
        for d in cand:
            if all(iou(d.bbox, k.bbox) < nms_iou and ios(d.bbox, k.bbox) < contain_ios for k in keep):
                keep.append(d)
        out.extend(keep)
    return sorted(out, key=lambda d: (d.bbox[1], d.bbox[0]))


def offset_polygons(polys: Optional[list], dx: float, dy: float) -> Optional[list]:
    if polys is None:
        return None
    return [np.asarray(p, dtype=np.float64) + [dx, dy] for p in polys]
