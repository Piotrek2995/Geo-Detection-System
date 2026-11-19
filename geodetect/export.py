"""Eksport wynikow przebiegu.

export_all(batch, dir)  — zestaw podstawowy (GeoJSON, CSV, mapa, wykres rozrzutu); uzywany tez
                          przez eksperymenty (per wariant).
export_run(batch, dir)  — pelny eksport jednego przebiegu (geodetect/run.py): dodatkowo GeoPackage
                          (warstwy objects, detections, cameras, rays), KML, CSV detekcji, miniatury
                          obiektow, run_report.json i run_report.html.

Wspolrzedne: WGS84 (lon, lat) i EPSG:2180 (x_2180, y_2180), wysokosc h_npm [m n.p.m.].
GeoJSON w WGS84 (RFC 7946, [lon, lat, h]); GPKG w EPSG:2180 (POINT Z / LINESTRING Z); KML w WGS84
(punkty przyklejone do terenu, wysokosc npm w atrybutach).
"""

from __future__ import annotations

import base64
import csv
import dataclasses
import html
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .aggregate import AggregatedObject
from .batch import BatchResult
from .pipeline import GeoDetection

log = logging.getLogger(__name__)

# Kolory klas: sloty 1-2 palety kategorycznej (walidowane pod CVD); kolejne klasy -> dalsze sloty.
CLASS_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
CLASS_MARKERS = ["o", "^", "s", "D", "v", "P", "X"]
CAMERA_COLOR = "#52514e"
REJECTED_COLOR = "#8a8984"
SINGLE_VIEW_COLOR = "#8a8984"
FLAG_COLOR = "#d03b3b"

THUMB_PX = 256           # miniatura obiektu w raporcie (dluzszy bok)
POPUP_THUMB_PX = 160     # miniatura w dymku mapy (base64 w HTML)


def class_colors(batch: BatchResult) -> dict[str, str]:
    """Kolor klasy: detect.colors z configu, a dla pozostalych kolejne sloty palety."""
    names = sorted({gd.detection.cls_name for gd in batch.detections}
                   | {o.cls_name for o in batch.objects})
    custom = batch.config.detect.colors if batch.config is not None else {}
    free = iter([c for c in CLASS_PALETTE if c not in custom.values()] * 3)
    return {n: custom[n] if n in custom else next(free) for n in names}


def class_label(batch: BatchResult, cls_name: str) -> str:
    """Nazwa wyswietlana klasy (detect.display_names), domyslnie nazwa z modelu."""
    return batch.config.detect.display(cls_name) if batch.config is not None else cls_name


def _clean(v):
    """NaN/inf -> None, numpy -> python (poprawny JSON)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, Path):
        return str(v)
    return v


def _write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("-> %s", path)
    return path


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


# ───────────── informacje o przebiegu ─────────────

def georef_mode(batch: BatchResult) -> str:
    """Opis georeferencji: tryb georef + kalibracja Z (z_anchor)."""
    cfg, zc = batch.config, batch.z_calibration
    mode = cfg.georef.mode if cfg is not None else "?"
    if mode == "similarity_nmpt":
        return "similarity_nmpt (7p do NMPT" + ("" if cfg.georef.use_baro_prior else ", bez barometru") + ")"
    return f"{mode} + z_anchor {zc.method if zc else '?'}"


def ray_surface(batch: BatchResult) -> str:
    s = sorted({gd.hit.surface for gd in batch.detections if gd.hit.surface})
    if s:
        return ",".join(s)
    return batch.config.surface.source if batch.config is not None else ""


def qc_status(batch: BatchResult) -> Optional[str]:
    return batch.qc.status if batch.qc is not None else None


# ───────────── atrybuty ─────────────

def detection_properties(gd: GeoDetection) -> dict:
    h = gd.hit
    sz = gd.size
    return {
        "image": gd.image_name,
        "class": gd.detection.cls_name,
        "conf": round(gd.detection.conf, 3),
        "det_source": gd.detection.source,
        "tile_truncated": gd.detection.tile_truncated,
        "object_id": gd.object_id,
        "accepted": gd.accepted,
        "rejected_reason": gd.rejected_reason,
        "altitude_m": round(h.h, 2),
        "surface": h.surface,
        "surface_h": None if h.surface_h != h.surface_h else round(h.surface_h, 2),
        "ground_h": None if h.ground_h is None else round(h.ground_h, 2),
        "above_ground_m": None if h.above_ground_m is None else round(h.above_ground_m, 2),
        "x_2180": round(h.x_2180, 3),
        "y_2180": round(h.y_2180, 3),
        "range_m": round(h.range_m, 2),
        "incidence_deg": round(gd.incidence_deg, 2),
        "size_m": _r(sz.size_m) if sz else None,
        "size_simple_m": _r(sz.simple_m) if sz else None,
        "size_method": sz.method if sz else None,
        "touches_border": gd.touches_border,
        "point_method": gd.point.method if gd.point else "bbox_center",
        "point_used": gd.point.used if gd.point else "bbox_center",
        "point_fallback": gd.point.fallback if gd.point else None,
        "mask_area_ratio": (round(gd.point.mask_quality.area_ratio, 3)
                            if gd.point and gd.point.mask_quality else None),
        "n_rays": int(len(gd.ray_uvs)),
        "pixel_u": round(gd.uv[0], 1),
        "pixel_v": round(gd.uv[1], 1),
        "bbox": [round(v, 1) for v in gd.detection.bbox],
    }


def object_properties(o: AggregatedObject, batch: Optional[BatchResult] = None) -> dict:
    p = {
        "object_id": o.object_id,
        "class": o.cls_name,
        "status": o.status,
        "n_views": o.n_views,
        "images": o.images,
        "rms_m": round(o.rms_m, 3),
        "max_m": round(o.max_m, 3),
        "mean_conf": round(o.mean_conf, 3),
        "mean_incidence_deg": round(o.mean_incidence_deg, 2),
        "mean_range_m": round(o.mean_range_m, 2),
        "size_m": _r(o.size_m),
        "flags": list(o.flags),
        "altitude_m": round(o.h, 2),
        "x_2180": round(o.x_2180, 3),
        "y_2180": round(o.y_2180, 3),
        "lon": round(o.lon, 8),
        "lat": round(o.lat, 8),
        "duplicate_images": o.duplicate_images,
    }
    if batch is not None:
        p.update({"class_label": class_label(batch, o.cls_name), "surface": ray_surface(batch),
                  "georef_mode": georef_mode(batch), "qc_status": qc_status(batch)})
    return p


OBJECT_CSV_FIELDS = ["object_id", "class", "class_label", "status", "n_views", "images", "rms_m", "max_m",
                     "mean_conf", "size_m", "flags", "mean_incidence_deg", "mean_range_m", "altitude_m",
                     "x_2180", "y_2180", "lon", "lat", "surface", "georef_mode", "qc_status",
                     "duplicate_images"]


# ───────────── GeoJSON / CSV ─────────────

def detections_geojson(batch: BatchResult) -> dict:
    status = {o.object_id: o.status for o in batch.objects}
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [gd.hit.lon, gd.hit.lat, round(gd.hit.h, 2)]},
         "properties": {**detection_properties(gd),
                        "class_label": class_label(batch, gd.detection.cls_name),
                        "object_status": status.get(gd.object_id)}}
        for gd in batch.detections]}


def objects_geojson(batch: BatchResult) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [o.lon, o.lat, round(o.h, 2)]},
         "properties": object_properties(o, batch)}
        for o in batch.objects]}


def _csv_row(p: dict) -> dict:
    return {k: ";".join(map(str, v)) if isinstance(v, (list, tuple)) else v for k, v in p.items()}


def export_objects_csv(batch: BatchResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OBJECT_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(_csv_row(object_properties(o, batch)) for o in batch.objects)
    log.info("-> %s", path)
    return path


def export_detections_csv(batch: BatchResult, path: Path) -> Path:
    rows = []
    for gd in batch.detections:
        p = detection_properties(gd)
        p.update({"class_label": class_label(batch, gd.detection.cls_name),
                  "lon": round(gd.hit.lon, 8), "lat": round(gd.hit.lat, 8)})
        rows.append(_csv_row(p))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["image", "class"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("-> %s", path)
    return path


# ───────────── GeoPackage ─────────────

def export_gpkg(batch: BatchResult, path: Path, rays: bool = True) -> Path:
    from .gpkg import Layer, write_gpkg
    objs = Layer("objects", "POINT", 2180, description="Obiekty (agregacja wielowidokowa)")
    for o in batch.objects:
        objs.features.append(((o.x_2180, o.y_2180, o.h), {**object_properties(o, batch), "h_npm": round(o.h, 2)}))
    dets = Layer("detections", "POINT", 2180, description="Detekcje (trafienia promieni)")
    ray = Layer("rays", "LINESTRING", 2180, description="Promienie kamera -> trafienie")
    for gd in batch.detections:
        h = gd.hit
        a = {**detection_properties(gd), "class_label": class_label(batch, gd.detection.cls_name),
             "lon": round(h.lon, 8), "lat": round(h.lat, 8), "h_npm": round(h.h, 2)}
        dets.features.append(((h.x_2180, h.y_2180, h.h), a))
    cams = Layer("cameras", "POINT", 2180, description="Srodki rzutow zdjec (EO z ODM)")
    for r in batch.images:
        c = r.camera
        cams.features.append(((c.x_2180, c.y_2180, c.h), {
            "image": r.image_name, "x_2180": round(c.x_2180, 3), "y_2180": round(c.y_2180, 3),
            "h_npm": round(c.h, 2), "lon": round(c.lon, 8), "lat": round(c.lat, 8),
            "n_detections": len(r.detections)}))
        if rays:
            for gd in r.detections:
                ray.features.append((((c.x_2180, c.y_2180, c.h), (gd.hit.x_2180, gd.hit.y_2180, gd.hit.h)), {
                    "image": r.image_name, "class": gd.detection.cls_name, "object_id": gd.object_id,
                    "accepted": gd.accepted, "range_m": round(gd.hit.range_m, 2),
                    "incidence_deg": round(gd.incidence_deg, 2)}))
    layers = [objs, dets, cams] + ([ray] if rays else [])
    write_gpkg(path, layers)
    log.info("-> %s (%s)", path, ", ".join(f"{ly.name}: {len(ly.features)}" for ly in layers))
    return path


# ───────────── KML ─────────────

def _kml_color(hex_rgb: str, alpha: int = 255) -> str:
    h = hex_rgb.lstrip("#")
    return f"{alpha:02x}{h[4:6]}{h[2:4]}{h[0:2]}"          # KML: aabbggrr


def export_kml(batch: BatchResult, path: Path, detections: bool = True) -> Path:
    colors = class_colors(batch)
    esc = html.escape
    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2">', "<Document>",
         f"<name>{esc('Geo-Detection-System — wyniki')}</name>"]
    for cls, col in colors.items():
        for suf, c, scale in (("", col, 1.1), ("_flag", FLAG_COLOR, 1.1), ("_det", col, 0.6)):
            L.append(f'<Style id="{esc(cls.replace(" ", "_"))}{suf}"><IconStyle><color>{_kml_color(c)}</color>'
                     f"<scale>{scale}</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/"
                     f"placemark_circle.png</href></Icon></IconStyle></Style>")

    def ext(props: dict) -> str:
        return "<ExtendedData>" + "".join(
            f'<Data name="{esc(str(k))}"><value>{esc(str(";".join(map(str, v)) if isinstance(v, list) else v))}'
            f"</value></Data>" for k, v in props.items() if v is not None) + "</ExtendedData>"

    L.append("<Folder><name>Obiekty</name>")
    for o in batch.objects:
        p = object_properties(o, batch)
        p["h_npm"] = p.pop("altitude_m")
        sid = o.cls_name.replace(" ", "_") + ("_flag" if o.flags else "")
        name = f"{class_label(batch, o.cls_name)} #{o.object_id}" + (" (!)" if o.flags else "")
        desc = (f"{o.status}, {o.n_views} widok(i), RMS {o.rms_m:.2f} m, rozmiar "
                f"{'—' if not math.isfinite(o.size_m) else f'{o.size_m:.2f} m'}, H {o.h:.2f} m n.p.m."
                + (f", flagi: {', '.join(o.flags)}" if o.flags else ""))
        L.append(f"<Placemark><name>{esc(name)}</name><description>{esc(desc)}</description>"
                 f"<styleUrl>#{esc(sid)}</styleUrl>{ext(p)}<Point><altitudeMode>clampToGround</altitudeMode>"
                 f"<coordinates>{o.lon:.8f},{o.lat:.8f},{o.h:.2f}</coordinates></Point></Placemark>")
    L.append("</Folder>")
    if detections:
        L.append("<Folder><name>Detekcje</name><visibility>0</visibility>")
        for gd in batch.detections:
            p = detection_properties(gd)
            L.append(f"<Placemark><name>{esc(gd.image_name)} {esc(class_label(batch, gd.detection.cls_name))}"
                     f"</name><visibility>0</visibility><styleUrl>#{esc(gd.detection.cls_name.replace(' ', '_'))}_det"
                     f"</styleUrl>{ext(p)}<Point><altitudeMode>clampToGround</altitudeMode><coordinates>"
                     f"{gd.hit.lon:.8f},{gd.hit.lat:.8f},{gd.hit.h:.2f}</coordinates></Point></Placemark>")
        L.append("</Folder>")
    L += ["</Document>", "</kml>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    log.info("-> %s", path)
    return path


# ───────────── miniatury ─────────────

def best_view(o: AggregatedObject, batch: BatchResult) -> Optional[GeoDetection]:
    """Najlepszy widok obiektu: pelny (nie uciety krawedzia kadru), najwyzsze conf."""
    members = [gd for gd in batch.detections if gd.object_id == o.object_id]
    if not members:
        return None
    return max(members, key=lambda gd: (not gd.touches_border, gd.detection.conf))


def _read_image(path: Path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def crop_thumbnail(img, bbox, max_px: int, pad: float = 0.35, min_side_px: int = 96):
    import cv2
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    side = max(side, min_side_px)
    h, w = img.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[b:d, a:c].copy()
    s = max_px / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    # ramka bboxa
    bx1, by1, bx2, by2 = [int(round((v - o) * s)) for v, o in ((x1, a), (y1, b), (x2, a), (y2, b))]
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 0), 1)
    return crop


def _bgr(hex_rgb: str) -> tuple[int, int, int]:
    h = hex_rgb.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw_detections(img, boxes: list[dict], max_px: Optional[int] = 1600):
    """Rysuje boxy na kopii obrazu BGR. boxes: [{"bbox": (x1,y1,x2,y2), "label": str, "color": "#rrggbb"}].
    Zwraca obraz RGB (zmniejszony do max_px dluzszego boku)."""
    import cv2
    out = img.copy()
    s = 1.0
    if max_px and max(out.shape[:2]) > max_px:
        s = max_px / max(out.shape[:2])
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
    th = max(2, int(round(max(out.shape[:2]) / 600)))
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v * s)) for v in b["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), _bgr(b.get("color") or CLASS_PALETTE[0]), th)
    labeled = [b for b in boxes if b.get("label")]
    if not labeled:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # etykiety przez PIL (TrueType: polskie znaki; cv2.putText zna tylko ASCII)
    from PIL import Image, ImageDraw
    im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    font = _label_font(max(12, 7 * th))
    for b in labeled:
        x1, y1 = int(round(b["bbox"][0] * s)), int(round(b["bbox"][1] * s))
        l, t, r, btm = dr.textbbox((0, 0), b["label"], font=font)
        tw, tht = r - l, btm - t
        y0 = max(0, y1 - tht - 8)
        dr.rectangle([x1, y0, x1 + tw + 8, y0 + tht + 8], fill=b.get("color") or CLASS_PALETTE[0])
        dr.text((x1 + 4 - l, y0 + 4 - t), b["label"], fill="white", font=font)
    return np.asarray(im)


def _label_font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_image(path: Path):
    """Obraz BGR (sciezki z polskimi znakami na Windows)."""
    return _read_image(Path(path))


def make_thumbnails(batch: BatchResult, out_dir: Path) -> dict[int, dict]:
    """Miniatura kazdego obiektu z jego najlepszego widoku: {object_id: {path, image, b64}}."""
    import cv2
    if batch.config is None:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    cache, out = {}, {}
    for o in batch.objects:
        gd = best_view(o, batch)
        if gd is None:
            continue
        p = batch.config.photos_dir / gd.image_name
        if gd.image_name not in cache:
            try:
                cache[gd.image_name] = _read_image(p)
            except OSError as exc:
                log.warning("Miniatura: brak %s (%s)", p, exc)
                cache[gd.image_name] = None
        img = cache[gd.image_name]
        if img is None:
            continue
        thumb = crop_thumbnail(img, gd.detection.bbox, THUMB_PX)
        f = out_dir / f"obj_{o.object_id:03d}.jpg"
        cv2.imwrite(str(f), thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        small = crop_thumbnail(img, gd.detection.bbox, POPUP_THUMB_PX)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out[o.object_id] = {"path": f, "image": gd.image_name, "conf": gd.detection.conf,
                            "b64": base64.b64encode(buf.tobytes()).decode("ascii") if ok else None}
    log.info("Miniatury obiektow: %d -> %s", len(out), out_dir)
    return out


# ───────────── mapa folium ─────────────

def _popup_table(props: dict) -> str:
    def fmt(v):
        return ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v
    rows = "".join(f"<tr><th style='text-align:left;padding-right:8px'>{html.escape(str(k))}</th>"
                   f"<td>{html.escape(str(fmt(v)))}</td></tr>" for k, v in props.items())
    return f"<table style='font-size:12px'>{rows}</table>"


def _legend_html(batch: BatchResult, colors: dict) -> str:
    items = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};"
        f"border:2px solid #fff;box-shadow:0 0 0 1px #888;margin-right:6px;vertical-align:middle'></span>"
        f"{html.escape(class_label(batch, n))}</div>" for n, c in colors.items())
    extra = (f"<div style='margin-top:4px'><span style='display:inline-block;width:12px;height:12px;"
             f"border-radius:50%;background:{SINGLE_VIEW_COLOR};border:2px dashed #555;margin-right:6px;"
             f"vertical-align:middle'></span>obiekt z jednego widoku</div>"
             f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
             f"border:3px solid {FLAG_COLOR};margin-right:6px;vertical-align:middle'></span>"
             f"rozmiar poza zakresem klasy</div>"
             f"<div><span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
             f"background:{CAMERA_COLOR};margin:0 8px 0 2px;vertical-align:middle'></span>kamera</div>")
    return ("<div style='position:fixed;bottom:24px;left:12px;z-index:9999;background:rgba(255,255,255,.93);"
            "padding:8px 10px;border-radius:6px;font:12px/1.5 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'>"
            f"<b>Legenda</b>{items}{extra}</div>")


def make_folium_map(batch: BatchResult, thumbs: Optional[dict] = None):
    import folium
    from .basemaps import add_basemaps
    thumbs = thumbs or {}
    colors = class_colors(batch)
    cams = [r.camera for r in batch.images]
    center = [float(np.mean([c.lat for c in cams])), float(np.mean([c.lon for c in cams]))]
    m = folium.Map(location=center, zoom_start=19, max_zoom=23, tiles=None)
    add_basemaps(m, buildings=True)

    fg_cam = folium.FeatureGroup(name="Kamery")
    for r in batch.images:
        c = r.camera
        folium.CircleMarker(
            [c.lat, c.lon], radius=4, color=CAMERA_COLOR, fill=True, fill_opacity=0.9,
            tooltip=f"kamera {r.image_name}",
            popup=folium.Popup(_popup_table({
                "image": r.image_name, "x_2180": round(c.x_2180, 2), "y_2180": round(c.y_2180, 2),
                "h_npm": round(c.h, 2), "detekcje": len(r.detections)}), max_width=300),
        ).add_to(fg_cam)

    objects = {o.object_id: o for o in batch.objects}
    fg_det = folium.FeatureGroup(name="Detekcje pojedyncze")
    fg_lines = folium.FeatureGroup(name="Powiazania detekcja-obiekt")
    fg_rej = folium.FeatureGroup(name="Detekcje odrzucone", show=False)
    for gd in batch.detections:
        props = detection_properties(gd)
        if gd.accepted:
            color, layer = colors[gd.detection.cls_name], fg_det
        else:
            color, layer = REJECTED_COLOR, fg_rej
        folium.CircleMarker(
            [gd.hit.lat, gd.hit.lon], radius=4, color=color, weight=1,
            fill=True, fill_opacity=0.6,
            tooltip=f"{class_label(batch, gd.detection.cls_name)} {gd.detection.conf:.2f} — "
                    f"{gd.image_name}",
            popup=folium.Popup(_popup_table(props), max_width=350),
        ).add_to(layer)
        if gd.object_id is not None:
            o = objects[gd.object_id]
            folium.PolyLine([[gd.hit.lat, gd.hit.lon], [o.lat, o.lon]],
                            color=colors[gd.detection.cls_name], weight=1, opacity=0.6
                            ).add_to(fg_lines)

    # confirmed: pelny kolor klasy; single_view: obrys w kolorze klasy, szare wypelnienie,
    # przerywana linia; flaga rozmiaru: czerwony pierscien + osobna warstwa
    fg_obj = folium.FeatureGroup(name="Obiekty potwierdzone (confirmed)")
    fg_single = folium.FeatureGroup(name="Obiekty z jednego widoku (single_view)")
    fg_flag = folium.FeatureGroup(name="Obiekty z flaga (rozmiar poza zakresem)")
    for o in batch.objects:
        single = not o.confirmed
        props = {k: v for k, v in object_properties(o, batch).items() if k not in ("lon", "lat")}
        th = thumbs.get(o.object_id)
        img = (f"<img src='data:image/jpeg;base64,{th['b64']}' style='display:block;margin-bottom:4px'>"
               f"<div style='font-size:11px;color:#555'>najlepszy widok: {html.escape(th['image'])}</div>"
               if th and th.get("b64") else "")
        size = "—" if not math.isfinite(o.size_m) else f"{o.size_m:.2f} m"
        tip = (f"obiekt {o.object_id}: {class_label(batch, o.cls_name)} [{o.status}], {o.n_views} widokow, "
               f"RMS {o.rms_m:.2f} m, rozmiar {size}" + (f" — {', '.join(o.flags)}" if o.flags else ""))
        layer = fg_flag if o.flags else fg_single if single else fg_obj
        folium.CircleMarker(
            [o.lat, o.lon], radius=8,
            color=FLAG_COLOR if o.flags else colors[o.cls_name] if single else "#ffffff",
            weight=3 if o.flags else 2, dash_array="4 3" if single else None, fill=True,
            fill_color=SINGLE_VIEW_COLOR if single else colors[o.cls_name],
            fill_opacity=0.5 if single else 1.0, tooltip=tip,
            popup=folium.Popup(img + _popup_table(props), max_width=420),
        ).add_to(layer)

    for fg in (fg_cam, fg_lines, fg_det, fg_rej, fg_obj, fg_single, fg_flag):
        fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(batch, colors)))
    return m


# ───────────── wykres rozrzutu ─────────────

def spread_residuals(batch: BatchResult, min_views: int = 2) -> list[tuple[GeoDetection, float]]:
    """(detekcja, odleglosc pozioma od srodka jej obiektu) dla obiektow z >= min_views."""
    objects = {o.object_id: o for o in batch.objects if o.n_views >= min_views}
    return [(gd, float(np.hypot(gd.hit.x_2180 - objects[gd.object_id].x_2180,
                                gd.hit.y_2180 - objects[gd.object_id].y_2180)))
            for gd in batch.detections if gd.object_id in objects]


def plot_spread(batch: BatchResult, out_png: Path) -> Path | None:
    """Odchylenie detekcji od srodka obiektu vs incidence_deg i range_m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = spread_residuals(batch)
    if not res:
        log.info("Brak obiektow z >=2 widokami — pomijam wykres rozrzutu")
        return None
    colors = class_colors(batch)
    classes = sorted({gd.detection.cls_name for gd, _ in res})

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="#fcfcfb")
    panels = [("incidence_deg", "Kat promienia od pionu [deg]", lambda gd: gd.incidence_deg),
              ("range_m", "Odleglosc kamera-obiekt [m]", lambda gd: gd.hit.range_m)]
    for ax, (_, xlabel, getter) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for i, cls in enumerate(classes):
            pts = [(getter(gd), r) for gd, r in res if gd.detection.cls_name == cls]
            x, y = zip(*pts)
            ax.scatter(x, y, s=42, marker=CLASS_MARKERS[i % len(CLASS_MARKERS)],
                       color=colors[cls], edgecolor="#fcfcfb", linewidth=1.2,
                       label=f"{class_label(batch, cls)} (n={len(pts)})", zorder=3)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Odchylenie od srodka obiektu [m]")
    axes[0].legend(frameon=False, loc="upper left")
    n_obj = len({gd.object_id for gd, _ in res})
    fig.suptitle(f"Rozrzut wielowidokowy: {len(res)} detekcji w {n_obj} obiektach "
                 f"z >=2 widokami", x=0.01, ha="left", fontsize=12, color="#0b0b0b")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("-> %s", out_png)
    return out_png


# ───────────── raport przebiegu ─────────────

def run_parameters(batch: BatchResult) -> dict:
    cfg = batch.config
    if cfg is None:
        return {}
    zc = batch.z_calibration
    d = cfg.detect
    z = {"method": zc.method, "z_offset_m": zc.offset,
         "stats": {k: v for k, v in zc.stats.items() if not isinstance(v, dict)}} if zc else None
    if zc is not None and zc.tilt is not None:
        z["tilt"] = {"a": zc.tilt.a, "b": zc.tilt.b}
    return {
        "model": str(cfg.model_path), "detect_source": getattr(d, "source", "model"),
        "classes": d.classes if d.classes not in (None, "all") else "all",
        "display_names": d.display_names, "class_colors": class_colors(batch),
        "conf": d.conf, "conf_per_class": d.conf_per_class,
        "iou": d.iou, "imgsz": d.imgsz, "tiling": bool(getattr(d.tiling, "enabled", False)),
        "point_method": cfg.point.method, "point_per_class": cfg.point.per_class,
        "surface": cfg.surface.source, "georef_mode": georef_mode(batch),
        "georef_decision": getattr(batch, "georef_decision", None),
        "georef_comparison": (dataclasses.asdict(batch.georef_comparison)
                              if getattr(batch, "georef_comparison", None) is not None else None),
        "z_calibration": z,
        "aggregate": {"eps_m": cfg.aggregate.eps_m, "min_samples": cfg.aggregate.min_samples,
                      "min_views": cfg.aggregate.min_views, "weights": cfg.aggregate.weights},
        "filters": cfg.filters.__dict__, "size_ranges_m": cfg.size.ranges_m if cfg.size.enabled else None,
        "photos_dir": str(cfg.photos_dir), "odm_dir": str(cfg.odm_dir),
    }


def run_statistics(batch: BatchResult) -> dict:
    per_cls = Counter(o.cls_name for o in batch.objects)
    return {
        "images_processed": len(batch.images), "images_skipped": len(batch.skipped),
        "skipped": batch.skipped,
        "detections": len(batch.detections), "accepted": len(batch.accepted), "rejected": len(batch.rejected),
        "rejected_by_reason": dict(Counter(gd.rejected_reason for gd in batch.rejected)),
        "misses": len(batch.misses),
        "objects": len(batch.objects), "confirmed": sum(o.confirmed for o in batch.objects),
        "single_view": sum(not o.confirmed for o in batch.objects),
        "flagged": sum(bool(o.flags) for o in batch.objects),
        "objects_per_class": {class_label(batch, c): n for c, n in sorted(per_cls.items())},
        "suspicious_merges": [o.object_id for o in batch.suspicious_objects],
    }


def run_report_dict(batch: BatchResult, paths: dict, out_dir: Path, thumbs: dict, meta: Optional[dict]) -> dict:
    objs = []
    for o in batch.objects:
        p = object_properties(o, batch)
        th = thumbs.get(o.object_id)
        if th:
            p["thumbnail"] = str(Path(th["path"]).relative_to(out_dir)).replace("\\", "/")
            p["best_view"] = th["image"]
        objs.append(p)
    return {
        "created": datetime.now().isoformat(timespec="seconds"), **(meta or {}),
        "parameters": run_parameters(batch),
        "qc": batch.qc.to_dict() if batch.qc is not None else None,
        "statistics": run_statistics(batch),
        "objects": objs,
        "files": {k: str(Path(v).relative_to(out_dir)).replace("\\", "/") for k, v in paths.items()
                  if Path(v).is_relative_to(out_dir)},
    }


_STATUS_CSS = {"OK": "ok", "OSTRZEŻENIE": "warn", "ZŁY": "bad", "brak danych": "na"}

_GLOSSARY = [
    ("obiekt", "Jeden fizyczny obiekt w terenie. Ten sam obiekt widziany na kilku zdjęciach daje kilka detekcji, "
               "które są łączone (grupowanie przestrzenne DBSCAN w promieniu eps) w jeden obiekt."),
    ("confirmed / single_view", "confirmed — obiekt widziany na co najmniej min_views zdjęciach (potwierdzony); "
                                "single_view — tylko jedno zdjęcie: możliwa fałszywa detekcja."),
    ("RMS / max rozrzutu", "Jak daleko od siebie leżą położenia tego samego obiektu wyznaczone z różnych zdjęć "
                           "(w poziomie). Miara spójności georeferencji, nie błąd bezwzględny."),
    ("rozmiar", "Większy wymiar poziomy obiektu oszacowany z ramki detekcji, odległości kamery i kąta patrzenia "
                "(mediana po zdjęciach). Dla obiektów wysokich oglądanych ukośnie zawyżony."),
    ("size_out_of_range", "Rozmiar poza typowym zakresem dla klasy (z konfiguracji) — obiekt prawdopodobnie "
                          "fałszywy (np. dach budynku rozpoznany jako namiot). Obiekt nie jest usuwany."),
    ("powierzchnia", "Model terenu, na który rzutowane są promienie z kamery: NMT (grunt), NMPT (grunt z "
                     "budynkami i roślinnością, GUGiK) lub DSM z rekonstrukcji ODM."),
    ("georeferencja", "Sposób osadzenia rekonstrukcji zdjęć w układzie państwowym: gps_only + kalibracja wysokości "
                      "(z_anchor) albo dodatkowo wyrównanie 7 parametrów do NMPT (similarity_nmpt)."),
    ("kontrola jakości (QC)", "Automatyczna ocena rekonstrukcji: skala pionowa vs barometr, ogniskowa, zgodność "
                              "modelu z NMPT GUGiK (przed i po georeferencji), baza lotu względem szumu GPS."),
]


def _fmt(v, nd=2, unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{unit}"
    return f"{v}{unit}"


def run_report_html(rep: dict, batch: BatchResult) -> str:
    esc = html.escape
    st, par = rep["statistics"], rep["parameters"]
    colors = class_colors(batch)
    css = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--card:#ffffff;--ok:#1baf7a;--warn:#eda100;
--bad:#d03b3b;--na:#8a8984}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#f1f0ec;--mut:#b5b3ab;--line:#33322f;--card:#1f1f1d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:19px;margin:36px 0 10px;padding-bottom:4px;border-bottom:1px solid var(--line)}
.mut{color:var(--mut)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile b{display:block;font-size:24px}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}.wrap{overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12.5px;font-weight:600;color:#fff}
.ok{background:var(--ok)}.warn{background:var(--warn)}.bad{background:var(--bad)}.na{background:var(--na)}
.flag{color:var(--bad);font-weight:600}img.th{width:128px;height:auto;border-radius:4px;display:block}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
iframe{width:100%;height:560px;border:1px solid var(--line);border-radius:8px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px}dt{font-weight:600}dd{margin:0}
code{font-size:13px}
"""
    L = ["<!doctype html><html lang='pl'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>Raport przebiegu</title>", f"<style>{css}</style></head><body><main>",
         "<h1>Raport przebiegu detekcji i georeferencji</h1>",
         f"<p class='mut'>{esc(rep['created'])} · zdjęcia: <code>{esc(par.get('photos_dir', ''))}</code></p>",
         "<p>System wykrywa obiekty (np. namioty, ludzi) na zdjęciach z drona modelem YOLO, a następnie wyznacza "
         "ich położenie w terenie: z każdego zdjęcia prowadzony jest promień od kamery przez wykryty obiekt aż do "
         "przecięcia z modelem terenu. Detekcje tego samego obiektu z różnych zdjęć są łączone w jeden obiekt.</p>"]

    # podsumowanie
    qc = rep.get("qc")
    L.append("<h2>Podsumowanie</h2><div class='tiles'>")
    for lab, v in (("zdjęcia przetworzone", st["images_processed"]), ("detekcje", st["detections"]),
                   ("obiekty", st["objects"]), ("potwierdzone", st["confirmed"]),
                   ("z jednego widoku", st["single_view"]), ("z flagą rozmiaru", st["flagged"])):
        L.append(f"<div class='tile'><span class='mut'>{esc(lab)}</span><b>{v}</b></div>")
    if qc:
        L.append(f"<div class='tile'><span class='mut'>kontrola jakości</span><b><span class='badge "
                 f"{_STATUS_CSS.get(qc['status'], 'na')}'>{esc(qc['status'])}</span></b></div>")
    L.append("</div>")
    if st["objects_per_class"]:
        L.append("<p>Obiekty wg klas: " + ", ".join(f"{esc(k)}: {v}" for k, v in st["objects_per_class"].items())
                 + ".</p>")

    # obiekty
    L.append("<h2>Obiekty</h2><p class='mut'>Miniatura z najlepszego widoku (najwyższa pewność, obiekt w całości "
             "w kadrze). Współrzędne: WGS84 i PUWG 1992 (EPSG:2180), wysokość w m n.p.m.</p><div class='wrap'><table>"
             "<tr><th></th><th>#</th><th>klasa</th><th>status</th><th>widoki</th><th>rozrzut RMS / max [m]</th>"
             "<th>pewność</th><th>rozmiar [m]</th><th>flagi</th><th>położenie</th></tr>")
    for o in rep["objects"]:
        c = colors.get(o["class"], "#888")
        img = f"<img class='th' src='{esc(o['thumbnail'])}' alt='obiekt {o['object_id']}'>" if o.get("thumbnail") else ""
        flags = "".join(f"<span class='flag'>{esc(f)}</span>" for f in o["flags"]) or "—"
        L.append(f"<tr><td>{img}</td><td>{o['object_id']}</td><td><span class='dot' style='background:{c}'></span>"
                 f"{esc(o.get('class_label', o['class']))}</td><td>{esc(o['status'])}</td><td>{o['n_views']}</td>"
                 f"<td>{_fmt(o['rms_m'])} / {_fmt(o['max_m'])}</td><td>{_fmt(o['mean_conf'])}</td>"
                 f"<td>{_fmt(o['size_m'])}</td><td>{flags}</td>"
                 f"<td>{o['lat']:.6f} N, {o['lon']:.6f} E<br><span class='mut'>X {o['x_2180']:.2f}, "
                 f"Y {o['y_2180']:.2f}, H {_fmt(o['altitude_m'])} m</span></td></tr>")
    L.append("</table></div>")

    # mapa
    mp = rep["files"].get("map_html")
    if mp:
        L.append(f"<h2>Mapa</h2><p class='mut'>Warstwy i podkłady w prawym górnym rogu mapy; kliknięcie obiektu "
                 f"pokazuje jego miniaturę i atrybuty. <a href='{esc(mp)}'>Otwórz mapę w osobnym oknie</a>.</p>"
                 f"<iframe src='{esc(mp)}' loading='lazy'></iframe>")

    # QC
    if qc:
        L.append("<h2>Kontrola jakości rekonstrukcji</h2><table><tr><th>kontrola</th><th>status</th><th>wynik</th></tr>")
        for ch in qc["checks"]:
            L.append(f"<tr><td>{esc(ch['name'])}</td><td><span class='badge {_STATUS_CSS.get(ch['status'], 'na')}'>"
                     f"{esc(ch['status'])}</span></td><td>{esc(ch['text'])}</td></tr>")
        L.append("</table>")
        if qc["recommendations"]:
            L.append("<p><b>Zalecenia:</b></p><ul>" + "".join(f"<li>{esc(r)}</li>" for r in qc["recommendations"])
                     + "</ul>")

    # parametry
    z = par.get("z_calibration") or {}
    rows = [("model detekcji", par.get("model")), ("klasy", par.get("classes")),
            ("progi pewności", f"domyślny {par.get('conf')}"
             + "".join(f", {k}: {v}" for k, v in (par.get("conf_per_class") or {}).items())),
            ("punkt promienia", f"{par.get('point_method')}"
             + "".join(f", {k}: {v}" for k, v in (par.get("point_per_class") or {}).items())),
            ("powierzchnia ray castingu", par.get("surface")), ("georeferencja", par.get("georef_mode")),
            ("kalibracja wysokości", f"{z.get('method')}: z_offset {_fmt(z.get('z_offset_m'), 3)} m" if z else "—")]
    zs = z.get("stats") or {}
    if str(z.get("method")) == "similarity_nmpt":
        rows.append(("transformacja 7p", ", ".join(f"{k} {_fmt(zs.get(k), 3)}" for k in
                                                  ("tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg", "scale", "tilt_deg")
                                                  if k in zs)
                     + f"; NMAD przed/po {_fmt(zs.get('res_nmad_before'))}/{_fmt(zs.get('res_nmad_after'))} m"))
    ag = par.get("aggregate") or {}
    rows += [("łączenie widoków", f"eps {ag.get('eps_m')} m, potwierdzony od {ag.get('min_views')} widoków"),
             ("zakresy rozmiaru [m]", ", ".join(f"{k}: {v[0]}–{v[1]}" for k, v in (par.get("size_ranges_m") or {}).items())
              or "—"),
             ("rekonstrukcja ODM", par.get("odm_dir"))]
    L.append("<h2>Parametry przebiegu</h2><dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(str(v))}</dd>" for k, v in rows)
             + "</dl>")

    # statystyki
    L.append("<h2>Statystyki</h2><dl>")
    L.append(f"<dt>detekcje</dt><dd>{st['detections']} (przyjęte {st['accepted']}, odrzucone {st['rejected']}; "
             f"bez przecięcia z terenem {st['misses']})</dd>")
    for k, v in st["rejected_by_reason"].items():
        L.append(f"<dt>odrzucone: {esc(str(k))}</dt><dd>{v}</dd>")
    if st["skipped"]:
        L.append("<dt>pominięte zdjęcia</dt><dd>" + "<br>".join(f"{esc(k)}: {esc(v)}" for k, v in st["skipped"].items())
                 + "</dd>")
    if st["suspicious_merges"]:
        L.append(f"<dt>możliwe złe scalenie</dt><dd>obiekty {st['suspicious_merges']}</dd>")
    L.append("</dl>")

    # pliki
    L.append("<h2>Pliki wynikowe</h2><ul>" + "".join(
        f"<li><a href='{esc(v)}'>{esc(v)}</a> <span class='mut'>({esc(k)})</span></li>"
        for k, v in rep["files"].items()) + "</ul>")
    L.append("<h2>Objaśnienia</h2><dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in _GLOSSARY)
             + "</dl></main></body></html>")
    return "\n".join(L)


# ───────────── wszystko naraz ─────────────

def export_all(batch: BatchResult, out_dir: Path) -> dict[str, Path]:
    """Zestaw podstawowy (eksperymenty): GeoJSON, CSV obiektow, mapa, wykres rozrzutu."""
    out: dict[str, Path] = {
        "detections_geojson": _write_json(detections_geojson(batch), out_dir / "detections.geojson"),
        "objects_geojson": _write_json(objects_geojson(batch), out_dir / "objects.geojson"),
        "objects_csv": export_objects_csv(batch, out_dir / "objects.csv"),
    }
    if batch.images:
        html_p = out_dir / "detections_map.html"
        make_folium_map(batch).save(str(html_p))
        log.info("-> %s", html_p)
        out["map_html"] = html_p
    png = plot_spread(batch, out_dir / "spread.png")
    if png is not None:
        out["spread_png"] = png
    return out


def export_run(batch: BatchResult, out_dir: Path, meta: Optional[dict] = None, rays: bool = True,
               thumbnails: bool = True) -> dict[str, Path]:
    """Pelny eksport przebiegu do out_dir (GeoJSON, CSV, GPKG, KML, mapa, miniatury, raport)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = make_thumbnails(batch, out_dir / "thumbs") if thumbnails else {}
    out: dict[str, Path] = {
        "objects_geojson": _write_json(objects_geojson(batch), out_dir / "objects.geojson"),
        "detections_geojson": _write_json(detections_geojson(batch), out_dir / "detections.geojson"),
        "objects_csv": export_objects_csv(batch, out_dir / "objects.csv"),
        "detections_csv": export_detections_csv(batch, out_dir / "detections.csv"),
        "gpkg": export_gpkg(batch, out_dir / "results.gpkg", rays=rays),
        "kml": export_kml(batch, out_dir / "objects.kml"),
    }
    if batch.images:
        p = out_dir / "map.html"
        make_folium_map(batch, thumbs).save(str(p))
        log.info("-> %s", p)
        out["map_html"] = p
    png = plot_spread(batch, out_dir / "spread.png")
    if png is not None:
        out["spread_png"] = png
    out["report_json"] = out_dir / "run_report.json"
    out["report_html"] = out_dir / "run_report.html"
    rep = run_report_dict(batch, out, out_dir, thumbs, meta)
    _write_json(rep, out["report_json"])
    out["report_html"].write_text(run_report_html(_clean(rep), batch), encoding="utf-8")
    log.info("-> %s", out["report_html"])
    return out
