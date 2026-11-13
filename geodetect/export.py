"""Eksport wynikow: GeoJSON (detekcje, obiekty), CSV, mapa folium, wykres rozrzutu."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

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


def class_colors(batch: BatchResult) -> dict[str, str]:
    names = sorted({gd.detection.cls_name for gd in batch.detections})
    return {n: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, n in enumerate(names)}


def _write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("-> %s", path)
    return path


# ───────────── atrybuty ─────────────

def detection_properties(gd: GeoDetection) -> dict:
    h = gd.hit
    return {
        "image": gd.image_name,
        "class": gd.detection.cls_name,
        "conf": round(gd.detection.conf, 3),
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


def object_properties(o: AggregatedObject) -> dict:
    return {
        "object_id": o.object_id,
        "class": o.cls_name,
        "n_views": o.n_views,
        "images": o.images,
        "rms_m": round(o.rms_m, 3),
        "max_m": round(o.max_m, 3),
        "mean_conf": round(o.mean_conf, 3),
        "mean_incidence_deg": round(o.mean_incidence_deg, 2),
        "mean_range_m": round(o.mean_range_m, 2),
        "altitude_m": round(o.h, 2),
        "x_2180": round(o.x_2180, 3),
        "y_2180": round(o.y_2180, 3),
        "duplicate_images": o.duplicate_images,
    }


OBJECT_CSV_FIELDS = ["object_id", "class", "n_views", "images", "rms_m", "max_m", "mean_conf",
                     "mean_incidence_deg", "mean_range_m", "altitude_m", "x_2180", "y_2180",
                     "duplicate_images", "lon", "lat"]


# ───────────── GeoJSON / CSV ─────────────

def detections_geojson(batch: BatchResult) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [gd.hit.lon, gd.hit.lat]},
         "properties": detection_properties(gd)}
        for gd in batch.detections]}


def objects_geojson(batch: BatchResult) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [o.lon, o.lat]},
         "properties": object_properties(o)}
        for o in batch.objects]}


def export_objects_csv(batch: BatchResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for o in batch.objects:
        p = object_properties(o)
        p["lon"], p["lat"] = round(o.lon, 8), round(o.lat, 8)
        p["images"] = ";".join(o.images)
        p["duplicate_images"] = ";".join(o.duplicate_images)
        rows.append(p)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OBJECT_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    log.info("-> %s", path)
    return path


# ───────────── mapa folium ─────────────

def _popup_table(props: dict) -> str:
    rows = "".join(f"<tr><th style='text-align:left;padding-right:8px'>{k}</th><td>{v}</td></tr>"
                   for k, v in props.items())
    return f"<table style='font-size:12px'>{rows}</table>"


def make_folium_map(batch: BatchResult):
    import folium
    colors = class_colors(batch)
    cams = [r.camera for r in batch.images]
    center = [float(np.mean([c.lat for c in cams])), float(np.mean([c.lon for c in cams]))]
    m = folium.Map(location=center, zoom_start=18, max_zoom=22, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satelita (Esri)", max_zoom=22, max_native_zoom=19,
    ).add_to(m)

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
            tooltip=f"{gd.detection.cls_name} {gd.detection.conf:.2f} — {gd.image_name}",
            popup=folium.Popup(_popup_table(props), max_width=350),
        ).add_to(layer)
        if gd.object_id is not None:
            o = objects[gd.object_id]
            folium.PolyLine([[gd.hit.lat, gd.hit.lon], [o.lat, o.lon]],
                            color=colors[gd.detection.cls_name], weight=1, opacity=0.6
                            ).add_to(fg_lines)

    fg_obj = folium.FeatureGroup(name="Obiekty zagregowane")
    for o in batch.objects:
        folium.CircleMarker(
            [o.lat, o.lon], radius=8, color="#ffffff", weight=2,
            fill=True, fill_color=colors[o.cls_name], fill_opacity=1.0,
            tooltip=f"obiekt {o.object_id}: {o.cls_name}, {o.n_views} widokow, "
                    f"RMS {o.rms_m:.2f} m",
            popup=folium.Popup(_popup_table(object_properties(o)), max_width=400),
        ).add_to(fg_obj)

    for fg in (fg_cam, fg_lines, fg_det, fg_rej, fg_obj):
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
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
                       label=f"{cls} (n={len(pts)})", zorder=3)
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


# ───────────── wszystko naraz ─────────────

def export_all(batch: BatchResult, out_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {
        "detections_geojson": _write_json(detections_geojson(batch), out_dir / "detections.geojson"),
        "objects_geojson": _write_json(objects_geojson(batch), out_dir / "objects.geojson"),
        "objects_csv": export_objects_csv(batch, out_dir / "objects.csv"),
    }
    if batch.images:
        html = out_dir / "detections_map.html"
        make_folium_map(batch).save(str(html))
        log.info("-> %s", html)
        out["map_html"] = html
    png = plot_spread(batch, out_dir / "spread.png")
    if png is not None:
        out["spread_png"] = png
    return out
