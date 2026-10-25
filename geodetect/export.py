"""Eksport wynikow: GeoJSON (WGS84) i mapa podgladowa folium."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .pipeline import ImageResult

log = logging.getLogger(__name__)

CLASS_COLORS = {"namiot": "red", "people": "green"}


def to_geojson(results: list[ImageResult]) -> dict:
    features = []
    for res in results:
        for gd in res.detections:
            h = gd.hit
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [h.lon, h.lat]},
                "properties": {
                    "image": gd.image_name,
                    "class": gd.detection.cls_name,
                    "conf": round(gd.detection.conf, 3),
                    "altitude_m": round(h.h, 2),
                    "x_2180": round(h.x_2180, 3),
                    "y_2180": round(h.y_2180, 3),
                    "range_m": round(h.range_m, 2),
                    "pixel_u": round(gd.uv[0], 1),
                    "pixel_v": round(gd.uv[1], 1),
                },
            })
    return {"type": "FeatureCollection", "features": features}


def export_geojson(results: list[ImageResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_geojson(results), indent=2, ensure_ascii=False),
                        encoding="utf-8")
    log.info("GeoJSON -> %s", out_path)
    return out_path


def make_folium_map(results: list[ImageResult]):
    import folium
    first = results[0].camera
    m = folium.Map(location=[first.lat, first.lon], zoom_start=19, tiles="OpenStreetMap")
    for res in results:
        c = res.camera
        folium.Marker([c.lat, c.lon], tooltip=f"kamera {res.image_name}",
                      icon=folium.Icon(color="blue")).add_to(m)
        for gd in res.detections:
            folium.CircleMarker(
                [gd.hit.lat, gd.hit.lon], radius=6,
                color=CLASS_COLORS.get(gd.detection.cls_name, "orange"),
                fill=True, fill_opacity=0.8,
                tooltip=f"{gd.detection.cls_name} ({gd.detection.conf:.2f}) — {gd.image_name}",
            ).add_to(m)
    return m


def export_folium_map(results: list[ImageResult], out_html: Path) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    make_folium_map(results).save(str(out_html))
    log.info("Mapa -> %s", out_html)
    return out_html
