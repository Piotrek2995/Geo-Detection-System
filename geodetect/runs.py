"""Zapisane przebiegi (output/runs/<data_czas>/): lista, wczytanie, porownanie (GUI: Wyniki, Historia)."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

REPORT = "run_report.json"


@dataclass
class RunSummary:
    path: Path
    name: str
    created: str
    n_images: int
    n_objects: int
    n_confirmed: int
    n_flagged: int
    qc_status: Optional[str]
    georef_mode: Optional[str]
    model: Optional[str]


@dataclass
class RunData:
    path: Path
    report: dict
    objects: list[dict]                 # properties z objects.geojson (+ lon, lat)
    detections: list[dict]              # properties z detections.geojson (+ lon, lat)
    files: dict[str, Path] = field(default_factory=dict)

    def object(self, object_id: int) -> Optional[dict]:
        return next((o for o in self.objects if o["object_id"] == object_id), None)

    def views(self, object_id: int) -> list[dict]:
        """Detekcje obiektu, najpierw najlepsze (pelne w kadrze, najwyzsze conf)."""
        v = [d for d in self.detections if d.get("object_id") == object_id]
        return sorted(v, key=lambda d: (bool(d.get("touches_border")), -float(d.get("conf") or 0)))


def _qc_status(qc: Optional[dict]) -> Optional[str]:
    """Status QC przebiegu; stare raporty przeliczane wg biezacej reguly (QcReport.from_dict)."""
    if not qc:
        return None
    from .qc import QcReport
    try:
        return QcReport.from_dict(qc).status
    except (KeyError, TypeError, ValueError):
        return qc.get("status")


def _summary(path: Path, rep: dict) -> RunSummary:
    st, par = rep.get("statistics", {}), rep.get("parameters", {})
    return RunSummary(path, path.name, rep.get("created", ""), st.get("images_processed", 0),
                      st.get("objects", 0), st.get("confirmed", 0), st.get("flagged", 0),
                      _qc_status(rep.get("qc")), par.get("georef_mode"),
                      Path(par["model"]).name if par.get("model") else None)


def list_runs(output_dir: Path) -> list[RunSummary]:
    """Przebiegi z <output_dir>/runs, najnowsze pierwsze (katalogi bez raportu pomijane)."""
    root = Path(output_dir) / "runs"
    if not root.is_dir():
        return []
    out = []
    for d in root.iterdir():
        f = d / REPORT
        if d.is_dir() and f.exists():
            try:
                out.append(_summary(d, json.loads(f.read_text(encoding="utf-8"))))
            except (OSError, ValueError) as exc:
                log.warning("Pomijam %s: %s", d, exc)
    return sorted(out, key=lambda r: r.name, reverse=True)


def _features(path: Path) -> list[dict]:
    if not path.exists():
        return []
    fc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in fc.get("features", []):
        p = dict(f.get("properties") or {})
        c = (f.get("geometry") or {}).get("coordinates") or [None, None]
        p.setdefault("lon", c[0])
        p.setdefault("lat", c[1])
        out.append(p)
    return out


def load_run(path: Path) -> RunData:
    path = Path(path)
    rep = json.loads((path / REPORT).read_text(encoding="utf-8"))
    files = {k: path / v for k, v in (rep.get("files") or {}).items()}
    return RunData(path, rep, _features(path / "objects.geojson"), _features(path / "detections.geojson"), files)


# ───────────── porownanie przebiegow ─────────────

@dataclass
class ObjectMatch:
    cls: str
    a_id: Optional[int]
    b_id: Optional[int]
    shift_m: Optional[float]           # odleglosc pozioma A -> B
    dx_m: Optional[float] = None
    dy_m: Optional[float] = None
    dh_m: Optional[float] = None


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def compare_runs(a: RunData, b: RunData, max_dist_m: float = 10.0) -> tuple[list[ObjectMatch], dict]:
    """Pary obiektow tej samej klasy (najblizsze, zachlannie, do max_dist_m) i roznice parametrow
    {klucz: (A, B)}."""
    matches: list[ObjectMatch] = []
    used_b: set[int] = set()
    pairs = []
    for oa in a.objects:
        for ob in b.objects:
            if oa["class"] == ob["class"]:
                d = math.hypot(ob["x_2180"] - oa["x_2180"], ob["y_2180"] - oa["y_2180"])
                if d <= max_dist_m:
                    pairs.append((d, oa, ob))
    used_a: set[int] = set()
    for d, oa, ob in sorted(pairs, key=lambda t: t[0]):
        if oa["object_id"] in used_a or ob["object_id"] in used_b:
            continue
        used_a.add(oa["object_id"])
        used_b.add(ob["object_id"])
        ha, hb = oa.get("altitude_m"), ob.get("altitude_m")
        matches.append(ObjectMatch(oa["class"], oa["object_id"], ob["object_id"], d,
                                   ob["x_2180"] - oa["x_2180"], ob["y_2180"] - oa["y_2180"],
                                   None if ha is None or hb is None else hb - ha))
    matches += [ObjectMatch(o["class"], o["object_id"], None, None) for o in a.objects if o["object_id"] not in used_a]
    matches += [ObjectMatch(o["class"], None, o["object_id"], None) for o in b.objects if o["object_id"] not in used_b]
    pa, pb = _flatten(a.report.get("parameters", {})), _flatten(b.report.get("parameters", {}))
    diff = {k: (pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)}
    return matches, diff


# ───────────── mapa przebiegu wczytanego z dysku ─────────────

def run_map(run: RunData, selected: Optional[int] = None, show_detections: bool = True, basemap: str = "esri"):
    """Mapa folium przebiegu z plikow (podklady jak w eksporcie, legenda, obiekty, detekcje, kamery)."""
    import html as _html

    import folium

    from .basemaps import add_basemaps
    from .export import CAMERA_COLOR, CLASS_PALETTE, FLAG_COLOR, SINGLE_VIEW_COLOR

    par = run.report.get("parameters", {})
    classes = sorted({o["class"] for o in run.objects} | {d["class"] for d in run.detections})
    colors = dict(par.get("class_colors") or {})
    free = iter([c for c in CLASS_PALETTE if c not in colors.values()] * 3)
    colors = {c: colors.get(c) or next(free) for c in classes}
    labels = par.get("display_names") or {}
    pts = [(o["lat"], o["lon"]) for o in run.objects] or [(d["lat"], d["lon"]) for d in run.detections]
    cams = []
    gp = run.path / "results.gpkg"
    if gp.exists():
        from .gpkg import read_gpkg
        try:
            cams = read_gpkg(gp, "cameras")["cameras"]
        except Exception as exc:       # uszkodzony / starszy plik — mapa bez kamer
            log.warning("Kamery z %s: %s", gp, exc)
    pts += [(c["lat"], c["lon"]) for c in cams]
    center = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)] if pts else [52, 19]
    m = folium.Map(location=center, zoom_start=19, max_zoom=23, tiles=None)
    add_basemaps(m, basemap)
    fg_cam = folium.FeatureGroup(name="Kamery")
    for c in cams:
        folium.CircleMarker([c["lat"], c["lon"]], radius=4, color=CAMERA_COLOR, fill=True, fill_opacity=0.9,
                            tooltip=f"kamera {c['image']}").add_to(fg_cam)
    fg_det = folium.FeatureGroup(name="Detekcje pojedyncze", show=show_detections)
    for d in run.detections:
        folium.CircleMarker([d["lat"], d["lon"]], radius=3, color=colors[d["class"]], weight=1, fill=True,
                            fill_opacity=0.6, tooltip=f"{d['image']}: {labels.get(d['class'], d['class'])} "
                            f"{float(d.get('conf') or 0):.2f}").add_to(fg_det)
    fg_obj = folium.FeatureGroup(name="Obiekty")
    for o in run.objects:
        single = o.get("status") != "confirmed"
        flagged = bool(o.get("flags"))
        sel = o["object_id"] == selected
        size = o.get("size_m")
        folium.CircleMarker(
            [o["lat"], o["lon"]], radius=12 if sel else 8,
            color=FLAG_COLOR if flagged else ("#111111" if sel else (colors[o["class"]] if single else "#ffffff")),
            weight=4 if sel else (3 if flagged else 2), dash_array="4 3" if single else None, fill=True,
            fill_color=SINGLE_VIEW_COLOR if single else colors[o["class"]], fill_opacity=0.55 if single else 1.0,
            tooltip=_html.escape(f"#{o['object_id'] + 1} {labels.get(o['class'], o['class'])} [{o.get('status')}], "
                                 f"{o.get('n_views')} widoków"
                                 + (f", {size:.1f} m" if isinstance(size, (int, float)) else "")
                                 + (f" — {', '.join(o['flags'])}" if flagged else "")),
        ).add_to(fg_obj)
    for fg in (fg_cam, fg_det, fg_obj):
        fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    items = "".join(f"<div><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
                    f"background:{c};margin-right:6px'></span>{_html.escape(labels.get(n, n))}</div>"
                    for n, c in colors.items())
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:20px;left:10px;z-index:9999;background:rgba(255,255,255,.93);"
        "padding:6px 9px;border-radius:6px;font:12px/1.5 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'>"
        f"<b>Legenda</b>{items}<div><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
        f"border:3px solid {FLAG_COLOR};margin-right:6px'></span>nietypowy rozmiar</div>"
        f"<div><span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
        f"background:{SINGLE_VIEW_COLOR};border:2px dashed #555;margin-right:6px'></span>jeden widok</div></div>"))
    return m
