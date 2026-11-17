"""Eksperymenty porownawcze: jeden mechanizm zamiast osobnych skryptow *_compare.py.

Eksperyment (experiments/*.yaml) = bazowy config + nadpisania wspolne + warianty:
  - `axes`: iloczyn kartezjanski osi, np. {surface.source: [nmt_local, odm_dsm]},
  - `variants`: jawna lista {name, set: {pole.z.kropkami: wartosc}, derive: {...}}.
Detekcje YOLO (i maski SAM — cache dyskowy) sa liczone raz dla kazdej unikalnej
kombinacji (model, sekcja detect) i wspoldzielone przez warianty rozniace sie tylko
geometria (powierzchnia, kalibracja Z, punkt reprezentatywny). DEM, rekonstrukcja ODM
i powierzchnie tez sa wspoldzielone.

Wynik: output/experiments/<nazwa>/<data_czas>/ — config i eksport kazdego wariantu,
summary.csv, summary.md (tabele do pracy), wykresy PNG, meta.json (commit, wersje).
Opis metryk: docs/experiments.md.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import logging
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .aggregate import AggregatedObject, aggregate
from .batch import BatchResult, run_batch
from .config import _PATH_FIELDS, PipelineConfig, _to_plain
from .detect import Detection, Detector, make_detector
from .diagnostics import ZScan, fit_along_vs_tan, multiview_z_scan, ray_residuals

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"detect.model": "model_path", "model": "model_path"}
EXPERIMENT_KEYS = {"name", "description", "base_config", "overrides", "axes", "variants",
                   "reference", "metrics", "images", "exclude"}


# ───────────── definicja eksperymentu ─────────────

@dataclass
class MetricsSpec:
    static_classes: list[str] = field(default_factory=lambda: ["namiot"])
    moving_classes: list[str] = field(default_factory=lambda: ["people"])
    min_views_static: int = 2        # obiekt statyczny w tabelach: confirmed i >= tylu widokow
    min_views_scan: int = 3          # multiview_check / RMS zbiorczy: >= tylu widokow
    multiview: Optional[dict] = None     # {span_m, step_m, max_rays} — skan z_offset per wariant
    ground_truth: bool = False           # P/R/F1 wzgledem cfg.eval (etykiety)
    gt_previews: bool = True             # podglady FP/FN
    missing_label_conf: float = 0.7      # FP powyzej -> possible_missing_labels/
    tree_profile: Optional[dict] = None  # {cover_threshold_m, step_m, n_plots}
    point_previews: int = 0              # podglady metod punktu (maski SAM)
    dem_alignment_plot: bool = False
    reference_point: Optional[dict] = None   # walidacja wzgledem punktu GT (geodetect.validation)


@dataclass
class Variant:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    derive: Optional[dict] = None    # {z_offset: multiview_check, from: <wariant>, span_m, step_m}


@dataclass
class Experiment:
    name: str
    path: Path
    base_config: Path
    description: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    variants: list[Variant] = field(default_factory=list)
    reference: Optional[str] = None
    metrics: MetricsSpec = field(default_factory=MetricsSpec)
    images: Optional[list[str]] = None

    @property
    def reference_variant(self) -> str:
        return self.reference or self.variants[0].name


def _short(v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "+".join(_short(x) for x in v) or "[]"
    if isinstance(v, dict):
        return ",".join(f"{k}={_short(x)}" for k, x in v.items()) or "{}"
    s = str(v)
    if "/" in s or "\\" in s or s.endswith((".pt", ".yaml", ".tif")):
        return Path(s).stem
    return s


def expand_variants(axes: Optional[dict], variants: Optional[list],
                    exclude: Optional[list] = None) -> list[Variant]:
    """Warianty z osi (iloczyn kartezjanski) i/lub jawnej listy. `exclude`: lista slownikow
    {pole: wartosc}; kombinacja z osi pasujaca do WSZYSTKICH par ktoregos z nich jest pomijana
    (np. odm_dsm x dem_alignment, gdy z_offset nie wplywa na wynik)."""
    out: list[Variant] = []
    if axes:
        keys = list(axes)
        for combo in itertools.product(*(axes[k] if isinstance(axes[k], list) else [axes[k]]
                                         for k in keys)):
            ov = dict(zip(keys, combo))
            if any(all(k in ov and ov[k] == v for k, v in ex.items()) for ex in exclude or []):
                continue
            name = " | ".join(_short(v) for v in combo)
            out.append(Variant(name, ov))
    for i, v in enumerate(variants or []):
        if not isinstance(v, dict):
            raise ValueError(f"Wariant #{i}: oczekiwano slownika {{name, set, derive}}")
        unknown = set(v) - {"name", "set", "derive"}
        if unknown:
            raise ValueError(f"Wariant #{i}: nieznane klucze {sorted(unknown)}")
        sets = dict(v.get("set") or {})
        name = v.get("name") or " | ".join(_short(x) for x in sets.values()) or f"wariant_{i}"
        out.append(Variant(str(name), sets, v.get("derive")))
    names = [v.name for v in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"Powtorzone nazwy wariantow: {sorted(dup)}")
    if not out:
        raise ValueError("Eksperyment bez wariantow (podaj axes lub variants)")
    return out


def load_experiment(path: Path | str) -> Experiment:
    import yaml
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - EXPERIMENT_KEYS
    if unknown:
        raise ValueError(f"{path.name}: nieznane klucze {sorted(unknown)} "
                         f"(dozwolone: {sorted(EXPERIMENT_KEYS)})")
    if "name" not in data:
        raise ValueError(f"{path.name}: brak 'name'")
    base = Path(data.get("base_config", "../config.yaml"))
    base = base if base.is_absolute() else (path.parent / base).resolve()
    m = data.get("metrics") or {}
    known = {f.name for f in fields(MetricsSpec)}
    if set(m) - known:
        raise ValueError(f"{path.name}: nieznane metryki {sorted(set(m) - known)}")
    exp = Experiment(str(data["name"]), path, base, str(data.get("description", "")).strip(),
                     dict(data.get("overrides") or {}),
                     expand_variants(data.get("axes"), data.get("variants"), data.get("exclude")),
                     data.get("reference"), MetricsSpec(**m), data.get("images"))
    if exp.reference and exp.reference not in [v.name for v in exp.variants]:
        raise ValueError(f"{path.name}: reference={exp.reference!r} nie jest nazwa wariantu "
                         f"({[v.name for v in exp.variants]})")
    for v in exp.variants:
        if v.derive and v.derive.get("from") not in [w.name for w in exp.variants]:
            raise ValueError(f"Wariant {v.name}: derive.from musi wskazywac inny wariant")
    return exp


def apply_override(cfg: PipelineConfig, dotted: str, value: Any, base_dir: Path) -> None:
    """Ustawia pole configu po sciezce z kropkami (np. detect.tiling.enabled)."""
    dotted = ALIASES.get(dotted, dotted)
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise ValueError(f"Nadpisanie {dotted!r}: brak pola {p!r}")
        obj = getattr(obj, p)
    last = parts[-1]
    if not is_dataclass(obj) or last not in {f.name for f in fields(obj)}:
        raise ValueError(f"Nadpisanie {dotted!r}: brak pola {last!r} w {type(obj).__name__}")
    cur = getattr(obj, last)
    if is_dataclass(cur) and isinstance(value, dict):
        for k, v in value.items():
            apply_override(cfg, f"{dotted}.{k}", v, base_dir)
        return
    if last in _PATH_FIELDS and value is not None:
        value = Path(value)
        if not value.is_absolute():
            value = (base_dir / value).resolve()
    setattr(obj, last, copy.deepcopy(value))


def variant_config(base: PipelineConfig, exp: Experiment, v: Variant) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    for k, val in {**exp.overrides, **v.overrides}.items():
        apply_override(cfg, k, val, exp.base_config.parent)
    return cfg


def detection_key(cfg: PipelineConfig) -> str:
    """Klucz cache detekcji: model + cala sekcja detect (geometria nie wplywa na YOLO)."""
    return json.dumps({"model": str(Path(cfg.model_path).resolve()),
                       "detect": _to_plain(cfg.detect)}, sort_keys=True)


# ───────────── cache detekcji ─────────────

class CachingDetector:
    """Interfejs Detector; wyniki w cache runnera wg detection_key(cfg)."""

    def __init__(self, store: "DetectionStore", cfg: PipelineConfig):
        self.store, self.cfg, self.key = store, cfg, detection_key(cfg)

    @property
    def names(self) -> dict[int, str]:
        return self.store.detector(self.key, self.cfg).names

    def detect(self, image_path) -> list[Detection]:
        return self.store.detect(self.key, self.cfg, Path(image_path))


class DetectionStore:
    """Detektory (model ladowany raz na klucz) i ich wyniki per zdjecie."""

    def __init__(self, factory=None):
        self.factory = factory or (lambda cfg: make_detector(cfg.model_path, cfg.detect))
        self.detectors: dict[str, Any] = {}
        self.results: dict[str, dict[str, list[Detection]]] = {}
        self.time_s: dict[str, float] = {}
        self.calls = 0

    def detector(self, key: str, cfg: PipelineConfig):
        if key not in self.detectors:
            self.detectors[key] = self.factory(cfg)
        return self.detectors[key]

    def detect(self, key: str, cfg: PipelineConfig, image_path: Path) -> list[Detection]:
        res = self.results.setdefault(key, {})
        if image_path.name not in res:
            det = self.detector(key, cfg)
            t0 = time.perf_counter()
            res[image_path.name] = det.detect(image_path)
            self.time_s[key] = self.time_s.get(key, 0.0) + time.perf_counter() - t0
            self.calls += 1
        return res[image_path.name]


# ───────────── metryki ─────────────

def _rms(v) -> float:
    v = np.asarray(list(v), float)
    return math.sqrt(float(np.mean(v ** 2))) if len(v) else float("nan")


def static_objects_of(batch: BatchResult, spec: MetricsSpec, min_views: int) -> list[AggregatedObject]:
    return sorted([o for o in batch.objects if o.cls_name in spec.static_classes
                   and o.confirmed and o.n_views >= min_views], key=lambda o: -o.n_views)


def object_metrics(batch: BatchResult, o: AggregatedObject) -> dict:
    res = ray_residuals(batch, [o])
    fit = fit_along_vs_tan(res)
    return {"object_id": o.object_id, "cls": o.cls_name, "status": o.status, "n_views": o.n_views,
            "rms": o.rms_m, "max": o.max_m, "along": _rms(r.along for r in res),
            "across": _rms(r.across for r in res),
            "slope": fit.slope if fit else float("nan"), "r": fit.r if fit else float("nan"),
            "x": o.x_2180, "y": o.y_2180, "h": o.h, "residuals": res, "fit": fit}


def pooled_rms(batch: BatchResult, objs: list[AggregatedObject]) -> float:
    centers = {o.object_id: o for o in objs}
    return _rms(math.hypot(gd.hit.x_2180 - centers[gd.object_id].x_2180,
                           gd.hit.y_2180 - centers[gd.object_id].y_2180)
                for gd in batch.detections if gd.object_id in centers)


def match_object(o, candidates: list, max_dist: float):
    """Najblizszy obiekt tej samej klasy w odleglosci <= max_dist (albo None)."""
    best, bd = None, max_dist
    for c in candidates:
        if c.cls_name != o.cls_name:
            continue
        d = math.hypot(c.x_2180 - o.x_2180, c.y_2180 - o.y_2180)
        if d <= bd:
            best, bd = c, d
    return best


def shift(o, ref) -> dict:
    if o is None or ref is None:
        return {"dist": float("nan"), "dx": float("nan"), "dy": float("nan"), "dh": float("nan")}
    dx, dy = o.x_2180 - ref.x_2180, o.y_2180 - ref.y_2180
    return {"dist": math.hypot(dx, dy), "dx": dx, "dy": dy, "dh": o.h - ref.h}


def _det_key(gd) -> tuple:
    return gd.image_name, gd.detection.cls_name, tuple(round(v, 1) for v in gd.detection.bbox)


def detection_shifts(batch: BatchResult, ref: BatchResult, classes: list[str]) -> dict[str, dict]:
    """Przesuniecie tych samych detekcji (zdjecie + bbox) wzgledem wariantu bazowego."""
    rmap = {_det_key(g): g for g in ref.detections}
    out = {}
    for cls in classes:
        d = [math.hypot(g.hit.x_2180 - rmap[_det_key(g)].hit.x_2180,
                        g.hit.y_2180 - rmap[_det_key(g)].hit.y_2180)
             for g in batch.detections if g.detection.cls_name == cls and _det_key(g) in rmap]
        out[cls] = {"n": len(d), "median": float(np.median(d)) if d else float("nan"),
                    "max": float(np.max(d)) if d else float("nan")}
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if tp and (p + r) else (0.0 if tp + fp + fn else float("nan"))
    return p, r, f1


def gt_evaluate(dets_by_img: dict, gt: dict, classes: list[str], thr: float, iou: float,
                images: list[str]) -> dict:
    """{cls: {tp, fp, fn, p, r, f1, fp_list, fn_list, tp_conf, fp_conf}} na zdjeciach `images`."""
    from .gt_labels import match_detections
    out = {}
    for cls in classes:
        tp = fp = fn = 0
        fp_list, fn_list, tp_conf, fp_conf = [], [], [], []
        for name in images:
            g = gt.get(name)
            if g is None:
                continue
            d = [x for x in dets_by_img.get(name, []) if x.cls_name == cls and x.conf >= thr]
            pairs, missed = match_detections(d, [b for b in g.boxes if b.cls_name == cls], iou)
            for det, m, _ in pairs:
                if m is None:
                    fp += 1
                    fp_list.append((name, det))
                    fp_conf.append(det.conf)
                else:
                    tp += 1
                    tp_conf.append(det.conf)
            fn += len(missed)
            fn_list += [(name, b) for b in missed]
        p, r, f1 = prf(tp, fp, fn)
        out[cls] = dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, fp_list=fp_list, fn_list=fn_list,
                        tp_conf=tp_conf, fp_conf=fp_conf)
    return out


def suggest_threshold(tp_conf: list[float], fp_conf: list[float], n_gt: int,
                      floor: float = 0.0) -> tuple[float, float]:
    """Prog conf maksymalizujacy F1 (przy remisie najnizszy). Zwraca (prog, F1)."""
    cands = sorted({floor, *[c for c in tp_conf + fp_conf if c >= floor]})
    best = (floor, -1.0)
    for t in cands:
        tp = sum(c >= t for c in tp_conf)
        fp = sum(c >= t for c in fp_conf)
        _, _, f1 = prf(tp, fp, n_gt - tp)
        f1 = 0.0 if math.isnan(f1) else f1
        if f1 > best[1] + 1e-12:
            best = (t, f1)
    return best


def cover_flags(ray_h: np.ndarray, nmt: np.ndarray, nmpt: np.ndarray, thr: float) -> dict:
    """Profil promienia przed trafieniem w NMT: czy przeszedl NAD pokryciem (NMPT - NMT > thr)
    i czy PRZEZ pokrycie (promien ponizej NMPT w takim miejscu)."""
    cover = nmpt - nmt
    ok = np.isfinite(cover)
    over = ok & (cover > thr)
    through = over & (ray_h <= nmpt)
    return {"over_cover": bool(over.any()), "through_cover": bool(through.any()),
            "max_cover_m": float(np.nanmax(np.where(ok, cover, np.nan))) if ok.any() else float("nan"),
            "obstacle_m": float(cover[over].max()) if over.any() else 0.0,
            "min_clearance_m": float((ray_h - nmpt)[over].min()) if over.any() else float("nan"),
            "max_cover_through_m": float(cover[through].max()) if through.any() else 0.0}


# ───────────── wyniki ─────────────

@dataclass
class VariantResult:
    variant: Variant
    cfg: PipelineConfig
    ctx: Any
    batch: BatchResult
    det_key: str
    time_total_s: float
    time_detect_s: float
    derived_offset: Optional[float] = None
    statics: list[dict] = field(default_factory=list)
    main: Optional[dict] = None
    pooled: float = float("nan")
    scan: Optional[ZScan] = None
    gt: dict = field(default_factory=dict)          # (prog, podzbior) -> gt_evaluate
    leaky: bool = False
    dir: Optional[Path] = None


def _versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "torch", "torchvision", "ultralytics", "opencv-python",
                "pyproj", "rasterio", "PyYAML", "matplotlib", "folium"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        out["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "brak"
    except Exception:
        pass
    return out


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.rstrip()
        except Exception:
            return ""
    dirty = [l for l in run("status", "--porcelain").splitlines()
             if not l[3:].startswith(("zdjecia/", "output/"))]
    return {"commit": run("rev-parse", "HEAD").strip(),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty_files": [l[3:] for l in dirty]}


_PL = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _safe(name: str) -> str:
    name = name.translate(_PL)
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_") or "wariant"


def _f(v, fmt="{:.2f}") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return fmt.format(v)


# ───────────── runner ─────────────

class ExperimentRunner:
    def __init__(self, exp: Experiment, out_root: Optional[Path] = None,
                 images: Optional[list[str]] = None, store: Optional[DetectionStore] = None):
        self.exp = exp
        self.base = PipelineConfig.from_file(exp.base_config)
        self.images = images or exp.images
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.out = (out_root or self.base.output_dir / "experiments") / _safe(exp.name) / stamp
        self.store = store or DetectionStore()
        self._dems: dict[str, Any] = {}
        self._recs: dict[str, Any] = {}
        self._surfaces: dict[str, Any] = {}
        self._segmenters: dict[str, Any] = {}
        self._gt: dict[str, Any] = {}
        self.results: dict[str, VariantResult] = {}
        self.tree: Optional[dict] = None
        self.validation = None

    # -- wspoldzielone zasoby --
    def dem(self, cfg):
        from .dem import resolve_dem
        k = json.dumps(_to_plain(cfg.dem), sort_keys=True)
        if k not in self._dems:
            self._dems[k] = resolve_dem(cfg.dem)
        return self._dems[k]

    def reconstruction(self, cfg):
        from .odm import OdmReconstruction
        k = str(cfg.reconstruction_path)
        if k not in self._recs:
            self._recs[k] = OdmReconstruction(cfg.reconstruction_path, cfg.reference_lla_path)
        return self._recs[k]

    def surface(self, cfg):
        from .geo import EnuFrame
        from .surface import area_bounds_2180, build_surface
        k = json.dumps([_to_plain(cfg.surface), _to_plain(cfg.dem), str(cfg.odm_dir)], sort_keys=True)
        if k not in self._surfaces:
            rec = self.reconstruction(cfg)
            fr = EnuFrame(rec.ref)
            cams = np.array([fr.enu_to_puwg92(*rec.camera(n)[1].C_enu)[:2] for n in rec.image_names])
            self._surfaces[k] = build_surface(cfg, area_bounds_2180(cfg, cams), ref_dem=self.dem(cfg))
        return self._surfaces[k]

    def segmenter(self, cfg):
        from .segment import Segmenter
        k = f"{cfg.point.sam_model}|{cfg.point.mask_cache_dir}"
        if k not in self._segmenters:
            self._segmenters[k] = Segmenter(cfg.point)
        return self._segmenters[k]

    def ground_truth(self, cfg, names: list[str], images: list[str]):
        from .gt_labels import load_ground_truth
        if cfg.eval.labels_dir is None:
            return {}
        k = f"{cfg.eval.labels_dir}|{','.join(names)}"
        if k not in self._gt:
            self._gt[k] = load_ground_truth(cfg.eval.labels_dir, images, cfg.photos_dir, names)
        return self._gt[k]

    # -- przebieg --
    def run(self) -> Path:
        from .pipeline import build_context
        self.out.mkdir(parents=True, exist_ok=True)
        order = self._order()
        for v in order:
            cfg = variant_config(self.base, self.exp, v)
            derived = None
            if v.derive:
                derived = self._derive_offset(v, cfg)
            det = CachingDetector(self.store, cfg)
            ctx = build_context(cfg, detector=det, dem=self.dem(cfg),
                                reconstruction=self.reconstruction(cfg), surface=self.surface(cfg))
            if cfg.point.needs_masks():
                ctx.segmenter = self.segmenter(cfg)
            t_det0 = self.store.time_s.get(det.key, 0.0)
            t0 = time.perf_counter()
            batch = run_batch(cfg, ctx, images=self.images)
            total = time.perf_counter() - t0
            vr = VariantResult(v, cfg, ctx, batch, det.key, total,
                               self.store.time_s.get(det.key, 0.0) - t_det0, derived)
            self.results[v.name] = vr
            log.warning("[%s] %s: %d detekcji, %.1f s (detekcja %.1f s)", self.exp.name, v.name,
                        len(batch.detections), total, vr.time_detect_s)
        for v in self.exp.variants:
            self._metrics(self.results[v.name])
        if self.exp.metrics.tree_profile is not None:
            self.tree = self._tree_profile()
        if self.exp.metrics.reference_point:
            from .validation import build_report
            self.validation = build_report(self)
        self._write()
        return self.out

    def _order(self) -> list[Variant]:
        """Warianty z derive po swoich zrodlach."""
        done, out, pending = set(), [], list(self.exp.variants)
        while pending:
            for v in list(pending):
                if not v.derive or v.derive["from"] in done:
                    out.append(v)
                    done.add(v.name)
                    pending.remove(v)
                    break
            else:
                raise ValueError("Cykl w derive.from")
        return out

    def _derive_offset(self, v: Variant, cfg: PipelineConfig) -> float:
        d = v.derive
        if d.get("z_offset") != "multiview_check":
            raise ValueError(f"Wariant {v.name}: obslugiwane derive.z_offset = multiview_check")
        src = self.results[d["from"]]
        scan = multiview_z_scan(src.batch, src.ctx, span_m=float(d.get("span_m", 15.0)),
                                step_m=float(d.get("step_m", 0.25)), eps_m=src.cfg.aggregate.eps_m,
                                cls_name=self.exp.metrics.static_classes[0],
                                min_views=self.exp.metrics.min_views_scan)
        src.batch.objects = aggregate(src.batch.detections, src.cfg.aggregate, src.cfg.size)
        if not scan.valid:
            raise ValueError(f"Wariant {v.name}: multiview_check bez obiektow statycznych z >= "
                             f"{self.exp.metrics.min_views_scan} widokami w wariancie {d['from']}")
        off = scan.refined_offset if d.get("use") == "refined" else scan.best_offset
        cfg.z_anchor.mode, cfg.z_anchor.offset_m = "fixed", off
        cfg.z_anchor.fixed_label = f"multiview_check z wariantu {d['from']}"
        return off

    def _metrics(self, vr: VariantResult) -> None:
        spec, b = self.exp.metrics, vr.batch
        statics = static_objects_of(b, spec, spec.min_views_static)
        vr.statics = [object_metrics(b, o) for o in statics]
        vr.main = vr.statics[0] if vr.statics else None
        vr.pooled = pooled_rms(b, [o for o in statics if o.n_views >= spec.min_views_scan])
        if spec.multiview:
            mv = spec.multiview
            vr.scan = multiview_z_scan(b, vr.ctx, span_m=float(mv.get("span_m", 3.0)),
                                       step_m=float(mv.get("step_m", 0.05)),
                                       eps_m=vr.cfg.aggregate.eps_m, cls_name=spec.static_classes[0],
                                       min_views=spec.min_views_scan,
                                       max_rays_per_detection=int(mv.get("max_rays", 30)))
            b.objects = aggregate(b.detections, vr.cfg.aggregate, vr.cfg.size)   # skan nadpisuje object_id
            vr.statics = [object_metrics(b, o) for o in static_objects_of(b, spec, spec.min_views_static)]
            vr.main = vr.statics[0] if vr.statics else None
        if spec.ground_truth and vr.cfg.eval.labels_dir is not None:
            names = list(self.store.detector(vr.det_key, vr.cfg).names.values())
            imgs = [r.image_name for r in b.images]
            gt = self.ground_truth(vr.cfg, names, imgs)
            train = set(vr.cfg.eval.train_images)
            labeled = [n for n in imgs if n in gt]
            subsets = {"wszystkie z GT": labeled,
                       "train (v7)": [n for n in labeled if n in train],
                       "valid+test": [n for n in labeled if n not in train]}
            dets = self.store.results.get(vr.det_key, {})
            vr.leaky = Path(vr.cfg.model_path).name in vr.cfg.eval.leaky_models
            for thr in vr.cfg.eval.thresholds:
                for sub, ims in subsets.items():
                    vr.gt[(thr, sub)] = gt_evaluate(dets, gt, names, thr, vr.cfg.eval.iou, ims)

    # -- drzewa / NMPT --
    def _tree_profile(self) -> Optional[dict]:
        from .raycast import SurfaceProjector, pixel_to_ray, raycast
        spec = self.exp.metrics.tree_profile or {}
        thr = float(spec.get("cover_threshold_m", 2.0))
        step = float(spec.get("step_m", 0.5))
        ref = self.results[self.exp.reference_variant]
        cfgs = {}
        for src in ("nmt_local", "nmpt_local", "odm_dsm"):
            c = copy.deepcopy(ref.cfg)
            c.surface.source = src
            try:
                cfgs[src] = self.surface(c)
            except (FileNotFoundError, ValueError) as exc:
                if src != "odm_dsm":
                    return {"error": f"brak {src}: {exc}"}
        nmt, nmpt, dsm = cfgs["nmt_local"], cfgs["nmpt_local"], cfgs.get("odm_dsm")
        frame = ref.ctx.frame
        p_nmt = SurfaceProjector(frame, nmt)
        p_dsm = SurfaceProjector(frame, dsm) if dsm is not None else None
        # odsetek obszaru lotu z pokryciem > thr (bbox kamer + 100 m, siatka NMT)
        cams = np.array([[r.camera.x_2180, r.camera.y_2180] for r in ref.batch.images])
        g = nmt.grid
        x0, x1 = cams[:, 0].min() - 100, cams[:, 0].max() + 100
        y0, y1 = cams[:, 1].min() - 100, cams[:, 1].max() + 100
        xs = np.arange(math.ceil((x0 - g.x0) / g.res_x), math.floor((x1 - g.x0) / g.res_x)) * g.res_x + g.x0 + g.res_x / 2
        ys = g.y0 - (np.arange(math.ceil((g.y0 - y1) / g.res_y), math.floor((g.y0 - y0) / g.res_y)) + 0.5) * g.res_y
        X, Y = np.meshgrid(xs, ys)
        cov = nmpt.sample_many(X.ravel(), Y.ravel()) - nmt.sample_many(X.ravel(), Y.ravel())
        area_frac = float(np.nanmean(cov > thr))
        others = {n: r for n, r in self.results.items()}
        rows, profiles = [], []
        for gd in ref.batch.detections:
            io, eo = ref.ctx.reconstruction.camera(gd.image_name)
            o, d = pixel_to_ray(gd.uv, io, eo)
            h_nmt = raycast(o, d, nmt, frame, ref.cfg.raycast)
            h_nmpt = raycast(o, d, nmpt, frame, ref.cfg.raycast)
            if h_nmt is None:
                continue
            ts = np.arange(0.0, h_nmt.range_m, step)
            P = o[None, :] + ts[:, None] * d[None, :]
            _, _, _, x, y, H, _, _ = p_nmt.project(P)
            zn, zp = nmt.sample_many(x, y), nmpt.sample_many(x, y)
            flags = cover_flags(H, zn, zp, thr)
            row = {"image": gd.image_name, "class": gd.detection.cls_name,
                   "conf": round(gd.detection.conf, 3), "u": round(gd.uv[0], 1),
                   "v": round(gd.uv[1], 1), "n_samples": len(ts), **flags,
                   "nmt_x": h_nmt.x_2180, "nmt_y": h_nmt.y_2180,
                   "d_nmt_nmpt_m": (math.hypot(h_nmpt.x_2180 - h_nmt.x_2180,
                                               h_nmpt.y_2180 - h_nmt.y_2180)
                                    if h_nmpt is not None else float("nan")),
                   "nmpt_hit_above_nmt_m": (h_nmpt.h - (nmt.sample(h_nmpt.x_2180, h_nmpt.y_2180) or np.nan)
                                            if h_nmpt is not None else float("nan"))}
            k = _det_key(gd)
            for n, r in others.items():      # przesuniecie trafienia kazdego wariantu wzgl. NMT
                m = next((g for g in r.batch.detections if _det_key(g) == k), None)
                row[f"d_vs_nmt[{n}]"] = (math.hypot(m.hit.x_2180 - h_nmt.x_2180,
                                                    m.hit.y_2180 - h_nmt.y_2180)
                                         if m is not None else float("nan"))
            rows.append(row)
            profiles.append((gd, o, d, h_nmt, h_nmpt, flags))
        return {"threshold_m": thr, "area_cover_frac": area_frac, "n_rays": len(rows),
                "rows": rows, "profiles": profiles, "nmt": nmt, "nmpt": nmpt, "dsm": dsm,
                "p_nmt": p_nmt, "p_dsm": p_dsm, "frame": frame,
                "n_plots": int(spec.get("n_plots", 3))}

    def _profile_series(self, tree, gd, o, d, h_nmt, h_nmpt) -> dict:
        t_end = h_nmt.range_m * 1.12
        ts = np.linspace(0, t_end, 600)
        P = o[None, :] + ts[:, None] * d[None, :]
        _, _, _, x, y, H, _, _ = tree["p_nmt"].project(P)
        cam = tree["p_nmt"].project(o[None, :])
        dist = np.hypot(x - cam[3][0], y - cam[4][0])
        zn, zp = tree["nmt"].sample_many(x, y), tree["nmpt"].sample_many(x, y)
        zd = None
        if tree["p_dsm"] is not None:
            sx, sy, Hc, _, _, Hn, _, _ = tree["p_dsm"].project(P)
            zd = tree["dsm"].sample_many(sx, sy) + (Hn - Hc)       # ODM -> npm
        keep = H <= np.nanmax(zp) + 15                              # dolna czesc promienia
        if keep.sum() < 20:
            keep[:] = True
        i0 = int(np.argmax(keep))
        sl = slice(i0, None)

        def hd(h):
            return None if h is None else float(np.hypot(h.x_2180 - cam[3][0], h.y_2180 - cam[4][0]))
        return {"d": dist[sl], "ray_h": H[sl], "nmt": zn[sl], "nmpt": zp[sl],
                "dsm": None if zd is None else zd[sl],
                "hits": {"NMT": hd(h_nmt), "NMPT": hd(h_nmpt)},
                "title": f"Profil promienia: {gd.image_name}, {gd.detection.cls_name} "
                         f"(conf {gd.detection.conf:.2f})"}

    # -- zapis --
    def _write(self) -> None:
        import yaml
        from .export import export_all
        from . import expplots as P
        spec = self.exp.metrics
        ref = self.results[self.exp.reference_variant]
        meta = {"experiment": self.exp.name, "definition": str(self.exp.path),
                "base_config": str(self.exp.base_config), "created": datetime.now().isoformat(),
                "git": _git(), "versions": _versions(),
                "detection_calls": self.store.calls}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        (self.out / "experiment.yaml").write_text(self.exp.path.read_text(encoding="utf-8"),
                                                  encoding="utf-8")
        for name, vr in self.results.items():
            vr.dir = self.out / "variants" / _safe(name)
            vr.dir.mkdir(parents=True, exist_ok=True)
            (vr.dir / "config.yaml").write_text(
                yaml.safe_dump(vr.cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
            export_all(vr.batch, vr.dir)
            if getattr(vr.ctx, "alignment", None) is not None:
                (vr.dir / "alignment.json").write_text(json.dumps(vr.ctx.alignment.result.report(), indent=2),
                                                       encoding="utf-8")

        plots = {}
        mains = [{"variant": n, **({k: vr.main[k] for k in ("rms", "max", "along", "across")}
                                   if vr.main else {"rms": None})}
                 for n, vr in self.results.items()]
        plots["static"] = P.plot_static(mains, self.out / "static_objects.png",
                                        f"główny {spec.static_classes[0]}")
        plots["along"] = P.plot_along([(n, vr.main["residuals"], vr.main["fit"])
                                       for n, vr in self.results.items() if vr.main],
                                      self.out / "along_vs_tan.png")
        if spec.multiview:
            plots["multiview"] = P.plot_multiview(
                [(n, vr.scan) for n, vr in self.results.items()],
                ref.ctx.z_calibration.offset, self.out / "multiview.png")
        if spec.dem_alignment_plot:
            al = next((vr.ctx.z_calibration.alignment for vr in self.results.values()
                       if vr.ctx.z_calibration.alignment is not None), None)
            if al is not None:
                plots["dem_alignment"] = P.plot_dem_alignment(al, self.out / "dem_alignment.png")
        if any(vr.gt for vr in self.results.values()):
            thr0 = min(self.base.eval.thresholds)
            rows = [{"variant": n, "subset": sub, "cls": cls, "p": m["p"], "r": m["r"], "f1": m["f1"]}
                    for n, vr in self.results.items() for (t, sub), res in vr.gt.items()
                    if t == thr0 and sub != "wszystkie z GT" for cls, m in res.items()]
            plots["pr"] = P.plot_pr(rows, self.out / "detection_pr.png")
            self._gt_previews()
        if self.tree and "rows" in self.tree:
            self._tree_plots(plots)
        if spec.point_previews:
            self._point_previews(spec.point_previews)
        if self.validation is not None:
            from .validation import write_outputs
            write_outputs(self.validation, self, self.out)
        self._write_csv()
        (self.out / "summary.md").write_text(self._summary_md(plots, meta), encoding="utf-8")

    def _gt_previews(self) -> None:
        from . import expplots as P
        spec = self.exp.metrics
        for n, vr in self.results.items():
            if not vr.gt:
                continue
            thr0 = min(vr.cfg.eval.thresholds)
            res = vr.gt[(thr0, "wszystkie z GT")]
            gt = self.ground_truth(vr.cfg, list(self.store.detector(vr.det_key, vr.cfg).names.values()),
                                   [r.image_name for r in vr.batch.images])
            for cls, m in res.items():
                for k, (img, det) in enumerate(m["fp_list"]):
                    lines = [img, f"wariant: {n}", f"FP {cls} conf {det.conf:.2f}",
                             f"box {det.bbox[2] - det.bbox[0]:.0f}x{det.bbox[3] - det.bbox[1]:.0f} px"]
                    gtb = [b.bbox for b in gt[img].boxes]
                    if spec.gt_previews:
                        P.detection_preview(vr.cfg.photos_dir / img,
                                            vr.dir / "gt_previews" / f"fp_{cls}_{k:02d}_{Path(img).stem}.png",
                                            det.bbox, lines, gtb)
                    if det.conf > spec.missing_label_conf:
                        P.detection_preview(vr.cfg.photos_dir / img,
                                            self.out / "possible_missing_labels" /
                                            f"{_safe(n)}_{cls}_{Path(img).stem}_{det.conf:.2f}.png",
                                            det.bbox, lines + ["mozliwa brakujaca etykieta?"], gtb)
                if spec.gt_previews:
                    for k, (img, b) in enumerate(m["fn_list"]):
                        P.detection_preview(vr.cfg.photos_dir / img,
                                            vr.dir / "gt_previews" / f"fn_{cls}_{k:02d}_{Path(img).stem}.png",
                                            b.bbox, [img, f"wariant: {n}", f"FN {cls} (brak detekcji)"],
                                            [x.bbox for x in gt[img].boxes], color=(80, 80, 80))

    def _tree_plots(self, plots: dict) -> None:
        from . import expplots as P
        t = self.tree
        scored = sorted(t["profiles"], key=lambda p: (not p[5]["through_cover"], not p[5]["over_cover"],
                                                      p[5]["min_clearance_m"] if np.isfinite(p[5]["min_clearance_m"]) else 1e9,
                                                      -(np.hypot(p[4].x_2180 - p[3].x_2180, p[4].y_2180 - p[3].y_2180)
                                                        if p[4] is not None else 0)))
        t["plots"] = []
        for k, (gd, o, d, hn, hp, _) in enumerate(scored[:t["n_plots"]]):
            prof = self._profile_series(t, gd, o, d, hn, hp)
            t["plots"].append(P.plot_ray_profile(
                prof, self.out / f"ray_profile_{k + 1}_{Path(gd.image_name).stem}.png"))

    def _point_previews(self, n: int) -> None:
        from . import expplots as P
        from .segment import clean_mask
        vr = next((r for r in self.results.values() if r.cfg.point.needs_masks()), None)
        if vr is None:
            return
        seg = self.segmenter(vr.cfg)
        b = vr.batch
        dets = sorted(b.detections, key=lambda g: (g.detection.cls_name, g.incidence_deg))
        tents = [g for g in dets if g.detection.cls_name in self.exp.metrics.static_classes]
        others = [g for g in dets if g.detection.cls_name not in self.exp.metrics.static_classes]
        pick = [tents[int(i)] for i in np.linspace(0, len(tents) - 1, max(1, n - 2))] if tents else []
        pick += [others[int(i)] for i in np.linspace(0, len(others) - 1, 2)] if others else []
        for k, gd in enumerate(pick):
            io, eo = vr.ctx.reconstruction.camera(gd.image_name)
            raw = seg.masks(vr.cfg.photos_dir / gd.image_name, [gd.detection.bbox])[0]
            mask, q = clean_mask(raw, gd.detection.bbox, vr.cfg.point) if raw is not None else (None, None)
            P.point_methods_preview(vr.cfg.photos_dir / gd.image_name,
                                    self.out / "point_previews" /
                                    f"preview_{k + 1}_{gd.detection.cls_name}_{Path(gd.image_name).stem}.png",
                                    gd.detection.bbox, io, eo, vr.cfg.point, mask, q,
                                    [f"{gd.image_name} {gd.detection.cls_name}",
                                     f"incidence {gd.incidence_deg:.1f} deg"])

    def _classes(self) -> list[str]:
        return sorted({gd.detection.cls_name for vr in self.results.values() for gd in vr.batch.detections})

    def _write_csv(self) -> None:
        ref = self.results[self.exp.reference_variant]
        classes = self._classes()
        rows = []
        for n, vr in self.results.items():
            b = vr.batch
            r = {"variant": n, "z_offset": vr.ctx.z_calibration.offset,
                 "z_method": vr.ctx.z_calibration.method,
                 "surface": vr.cfg.surface.source, "model": Path(vr.cfg.model_path).name,
                 "point_method": vr.cfg.point.method, "tiling": vr.cfg.detect.tiling.enabled,
                 "time_total_s": vr.time_total_s, "time_detect_s": vr.time_detect_s,
                 "misses": len(b.misses), "pooled_rms": vr.pooled}
            for c in classes:
                r[f"det_{c}"] = sum(1 for g in b.detections if g.detection.cls_name == c)
                r[f"obj_{c}_confirmed"] = sum(1 for o in b.objects if o.cls_name == c and o.confirmed)
                r[f"obj_{c}_single"] = sum(1 for o in b.objects if o.cls_name == c and not o.confirmed)
            if vr.main:
                for k in ("n_views", "rms", "max", "along", "across", "slope", "r", "x", "y", "h"):
                    r[f"main_{k}"] = vr.main[k]
                s = shift(self._obj(vr.main), self._obj(ref.main))
                r.update({f"main_shift_{k}": v for k, v in s.items()})
            if vr.scan is not None:
                r["multiview_offset"] = vr.scan.refined_offset
                r["multiview_minus_z"] = vr.scan.refined_offset - vr.ctx.z_calibration.offset
            for (t, sub), res in vr.gt.items():
                for c, m in res.items():
                    for k in ("tp", "fp", "fn", "p", "r", "f1"):
                        r[f"gt_{t:g}_{sub}_{c}_{k}"] = m[k]
            rows.append(r)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(self.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(self.out / "static_objects.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "object_id", "class", "n_views", "rms_m", "max_m", "along_m",
                        "across_m", "slope", "r", "x_2180", "y_2180", "h", "shift_m", "dx", "dy", "dh"])
            for n, vr in self.results.items():
                for s in vr.statics:
                    m = match_object(self._obj(s), [self._obj(x) for x in ref.statics],
                                     3 * vr.cfg.aggregate.eps_m)
                    sh = shift(self._obj(s), m)
                    w.writerow([n, s["object_id"], s["cls"], s["n_views"], *(round(s[k], 4) for k in
                                ("rms", "max", "along", "across", "slope", "r", "x", "y", "h")),
                                *(round(sh[k], 4) for k in ("dist", "dx", "dy", "dh"))])
        if self.tree and "rows" in self.tree:
            rows = self.tree["rows"]
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(self.out / "tree_profile.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    @staticmethod
    def _obj(m: Optional[dict]):
        if m is None:
            return None
        from types import SimpleNamespace
        return SimpleNamespace(cls_name=m["cls"], x_2180=m["x"], y_2180=m["y"], h=m["h"])

    # -- summary.md --
    def _summary_md(self, plots: dict, meta: dict) -> str:
        e, spec = self.exp, self.exp.metrics
        ref = self.results[e.reference_variant]
        classes = self._classes()
        L = [f"# Eksperyment: {e.name}", ""]
        if e.description:
            L += [e.description, ""]
        g = meta["git"]
        L += [f"- definicja: `{e.path.relative_to(ROOT) if e.path.is_relative_to(ROOT) else e.path}`, "
              f"bazowy config: `{e.base_config.name}`",
              f"- commit: `{g['commit'][:10]}` ({g['branch']})"
              + (f", **niezacommitowane zmiany**: {', '.join(g['dirty_files'][:8])}"
                 if g["dirty_files"] else ", drzewo czyste"),
              "- wersje: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items()
                                       if k not in ("platform",)),
              f"- wariant bazowy (przesuniecia): **{e.reference_variant}**",
              f"- wywolania detektora: {meta['detection_calls']} (cache miedzy wariantami)", ""]
        L += ["## Warianty", "", "| wariant | nadpisania |", "|---|---|"]
        for v in e.variants:
            ov = ", ".join(f"`{k}={_short(x)}`" for k, x in v.overrides.items()) or "–"
            if v.derive:
                ov += f", z_offset z multiview_check wariantu `{v.derive['from']}`"
            L.append(f"| {v.name} | {ov} |")
        L.append("")

        qcs: dict = {}
        for n, vr in self.results.items():
            q = getattr(vr.ctx, "qc", None)
            if q is not None:
                qcs.setdefault(id(q), (q, str(vr.cfg.odm_dir), []))[2].append(n)
        for q, odm, used in qcs.values():
            L += [f"## Kontrola jakości rekonstrukcji — `{Path(odm).name}` (z_offset wariantów: "
                  f"{', '.join(used[:4])}{' …' if len(used) > 4 else ''})", ""]
            L += [f"- {x.strip()}" for x in q.lines()] + [""]
        seen = set()
        for n, vr in self.results.items():
            al = getattr(vr.ctx, "alignment", None)
            if al is None or id(al) in seen:
                continue
            seen.add(id(al))
            from .align import alignment_md
            used = [m for m, v in self.results.items() if getattr(v.ctx, "alignment", None) is al]
            L += alignment_md(al, f"Wyrównanie 7p do NMPT — {', '.join(used)}")

        so = spec.static_classes[0]
        L += [f"## Wyniki — główny obiekt statyczny ({so}, confirmed, najwięcej widoków)", "",
              f"| wariant | z_offset [m] | widoki | RMS [m] | max [m] | wzdłuż [m] | w poprzek [m] | "
              f"nachyl. vs tan(inc) [m] (r) | RMS zbiorczy [m] | przesunięcie vs {e.reference_variant} [m] "
              f"(dx, dy, dH) | czas: detekcja / całość [s] |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, vr in self.results.items():
            m = vr.main
            s = shift(self._obj(m), self._obj(ref.main))
            sh = "–" if m is None else (f"{_f(s['dist'])} ({_f(s['dx'], '{:+.2f}')}, "
                                        f"{_f(s['dy'], '{:+.2f}')}, {_f(s['dh'], '{:+.2f}')})")
            L.append(f"| {n} | {_f(vr.ctx.z_calibration.offset)} | {m['n_views'] if m else '–'} | "
                     f"{_f(m['rms']) if m else '–'} | {_f(m['max']) if m else '–'} | "
                     f"{_f(m['along']) if m else '–'} | {_f(m['across']) if m else '–'} | "
                     + (f"{_f(m['slope'], '{:+.2f}')} ({_f(m['r'], '{:+.2f}')})" if m else "–")
                     + f" | {_f(vr.pooled)} | {sh} | {vr.time_detect_s:.1f} / {vr.time_total_s:.1f} |")
        L += ["", "RMS zbiorczy: po wszystkich obiektach statycznych z ≥ "
              f"{spec.min_views_scan} widokami. Czas detekcji 0 = detekcje z cache (inny wariant).", ""]

        L += ["## Liczby detekcji i obiektów", "",
              "| wariant | " + " | ".join(f"detekcje {c}" for c in classes) + " | "
              + " | ".join(f"obiekty {c} (confirmed / single_view)" for c in classes)
              + " | bez przecięcia |", "|---" * (2 * len(classes) + 2) + "|"]
        for n, vr in self.results.items():
            b = vr.batch
            L.append(f"| {n} | " + " | ".join(str(sum(1 for g in b.detections if g.detection.cls_name == c))
                                               for c in classes) + " | "
                     + " | ".join(f"{sum(1 for o in b.objects if o.cls_name == c and o.confirmed)} / "
                                  f"{sum(1 for o in b.objects if o.cls_name == c and not o.confirmed)}"
                                  for c in classes) + f" | {len(b.misses)} |")
        L.append("")

        L += [f"## Wszystkie obiekty statyczne ({', '.join(spec.static_classes)}, confirmed, "
              f"≥ {spec.min_views_static} widoki)", "",
              "| wariant | obiekt | widoki | RMS [m] | max [m] | wzdłuż / w poprzek [m] | X, Y EPSG:2180 | "
              "H [m npm] | przesunięcie vs bazowy [m] |", "|---|---|---|---|---|---|---|---|---|"]
        for n, vr in self.results.items():
            for s in vr.statics:
                mm = match_object(self._obj(s), [self._obj(x) for x in ref.statics],
                                  3 * vr.cfg.aggregate.eps_m)
                sh = shift(self._obj(s), mm)
                L.append(f"| {n} | {s['cls']} #{s['object_id']} | {s['n_views']} | {_f(s['rms'])} | "
                         f"{_f(s['max'])} | {_f(s['along'])} / {_f(s['across'])} | "
                         f"{s['x']:.2f}, {s['y']:.2f} | {s['h']:.2f} | {_f(sh['dist'])} |")
        L.append("")

        if spec.multiview:
            L += ["## multiview_check (diagnostyka: z_offset minimalizujący rozrzut)", "",
                  "| wariant | z_offset wariantu | optimum | różnica | min RMS [m] |", "|---|---|---|---|---|"]
            for n, vr in self.results.items():
                s = vr.scan
                if s is None:
                    continue
                z = vr.ctx.z_calibration.offset
                L.append(f"| {n} | {z:.2f} | {_f(s.refined_offset)} | "
                         f"{_f(s.refined_offset - z, '{:+.2f}')} | {_f(s.best_rms)} |")
            L.append("")

        mc = [c for c in spec.moving_classes if c in classes]
        if mc:
            L += [f"## Klasy ruchome ({', '.join(mc)}) — tylko liczby i przesunięcia", "",
                  "Osoby poruszały się między zdjęciami, więc rozrzut klastra nie jest miarą jakości.", "",
                  f"| wariant | klasa | detekcje | obiekty (confirmed / single) | przesunięcie detekcji "
                  f"vs {e.reference_variant}: mediana / max [m] (n — detekcje o identycznym bboxie) | przesunięcie środka największego "
                  f"klastra [m] |", "|---|---|---|---|---|---|"]
            for n, vr in self.results.items():
                ds = detection_shifts(vr.batch, ref.batch, mc)
                for c in mc:
                    ob = [o for o in vr.batch.objects if o.cls_name == c]
                    rb = [o for o in ref.batch.objects if o.cls_name == c]
                    big = max(ob, key=lambda o: o.n_views) if ob else None
                    rbig = max(rb, key=lambda o: o.n_views) if rb else None
                    L.append(f"| {n} | {c} | {sum(1 for g in vr.batch.detections if g.detection.cls_name == c)} | "
                             f"{sum(1 for o in ob if o.confirmed)} / {sum(1 for o in ob if not o.confirmed)} | "
                             f"{_f(ds[c]['median'])} / {_f(ds[c]['max'])} ({ds[c]['n']}) | "
                             f"{_f(shift(big, rbig)['dist'])} |")
            L.append("")

        if any(vr.gt for vr in self.results.values()):
            L += self._gt_md()

        if self.tree is not None:
            L += self._tree_md()

        if self.validation is not None:
            from .validation import summary_md
            L += summary_md(self.validation, self)

        L += ["## Pliki", "", "- `summary.csv` — jeden wiersz na wariant (wszystkie metryki)",
              "- `static_objects.csv` — obiekty statyczne per wariant",
              "- `variants/<wariant>/` — config.yaml i eksport (GeoJSON, CSV, mapa, spread.png)",
              "- `meta.json` — commit, wersje bibliotek, GPU"]
        for k, p in plots.items():
            if p:
                L.append(f"- `{Path(p).name}`")
        if self.tree and self.tree.get("plots"):
            L += [f"- `{p.name}`" for p in self.tree["plots"]]
        L.append("")
        return "\n".join(L)

    def _gt_md(self) -> list[str]:
        L = ["## Detekcja względem etykiet (ground truth)", ""]
        anyvr = next(vr for vr in self.results.values() if vr.gt)
        ev = anyvr.cfg.eval
        L += [f"Etykiety: `{ev.labels_dir}`, IoU {ev.iou}. Podzbiory: **wszystkie z GT**, "
              f"**train (v7)** — zdjęcia ze splitu treningowego eksportu "
              f"({len(ev.train_images)} zdjęć), **valid+test** — zbiór uczciwy.", ""]
        leaky = sorted({Path(vr.cfg.model_path).name for vr in self.results.values() if vr.leaky})
        if leaky:
            L += [f"> **Uwaga — wyciek danych:** modele {', '.join(leaky)} były trenowane na tym "
                  "eksporcie. Wyniki na podzbiorach „wszystkie z GT” i „train (v7)” są dla nich "
                  "zawyżone; miarodajny jest tylko **valid+test** (mało zdjęć — duża niepewność).", ""]
        classes = sorted({c for vr in self.results.values() for res in vr.gt.values() for c in res})
        for thr in anyvr.cfg.eval.thresholds:
            L += [f"### conf ≥ {thr:g}", "",
                  "| wariant | podzbiór (zdjęcia) | " + " | ".join(f"{c}: P / R / F1 (TP/FP/FN)" for c in classes)
                  + " |", "|---|---|" + "---|" * len(classes)]
            for n, vr in self.results.items():
                for (t, sub), res in vr.gt.items():
                    if t != thr:
                        continue
                    nim = len({img for c in res.values() for img, _ in c["fn_list"] + c["fp_list"]})
                    cells = [f"{_f(res[c]['p'])} / {_f(res[c]['r'])} / {_f(res[c]['f1'])} "
                             f"({res[c]['tp']}/{res[c]['fp']}/{res[c]['fn']})" for c in classes]
                    flag = " ⚠" if vr.leaky and sub != "valid+test" else ""
                    L.append(f"| {n} | {sub}{flag} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["### Rozkład conf TP / FP i proponowany próg (max F1, wszystkie z GT)", "",
              "| wariant | klasa | conf TP: min / mediana | conf FP | próg (F1) | próg (F1) valid+test |",
              "|---|---|---|---|---|---|"]
        for n, vr in self.results.items():
            if not vr.gt:
                continue
            thr0 = min(vr.cfg.eval.thresholds)
            res, hon = vr.gt[(thr0, "wszystkie z GT")], vr.gt[(thr0, "valid+test")]
            for c, m in res.items():
                tpc, fpc = m["tp_conf"], m["fp_conf"]
                t, f1 = suggest_threshold(tpc, fpc, m["tp"] + m["fn"], thr0)
                h = hon[c]
                th, f1h = suggest_threshold(h["tp_conf"], h["fp_conf"], h["tp"] + h["fn"], thr0)
                L.append(f"| {n} | {c} | {_f(min(tpc) if tpc else None)} / "
                         f"{_f(float(np.median(tpc)) if tpc else None)} | "
                         f"{', '.join(f'{x:.2f}' for x in sorted(fpc)) or '–'} | {t:.2f} ({f1:.2f}) | "
                         f"{th:.2f} ({f1h:.2f}) |")
        L.append("")
        miss = sorted((self.out / "possible_missing_labels").glob("*.png")) \
            if (self.out / "possible_missing_labels").exists() else []
        L += [f"### Możliwe brakujące etykiety (FP z conf > {self.exp.metrics.missing_label_conf})", ""]
        if miss:
            L += [f"- `possible_missing_labels/{p.name}`" for p in miss]
        else:
            L.append("Brak.")
        L += ["", "Podglądy wszystkich FP / FN: `variants/<wariant>/gt_previews/`.", ""]
        return L

    def _tree_md(self) -> list[str]:
        t = self.tree
        L = ["## Promienie a pokrycie terenu (NMPT − NMT)", ""]
        if "error" in t:
            return L + [f"Pominięte: {t['error']}", ""]
        thr = t["threshold_m"]
        over = [r for r in t["rows"] if r["over_cover"]]
        through = [r for r in t["rows"] if r["through_cover"]]
        L += [f"Sprawdzono **{t['n_rays']} promieni** (detekcje wariantu {self.exp.reference_variant}). "
              f"Pokrycie NMPT − NMT > {thr:g} m zajmuje **{100 * t['area_cover_frac']:.1f}%** obszaru "
              f"lotu (bbox kamer + 100 m).", "",
              f"- promienie przechodzące **nad** pokryciem > {thr:g} m przed trafieniem w NMT: {len(over)}",
              f"- promienie przechodzące **przez** pokrycie (poniżej NMPT): {len(through)}", ""]
        if over:
            L += [f"Promienie nad pokryciem > {thr:g} m (przed trafieniem w NMT):", "",
                  "| zdjęcie | klasa | conf | wys. przeszkody (NMPT − NMT) [m] | min. prześwit promienia "
                  "nad NMPT [m] | przez pokrycie | trafienie NMT vs NMPT [m] | trafienie NMPT nad NMT [m] |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in sorted(over, key=lambda r: r["min_clearance_m"]):
                L.append(f"| {r['image']} | {r['class']} | {r['conf']:.2f} | {r['obstacle_m']:.1f} | "
                         f"{_f(r['min_clearance_m'], '{:.1f}')} | {'TAK' if r['through_cover'] else 'nie'} | "
                         f"{_f(r['d_nmt_nmpt_m'])} | {_f(r['nmpt_hit_above_nmt_m'])} |")
            L.append("")
        if not through:
            L.append(f"**Żaden promień nie przeszedł przez pokrycie** (koronę drzewa / budynek, "
                     f"NMPT − NMT > {thr:g} m) przed trafieniem w grunt"
                     + (" — promienie przechodzące nad pokryciem miały nad nim prześwit, więc NMT "
                        "i NMPT dają to samo trafienie." if over else
                        " — żaden nie przechodził nawet nad takim pokryciem."))
        d = [r["d_nmt_nmpt_m"] for r in t["rows"] if np.isfinite(r["d_nmt_nmpt_m"])]
        if d:
            L += ["", f"Odległość pozioma trafienia NMT vs NMPT: mediana {np.median(d):.2f} m, "
                      f"max {np.max(d):.2f} m (n={len(d)})."]
        if t.get("plots"):
            L += ["", "Profile promienia (NMT, NMPT, DSM ODM, linia promienia): "
                  + ", ".join(f"`{p.name}`" for p in t["plots"])]
        L += ["", "Pełna lista: `tree_profile.csv`.", ""]
        return L


def run_experiment(path: Path | str, out_root: Optional[Path] = None,
                   images: Optional[list[str]] = None) -> Path:
    exp = load_experiment(path)
    return ExperimentRunner(exp, out_root, images).run()
