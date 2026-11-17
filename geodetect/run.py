"""API przebiegu dla GUI i CLI: run_pipeline(cfg, progress_cb, cancel) -> RunResult.

Bez print — postep przez callback, diagnostyka przez logging. Przerwanie: obiekt CancelToken
(albo dowolna funkcja bez argumentow zwracajaca True), sprawdzany przed kazdym zdjeciem i
miedzy etapami; przerwany przebieg zwraca RunResult(cancelled=True) bez eksportu.

Etapy (Progress.stage): context (NMT, rekonstrukcja, kalibracja Z / 7p, QC, model) ->
images (detekcja + ray casting, i/n) -> aggregate -> export -> done.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

from .aggregate import AggregatedObject
from .batch import BatchResult, Cancelled, run_batch
from .config import PipelineConfig
from .pipeline import GeoDetection, PipelineContext, build_context

log = logging.getLogger(__name__)

STAGES = ("context", "images", "aggregate", "export", "done")
# udzial etapow w pasku postepu (0..1)
_SPAN = {"context": (0.0, 0.10), "images": (0.10, 0.90), "aggregate": (0.90, 0.92),
         "export": (0.92, 1.0), "done": (1.0, 1.0)}


@dataclass
class Progress:
    stage: str             # jeden z STAGES
    i: int                 # krok w etapie (od 1), 0 = poczatek etapu
    n: int                 # liczba krokow etapu (0 = nieznana)
    message: str
    fraction: float        # postep calkowity 0..1


ProgressCb = Callable[[Progress], None]


class CancelToken:
    """Flaga przerwania (bezpieczna miedzy watkami: GUI ustawia, watek roboczy sprawdza)."""

    def __init__(self):
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def __call__(self) -> bool:
        return self._ev.is_set()


@dataclass
class RunResult:
    objects: list[AggregatedObject] = field(default_factory=list)
    detections: list[GeoDetection] = field(default_factory=list)
    qc: Optional[object] = None                 # qc.QcReport
    export: dict[str, Path] = field(default_factory=dict)   # klucz -> plik (report_html, gpkg, ...)
    out_dir: Optional[Path] = None
    batch: Optional[BatchResult] = None
    cancelled: bool = False
    error: Optional[str] = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.cancelled and self.error is None


def run_dir(cfg: PipelineConfig, name: Optional[str] = None) -> Path:
    """Katalog przebiegu: <output_dir>/runs/<RRRR-MM-DD_HHMMSS>[_name]."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(cfg.output_dir) / "runs" / (stamp + (f"_{name}" if name else ""))


def run_pipeline(cfg: PipelineConfig, progress_cb: Optional[ProgressCb] = None,
                 cancel: Union[CancelToken, Callable[[], bool], None] = None,
                 images: Optional[list[str]] = None, all_images: bool = False,
                 out_dir: Optional[Path] = None, export: bool = True,
                 ctx: Optional[PipelineContext] = None, raise_errors: bool = False) -> RunResult:
    """Pelny przebieg: kontekst -> zdjecia -> agregacja -> eksport.

    images: lista zdjec (None = cfg.images, chyba ze all_images=True: wszystkie z rekonstrukcji).
    out_dir: None = run_dir(cfg). raise_errors=False: wyjatek -> RunResult(error=...) (GUI);
    CLI moze chciec raise_errors=True, zeby zobaczyc traceback."""
    t0 = time.perf_counter()
    is_cancelled = cancel if cancel is not None else (lambda: False)

    def report(stage: str, i: int = 0, n: int = 0, msg: str = "") -> None:
        if progress_cb is None:
            return
        a, b = _SPAN[stage]
        frac = a + (b - a) * (i / n if n else 0.0)
        try:
            progress_cb(Progress(stage, i, n, msg, min(1.0, frac)))
        except Exception:                       # blad w GUI nie przerywa obliczen
            log.exception("progress_cb zglosil wyjatek")

    def check() -> None:
        if is_cancelled():
            raise Cancelled("przerwano na zadanie")

    res = RunResult()
    try:
        report("context", 0, 0, "Przygotowanie: NMT, rekonstrukcja ODM, kalibracja, model")
        check()
        if ctx is None:
            ctx = build_context(cfg)
        res.qc = getattr(ctx, "qc", None)
        check()
        todo = None if all_images else (images if images is not None else cfg.images)
        batch = run_batch(cfg, ctx, images=todo,
                          progress_cb=lambda i, n, name: report("images", i, n, name),
                          cancel=is_cancelled)
        report("aggregate", 1, 1, f"Agregacja: {len(batch.objects)} obiektow")
        res.batch, res.objects, res.detections = batch, batch.objects, batch.detections
        res.qc = batch.qc if batch.qc is not None else res.qc
        check()
        if export:
            from .export import export_run
            res.out_dir = Path(out_dir) if out_dir is not None else run_dir(cfg)
            report("export", 0, 1, f"Eksport do {res.out_dir}")
            res.export = export_run(batch, res.out_dir, meta={
                "config_file": str(getattr(cfg, "source_path", "") or ""),
                "images": todo if todo is not None else "all"})
            report("export", 1, 1, "Eksport zakonczony")
        report("done", 1, 1, "Gotowe")
    except Cancelled as exc:
        res.cancelled = True
        res.error = None
        log.warning("Przebieg przerwany: %s", exc)
    except Exception as exc:
        if raise_errors:
            raise
        log.exception("Przebieg zakonczony bledem")
        res.error = f"{type(exc).__name__}: {exc}"
    res.duration_s = time.perf_counter() - t0
    return res
