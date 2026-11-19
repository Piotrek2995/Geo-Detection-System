"""Wspolne elementy stron GUI (streamlit): config w sesji, zasoby w cache, komunikaty."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

from app import state as S
from geodetect.config import PipelineConfig

STATUS_COLORS = {"OK": "#1baf7a", "OSTRZEŻENIE": "#d99a00", "ZŁY": "#d03b3b", "brak danych": "#8a8984",
                 "informacja": "#5b7f95"}
PRIMARY = "#004A6B"


# ───────────── config w sesji ─────────────

def cfg() -> PipelineConfig:
    """Biezacy PipelineConfig (sesja): ostatni projekt albo config.yaml."""
    if "cfg" not in st.session_state:
        st.session_state.cfg = S.last_project() or S.base_config()
    return st.session_state.cfg


def set_cfg(new: PipelineConfig) -> None:
    st.session_state.cfg = new
    save()


def save() -> None:
    """Zapamietuje projekt lokalnie (otwarcie GUI wraca do niego)."""
    try:
        S.remember_project(st.session_state.cfg)
    except Exception as exc:          # noqa: BLE001 — brak zapisu nie blokuje pracy
        st.toast(f"Nie zapisano ostatniego projektu: {exc}")


def key_suffix() -> str:
    """Zmienia sie po wczytaniu innego configu — wymusza odswiezenie widgetow."""
    return str(st.session_state.get("cfg_version", 0))


def bump_version() -> None:
    st.session_state.cfg_version = st.session_state.get("cfg_version", 0) + 1


def prefs() -> dict:
    if "prefs" not in st.session_state:
        st.session_state.prefs = S.load_prefs()
    return st.session_state.prefs


def set_pref(key: str, value) -> None:
    p = prefs()
    if p.get(key) != value:
        p[key] = value
        try:
            S.save_prefs(p)
        except OSError:
            pass


# ───────────── zasoby (cache) ─────────────

@st.cache_data(show_spinner="Odczyt metadanych modeli…")
def models(folder: str) -> list:
    from geodetect.detect import list_models
    return list_models(folder)


@st.cache_data(show_spinner="Odczyt GPS ze zdjęć…")
def photos(folder: str, _mtime: float) -> list:
    from geodetect.photos import scan_photos
    return scan_photos(Path(folder))


def photos_for(c: PipelineConfig) -> list:
    p = Path(c.photos_dir)
    return photos(str(p), p.stat().st_mtime if p.is_dir() else 0.0)


@st.cache_resource(show_spinner="Wczytywanie modelu detekcji (GPU)…", max_entries=2)
def detector(model_path: str, detect_json: str):
    from geodetect.config import DetectConfig, _build
    from geodetect.detect import Detector
    d = _build(DetectConfig, json.loads(detect_json))
    return Detector(Path(model_path), d)


def detector_for(c: PipelineConfig):
    return detector(str(c.model_path), json.dumps(c.to_dict()["detect"], sort_keys=True))


@st.cache_data(ttl=600, show_spinner="Sprawdzanie usługi GUGiK…")
def rest_status(url: str):
    from geodetect.coverage import gugik_rest_status
    return gugik_rest_status(url)


@st.cache_data(ttl=300, show_spinner=False)
def docker_status():
    from geodetect.odmrun import docker_status as ds
    return ds()


@st.cache_data(show_spinner="Sprawdzanie pokrycia arkuszy NMT/NMPT…")
def coverage(odm_dir: str, nmt_dir: str, nmpt_dir: str, _cfg_json: str):
    from geodetect.coverage import surface_coverage
    c = PipelineConfig.from_dict(json.loads(_cfg_json))
    try:
        return surface_coverage(c)
    except FileNotFoundError:
        return None


def coverage_for(c: PipelineConfig):
    return coverage(str(c.odm_dir), str(c.surface.nmt_dir), str(c.surface.nmpt_dir), json.dumps(c.to_dict()))


def odm_ok(c: PipelineConfig) -> dict:
    from geodetect.odmrun import odm_outputs_ok
    return odm_outputs_ok(c.odm_dir)


# ───────────── natywne okno wyboru folderu ─────────────

def pick_folder(initial: Optional[Path] = None, title: str = "Wybierz folder") -> Optional[str]:
    """Okno tkinter w osobnym procesie (tkinter nie dziala w watku serwera streamlit)."""
    code = ("import tkinter as tk, tkinter.filedialog as fd, sys\n"
            "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            "p = fd.askdirectory(initialdir=sys.argv[1] or None, title=sys.argv[2])\n"
            "print(p or '')\n")
    try:
        out = subprocess.run([sys.executable, "-c", code, str(initial or ""), title], capture_output=True,
                             text=True, timeout=600)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def folder_input(label: str, value: Path, key: str, help: Optional[str] = None) -> Path:
    """Pole tekstowe + przycisk „Wybierz…” (natywne okno)."""
    k = f"{key}_{key_suffix()}"
    if st.session_state.get(f"{k}_picked"):
        st.session_state[k] = st.session_state.pop(f"{k}_picked")
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    with c1:
        txt = st.text_input(label, value=str(value), key=k, help=help)
    with c2:
        if st.button("Wybierz…", key=f"{k}_btn", width="stretch"):
            p = pick_folder(Path(txt) if txt else None, label)
            if p:
                st.session_state[f"{k}_picked"] = str(Path(p))
                st.rerun()
    return Path(txt.strip().strip('"')) if txt.strip() else value


# ───────────── komunikaty ─────────────

def show_issues(issues: list, ok_text: Optional[str] = None) -> None:
    for i in issues:
        {"error": st.error, "warning": st.warning, "info": st.info}[i.level](i.message,
                                                                           icon={"error": "⛔", "warning": "⚠️",
                                                                                 "info": "ℹ️"}[i.level])
    if not issues and ok_text:
        st.success(ok_text, icon="✅")


def badge(status: str) -> str:
    col = STATUS_COLORS.get(status, "#8a8984")
    return (f"<span style='background:{col};color:#fff;padding:2px 10px;border-radius:10px;"
            f"font-weight:600;font-size:0.85rem'>{status}</span>")


def qc_panel(rep, key: str, extra_details: Optional[list[str]] = None, rerun: bool = False) -> None:
    """Panel QC: status ogolny, najwazniejsze punkty, reszta zwinieta, zalecenia z przyciskiem akcji.
    rerun=True: akcja "Zastosuj i uruchom ponownie" (strona Wyniki)."""
    from geodetect.qc import explain_report
    v = explain_report(rep)
    st.markdown(f"{badge(v['status'])} &nbsp; {v['summary']}", unsafe_allow_html=True)
    checks = v["checks"]
    for title, status, text in checks[:v["top"]]:
        st.markdown(f"{badge(status)} &nbsp; **{title}:** {text}", unsafe_allow_html=True)
    for i, (text, action) in enumerate(v["actions"]):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"**Zalecenie:** {text}")
        target = {"georef_nmpt": "similarity_nmpt", "georef_auto": "auto"}.get(action)
        if target and cfg().georef.mode != target:
            label = "Zastosuj i uruchom ponownie" if rerun else "Zastosuj"
            if c2.button(label, key=f"{key}_act_{i}", width="stretch"):
                msg = S.apply_action(cfg(), action)
                save()
                bump_version()
                if rerun:
                    st.session_state.run_job = S.start_run_job(cfg())
                    st.session_state.toast = msg + " Przetwarzanie uruchomione."
                    st.switch_page("views/uruchom.py")
                st.toast(msg)
                st.rerun()
    with st.expander("Szczegóły"):
        for title, status, text in checks[v["top"]:]:
            st.markdown(f"{badge(status)} &nbsp; **{title}:** {text}", unsafe_allow_html=True)
        st.code("\n".join((extra_details or []) + rep.lines()), language=None)


def page_header(title: str, intro: Optional[str] = None) -> None:
    st.title(title)
    if intro:
        st.caption(intro)
    msg = st.session_state.pop("toast", None)
    if msg:
        st.toast(msg)


def georef_inputs(c: PipelineConfig) -> tuple:
    """(ok ODM, RestStatus, pokrycie arkuszy) dla walidacji; przy wyborze automatycznym ustawia powierzchnie."""
    ok = odm_ok(c)
    rest = rest_status(c.dem.gugik_url)
    cov = coverage_for(c) if ok.get("rekonstrukcja") else None
    if prefs().get("surface_auto", True):
        c.surface.source = S.resolve_surface(S.surface_availability(rest, cov, ok))[0]
    return ok, rest, cov
