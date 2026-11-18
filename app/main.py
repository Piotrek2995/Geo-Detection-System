"""Geo-Detection-System — GUI (Streamlit). Uruchomienie: start_gui.bat albo
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import state as S  # noqa: E402
from app import ui  # noqa: E402

st.set_page_config(page_title="Geo-Detection-System", page_icon="🛰️", layout="wide")

VIEWS = Path(__file__).parent / "views"
pages = {
    "Przygotowanie": [
        st.Page(VIEWS / "projekt.py", title="1. Projekt", icon=":material/folder_open:", default=True),
        st.Page(VIEWS / "detekcja.py", title="2. Detekcja", icon=":material/center_focus_strong:"),
        st.Page(VIEWS / "georeferencja.py", title="3. Georeferencja", icon=":material/public:"),
    ],
    "Przetwarzanie": [
        st.Page(VIEWS / "uruchom.py", title="4. Uruchom", icon=":material/play_circle:"),
        st.Page(VIEWS / "wyniki.py", title="5. Wyniki", icon=":material/map:"),
    ],
}
if S.research_mode(ui.prefs()):
    pages["Badania"] = [st.Page(VIEWS / "eksperymenty.py", title="Eksperymenty", icon=":material/science:")]

nav = st.navigation(pages)

c = ui.cfg()
with st.sidebar:
    st.markdown(f"<div style='color:{ui.PRIMARY};font-weight:700;font-size:1.1rem'>Geo-Detection-System</div>",
                unsafe_allow_html=True)
    try:
        n_photos = len(ui.photos_for(c)) if Path(c.photos_dir).is_dir() else 0
    except Exception:                 # noqa: BLE001 — status nie moze zablokowac GUI
        n_photos = 0
    st.caption(S.project_status(c, n_photos, ui.odm_ok(c)))
    job = st.session_state.get("run_job")
    if job is not None and job.running:
        st.info(f"Przetwarzanie: {job.progress:.0%}")
    with st.expander("Ustawienia"):
        presets = S.list_presets()
        sel = st.selectbox("Preset detekcji", ["(wybierz)"] + presets, key="preset_sel",
                           help="Preset zmienia model i ustawienia detekcji, nie zmienia projektu.")
        if st.button("Wczytaj preset", disabled=sel == "(wybierz)", width="stretch"):
            try:
                ui.set_cfg(S.load_preset(sel, c))
                ui.bump_version()
                st.session_state.toast = f"Wczytano preset: {sel}"
                st.rerun()
            except Exception as exc:          # noqa: BLE001
                st.error(f"Nie wczytano presetu: {exc}")
        name = st.text_input("Zapisz bieżące ustawienia detekcji jako", placeholder="nazwa presetu")
        if st.button("Zapisz preset", disabled=not name.strip(), width="stretch"):
            try:
                p = S.save_preset(c, name)
                st.toast(f"Zapisano preset: {p.stem}")
            except Exception as exc:          # noqa: BLE001
                st.error(str(exc))
        if st.button("Przywróć ustawienia domyślne", width="stretch"):
            ui.set_cfg(S.base_config())
            ui.bump_version()
            st.rerun()
        rm = st.toggle("Tryb badawczy", value=bool(ui.prefs().get("research_mode")),
                       help="Pokazuje stronę Eksperymenty (porównania wariantów metod).")
        if rm != bool(ui.prefs().get("research_mode")):
            ui.set_pref("research_mode", rm)
            st.rerun()

nav.run()
