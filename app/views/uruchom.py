"""Strona 4 — Uruchom: podsumowanie ustawien, walidacja, przebieg w tle (postep, log, miniatury, przerwanie)."""

from pathlib import Path

import streamlit as st

from app import state as S
from app import ui
from geodetect.export import draw_detections, read_image

ui.page_header("Uruchom", "Sprawdź ustawienia i uruchom przetwarzanie.")
c = ui.cfg()

photos = ui.photos_for(c) if Path(c.photos_dir).is_dir() else []
names = [p.name for p in photos if p.name not in c.excluded_images()]
ok, rest, cov = ui.georef_inputs(c)
info = None
try:
    from geodetect.detect import model_info
    info = model_info(c.model_path)
except Exception:                     # noqa: BLE001 — walidacja zglosi brak modelu
    pass

summary = S.settings_summary(c, len(names))
st.table({"Ustawienie": [k for k, _ in summary], "Wybór": [v for _, v in summary]})
with st.expander("Pozostałe ustawienia"):
    det = S.settings_details(c)
    st.table({"Ustawienie": [k for k, _ in det], "Wartość": [v for _, v in det]})

issues = S.validate_project(c, photos, ok) + S.validate_detection(c, info) + S.validate_georef(c, cov, rest, ok)
for i in issues:
    i.message = f"{i.where}: {i.message}"
ui.show_issues(issues)

scope = st.radio("Zdjęcia", ["wszystkie", "wybrane"], horizontal=True)
images = None
if scope == "wybrane":
    images = st.multiselect("Zdjęcia do przetworzenia", names, default=names[:3])

job = st.session_state.get("run_job")
busy = job is not None and job.running
if st.button("Uruchom przetwarzanie", type="primary",
             disabled=S.has_errors(issues) or busy or (images is not None and not images)):
    st.session_state.run_job = S.start_run_job(c, all_images=images is None, images=images)
    ui.save()
    st.rerun()


@st.fragment(run_every=1.0)
def progress_panel():
    job = st.session_state.get("run_job")
    if job is None:
        return
    st.progress(min(job.progress, 1.0), text=f"{job.message or 'Start…'} ({job.elapsed_s:.0f} s)")
    if job.running:
        if st.button("Przerwij", key="run_cancel"):
            job.cancel()
    elif job.status == "done":
        r = job.result
        st.success(f"Gotowe w {r.duration_s:.0f} s: {len(r.objects)} obiektów.")
        if st.button("Pokaż wyniki", type="primary"):
            st.session_state.result_dir = str(r.out_dir)
            st.switch_page("views/wyniki.py")
    elif job.status == "cancelled":
        st.warning("Przetwarzanie przerwane, wyniki nie zostały zapisane.")
    elif job.status == "error":
        st.error(f"Błąd przetwarzania: {job.error}")
    col_img, col_log = st.columns([3, 2])
    with col_img:
        st.caption(f"Przetworzone zdjęcia: {len(job.items)}")
        cols = st.columns(3)
        for i, name in enumerate(job.items[-6:]):
            p = Path(c.photos_dir) / str(name).split()[0]
            if p.exists():
                key = f"thumb_{p.name}"
                if key not in st.session_state:
                    st.session_state[key] = draw_detections(read_image(p), [], max_px=320)
                cols[i % 3].image(st.session_state[key], caption=p.name)
    with col_log:
        with st.expander("Log", expanded=job.status == "error"):
            st.code("\n".join(list(job.logs)[-25:]) or "…", language=None, height=320)


progress_panel()
