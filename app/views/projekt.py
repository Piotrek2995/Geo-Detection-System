"""Strona 1 — Projekt: zdjecia, punkt startu, rekonstrukcja ODM (wskazanie albo uruchomienie)."""

from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from app import state as S
from app import ui
from geodetect.basemaps import add_basemaps
from geodetect.photos import suggest_start_image

ui.page_header("Projekt", "Wskaż zdjęcia z lotu i ich rekonstrukcję.")
c = ui.cfg()
v = ui.key_suffix()

# ───────────── zdjecia ─────────────
st.subheader("Zdjęcia z lotu")
new_dir = ui.folder_input("Folder ze zdjęciami", c.photos_dir, "photos_dir",
                          help="Zdjęcia z drona z pozycją GPS zapisaną w pliku (EXIF).")
if new_dir != Path(c.photos_dir):
    c.photos_dir = new_dir
    ui.save()
photos = ui.photos_for(c) if Path(c.photos_dir).is_dir() else []
with_gps = [p for p in photos if p.has_gps]
alts = [p.alt for p in with_gps if p.alt is not None]
if photos:
    st.markdown(f"**{len(photos)}** zdjęć, z pozycją GPS: **{len(with_gps)}**"
                + (f", wysokość lotu **{min(alts):.0f} do {max(alts):.0f} m**" if alts else ""))

# ───────────── punkt startu ─────────────
st.subheader("Punkt startu drona")
st.caption("Punkt startu poprawia dokładność wysokości.")
st_cfg = c.z_anchor.start
names = [p.name for p in photos]
sugg = suggest_start_image(photos)
modes = ["brak", "zdjęcie startowe", "współrzędne"]
cur = 1 if st_cfg.image or (sugg and not st_cfg.latlon) else (2 if st_cfg.latlon else 0)
mode = st.radio("Sposób wskazania", modes, index=cur, horizontal=True, key=f"start_mode_{v}",
                help="Zdjęcie startowe: zrobione tuż nad ziemią w miejscu startu; służy tylko do lokalizacji "
                     "startu i jest pomijane w dalszym przetwarzaniu. Współrzędne można wpisać albo kliknąć na mapie.")
img, latlon = st_cfg.image, st_cfg.latlon
if mode == modes[0]:
    img, latlon = None, None
elif mode == modes[1]:
    opts = names or ([img] if img else [])
    idx = opts.index(img) if img in opts else (opts.index(sugg) if sugg in opts else 0)
    img = st.selectbox("Zdjęcie startowe", opts, index=idx if opts else None, key=f"start_img_{v}",
                       help="Wykrywane automatycznie: zdjęcie z najniższą wysokością na początku lotu.")
    if sugg and img == sugg:
        st.caption(f"Wykryto automatycznie: {sugg}")
    latlon = None
else:
    img = None
    clicked = st.session_state.pop("map_click", None)
    base = clicked or latlon or ([with_gps[0].lat, with_gps[0].lon] if with_gps else [52.0, 21.0])
    a, b = st.columns(2)
    lat = a.number_input("Szerokość geograficzna", value=float(base[0]), format="%.7f", key=f"start_lat_{v}_{clicked}")
    lon = b.number_input("Długość geograficzna", value=float(base[1]), format="%.7f", key=f"start_lon_{v}_{clicked}")
    latlon = (lat, lon)
if (img, list(latlon) if latlon else None) != (st_cfg.image, st_cfg.latlon):
    S.set_start(c, img, latlon)
    ui.save()

# mapa zdjec
if with_gps:
    pts = [(p.lat, p.lon) for p in with_gps]
    m = folium.Map(location=[sum(a for a, _ in pts) / len(pts), sum(b for _, b in pts) / len(pts)],
                   zoom_start=18, max_zoom=23, tiles=None)
    add_basemaps(m, "esri")
    for p in with_gps:
        is_start = p.name == c.z_anchor.start.image
        folium.CircleMarker([p.lat, p.lon], radius=7 if is_start else 4,
                            color="#d03b3b" if is_start else ui.PRIMARY, fill=True, fill_opacity=0.85,
                            tooltip=p.name + (f", {p.alt:.1f} m" if p.alt is not None else "")
                            + (", zdjęcie startowe" if is_start else "")).add_to(m)
    if c.z_anchor.start.latlon:
        folium.Marker(c.z_anchor.start.latlon, tooltip="punkt startu",
                      icon=folium.Icon(color="red", icon="flag")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    if mode == modes[2]:
        st.caption("Kliknij na mapie, aby ustawić punkt startu.")
    out = st_folium(m, height=420, use_container_width=True, returned_objects=["last_clicked"],
                    key=f"photo_map_{v}")
    lc = (out or {}).get("last_clicked")
    if mode == modes[2] and lc and [round(lc["lat"], 7), round(lc["lng"], 7)] != \
            [round(x, 7) for x in (c.z_anchor.start.latlon or [0, 0])]:
        st.session_state.map_click = [lc["lat"], lc["lng"]]
        S.set_start(c, None, (lc["lat"], lc["lng"]))
        ui.save()
        st.rerun()
elif photos:
    st.warning("Zdjęcia nie mają pozycji GPS, mapa jest niedostępna.")

# ───────────── rekonstrukcja ─────────────
st.subheader("Rekonstrukcja")
odm_mode = st.radio("Rekonstrukcja", ["mam gotowe wyniki", "uruchom teraz"], horizontal=True,
                    key=f"odm_mode_{v}", label_visibility="collapsed",
                    help="Rekonstrukcja (OpenDroneMap) wyznacza położenie kamer i model powierzchni.")
new_odm = ui.folder_input("Folder rekonstrukcji", c.odm_dir, "odm_dir",
                          help="Folder projektu OpenDroneMap; przy nowej rekonstrukcji folder docelowy.")
if new_odm != Path(c.odm_dir):
    c.odm_dir = new_odm
    ui.save()
ok = ui.odm_ok(c)
ready, text = S.odm_status(ok)
(st.success if ready and all(ok.values()) else st.info if ready else st.warning)(text)
with st.expander("Szczegóły rekonstrukcji"):
    for k, val in ok.items():
        st.markdown(f"{'✓' if val else '✗'} {S.ODM_PARTS.get(k, k)}")

if odm_mode == "uruchom teraz":
    ds = ui.docker_status()
    if not ds.ok:
        st.error(f"Docker nie działa: {ds.message}. Uruchom Docker Desktop i odśwież stronę.")
    fixed = st.checkbox("Użyj kalibracji kamery z innej rekonstrukcji",
                        help="Zalecane przy krótkich, niskich lotach: kalibracja z lotu o dobrej geometrii "
                             "zamiast wyznaczania jej od nowa.")
    src = None
    if fixed:
        src = ui.folder_input("Rekonstrukcja z dobrą kalibracją", Path(c.odm_dir).resolve(), "cal_src")
    job = st.session_state.get("odm_job")
    if st.button("Uruchom rekonstrukcję", type="primary",
                 disabled=not ds.ok or (job is not None and job.running) or not with_gps):
        st.session_state.odm_job = S.start_odm_job(c, src)
        st.rerun()

    @st.fragment(run_every=1.0)
    def odm_progress():
        job = st.session_state.get("odm_job")
        if job is None:
            return
        st.progress(min(job.progress, 1.0), text=f"{job.message} ({job.elapsed_s / 60:.1f} min)")
        with st.expander("Log", expanded=job.running):
            st.code("\n".join(list(job.logs)[-20:]) or "…", language=None)
        if job.running:
            if st.button("Przerwij", key="odm_cancel"):
                job.cancel()
        elif job.status == "done":
            st.success(f"Rekonstrukcja zakończona ({job.elapsed_s / 60:.1f} min).")
        elif job.status == "cancelled":
            st.warning("Rekonstrukcja przerwana.")
        elif job.status == "error":
            st.error(job.error)
    odm_progress()

st.divider()
issues = S.validate_project(c, photos, ok)
ui.show_issues(issues, "Projekt kompletny, przejdź do strony Detekcja.")
