"""Strona 5 — Wyniki: mapa, tabela obiektow, widoki obiektu, kontrola jakosci, porownanie, pobieranie."""

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app import state as S
from app import ui
from geodetect.export import crop_thumbnail, draw_detections, read_image
from geodetect.qc import QcReport
from geodetect.runs import compare_runs, list_runs, load_run, run_map

ui.page_header("Wyniki")
c = ui.cfg()

runs = list_runs(Path(c.output_dir))
if not runs:
    st.info("Brak wyników, uruchom przetwarzanie na stronie Uruchom.")
    st.stop()
paths = [str(r.path) for r in runs]
want = st.session_state.get("result_dir")
idx = paths.index(str(Path(want))) if want and str(Path(want)) in paths else 0
c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
sel = c1.selectbox("Przebieg", range(len(runs)), index=idx, format_func=lambda k: S.run_label(runs[k]))
if runs[sel].qc_status:
    c2.markdown(f"Jakość: {ui.badge(runs[sel].qc_status)}", unsafe_allow_html=True)
st.session_state.result_dir = paths[sel]


@st.cache_data(show_spinner="Wczytywanie wyników…")
def _load(path: str, _mtime: float):
    return load_run(Path(path))


def load(path: str):
    return _load(path, Path(path, "run_report.json").stat().st_mtime)


run = load(paths[sel])
par = run.report.get("parameters", {})
labels = par.get("display_names") or {}
stats = run.report.get("statistics", {})
lab = lambda n: labels.get(n, n)      # noqa: E731

st.markdown(f"**{len(run.objects)}** obiektów (potwierdzonych: **{stats.get('confirmed', 0)}**, "
            f"z flagą: **{stats.get('flagged', 0)}**) z **{stats.get('detections', len(run.detections))}** "
            f"wykryć na **{stats.get('images_processed', 0)}** zdjęciach.")
if par.get("georef_decision"):
    st.caption(par["georef_decision"])

# ───────────── tabela + mapa ─────────────
df = pd.DataFrame([{
    "nr": o["object_id"] + 1, "klasa": lab(o["class"]), "status": S.STATUS_PL.get(o.get("status"), o.get("status")),
    "widoki": o.get("n_views"), "pewność": o.get("mean_conf"), "flaga": bool(o.get("flags")),
} for o in run.objects])

f1, f2, f3 = st.columns(3)
classes = sorted(df["klasa"].unique()) if len(df) else []
fc = f1.multiselect("Klasa", classes, default=classes)
statuses = sorted(df["status"].unique()) if len(df) else []
fs = f2.multiselect("Status", statuses, default=statuses)
ff = f3.selectbox("Flagi", ["wszystkie", "tylko z flagą", "bez flag"])
view = df[df["klasa"].isin(fc) & df["status"].isin(fs)] if len(df) else df
if ff == "tylko z flagą":
    view = view[view["flaga"]]
elif ff == "bez flag":
    view = view[~view["flaga"]]

selected = st.session_state.get("sel_obj")
col_map, col_tab = st.columns([1, 1])
with col_tab:
    ev = st.dataframe(view, hide_index=True, width="stretch", height=470, on_select="rerun",
                      selection_mode="single-row", key=f"tab_{sel}",
                      column_config={"nr": st.column_config.NumberColumn(width="small"),
                                     "widoki": st.column_config.NumberColumn(width="small"),
                                     "pewność": st.column_config.NumberColumn(format="%.2f", width="small"),
                                     "flaga": st.column_config.CheckboxColumn(width="small")})
    st.caption("Kliknij wiersz albo punkt na mapie, aby zobaczyć szczegóły.")
    rows = ev.selection.rows if ev is not None else []
    if rows:
        selected = int(view.iloc[rows[0]]["nr"]) - 1
        st.session_state.sel_obj = selected
with col_map:
    ids = {int(n) - 1 for n in view["nr"]} if len(view) else set()
    sub = type(run)(run.path, run.report, [o for o in run.objects if o["object_id"] in ids],
                    [d for d in run.detections if d.get("object_id") in ids or d.get("object_id") is None],
                    run.files)
    out = st_folium(run_map(sub, selected), height=470, use_container_width=True,
                    returned_objects=["last_object_clicked"], key=f"map_{sel}")
    clk = (out or {}).get("last_object_clicked")
    if clk and sub.objects:
        near = min(sub.objects, key=lambda o: (o["lat"] - clk["lat"]) ** 2 + (o["lon"] - clk["lng"]) ** 2)
        if near["object_id"] != selected:
            st.session_state.sel_obj = near["object_id"]
            st.rerun()

# ───────────── obiekt ─────────────
obj = run.object(selected) if selected is not None else None
if obj is not None:
    st.subheader(f"Obiekt {obj['object_id'] + 1}: {lab(obj['class'])}")
    facts = [S.STATUS_PL.get(obj.get("status"), obj.get("status")), f"widoki: {obj.get('n_views')}",
             f"pewność {obj['mean_conf']:.2f}" if obj.get("mean_conf") is not None else None,
             f"rozmiar {obj['size_m']:.1f} m" if obj.get("size_m") is not None else None,
             f"rozrzut {obj['rms_m']:.2f} m" if obj.get("rms_m") is not None else None]
    st.markdown(" · ".join(f for f in facts if f))
    if obj.get("flags"):
        st.warning("Flagi: " + ", ".join({"size_out_of_range": "nietypowy rozmiar"}.get(f, f) for f in obj["flags"]))
    with st.expander("Współrzędne"):
        st.markdown(f"X = {obj.get('x_2180')}, Y = {obj.get('y_2180')} (PUWG 1992)  \n"
                    f"H = {obj.get('altitude_m')} m n.p.m.  \n"
                    f"szerokość {obj['lat']:.7f}, długość {obj['lon']:.7f}")
    photos_dir = Path(par.get("photos_dir") or c.photos_dir)
    views = run.views(obj["object_id"])
    n_show = st.slider("Liczba widoków", 1, len(views), min(8, len(views)), key="nviews") if len(views) > 1 else 1
    cols = st.columns(4)
    for i, dv in enumerate(views[:n_show]):
        p = photos_dir / dv["image"]
        with cols[i % 4]:
            if p.exists() and dv.get("bbox"):
                import cv2
                st.image(cv2.cvtColor(crop_thumbnail(read_image(p), dv["bbox"], 320), cv2.COLOR_BGR2RGB),
                         width="stretch")
            st.caption(f"{dv['image']} · pewność {float(dv.get('conf') or 0):.2f} · "
                       f"{float(dv.get('range_m') or 0):.0f} m" + (" · ucięty" if dv.get("touches_border") else ""))
    if views:
        with st.expander("Całe zdjęcie z ramką"):
            dv = views[0]
            p = photos_dir / dv["image"]
            if p.exists():
                st.image(draw_detections(read_image(p), [{"bbox": dv["bbox"], "label": lab(obj["class"]),
                                                          "color": "#ff2d55"}]), width="stretch")

# ───────────── jakosc, wykresy, porownanie ─────────────
st.divider()
t_qc, t_plot = st.tabs(["Jakość", "Wykresy"])
with t_qc:
    q = run.report.get("qc")
    if q:
        ui.qc_panel(QcReport.from_dict(q), f"run_qc_{sel}", rerun=True)
    else:
        st.info("Ten przebieg nie ma kontroli jakości.")
with t_plot:
    if len(run.objects):
        pdf = pd.DataFrame([{"klasa": lab(o["class"]), "widoki": o.get("n_views"), "rozrzut [m]": o.get("rms_m")}
                            for o in run.objects])
        g1, g2 = st.columns(2)
        g1.caption("Rozrzut położeń a liczba widoków")
        g1.scatter_chart(pdf, x="widoki", y="rozrzut [m]", color="klasa", height=300)
        dd = pd.DataFrame([{"klasa": lab(x["class"]), "pewność": float(x.get("conf") or 0)} for x in run.detections])
        g2.caption("Średnia pewność wykrycia według klasy")
        g2.bar_chart(dd.groupby("klasa")["pewność"].mean(), height=300)
    sp = run.files.get("spread_png")
    if sp and Path(sp).exists():
        st.image(str(sp), caption="Położenia z poszczególnych widoków wokół obiektów", width="stretch")
with st.expander("Porównaj z innym przebiegiem"):
    others = [k for k in range(len(runs)) if k != sel]
    if not others:
        st.info("Brak innych przebiegów do porównania.")
    else:
        k = st.selectbox("Porównaj z", others, format_func=lambda k: S.run_label(runs[k]), key="cmp_sel")
        other = load(paths[k])
        matches, diff = compare_runs(run, other)
        paired = [m for m in matches if m.shift_m is not None]
        if paired:
            st.markdown(f"Wspólne obiekty: **{len(paired)}**, średnie przesunięcie "
                        f"**{sum(m.shift_m for m in paired) / len(paired):.2f} m**, największe "
                        f"**{max(m.shift_m for m in paired):.2f} m**.")
        st.dataframe(pd.DataFrame(S.comparison_rows(matches, run, other, labels)), hide_index=True, width="stretch")
        skip = {"photos_dir", "odm_dir"}
        rows = [{"ustawienie": S.param_label(key), "A (ten przebieg)": str(a), "B": str(b)}
                for key, (a, b) in diff.items() if key not in skip]
        if rows:
            st.caption("Różnice ustawień")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ───────────── pobieranie ─────────────
st.subheader("Pobierz")
dl = [("gpkg", "GeoPackage (QGIS)", "application/geopackage+sqlite3"),
      ("objects_geojson", "GeoJSON: obiekty", "application/geo+json"),
      ("detections_geojson", "GeoJSON: wykrycia", "application/geo+json"),
      ("objects_csv", "CSV: obiekty", "text/csv"),
      ("kml", "KML (Google Earth)", "application/vnd.google-earth.kml+xml"),
      ("report_html", "Raport HTML", "text/html")]
cols = st.columns(len(dl))
for col, (key, label, mime) in zip(cols, dl):
    p = run.files.get(key)
    if p and Path(p).exists():
        col.download_button(label, Path(p).read_bytes(), file_name=f"{run.path.name}_{Path(p).name}", mime=mime,
                            width="stretch", key=f"dl_{key}")
st.caption(f"Folder wyników: {run.path}")
