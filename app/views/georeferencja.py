"""Strona 3 — Georeferencja: powierzchnia terenu, kalibracja wysokosci, tryb georeferencji, kontrola jakosci."""


import streamlit as st

from app import state as S
from app import ui
from geodetect.config import GEOREF_MODES, SURFACE_SOURCES, Z_ANCHOR_MODES

ui.page_header("Georeferencja", "Jak wykryte obiekty są przenoszone na mapę.")
c = ui.cfg()
v = ui.key_suffix()
before = c.to_dict()
ok = ui.odm_ok(c)

# ───────────── powierzchnia ─────────────
st.subheader("Powierzchnia terenu")
rest = ui.rest_status(c.dem.gugik_url)
cov = ui.coverage_for(c) if ok.get("rekonstrukcja") else None
avail = S.surface_availability(rest, cov, ok)
auto_src, auto_why = S.resolve_surface(avail)
AUTO = "auto"
opts = [AUTO, *SURFACE_SOURCES]
surface_auto = bool(ui.prefs().get("surface_auto", True))


def surface_label(o: str) -> str:
    if o == AUTO:
        return f"Automatycznie (zalecane): {S.SURFACE_SHORT[auto_src]}"
    return S.SURFACE_LABELS[o] + ("" if avail[o][0] else " (niedostępne)")


choice = st.radio("Powierzchnia", opts, index=0 if surface_auto else opts.index(c.surface.source),
                  format_func=surface_label, key=f"surf_{v}", label_visibility="collapsed",
                  help="Promień od kamery przez wykryty obiekt jest przecinany z tą powierzchnią. "
                       "NMPT zawiera budynki i drzewa, NMT tylko grunt.")
ui.set_pref("surface_auto", choice == AUTO)
c.surface.source = auto_src if choice == AUTO else choice
if choice == AUTO:
    st.caption(f"Wybrano {S.SURFACE_SHORT[auto_src]}: {auto_why}.")
with st.expander("Dostępność danych"):
    for src in SURFACE_SOURCES:
        good, msg = avail[src]
        st.markdown(f"{'✓' if good else '✗'} {S.SURFACE_LABELS[src]}: {msg}")

# ───────────── georeferencja / wysokosc ─────────────
st.subheader("Dowiązanie do układu współrzędnych")
gm = list(GEOREF_MODES)
c.georef.mode = st.radio("Georeferencja", gm, index=gm.index(c.georef.mode), format_func=lambda m: S.GEOREF_LABELS[m],
                         key=f"gm_{v}",
                         help="Automatycznie: kontrola jakości decyduje, czy wystarczy GPS drona, czy model trzeba "
                              "wyrównać do NMPT GUGiK (gdy jest pochylony lub zdeformowany).")
zm = list(Z_ANCHOR_MODES)
c.z_anchor.mode = st.selectbox("Kalibracja wysokości", zm, index=zm.index(c.z_anchor.mode),
                               format_func=lambda m: S.Z_MODE_LABELS[m], key=f"zm_{v}",
                               help="Wysokości z drona są względne; kalibracja przesuwa model do wysokości "
                                    "nad poziomem morza.")
if c.z_anchor.mode == "agl_assumption":
    c.z_anchor.agl_m = st.number_input("Wysokość lotu nad terenem [m]", value=float(c.z_anchor.agl_m), step=1.0,
                                       key=f"agl_{v}")
if c.z_anchor.mode == "fixed":
    c.z_anchor.offset_m = st.number_input("Przesunięcie wysokości [m]", value=float(c.z_anchor.offset_m or 0.0),
                                          step=0.1, key=f"zoff_{v}")

# ───────────── zaawansowane ─────────────
with st.expander("Zaawansowane"):
    a, b, d = st.columns(3)
    c.aggregate.eps_m = a.number_input("Odległość łączenia widoków [m]", value=float(c.aggregate.eps_m),
                                       min_value=0.1, step=0.5, key=f"eps_{v}",
                                       help="Wykrycia tej samej klasy bliżej siebie są jednym obiektem.")
    c.aggregate.min_views = int(b.number_input("Widoki do potwierdzenia obiektu", value=int(c.aggregate.min_views),
                                               min_value=1, step=1, key=f"mv_{v}"))
    wopts = ["equal", "conf", "cos_incidence"]
    wlab = {"equal": "równe", "conf": "według pewności", "cos_incidence": "według kąta widzenia"}
    c.aggregate.weights = d.selectbox("Wagi widoków", wopts, index=wopts.index(c.aggregate.weights),
                                      format_func=wlab.get, key=f"w_{v}")
    a, b, d = st.columns(3)
    c.raycast.step_m = a.number_input("Krok promienia [m]", value=float(c.raycast.step_m), min_value=0.1,
                                      step=0.5, key=f"step_{v}")
    c.surface.area_buffer_m = b.number_input("Margines obszaru [m]", value=float(c.surface.area_buffer_m),
                                             min_value=0.0, step=50.0, key=f"buf_{v}")
    ilab = {"bilinear": "dwuliniowa", "nearest": "najbliższy sąsiad"}
    iopts = list(ilab)
    c.surface.interpolation = d.selectbox("Interpolacja", iopts, index=iopts.index(c.surface.interpolation),
                                          format_func=ilab.get, key=f"interp_{v}")
    a, b, d = st.columns(3)
    c.filters.exclude_border = a.checkbox("Pomijaj obiekty ucięte krawędzią zdjęcia",
                                          value=c.filters.exclude_border, key=f"exb_{v}")
    mr = b.number_input("Maks. odległość od kamery [m] (0: bez limitu)", value=float(c.filters.max_range_m or 0),
                        min_value=0.0, step=10.0, key=f"mr_{v}")
    c.filters.max_range_m = mr or None
    mi = d.number_input("Maks. kąt widzenia [°] (0: bez limitu)", value=float(c.filters.max_incidence_deg or 0),
                        min_value=0.0, max_value=89.0, step=5.0, key=f"mi_{v}")
    c.filters.max_incidence_deg = mi or None
    c.surface.nmt_dir = ui.folder_input("Folder arkuszy NMT", c.surface.nmt_dir, "nmt_dir")
    c.surface.nmpt_dir = ui.folder_input("Folder arkuszy NMPT", c.surface.nmpt_dir, "nmpt_dir")

if c.to_dict() != before:
    ui.save()

issues = S.validate_georef(c, cov, rest, ok)
ui.show_issues(issues)

# ───────────── kontrola jakosci ─────────────
st.subheader("Kontrola jakości")
if st.button("Sprawdź jakość rekonstrukcji", type="primary", disabled=not ok.get("rekonstrukcja"),
             help="Porównuje model z barometrem drona i NMPT GUGiK. Trwa do minuty."):
    from geodetect.pipeline import check_reconstruction
    try:
        with st.spinner("Sprawdzanie…"):
            g = check_reconstruction(S.clone(c))
        st.session_state.qc = (c.to_dict(), g.qc, g.z_calibration_gps.describe() if g.z_calibration_gps
                               else g.z_calibration.describe(), g.decision)
    except Exception as exc:          # noqa: BLE001
        st.error(f"Kontrola jakości nie powiodła się: {exc}")
qc = st.session_state.get("qc")
if qc:
    if qc[0] != c.to_dict():
        st.caption("Wynik dla wcześniejszych ustawień, sprawdź ponownie po zmianach.")
    if qc[3]:
        st.info(qc[3])
    if qc[1] is not None:
        ui.qc_panel(qc[1], "georef_qc", extra_details=[f"Kalibracja wysokości: {qc[2]}"])
