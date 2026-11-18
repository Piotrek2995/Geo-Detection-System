"""Strona 2 — Detekcja: model, klasy z ustawieniami per klasa, prompt (modele otwartego slownika), podglad."""

from pathlib import Path

import streamlit as st

from app import state as S
from app import ui
from geodetect.export import CLASS_PALETTE, draw_detections, read_image

ui.page_header("Detekcja", "Wybierz model i obiekty, które ma wykrywać.")
c = ui.cfg()
v = ui.key_suffix()
before = c.to_dict()
s = S.detection_settings(c)
MASK_METHODS = [m for m in S.POINT_LABELS if m not in S.POINT_SIMPLE]

# ───────────── model ─────────────
infos = ui.models(str(S.ROOT / "models"))
paths = [str(i.path.resolve()) for i in infos]
cur = str(Path(c.model_path).resolve())
if cur not in paths and Path(cur).exists():
    from geodetect.detect import model_info
    extra = model_info(cur)
    if extra:
        infos, paths = infos + [extra], paths + [cur]
if not infos:
    st.error("Brak modeli w folderze models/.")
    st.stop()
sel = st.selectbox("Model", range(len(infos)), index=paths.index(cur) if cur in paths else 0,
                   format_func=lambda k: infos[k].path.stem, key=f"model_{v}",
                   help="Pliki modeli z folderu models/.")
info = infos[sel]
s.model_path = info.path
st.caption(S.model_summary(info, c))

# ───────────── prompt (modele otwartego slownika) ─────────────
photos = ui.photos_for(c) if Path(c.photos_dir).is_dir() else []
names_photos = [p.name for p in photos if p.name not in c.excluded_images()]
if info.open_vocab:
    st.subheader("Czego szukać")
    kinds = ["opis tekstowy", "zaznaczenie na zdjęciu"]
    can_visual = "yoloe" in info.name.lower()
    kind = st.radio("Sposób", kinds if can_visual else kinds[:1], horizontal=True, key=f"pk_{v}",
                    index=1 if (s.visual_prompt_boxes and can_visual) else 0, label_visibility="collapsed")
    if kind == kinds[0]:
        txt = st.text_input("Nazwy obiektów po angielsku, rozdzielone przecinkami",
                            value=", ".join(s.prompt_text or []), placeholder="np. person, car, ball",
                            key=f"prompt_{v}")
        s.prompt_text = S.parse_prompt(txt)
        s.visual_prompt_image = s.visual_prompt_boxes = s.visual_prompt_classes = None
    elif names_photos:
        import numpy as np
        import plotly.graph_objects as go
        cur_ref = Path(s.visual_prompt_image).name if s.visual_prompt_image else None
        ref_name = st.selectbox("Zdjęcie wzorcowe", names_photos,
                                index=names_photos.index(cur_ref) if cur_ref in names_photos else 0,
                                key=f"vp_img_{v}")
        cls_name = st.text_input("Nazwa obiektu", value=(s.visual_prompt_classes or ["obiekt"])[0],
                                 key=f"vp_cls_{v}")
        img = read_image(Path(c.photos_dir) / ref_name)
        sc = 1280 / max(img.shape[:2])
        small = draw_detections(img, [{"bbox": bb, "color": "#ff2d55"} for bb in (s.visual_prompt_boxes or [])
                                      if cur_ref == ref_name], max_px=1280)
        fig = go.Figure(go.Image(z=small))
        gx, gy = np.meshgrid(np.arange(0, small.shape[1], 12), np.arange(0, small.shape[0], 12))
        fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=2, opacity=0.01),
                                 hoverinfo="skip", showlegend=False))
        fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=0, b=0), height=520,
                          xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        st.caption("Zaznacz obiekt prostokątem na zdjęciu.")
        ev = st.plotly_chart(fig, on_select="rerun", selection_mode="box", key=f"vp_chart_{v}", width="stretch")
        box = (s.visual_prompt_boxes or [[0, 0, 0, 0]])[0]
        sel_box = (getattr(ev, "selection", None) or {}).get("box") or []
        if sel_box:
            xs, ys = sel_box[0]["x"], sel_box[0]["y"]
            box = [min(xs) / sc, min(ys) / sc, max(xs) / sc, max(ys) / sc]
        with st.expander("Współrzędne ramki (piksele)"):
            cols = st.columns(4)
            box = [cols[i].number_input(lab, value=float(box[i]), min_value=0.0, step=1.0,
                                        key=f"vp_{lab}_{v}_{round(box[i])}")
                   for i, lab in enumerate(("x1", "y1", "x2", "y2"))]
        if box[2] > box[0] and box[3] > box[1]:
            s.visual_prompt_image = (Path(c.photos_dir) / ref_name).resolve()
            s.visual_prompt_boxes = [box]
            s.visual_prompt_classes = [cls_name.strip() or "obiekt"]
            s.prompt_text = None
            y0, x0 = max(0, int(box[1]) - 40), max(0, int(box[0]) - 40)
            st.image(draw_detections(img[y0:int(box[3]) + 40, x0:int(box[2]) + 40], [], max_px=200),
                     caption=f"Wzorzec z {ref_name}")
    else:
        st.info("Najpierw wskaż folder zdjęć na stronie Projekt.")
    tmp = S.clone(c)
    S.apply_detection(tmp, s)
    available = S.model_class_names(info, tmp)
else:
    s.prompt_text = s.visual_prompt_image = s.visual_prompt_boxes = s.visual_prompt_classes = None
    available = list(info.names.values())

# ───────────── klasy ─────────────
st.subheader("Obiekty")
if info.open_vocab:
    chosen = available
else:
    default = [x for x in (s.classes or available) if x in available]
    chosen = st.multiselect("Wykrywane klasy", available, default=default, key=f"classes_{v}_{info.name}",
                            format_func=lambda n: s.display_names.get(n, n))
s.classes = chosen if (info.open_vocab or len(chosen) < len(available)) else None


def class_card(i: int, cls: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{s.display_names.get(cls, cls)}**")
        c1, c2, c3 = st.columns([3, 1, 3])
        s.display_names[cls] = c1.text_input("Nazwa wyświetlana", value=s.display_names.get(cls, cls),
                                             key=f"dn_{cls}_{v}")
        s.colors[cls] = c2.color_picker("Kolor", value=s.colors.get(cls, CLASS_PALETTE[i % len(CLASS_PALETTE)]),
                                        key=f"col_{cls}_{v}")
        s.conf_per_class[cls] = c3.slider("Próg pewności", 0.001, 1.0, s.class_conf(cls), 0.001, format="%.3f",
                                          key=f"cc_{cls}_{v}",
                                          help="Niżej: więcej wykrytych obiektów, ale też więcej pomyłek.")
        pm = s.class_point(cls)
        simple = st.radio("Punkt lokalizacji", S.POINT_SIMPLE, horizontal=True,
                          index=S.POINT_SIMPLE.index(pm) if pm in S.POINT_SIMPLE else 0,
                          format_func=lambda m: S.POINT_LABELS[m], key=f"pp_{cls}_{v}",
                          help="Który punkt obiektu jest lokalizowany w terenie. Dla ludzi: podstawa (stopy).")
        with st.expander("Zaawansowane"):
            use_mask = st.checkbox("Punkt z maski obiektu (dokładniejszy, wolniejszy)", value=pm in MASK_METHODS,
                                   key=f"mask_{cls}_{v}",
                                   help="Kształt obiektu wyznaczany modelem segmentacji (SAM), "
                                        "pobieranym przy pierwszym użyciu.")
            if use_mask:
                pm = st.selectbox("Punkt z maski", MASK_METHODS,
                                  index=MASK_METHODS.index(pm) if pm in MASK_METHODS else 0,
                                  format_func=lambda m: S.POINT_LABELS[m], key=f"pmask_{cls}_{v}")
            else:
                pm = simple
            rng = s.size_ranges.get(cls)
            use = st.checkbox("Oznaczaj obiekty o nietypowym rozmiarze", value=rng is not None,
                              key=f"sz_on_{cls}_{v}",
                              help="Obiekty spoza zakresu dostają flagę, nie są usuwane.")
            if use:
                a, b = st.columns(2)
                lo = a.number_input("Minimum [m]", value=float(rng[0]) if rng else 0.3, min_value=0.0, step=0.1,
                                    key=f"szmin_{cls}_{v}")
                hi = b.number_input("Maksimum [m]", value=float(rng[1]) if rng else 5.0, min_value=0.0, step=0.1,
                                    key=f"szmax_{cls}_{v}")
                s.size_ranges[cls] = [lo, hi]
            else:
                s.size_ranges.pop(cls, None)
        s.point_per_class[cls] = pm


for i, cls in enumerate(chosen[:20]):
    class_card(i, cls)
if len(chosen) > 20:
    st.caption(f"Pozostałe {len(chosen) - 20} klas używa ustawień domyślnych.")

with st.expander("Małe obiekty"):
    s.tiling = st.checkbox("Wykrywaj także na powiększonych fragmentach zdjęcia", value=s.tiling, key=f"tile_{v}",
                           help="Pomaga wykryć małe obiekty (np. ludzi z dużej wysokości), ale trwa dłużej.")
    if s.tiling:
        tc = st.multiselect("Tylko dla klas (puste: wszystkie)", chosen, format_func=lambda n: s.display_names.get(n, n),
                            default=[x for x in (s.tiling_classes or []) if x in chosen], key=f"tilec_{v}")
        s.tiling_classes = tc or None

S.apply_detection(c, s, available)
if c.to_dict() != before:
    ui.save()

issues = S.validate_detection(c, info)
ui.show_issues(issues)

# ───────────── podglad ─────────────
st.subheader("Podgląd")
if names_photos:
    pc1, pc2 = st.columns([3, 1], vertical_alignment="bottom")
    pname = pc1.selectbox("Zdjęcie", names_photos, key="preview_img")
    if pc2.button("Pokaż detekcje", type="primary", disabled=S.has_errors(issues), width="stretch"):
        try:
            det = ui.detector_for(c)
            with st.spinner("Wykrywanie…"):
                st.session_state.preview = (pname, det.detect(Path(c.photos_dir) / pname))
        except Exception as exc:      # noqa: BLE001
            st.error(f"Wykrywanie nie powiodło się: {exc}")
    pv = st.session_state.get("preview")
    if pv and pv[0] == pname:
        dets = pv[1]
        boxes = [{"bbox": d.bbox, "label": f"{c.detect.display(d.cls_name)} {d.conf:.2f}",
                  "color": c.detect.colors.get(d.cls_name, CLASS_PALETTE[0])} for d in dets]
        st.image(draw_detections(read_image(Path(c.photos_dir) / pname), boxes),
                 caption=f"{pname}: wykryto {len(dets)}", width="stretch")
else:
    st.info("Najpierw wskaż folder zdjęć na stronie Projekt.")
