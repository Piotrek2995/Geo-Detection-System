"""Strona Eksperymenty (tryb badawczy): experiments/*.yaml, uruchomienie w tle, summary.md i wykresy."""

from pathlib import Path

import streamlit as st

from app import state as S
from app import ui

ui.page_header("Eksperymenty", "Porównanie wariantów metod na bieżącym projekcie.")

exps = S.list_experiments()
if not exps:
    st.info("Brak plików eksperymentów w folderze experiments/.")
    st.stop()
sel = st.selectbox("Eksperyment", exps, format_func=lambda p: p.stem)
with st.expander("Definicja"):
    st.code(Path(sel).read_text(encoding="utf-8"), language="yaml")

job = st.session_state.get("exp_job")
if st.button("Uruchom eksperyment", type="primary", disabled=job is not None and job.running):
    st.session_state.exp_job = S.start_experiment_job(sel)
    st.rerun()


@st.fragment(run_every=2.0)
def exp_progress():
    job = st.session_state.get("exp_job")
    if job is None:
        return
    if job.running:
        st.info(f"{job.message or 'Trwa…'} ({job.elapsed_s / 60:.1f} min)")
    elif job.status == "error":
        st.error(job.error)
    elif job.status == "done":
        st.success(f"Zakończono: {job.result}")
    with st.expander("Log", expanded=job.status == "error"):
        st.code("\n".join(list(job.logs)[-25:]) or "…", language=None, height=280)


exp_progress()

from geodetect.experiments import load_experiment  # noqa: E402

try:
    name = load_experiment(sel).name
except Exception as exc:              # noqa: BLE001
    st.error(f"Nie wczytano eksperymentu: {exc}")
    st.stop()
out = S.latest_experiment_output(name)
st.subheader("Ostatni wynik")
if out is None:
    st.caption("Ten eksperyment nie był jeszcze uruchamiany.")
else:
    st.caption(str(out))
    st.markdown((out / "summary.md").read_text(encoding="utf-8"))
    pngs = sorted(out.glob("*.png"))
    for i in range(0, len(pngs), 2):
        cols = st.columns(2)
        for col, p in zip(cols, pngs[i:i + 2]):
            col.image(str(p), caption=p.stem, width="stretch")
