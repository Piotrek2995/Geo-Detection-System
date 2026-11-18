"""Zrzuty ekranu GUI (Playwright) do docs/gui/*.png.

Wymaga uruchomionego GUI:  streamlit run app/main.py --server.port 8599
    python scripts/gui_screenshots.py [--url http://localhost:8599] [--out docs/gui] [--pages projekt,detekcja]

Zrzuty robione na biezacym projekcie GUI (dla dokumentacji: lot z namiotami, config.yaml).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

log = logging.getLogger("gui_screenshots")

PAGES = {           # slug strony -> (sciezka URL, akcja przed zrzutem)
    "projekt": ("", None),             # strona domyslna: URL bez sciezki
    "detekcja": ("detekcja", "preview"),
    "georeferencja": ("georeferencja", "qc"),
    "uruchom": ("uruchom", None),
    "wyniki": ("wyniki", "select_object"),
    "eksperymenty": ("eksperymenty", None),
}


def wait_idle(page, timeout_s: float = 120.0) -> None:
    """Czeka, az Streamlit skonczy rerun (znika wskaznik 'Running')."""
    t0 = time.time()
    page.wait_for_timeout(2500)
    while time.time() - t0 < timeout_s:
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(3000)       # mapy (kafelki WMS) i obrazy


def full_height(page, width: int, max_h: int = 9000) -> None:
    """Streamlit przewija tresc w wewnetrznym kontenerze — viewport powiekszany do wysokosci tresci."""
    h = page.evaluate("""() => {
        const el = document.querySelector("[data-testid='stMain']") || document.querySelector("section.main");
        const inner = document.querySelector("[data-testid='stMainBlockContainer']");
        return Math.max(el ? el.scrollHeight : 0, inner ? inner.scrollHeight + 120 : 0, 1000);
    }""")
    page.set_viewport_size({"width": width, "height": int(min(h, max_h))})
    wait_idle(page, 60)


def action(page, kind: str | None) -> None:
    if kind == "preview":
        btn = page.get_by_role("button", name="Pokaż detekcje")
        if btn.count() and btn.first.is_enabled():
            btn.first.click()
            page.get_by_text("wykryto", exact=False).first.wait_for(timeout=300_000)
            wait_idle(page, 60)
    elif kind == "qc":
        btn = page.get_by_role("button", name="Sprawdź jakość rekonstrukcji")
        if btn.count() and btn.first.is_enabled():
            btn.first.click()        # kontrola + ewentualne wyrownanie: do ~1 min
            page.get_by_text("Szczegóły", exact=True).first.wait_for(timeout=300_000)
            wait_idle(page, 60)
    elif kind == "select_object":
        cells = page.locator("[data-testid='stDataFrame'] canvas")
        if cells.count():
            box = cells.first.bounding_box()
            if box:     # pierwszy wiersz danych (kolumna wyboru wiersza po lewej)
                page.mouse.click(box["x"] + 8, box["y"] + 52)
                wait_idle(page, 120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--out", type=Path, default=Path("docs/gui"))
    ap.add_argument("--pages", default=",".join(PAGES))
    ap.add_argument("--width", type=int, default=1500)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        page = br.new_page(viewport={"width": args.width, "height": 1000}, device_scale_factor=1)
        for i, slug in enumerate(args.pages.split(","), 1):
            path, act = PAGES[slug]
            page.goto(f"{args.url.rstrip('/')}/{path}")
            wait_idle(page, 180)
            if page.get_by_text("Page not found").count():
                log.warning("Brak strony %s — pomijam", slug)
                continue
            action(page, act)
            out = args.out / f"{list(PAGES).index(slug) + 1:02d}_{slug}.png"
            page.mouse.move(0, 0)           # bez dymkow pomocy na zrzucie
            full_height(page, args.width)
            page.wait_for_timeout(5000)      # kafelki WMS po zmianie rozmiaru
            page.screenshot(path=str(out), full_page=True)
            page.set_viewport_size({"width": args.width, "height": 1000})
            log.info("%s", out)
        br.close()


if __name__ == "__main__":
    main()
