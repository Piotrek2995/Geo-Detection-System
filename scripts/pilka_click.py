"""Reczne zaznaczanie pilki na zdjeciach -> data/pilka_pixels.csv (niezalezne odniesienie).

    python scripts/pilka_click.py [--config config_pilka.yaml]

Dwa okna:
  "zdjecie" — cale zdjecie (pomniejszone). Klik LPM = wybierz okolice pilki (otwiera zoom).
  "zoom"    — powiekszenie 8x fragmentu 120x120 px. Klik LPM = dokladny srodek pilki.
Zolty krzyzyk = obecny punkt z CSV (wstepny/kandydat), zielony = Twoj punkt.

Klawisze (w dowolnym oknie):
  n / spacja / Enter  nastepne zdjecie        b  poprzednie
  x  pilka NIEWIDOCZNA na tym zdjeciu          c  zatwierdz obecny (zolty) punkt bez zmian
  z  zoom wokol obecnego punktu                q / Esc  zapisz i wyjdz
Zapis po kazdej zmianie; kopia poprzedniego CSV: data/pilka_pixels.csv.bak
Zdjecia z kolumna source zaczynajaca sie od "recznie" zostaly zaznaczone tym narzedziem.

Tryb --ortho (GT_transfer): pary punktow dachu na ortofotomapie GUGiK 2025 i ODM z lotu
+ polozenie pilki na ortofotomapie ODM -> data/pilka_ortho_pairs.csv (wspolrzedne EPSG:2180).
  Klik w oknie "GUGiK", potem ten sam punkt w oknie "ODM" = jedna para (2-4 pary: narozniki
  dachu, komin, koniec kalenicy — punkty NA DACHU, na tej samej wysokosci co pilka).
  p  nastepny klik w oknie ODM = pilka      u  cofnij ostatni klik      q / Esc  zapisz i wyjdz
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.config import PipelineConfig  # noqa: E402

FIELDS = ["image", "u", "v", "visible", "source", "conf"]
VIEW_W = 1400
ZOOM_HALF, ZOOM_K = 60, 8


def load(path: Path, names: list[str]) -> dict[str, dict]:
    rows = {n: {"image": n, "u": "", "v": "", "visible": "", "source": "", "conf": ""} for n in names}
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["image"] in rows:
                rows[r["image"]].update({k: r.get(k, "") or "" for k in FIELDS})
    return rows


def save(path: Path, rows: dict[str, dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for n in sorted(rows):
            w.writerow(rows[n])


def cross(img, x, y, col, s=10, t=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - s, y), (x + s, y), col, t)
    cv2.line(img, (x, y - s), (x, y + s), col, t)


def read_pgw(png: Path) -> np.ndarray:
    a, d, b, e, c, f = map(float, png.with_suffix(".pgw").read_text().split())
    return np.array([[a, b, c], [d, e, f]])


def ortho_mode(out_csv: Path, ref_dir: Path, half_m: float = 15.0, k: float = 1.5) -> int:
    import json
    ref = json.loads((ref_dir / "reference.json").read_text(encoding="utf-8"))
    gx, gy = ref["gt_x_2180"], ref["gt_y_2180"]
    wins = {}
    for name, fn in (("GUGiK", "ortho_gugik.png"), ("ODM", "ortho_odm.png")):
        img = cv2.imread(str(ref_dir / fn))
        T = read_pgw(ref_dir / fn)
        res = abs(T[0, 0])
        c0 = int((gx - half_m - T[0, 2]) / T[0, 0])
        r0 = int((gy + half_m - T[1, 2]) / T[1, 1])
        n = int(2 * half_m / res)
        crop = img[max(r0, 0):r0 + n, max(c0, 0):c0 + n]
        wins[name] = {"img": cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC),
                      "T": T, "c0": max(c0, 0), "r0": max(r0, 0)}
    clicks: list[dict] = []          # {"win", "x", "y", "kind"}
    if out_csv.exists():             # wczytaj wczesniej zapisane punkty (kontynuacja)
        for r in csv.DictReader(open(out_csv, encoding="utf-8")):
            if r["kind"] == "pilka":
                clicks.append({"win": "ODM", "x": float(r["odm_x"]), "y": float(r["odm_y"]), "kind": "ball"})
            else:
                clicks.append({"win": "GUGiK", "x": float(r["gugik_x"]), "y": float(r["gugik_y"]), "kind": "pt"})
                clicks.append({"win": "ODM", "x": float(r["odm_x"]), "y": float(r["odm_y"]), "kind": "pt"})
    st = {"ball": False}

    def to_xy(w, px, py):
        W = wins[w]
        col, row = W["c0"] + px / k, W["r0"] + py / k
        return tuple(W["T"] @ np.array([col, row, 1.0]))

    def to_px(w, x, y):
        W = wins[w]
        A = W["T"][:, :2]
        col, row = np.linalg.solve(A, np.array([x, y]) - W["T"][:, 2])
        return (col - W["c0"]) * k, (row - W["r0"]) * k

    def draw():
        for w in wins:
            img = wins[w]["img"].copy()
            npair = 0
            for c in clicks:
                if c["win"] != w:
                    continue
                px, py = to_px(w, c["x"], c["y"])
                if c["kind"] == "ball":
                    cv2.circle(img, (int(px), int(py)), 9, (0, 255, 0), 2)
                    cv2.putText(img, "pilka", (int(px) + 10, int(py) - 8), 0, 0.6, (0, 255, 0), 2)
                else:
                    npair += 1
                    cross(img, px, py, (0, 220, 255), 9, 2)
                    cv2.putText(img, str(npair), (int(px) + 8, int(py) - 8), 0, 0.7, (0, 220, 255), 2)
            ng = sum(1 for c in clicks if c["win"] == "GUGiK" and c["kind"] == "pt")
            no = sum(1 for c in clicks if c["win"] == "ODM" and c["kind"] == "pt")
            nxt = "GUGiK" if ng <= no else "ODM"
            msg = ("nastepny klik w ODM = PILKA" if st["ball"] else
                   f"pary: {min(ng, no)} | teraz kliknij punkt dachu w oknie {nxt} | p: pilka | u: cofnij | q: koniec")
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, f"{w}: {msg}", (6, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(w, img)

    def on(w):
        def cb(ev, px, py, *_):
            if ev != cv2.EVENT_LBUTTONDOWN:
                return
            x, y = to_xy(w, px, py)
            if st["ball"] and w == "ODM":
                clicks[:] = [c for c in clicks if c["kind"] != "ball"]
                clicks.append({"win": w, "x": x, "y": y, "kind": "ball"})
                st["ball"] = False
            else:
                clicks.append({"win": w, "x": x, "y": y, "kind": "pt"})
            draw()
        return cb

    for w in wins:
        cv2.namedWindow(w, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(w, on(w))
    draw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            st["ball"] = True
            draw()
        elif key == ord("u") and clicks:
            clicks.pop()
            draw()
    cv2.destroyAllWindows()
    g = [c for c in clicks if c["win"] == "GUGiK" and c["kind"] == "pt"]
    o = [c for c in clicks if c["win"] == "ODM" and c["kind"] == "pt"]
    ball = [c for c in clicks if c["kind"] == "ball"]
    n = min(len(g), len(o))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "gugik_x", "gugik_y", "odm_x", "odm_y"])
        for i in range(n):
            w.writerow([i + 1, "punkt_dachu", f"{g[i]['x']:.3f}", f"{g[i]['y']:.3f}",
                        f"{o[i]['x']:.3f}", f"{o[i]['y']:.3f}"])
        for b in ball:
            w.writerow(["pilka", "pilka", "", "", f"{b['x']:.3f}", f"{b['y']:.3f}"])
    print(f"Zapisano {out_csv}: {n} par punktow dachu, pilka: {'tak' if ball else 'NIE'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config_pilka.yaml")
    ap.add_argument("--start", default=None, help="zacznij od tego zdjecia")
    ap.add_argument("--ortho", action="store_true", help="pary punktow GUGiK/ODM do GT_transfer")
    args = ap.parse_args()
    if args.ortho:
        return ortho_mode(ROOT / "data" / "pilka_ortho_pairs.csv", ROOT / "output" / "pilka" / "reference")
    cfg = PipelineConfig.from_file(args.config)
    excl = cfg.excluded_images()
    images = sorted(p for p in cfg.photos_dir.glob("*.JPG") if p.name not in excl)
    csv_path = cfg.detect.manual_csv
    if csv_path.exists():
        shutil.copy(csv_path, csv_path.with_suffix(".csv.bak"))
    rows = load(csv_path, [p.name for p in images])
    # kandydat sprzed reczniego zaznaczenia (do porownania) -> osobny plik
    cand = csv_path.with_name(csv_path.stem + "_kandydaci.csv")
    if not cand.exists() and csv_path.exists():
        shutil.copy(csv_path, cand)

    st = {"i": 0, "zoom": None, "img": None, "scale": 1.0}
    if args.start:
        st["i"] = next((k for k, p in enumerate(images) if p.name == args.start), 0)

    def cur():
        return rows[images[st["i"]].name]

    def point():
        r = cur()
        if r["u"] and r["v"] and str(r["visible"]).strip() in ("1", "true", "True"):
            return float(r["u"]), float(r["v"])
        return None

    def draw():
        img = st["img"].copy()
        r = cur()
        p = point()
        manual = r["source"].startswith("recznie")
        if p:
            cross(img, *p, (0, 220, 0) if manual else (0, 220, 255), 25, 3)
        s = st["scale"]
        view = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        status = ("NIEWIDOCZNA" if str(r["visible"]).strip() == "0" else
                  f"({p[0]:.1f}, {p[1]:.1f})" if p else "brak punktu")
        txt = (f"[{st['i'] + 1}/{len(images)}] {images[st['i']].name}  {status}  "
               f"{'RECZNIE' if manual else 'zrodlo: ' + (r['source'] or '-')}")
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(view, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, "LPM: okolica pilki -> zoom | n/b nast./poprz. | x niewidoczna | c zatwierdz | q koniec",
                    (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("zdjecie", view)
        if st["zoom"] is not None:
            cx, cy = st["zoom"]
            H, W = img.shape[:2]
            x0 = int(np.clip(cx - ZOOM_HALF, 0, W - 2 * ZOOM_HALF))
            y0 = int(np.clip(cy - ZOOM_HALF, 0, H - 2 * ZOOM_HALF))
            st["zoom_org"] = (x0, y0)
            crop = st["img"][y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]
            z = cv2.resize(crop, None, fx=ZOOM_K, fy=ZOOM_K, interpolation=cv2.INTER_NEAREST)
            if p and x0 <= p[0] < x0 + 2 * ZOOM_HALF and y0 <= p[1] < y0 + 2 * ZOOM_HALF:
                cross(z, (p[0] - x0 + 0.5) * ZOOM_K, (p[1] - y0 + 0.5) * ZOOM_K,
                      (0, 220, 0) if manual else (0, 220, 255), 30, 2)
            cv2.imshow("zoom", z)

    def open_image():
        st["img"] = cv2.imread(str(images[st["i"]]))
        st["scale"] = min(1.0, VIEW_W / st["img"].shape[1])
        st["zoom"] = point()
        if st["zoom"] is None:
            try:
                cv2.destroyWindow("zoom")
            except cv2.error:
                pass
        draw()

    def set_point(u, v):
        cur().update(u=f"{u:.1f}", v=f"{v:.1f}", visible="1", source="recznie", conf="")
        save(csv_path, rows)

    def on_view(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["zoom"] = (x / st["scale"], y / st["scale"])
            draw()

    def on_zoom(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and st.get("zoom_org"):
            x0, y0 = st["zoom_org"]
            set_point(x0 + x / ZOOM_K - 0.5, y0 + y / ZOOM_K - 0.5)
            draw()

    cv2.namedWindow("zdjecie", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("zdjecie", on_view)
    open_image()
    while True:
        if st["zoom"] is not None:
            cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("zoom", on_zoom)
        k = cv2.waitKey(50) & 0xFF
        if k == 255:
            if cv2.getWindowProperty("zdjecie", cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        if k in (ord("q"), 27):
            break
        if k in (ord("n"), ord(" "), 13):
            st["i"] = min(st["i"] + 1, len(images) - 1)
            open_image()
        elif k == ord("b"):
            st["i"] = max(st["i"] - 1, 0)
            open_image()
        elif k == ord("x"):
            cur().update(u="", v="", visible="0", source="recznie: niewidoczna", conf="")
            save(csv_path, rows)
            draw()
        elif k == ord("c") and point():
            set_point(*point())
            draw()
        elif k == ord("z") and point():
            st["zoom"] = point()
            draw()
    save(csv_path, rows)
    cv2.destroyAllWindows()
    n_man = sum(1 for r in rows.values() if r["source"].startswith("recznie"))
    print(f"Zapisano {csv_path}: {n_man}/{len(rows)} zdjec zaznaczonych recznie")
    return 0


if __name__ == "__main__":
    sys.exit(main())
