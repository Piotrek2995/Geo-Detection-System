"""Przeniesienie punktow wskazanych na ortofotomapie jednej rekonstrukcji ODM na ortofotomape
innej rekonstrukcji z tych samych zdjec (np. odm_pilka -> odm_pilka_fixcal).

Obie ortofotomapy powstaly z tych samych zdjec, wiec dopasowanie SIFT dziala (w odroznieniu od
pary ODM <-> GUGiK). Transformacja: afiniczna 2D (RANSAC) w ukladzie EPSG:2180, lokalnie —
z dopasowan w promieniu --radius wokol punktow. Kolumny gugik_x/gugik_y przepisywane bez zmian.

    python scripts/transfer_ortho_points.py --src odm_pilka --dst odm_pilka_fixcal \
        --pairs data/pilka_ortho_pairs.csv --out data/pilka_ortho_pairs_fixcal.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]


def render(ortho: Path, bounds, res: float) -> np.ndarray:
    """Wycinek ortofotomapy w EPSG:2180 (siatka res), szarosc uint8."""
    x0, y0, x1, y1 = bounds
    w, h = int((x1 - x0) / res), int((y1 - y0) / res)
    dst = np.zeros((3, h, w), np.uint8)
    with rasterio.open(ortho) as ds:
        for b in range(3):
            reproject(rasterio.band(ds, b + 1), dst[b], dst_transform=rasterio.transform.from_origin(x0, y1, res, res),
                      dst_crs="EPSG:2180", resampling=Resampling.bilinear)
    return cv2.cvtColor(np.transpose(dst, (1, 2, 0)), cv2.COLOR_RGB2GRAY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "odm_pilka")
    ap.add_argument("--dst", type=Path, default=ROOT / "odm_pilka_fixcal")
    ap.add_argument("--pairs", type=Path, default=ROOT / "data/pilka_ortho_pairs.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/pilka_ortho_pairs_fixcal.csv")
    ap.add_argument("--res", type=float, default=0.02, help="rozdzielczosc dopasowania [m]")
    ap.add_argument("--radius", type=float, default=12.0, help="obszar dopasowania wokol punktow [m]")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.pairs, encoding="utf-8")))
    P = np.array([[float(r["odm_x"]), float(r["odm_y"])] for r in rows])
    c = P.mean(0)
    b_src = (c[0] - a.radius, c[1] - a.radius, c[0] + a.radius, c[1] + a.radius)
    # rekonstrukcja docelowa moze byc przesunieta o kilka m: wiekszy wycinek
    b_dst = (b_src[0] - 8, b_src[1] - 8, b_src[2] + 8, b_src[3] + 8)
    A = render(a.src / "odm_orthophoto/odm_orthophoto.tif", b_src, a.res)
    B = render(a.dst / "odm_orthophoto/odm_orthophoto.tif", b_dst, a.res)
    sift = cv2.SIFT_create(8000)
    ka, da = sift.detectAndCompute(A, (A > 0).astype(np.uint8))
    kb, db = sift.detectAndCompute(B, (B > 0).astype(np.uint8))
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x for x, y in m if x.distance < 0.75 * y.distance]

    def to_xy(pts, b):
        return np.column_stack([b[0] + (pts[:, 0] + 0.5) * a.res, b[3] - (pts[:, 1] + 0.5) * a.res])
    sa = to_xy(np.float32([ka[g.queryIdx].pt for g in good]), b_src)
    sb = to_xy(np.float32([kb[g.trainIdx].pt for g in good]), b_dst)
    M, inl = cv2.estimateAffine2D(sa, sb, ransacReprojThreshold=0.10, maxIters=20000, confidence=0.999)
    if M is None:
        print("Dopasowanie nieudane", file=sys.stderr)
        return 1
    inl = inl.ravel().astype(bool)
    res = sb[inl] - (sa[inl] @ M[:, :2].T + M[:, 2])
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
    Q = P @ M[:, :2].T + M[:, 2]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r, q in zip(rows, Q):
            w.writerow({**r, "odm_x": f"{q[0]:.3f}", "odm_y": f"{q[1]:.3f}"})
    meta = {"src": str(a.src.name), "dst": str(a.dst.name), "matches": len(good), "inliers": int(inl.sum()),
            "rms_m": rms, "affine": M.tolist(),
            "shift_m": [[float(q[0] - p[0]), float(q[1] - p[1])] for p, q in zip(P, Q)]}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(good)} dopasowan, {inl.sum()} zgodnych, RMS {rms:.3f} m -> {a.out}")
    for r, p, q in zip(rows, P, Q):
        print(f"  {r['id']:>6}: ({p[0]:.2f}, {p[1]:.2f}) -> ({q[0]:.2f}, {q[1]:.2f})  d = {np.hypot(*(q - p)):.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
