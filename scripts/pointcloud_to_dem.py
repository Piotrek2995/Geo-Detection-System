"""WSTEPNY DSM/DTM z chmury punktow ODM (odm_filterpoints/point_cloud.ply).

Do uzycia, gdy ODM nie wygenerowal odm_dem/dtm.tif i dsm.tif (brak --dtm --dsm).
Docelowo lepiej dogenerowac DEM w ODM (klasyfikacja gruntu SMRF):
    docker run --rm -v "${PWD}\\odm_project:/datasets/code" opendronemap/odm \\
        --project-path /datasets --rerun-from odm_georeferencing --dtm --dsm --dem-resolution 10

Chmura w odm_filterpoints jest w ukladzie UTM z offsetem (odm_georeferencing/coords.txt,
proj.txt), z ta sama osia Z co kamery ODM. Rasteryzacja na siatce `--res` m:
  DSM = 90. percentyl z w komorce,
  DTM = 10. percentyl z w komorce, potem filtr percentylowy (10%) w oknie `--ground-window` m
        — usuwa obiekty mniejsze niz okno (namioty, ludzie, krzewy); to NIE jest SMRF.

Uzycie:
    python scripts/pointcloud_to_dem.py [--odm odm_project] [--out output/cache] [--res 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import binary_closing, distance_transform_edt, percentile_filter

PLY_DTYPE_FIELDS = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                    "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def read_ply_xyz(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.read(1)
        fields = []
        for line in header.decode("ascii").splitlines():
            parts = line.split()
            if parts[:1] == ["format"] and parts[1] != "binary_little_endian":
                raise ValueError(f"Obslugiwany tylko binary_little_endian PLY, jest {parts[1]}")
            if parts[:1] == ["property"]:
                fields.append((parts[2], PLY_DTYPE_FIELDS[parts[1]]))
        data = np.frombuffer(f.read(), dtype=np.dtype(fields))
    return np.c_[data["x"], data["y"], data["z"]].astype(np.float64)


def read_offset_and_crs(odm: Path) -> tuple[float, float, str]:
    coords = (odm / "odm_georeferencing" / "coords.txt").read_text().splitlines()
    ox, oy = map(float, coords[1].split()[:2])
    crs = (odm / "odm_georeferencing" / "proj.txt").read_text().strip()
    return ox, oy, crs


def grouped_percentile(cell: np.ndarray, z: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Percentyl q z wartosci z w kazdej grupie `cell`. Zwraca (unikalne komorki, wartosci)."""
    order = np.lexsort((z, cell))
    c, zs = cell[order], z[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    counts = np.diff(np.r_[starts, len(c)])
    idx = starts + np.floor(q * (counts - 1)).astype(int)
    return c[starts], zs[idx], counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odm", type=Path, default=Path("odm_project"))
    ap.add_argument("--out", type=Path, default=Path("output/cache"))
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--ground-window", type=float, default=8.0)
    ap.add_argument("--min-points", type=int, default=3)
    args = ap.parse_args()

    pts = read_ply_xyz(args.odm / "odm_filterpoints" / "point_cloud.ply")
    ox, oy, crs = read_offset_and_crs(args.odm)
    x, y, z = pts[:, 0] + ox, pts[:, 1] + oy, pts[:, 2]

    res = args.res
    x_min, y_max = np.floor(x.min()), np.ceil(y.max())
    cols = ((x - x_min) / res).astype(int)
    rows = ((y_max - y) / res).astype(int)
    ncols, nrows = cols.max() + 1, rows.max() + 1
    cell = rows * ncols + cols

    shape = (nrows, ncols)
    cells, z_lo, counts = grouped_percentile(cell, z, 0.10)
    _, z_hi, _ = grouped_percentile(cell, z, 0.90)
    ok = counts >= args.min_points
    lo = np.full(shape, np.nan)
    hi = np.full(shape, np.nan)
    lo.flat[cells[ok]] = z_lo[ok]
    hi.flat[cells[ok]] = z_hi[ok]

    valid = np.isfinite(lo)
    footprint = binary_closing(valid, iterations=3)  # domknij drobne dziury
    # wypelnij luki najblizsza wartoscia, zeby filtr percentylowy nie widzial NaN
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    lo_filled = lo[tuple(idx)]
    win = max(3, int(round(args.ground_window / res)) | 1)
    dtm = percentile_filter(lo_filled, percentile=10, size=win)
    dtm[~footprint] = np.nan
    hi_filled = hi[tuple(idx)]
    dsm = np.where(footprint, hi_filled, np.nan)

    transform = from_origin(x_min, y_max, res, res)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, arr in (("odm_pc_dtm.tif", dtm), ("odm_pc_dsm.tif", dsm)):
        path = args.out / name
        with rasterio.open(path, "w", driver="GTiff", height=nrows, width=ncols, count=1,
                           dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
                           compress="deflate") as dst:
            dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)
        print(f"{path}: {ncols}x{nrows} px @ {res} m, "
              f"z {np.nanmin(arr):.2f}..{np.nanmax(arr):.2f} (mediana {np.nanmedian(arr):.2f})")
    print(f"punkty: {len(pts)}, komorki z >= {args.min_points} pkt: {ok.sum()}, CRS: {crs}")


if __name__ == "__main__":
    main()
