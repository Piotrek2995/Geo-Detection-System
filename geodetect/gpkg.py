"""Minimalny zapis/odczyt GeoPackage (OGC 12-128r18) przez sqlite3 — bez GDAL/fiona.

Obslugiwane geometrie: POINT Z i LINESTRING Z (ISO WKB, little endian), bez obwiedni w
naglowku GP. Wystarcza do warstw wynikowych (obiekty, detekcje, kamery, promienie); plik
otwiera QGIS / ogr. Atrybuty: TEXT, REAL, INTEGER (bool -> INTEGER), listy -> TEXT ";".
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

APPLICATION_ID = 0x47504B47          # "GPKG"
USER_VERSION = 10300                 # GeoPackage 1.3
WKB_POINT_Z, WKB_LINESTRING_Z = 1001, 1002


@dataclass
class Layer:
    name: str
    geometry_type: str                   # POINT | LINESTRING
    srs_id: int
    features: list[tuple[Any, dict]] = field(default_factory=list)   # (wspolrzedne, atrybuty)
    description: str = ""


def _srs_wkt(epsg: int) -> str:
    from pyproj import CRS
    from pyproj.enums import WktVersion
    return CRS.from_epsg(epsg).to_wkt(WktVersion.WKT1_GDAL)


def geometry_blob(coords, srs_id: int, gtype: str) -> bytes:
    """Naglowek GP (wersja 0, flagi: little endian, bez obwiedni) + ISO WKB."""
    head = b"GP" + bytes([0, 0b00000001]) + struct.pack("<i", srs_id)
    if gtype == "POINT":
        x, y, z = (list(coords) + [0.0])[:3]
        wkb = struct.pack("<BIddd", 1, WKB_POINT_Z, x, y, z)
    elif gtype == "LINESTRING":
        pts = [(list(p) + [0.0])[:3] for p in coords]
        wkb = struct.pack("<BII", 1, WKB_LINESTRING_Z, len(pts)) + b"".join(struct.pack("<ddd", *p) for p in pts)
    else:
        raise ValueError(f"Nieobslugiwany typ geometrii: {gtype}")
    return head + wkb


def parse_geometry(blob: bytes):
    """Odwrotnosc geometry_blob: (typ, wspolrzedne, srs_id)."""
    if blob[:2] != b"GP":
        raise ValueError("To nie jest geometria GeoPackage")
    flags = blob[3]
    env = (flags >> 1) & 0b111
    srs_id = struct.unpack("<i", blob[4:8])[0]
    off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    order = "<" if blob[off] == 1 else ">"
    gtype = struct.unpack(order + "I", blob[off + 1:off + 5])[0]
    if gtype == WKB_POINT_Z:
        return "POINT", struct.unpack(order + "ddd", blob[off + 5:off + 29]), srs_id
    if gtype == WKB_LINESTRING_Z:
        n = struct.unpack(order + "I", blob[off + 5:off + 9])[0]
        v = struct.unpack(order + "d" * 3 * n, blob[off + 9:off + 9 + 24 * n])
        return "LINESTRING", [v[i:i + 3] for i in range(0, len(v), 3)], srs_id
    raise ValueError(f"Nieobslugiwany typ WKB {gtype}")


def _sql_type(values: list) -> str:
    vals = [v for v in values if v is not None]
    if vals and all(isinstance(v, (bool, int)) for v in vals):
        return "BOOLEAN" if all(isinstance(v, bool) for v in vals) else "INTEGER"
    if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "REAL"
    return "TEXT"


def _value(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, float) and v != v:
        return None                                  # NaN -> NULL
    if v is None or isinstance(v, (int, float, str)):
        return v
    return str(v)


def write_gpkg(path: Path, layers: list[Layer]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        c = con.cursor()
        c.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        c.execute(f"PRAGMA user_version = {USER_VERSION}")
        c.execute("""CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
                     organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
                     definition TEXT NOT NULL, description TEXT)""")
        c.execute("""CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
                     identifier TEXT UNIQUE, description TEXT DEFAULT '',
                     last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                     min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                     srs_id INTEGER, CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                     REFERENCES gpkg_spatial_ref_sys(srs_id))""")
        c.execute("""CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
                     geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL,
                     m TINYINT NOT NULL, CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
                     CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                     CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))""")
        srs = [("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
               ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None)]
        for epsg in sorted({ly.srs_id for ly in layers} | {4326}):
            from pyproj import CRS
            srs.append((CRS.from_epsg(epsg).name, epsg, "EPSG", epsg, _srs_wkt(epsg), None))
        c.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", srs)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for ly in layers:
            keys: list[str] = []
            for _, a in ly.features:
                keys += [k for k in a if k not in keys]
            types = {k: _sql_type([a.get(k) for _, a in ly.features]) for k in keys}
            cols = "".join(f', "{k}" {types[k]}' for k in keys)
            c.execute(f'CREATE TABLE "{ly.name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom {ly.geometry_type}{cols})')
            pts = []
            for geom, a in ly.features:
                g = [geom] if ly.geometry_type == "POINT" else list(geom)
                pts += [p[:2] for p in g]
                c.execute(f'INSERT INTO "{ly.name}" (geom{"".join(f", {chr(34)}{k}{chr(34)}" for k in keys)}) '
                          f'VALUES (?{",?" * len(keys)})',
                          [geometry_blob(geom, ly.srs_id, ly.geometry_type)] + [_value(a.get(k)) for k in keys])
            bb = ([min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
                  if pts else [None] * 4)
            c.execute("INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (ly.name, "features", ly.name, ly.description, now, *bb, ly.srs_id))
            c.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)",
                      (ly.name, "geom", ly.geometry_type, ly.srs_id, 1, 0))
        con.commit()
    finally:
        con.close()
    return path


def read_gpkg(path: Path, layer: Optional[str] = None) -> dict[str, list[dict]]:
    """Odczyt warstw (do testow / kontroli): {warstwa: [{"geometry": (typ, coords), **atrybuty}]}."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        names = [r[0] for r in con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
        out = {}
        for n in names if layer is None else [layer]:
            rows = []
            for r in con.execute(f'SELECT * FROM "{n}"'):
                d = dict(r)
                gt, coords, srs = parse_geometry(d.pop("geom"))
                d["geometry"] = (gt, coords)
                d["srs_id"] = srs
                rows.append(d)
            out[n] = rows
        return out
    finally:
        con.close()
