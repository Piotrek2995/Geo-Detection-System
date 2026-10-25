"""Transformacje miedzy lokalnym ukladem ODM (topocentric ENU) a EPSG:2180.

Lokalny uklad OpenSfM to topocentryczny ENU na elipsoidzie WGS84 z originem
w reference_lla.json. Jego osie NIE pokrywaja sie z osiami PUWG 1992: w rejonie
lotu (53.82N, 21.44E, poludnik osiowy 19E) zbieznosc poludnikow wynosi ok. 1.97 deg,
co przy naiwnym `ref_x + E, ref_y + N` daje ok. 3.4 cm bledu na kazdy metr
odleglosci od originu (10 m na 300 m). Dlatego kazdy punkt przeliczamy scisle:
ENU -> geodezyjne (lon, lat, h) -> EPSG:2180.

Wysokosci: H [m npm] = h + z_offset, gdzie h to wysokosc wzgledem elipsoidy z
odwrotnej transformacji topocentrycznej z h_0 = reference_lla.altitude. W originie
daje to dokladnie ref_alt + z_local + z_offset (logika z prototypu); dalej dochodzi
poprawka na krzywizne Ziemi (~1 cm na 300 m).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer
from pyproj.enums import TransformDirection


@dataclass(frozen=True)
class GeoRef:
    """Origin lokalnego ENU (reference_lla.json)."""
    lat: float
    lon: float
    alt: float = 0.0


class EnuFrame:
    """Przeliczenia punktow lokalnego ENU (ODM) na WGS84 i EPSG:2180."""

    def __init__(self, ref: GeoRef, z_offset: float = 0.0):
        self.ref = ref
        self.z_offset = z_offset
        # geodezyjne (deg) -> ECEF -> topocentric ENU; uzywane w kierunku odwrotnym
        self._enu = Transformer.from_pipeline(
            "+proj=pipeline "
            "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
            "+step +proj=cart +ellps=WGS84 "
            f"+step +proj=topocentric +ellps=WGS84 "
            f"+lat_0={ref.lat!r} +lon_0={ref.lon!r} +h_0={ref.alt!r}"
        )
        self._to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        self._from_2180 = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)

    def with_z_offset(self, z_offset: float) -> "EnuFrame":
        return EnuFrame(self.ref, z_offset)

    def enu_to_geodetic(self, e, n, u):
        """ENU [m] -> (lon, lat, h_ellipsoidal). Przyjmuje skalary lub tablice."""
        return self._enu.transform(e, n, u, direction=TransformDirection.INVERSE)

    def geodetic_to_enu(self, lon, lat, h):
        return self._enu.transform(lon, lat, h)

    def enu_to_puwg92(self, e, n, u):
        """ENU [m] -> (x=easting, y=northing) EPSG:2180 i H = h + z_offset [m npm]."""
        lon, lat, h = self.enu_to_geodetic(e, n, u)
        x, y = self._to_2180.transform(lon, lat)
        return x, y, h + self.z_offset

    def puwg92_to_lonlat(self, x, y):
        return self._from_2180.transform(x, y)

    def lonlat_to_puwg92(self, lon, lat):
        return self._to_2180.transform(lon, lat)
