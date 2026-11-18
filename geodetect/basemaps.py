"""Podklady map folium (mapa wynikowa przebiegu, mapa walidacji)."""

from __future__ import annotations

_GEOPORTAL = "https://mapy.geoportal.gov.pl/wss/service"
_KIEG = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"


BASEMAPS = (   # (klucz, nazwa, rodzaj, url, parametry)
    ("orto", "Ortofotomapa GUGiK (HR)", "wms", f"{_GEOPORTAL}/PZGIK/ORTO/WMS/HighResolution",
     {"layers": "Raster", "fmt": "image/jpeg", "attr": "GUGiK"}),
    ("orto_std", "Ortofotomapa GUGiK (standard)", "wms", f"{_GEOPORTAL}/PZGIK/ORTO/WMS/StandardResolution",
     {"layers": "Raster", "fmt": "image/jpeg", "attr": "GUGiK"}),
    ("esri", "Esri World Imagery", "xyz",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     {"attr": "Esri, Maxar, Earthstar Geographics", "max_native_zoom": 19}),
    ("osm", "OpenStreetMap (CARTO Voyager)", "xyz", "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
     {"attr": "&copy; OpenStreetMap contributors &copy; CARTO", "subdomains": "abcd", "max_native_zoom": 20}),
    ("osm_light", "OpenStreetMap (CARTO jasna)", "xyz", "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
     {"attr": "&copy; OpenStreetMap contributors &copy; CARTO", "subdomains": "abcd", "max_native_zoom": 20}),
    ("esri_topo", "Esri Topo", "xyz",
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
     {"attr": "Esri", "max_native_zoom": 19}),
    ("topo", "Mapa topograficzna GUGiK", "wms", f"{_GEOPORTAL}/img/guest/TOPO/MapServer/WMSServer",
     {"layers": "Raster", "fmt": "image/png", "attr": "GUGiK"}),
    ("relief", "Cieniowanie NMT GUGiK", "wms", f"{_GEOPORTAL}/PZGIK/NMT/GRID1/WMS/ShadedRelief",
     {"layers": "Raster", "fmt": "image/png", "attr": "GUGiK"}),
)


def add_basemaps(m, default: str = "orto", buildings: bool = False) -> None:
    """Podklady mapy. Widoczny jest tylko podklad `default` (klucz z BASEMAPS), pozostale
    do wyboru w przelaczniku warstw; inaczej w streamlit-folium wszystkie rysuja sie naraz.
    buildings: nakladka budynkow EGiB wlaczona na starcie.

    Standardowy OpenStreetMap (tile.openstreetmap.org) nie jest uzyty: jego polityka wymaga
    naglowka Referer, ktorego przegladarka nie wysyla dla pliku otwartego lokalnie (file://),
    wiec kafelki byly blokowane. Dane OSM sa dostepne przez CARTO (bez tego wymogu).
    W GUI domyslny jest Esri (kafle z cache, laduja sie niezawodnie); WMS GUGiK bywa wolny."""
    import folium
    if default not in {b[0] for b in BASEMAPS}:
        raise ValueError(f"Nieznany podklad {default!r}")
    for key, name, kind, url, kw in sorted(BASEMAPS, key=lambda b: b[0] != default):
        kw = dict(kw)
        show = key == default
        if kind == "wms":
            folium.raster_layers.WmsTileLayer(url=url, name=name, max_zoom=23, overlay=False, show=show,
                                              **kw).add_to(m)
        else:
            folium.TileLayer(url, name=name, max_zoom=23, show=show, **kw).add_to(m)
    # nakladki
    wms = folium.raster_layers.WmsTileLayer
    wms(url=_KIEG, layers="budynki", name="Budynki EGiB", fmt="image/png", transparent=True, max_zoom=23,
        overlay=True, show=buildings, attr="GUGiK KIEG").add_to(m)
    wms(url=_KIEG, layers="dzialki,numery_dzialek", name="Dzialki EGiB", fmt="image/png", transparent=True,
        max_zoom=23, overlay=True, show=False, attr="GUGiK KIEG").add_to(m)
