"""Podklady map folium (mapa wynikowa przebiegu, mapa walidacji)."""

from __future__ import annotations

_GEOPORTAL = "https://mapy.geoportal.gov.pl/wss/service"
_KIEG = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"


def add_basemaps(m) -> None:
    """Podklady mapy wynikowej. Pierwszy dodany jest domyslny (ortofotomapa GUGiK).

    Standardowy OpenStreetMap (tile.openstreetmap.org) nie jest uzyty: jego polityka wymaga
    naglowka Referer, ktorego przegladarka nie wysyla dla pliku otwartego lokalnie (file://),
    wiec kafelki byly blokowane. Dane OSM sa dostepne przez CARTO (bez tego wymogu)."""
    import folium
    wms = folium.raster_layers.WmsTileLayer
    wms(url=f"{_GEOPORTAL}/PZGIK/ORTO/WMS/HighResolution", layers="Raster", name="Ortofotomapa GUGiK (HR)",
        fmt="image/jpeg", max_zoom=23, overlay=False, attr="GUGiK").add_to(m)
    wms(url=f"{_GEOPORTAL}/PZGIK/ORTO/WMS/StandardResolution", layers="Raster",
        name="Ortofotomapa GUGiK (standard)", fmt="image/jpeg", max_zoom=23, overlay=False, attr="GUGiK").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     name="Esri World Imagery", attr="Esri, Maxar, Earthstar Geographics", max_zoom=23,
                     max_native_zoom=19).add_to(m)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
                     name="OpenStreetMap (CARTO Voyager)", subdomains="abcd", max_zoom=23, max_native_zoom=20,
                     attr="&copy; OpenStreetMap contributors &copy; CARTO").add_to(m)
    folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
                     name="OpenStreetMap (CARTO jasna)", subdomains="abcd", max_zoom=23, max_native_zoom=20,
                     attr="&copy; OpenStreetMap contributors &copy; CARTO").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                     name="Esri Topo", attr="Esri", max_zoom=23, max_native_zoom=19).add_to(m)
    wms(url=f"{_GEOPORTAL}/img/guest/TOPO/MapServer/WMSServer", layers="Raster", name="Mapa topograficzna GUGiK",
        fmt="image/png", max_zoom=23, overlay=False, attr="GUGiK").add_to(m)
    wms(url=f"{_GEOPORTAL}/PZGIK/NMT/GRID1/WMS/ShadedRelief", layers="Raster", name="Cieniowanie NMT GUGiK",
        fmt="image/png", max_zoom=23, overlay=False, attr="GUGiK").add_to(m)
    # nakladki
    wms(url=_KIEG, layers="budynki", name="Budynki EGiB", fmt="image/png", transparent=True, max_zoom=23,
        overlay=True, attr="GUGiK KIEG").add_to(m)
    wms(url=_KIEG, layers="dzialki,numery_dzialek", name="Dzialki EGiB", fmt="image/png", transparent=True,
        max_zoom=23, overlay=True, show=False, attr="GUGiK KIEG").add_to(m)
