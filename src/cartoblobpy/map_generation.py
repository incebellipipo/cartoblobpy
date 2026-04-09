from pathlib import Path
from typing import List, Tuple
import math
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from shapely.geometry import (
    box, LineString, MultiLineString, Polygon, MultiPolygon, Point,
    GeometryCollection,
)
from shapely.ops import polygonize, linemerge, unary_union, snap
from shapely.validation import make_valid
from pyproj import Transformer
import osmnx as ox


def get_utm_crs(lon: float, lat: float) -> str:
    """Derive the UTM EPSG code from a WGS84 lon/lat coordinate.

    Args:
        lon: Longitude in WGS84
        lat: Latitude in WGS84

    Returns:
        EPSG code string (e.g., "EPSG:32632" for UTM zone 32N)
    """
    zone = int(math.floor((lon + 180) / 6)) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def to_utm(lon: float, lat: float, utm_crs: str) -> Tuple[float, float]:
    """Convert a single WGS84 point to UTM coordinates.

    Args:
        lon: Longitude in WGS84
        lat: Latitude in WGS84
        utm_crs: Target UTM CRS string

    Returns:
        Tuple of (x_utm, y_utm) in meters
    """
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    return transformer.transform(lon, lat)


def gdf_to_utm(gdf: gpd.GeoDataFrame, utm_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame from WGS84 to UTM.

    Args:
        gdf: GeoDataFrame with WGS84 geometry
        utm_crs: Target UTM CRS string

    Returns:
        Reprojected GeoDataFrame in UTM coordinates
    """
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(utm_crs)


def extract_linestrings(geom) -> List[LineString]:
    """Recursively extract all LineStrings from any geometry.

    Args:
        geom: Shapely geometry object

    Returns:
        List of LineString objects
    """
    lines = []
    if geom.is_empty:
        return lines
    if isinstance(geom, LineString):
        if geom.length > 0:
            lines.append(geom)
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            lines.extend(extract_linestrings(g))
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            lines.extend(extract_linestrings(g))
    return lines


def classify_polygons(
    polygons: List[Polygon],
    coastline_segments: List[LineString],
    bbox_geom_utm: Polygon,
) -> Tuple[List[Polygon], List[Polygon]]:

    if not polygons:
        return [], []

    bbox_ring = bbox_geom_utm.boundary
    minx, miny, maxx, maxy = bbox_geom_utm.bounds

    # -- Build coastline union for edge classification --
    if coastline_segments:
        coast_union = unary_union(coastline_segments)
    else:
        coast_union = LineString()  # empty

    n = len(polygons)

    # -- Adjacency: find shared edges and classify them --
    # adj[i] = list of (j, is_coastline_edge)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shared = polygons[i].boundary.intersection(polygons[j].boundary)
            if shared.is_empty or shared.length < 0.01:
                continue
            # Is this shared edge along the coastline?
            overlap_with_coast = shared.intersection(coast_union)
            is_coast = (not overlap_with_coast.is_empty
                        and overlap_with_coast.length > 0.01)
            adj[i].append((j, is_coast))
            adj[j].append((i, is_coast))

    # -- Seed polygon: find the one containing top-center of bbox --
    # Top-center is typically open water for coastal maps
    top_center = Point((minx + maxx) / 2, maxy - 0.1)
    seed_idx = None
    for i, poly in enumerate(polygons):
        if poly.contains(top_center):
            seed_idx = i
            break

    # Fallback: largest polygon touching bbox = main land mass
    if seed_idx is None:
        best_idx, best_area = 0, 0
        for i, poly in enumerate(polygons):
            if poly.boundary.intersects(bbox_ring) and poly.area > best_area:
                best_idx = i
                best_area = poly.area
        seed_idx = best_idx
        # Largest touching polygon is likely land, not water
        seed_is_water = False
    else:
        seed_is_water = True

    # -- Flood-fill from seed --
    # label: True = water, False = land, None = unclassified
    label = [None] * n
    label[seed_idx] = seed_is_water

    queue = [seed_idx]
    while queue:
        current = queue.pop(0)
        for neighbor, is_coast_edge in adj[current]:
            if label[neighbor] is not None:
                continue
            if is_coast_edge:
                # Coastline separates land from water → flip
                label[neighbor] = not label[current]
            else:
                # Bbox edge or other non-coastline boundary → same type
                label[neighbor] = label[current]
            queue.append(neighbor)

    # -- Collect results --
    land = []
    water = []
    for i, poly in enumerate(polygons):
        if poly.is_empty or poly.area == 0:
            continue

        # Interior polygons (not touching bbox) = islands
        if not poly.boundary.intersects(bbox_ring):
            cx, cy = poly.centroid.x, poly.centroid.y
            print(f"  Island detected: area={poly.area:.1f} m2, "
                  f"centroid=({cx:.1f}, {cy:.1f}) UTM")
            land.append(poly)
            continue

        if label[i] is None:
            # Unreachable from seed — treat as water (conservative)
            water.append(poly)
        elif label[i]:
            water.append(poly)
        else:
            land.append(poly)

    return land, water


def generate_map(
    lat_top: float,
    lat_bottom: float,
    lon_left: float,
    lon_right: float,
    output_image: str = "map.png",
    output_yaml: str = "map.yaml",
    dpi: int = 300,
    fig_height_inches: float = 10,
    color_water: str = "#2A7FFF",
    color_land: str = "#3CB371",
) -> dict:
    """Generate a map from geographic coordinates.

    Automatically classifies land vs water regions using the coastline topology.
    No manual calibration points needed.

    Args:
        lat_top: Top latitude (WGS84)
        lat_bottom: Bottom latitude (WGS84)
        lon_left: Left longitude (WGS84)
        lon_right: Right longitude (WGS84)
        output_image: Output PNG filename
        output_yaml: Output YAML filename
        dpi: DPI for output image
        fig_height_inches: Figure height in inches
        color_water: Hex color for water regions
        color_land: Hex color for land regions

    Returns:
        Dictionary with map metadata (resolution, dimensions, paths)
    """
    bbox_tuple = (lon_left, lat_bottom, lon_right, lat_top)

    # Auto-detect UTM CRS from bbox center
    center_lon = (lon_left + lon_right) / 2
    center_lat = (lat_bottom + lat_top) / 2
    utm_crs = get_utm_crs(center_lon, center_lat)


    utm_bl = to_utm(lon_left, lat_bottom, utm_crs)    # bottom-left
    utm_tr = to_utm(lon_right, lat_top, utm_crs)      # top-right

    bbox_geom_utm = box(utm_bl[0], utm_bl[1], utm_tr[0], utm_tr[1])
    bbox_ring_utm = bbox_geom_utm.boundary

    width_m  = utm_tr[0] - utm_bl[0]
    height_m = utm_tr[1] - utm_bl[1]

    print(f"Auto-detected projection: {utm_crs}")
    print(f"UTM bounding box: {width_m:.1f} x {height_m:.1f} m")
    print(f"  BL: ({utm_bl[0]:.1f}, {utm_bl[1]:.1f})")
    print(f"  TR: ({utm_tr[0]:.1f}, {utm_tr[1]:.1f})")


    all_coast_geoms_utm = []
    island_polys_utm = []

    try:
        coast = ox.features_from_bbox(
            bbox=bbox_tuple,
            tags={"natural": "coastline"},
        )

        # Open coastline segments (LineString/MultiLineString)
        coast_lines = coast[
            coast.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ]
        if not coast_lines.empty:
            coast_lines_utm = gdf_to_utm(coast_lines, utm_crs)
            print(f"  {len(coast_lines_utm)} coastline line(s)")
            all_coast_geoms_utm.extend(coast_lines_utm.geometry.tolist())

        # Closed coastline ways -> island polygons
        coast_polys = coast[
            coast.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        if not coast_polys.empty:
            coast_polys_utm = gdf_to_utm(coast_polys, utm_crs)
            print(f"  {len(coast_polys_utm)} closed coastline polygon(s) (islands)")
            for geom in coast_polys_utm.geometry:
                if isinstance(geom, Polygon):
                    island_polys_utm.append(geom)
                elif isinstance(geom, MultiPolygon):
                    island_polys_utm.extend(geom.geoms)

    except Exception as e:
        print(f"  Error fetching coastline: {e}")

    # Fallback: place=island / place=islet
    try:
        islands = ox.features_from_bbox(
            bbox=bbox_tuple,
            tags={"place": ["island", "islet"]},
        )
        island_areas = islands[
            islands.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        if not island_areas.empty:
            island_areas_utm = gdf_to_utm(island_areas, utm_crs)
            print(f"  {len(island_areas_utm)} place=island/islet polygon(s)")
            for geom in island_areas_utm.geometry:
                if isinstance(geom, Polygon):
                    island_polys_utm.append(geom)
                elif isinstance(geom, MultiPolygon):
                    island_polys_utm.extend(geom.geoms)
    except Exception as e:
        print(f"  No place=island/islet features: {e}")

    if not all_coast_geoms_utm and not island_polys_utm:
        print("No coastline found -- rendering all as water.")
        _save_empty_map(width_m, height_m, output_image, color_water, dpi,
                       fig_height_inches)
        return _generate_yaml(output_image, output_yaml, height_m, width_m / height_m,
                             fig_height_inches, dpi)


    clipped_segments = []
    if all_coast_geoms_utm:
        all_coast = unary_union(all_coast_geoms_utm)
        merged = linemerge(all_coast)

        clipped_geom = merged.intersection(bbox_geom_utm)
        clipped_segments = extract_linestrings(clipped_geom)

    print(f"  {len(clipped_segments)} clipped segment(s)")
    print(f"  {len(island_polys_utm)} island polygon(s)")

    if not clipped_segments and not island_polys_utm:
        print("No usable coastline -- rendering all as water.")
        _save_empty_map(width_m, height_m, output_image, color_water, dpi,
                       fig_height_inches)
        return _generate_yaml(output_image, output_yaml, height_m, width_m / height_m,
                             fig_height_inches, dpi)


    # Snap endpoints to bbox (tolerance in meters — 0.1 m is fine)
    snapped_segments = []
    for seg in clipped_segments:
        snapped = snap(seg, bbox_ring_utm, tolerance=0.1)
        if isinstance(snapped, LineString):
            snapped_segments.append(snapped)
        else:
            snapped_segments.extend(extract_linestrings(snapped))

    all_lines = unary_union([*snapped_segments, bbox_ring_utm])
    candidate_polys = list(polygonize(all_lines))
    print(f"  {len(candidate_polys)} candidate polygon(s)")

    # Validate
    valid_polys = []
    for p in candidate_polys:
        if not p.is_valid:
            p = make_valid(p)
        if isinstance(p, Polygon) and p.area > 0:
            valid_polys.append(p)
        elif isinstance(p, (MultiPolygon, GeometryCollection)):
            for g in p.geoms:
                if isinstance(g, Polygon) and g.area > 0:
                    valid_polys.append(g)

    print(f"  {len(valid_polys)} valid polygon(s)")


    land_polys, water_polys = classify_polygons(
        valid_polys, snapped_segments, bbox_geom_utm
    )

    # Add island polygons (clipped to UTM bbox)
    for ip in island_polys_utm:
        clipped_island = ip.intersection(bbox_geom_utm)
        if isinstance(clipped_island, Polygon) and clipped_island.area > 0:
            land_polys.append(clipped_island)
        elif isinstance(clipped_island, MultiPolygon):
            for g in clipped_island.geoms:
                if g.area > 0:
                    land_polys.append(g)

    print(f"  {len(land_polys)} land (incl. islands), {len(water_polys)} water")


    aspect = width_m / height_m
    fig_w = fig_height_inches * aspect
    fig_h = fig_height_inches

    px_w = int(round(fig_w * dpi))
    px_h = int(round(fig_h * dpi))
    resolution = height_m / px_h  # m/px — identical in both axes

    print(f"\n  Extent: {width_m:.1f} x {height_m:.1f} m")
    print(f"  Figure: {fig_w:.2f} x {fig_h} in -> {px_w} x {px_h} px")
    print(f"  Resolution: {resolution:.6f} m/px")


    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=color_water)
    ax.set_facecolor(color_water)

    for lp in land_polys:
        gpd.GeoSeries([lp]).plot(ax=ax, color=color_land, edgecolor="none")


    print("Fetching inland water features...")
    water_tags = {
        "natural": ["water", "bay", "strait", "wetland"],
        "waterway": ["river", "stream", "canal", "drain", "ditch",
                      "riverbank", "dock", "boatyard"],
        "water": True,
        "landuse": ["reservoir", "basin"],
        "leisure": ["marina"],
    }
    try:
        water = ox.features_from_bbox(bbox=bbox_tuple, tags=water_tags)

        water_polys_feat = water[
            water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        if not water_polys_feat.empty:
            water_polys_utm = gdf_to_utm(water_polys_feat, utm_crs)
            print(f"  {len(water_polys_utm)} inland water polygon(s)")
            water_polys_utm.plot(ax=ax, color=color_water, edgecolor="none")

        water_lines_feat = water[
            water.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ]
        if not water_lines_feat.empty:
            water_lines_utm = gdf_to_utm(water_lines_feat, utm_crs)
            print(f"  {len(water_lines_utm)} waterway line(s)")
            water_lines_utm.plot(ax=ax, color=color_water, linewidth=1.5)

    except Exception as e:
        print(f"  Inland water error: {e}")

    # ------------------------------------------------------------------
    # 8. Frame and save image
    # ------------------------------------------------------------------
    ax.set_xlim([utm_bl[0], utm_tr[0]])
    ax.set_ylim([utm_bl[1], utm_tr[1]])
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_image, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0)


    return _generate_yaml(output_image, output_yaml, height_m, aspect,
                         fig_height_inches, dpi)


def _save_empty_map(width_m: float, height_m: float, output_image: str,
                    color_water: str, dpi: int, fig_height_inches: float) -> None:
    """Save an all-water map as fallback."""
    aspect = width_m / height_m
    fig_w = fig_height_inches * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_height_inches),
                           facecolor=color_water)
    ax.set_facecolor(color_water)
    ax.axis("off")
    plt.savefig(output_image, dpi=dpi, bbox_inches="tight", pad_inches=0)
    print(f"✓ Saved empty map to {output_image}")


def _generate_yaml(output_image: str, output_yaml: str, height_m: float,
                   aspect: float, fig_height_inches: float,
                   dpi: int) -> dict:
    """Generate YAML metadata file and return map info."""
    px_h = int(round(fig_height_inches * dpi))
    resolution = height_m / px_h

    map_yaml = {
        "image": f"./{Path(output_image).name}",
        "resolution": round(resolution, 6),
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    with open(output_yaml, "w") as f:
        yaml.dump(map_yaml, f, default_flow_style=False, sort_keys=False)

    return {
        "image_path": output_image,
        "yaml_path": output_yaml,
        "resolution": round(resolution, 6),
        "dimensions_pixels": (int(round(fig_height_inches * aspect * dpi)), px_h),
        "dimensions_meters": (aspect * height_m, height_m),
    }
