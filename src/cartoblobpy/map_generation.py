import math
import yaml
from pathlib import Path
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point, LineString
from shapely.ops import polygonize, unary_union, nearest_points, linemerge
from pyproj import Transformer


def create_map(
    lat_tl: float,
    lon_tl: float,
    lat_br: float,
    lon_br: float,
    resolution: float = 1.0,
    output: str = "map.png",
    output_yaml: str = "map.yaml",
    water_color: str = "#2A7FFF",
    land_color: str = "#3CB371",
    snap_tolerance: float = 100.0,
    dpi: int = 100
):
    """
    Generate a 2D occupancy grid map image and corresponding YAML metadata file
    for a specified geographic bounding box. It pops up a plot window during
    execution to show the map being generated, and saves the final map as a PNG
    image along with a YAML file containing metadata for ROS navigation.

    Parameters:
    - lat_tl, lon_tl: Latitude and Longitude of the top-left corner of the bounding box.
    - lat_br, lon_br: Latitude and Longitude of the bottom-right corner of the bounding box.
    - resolution: Desired map resolution in meters per pixel (default: 1.0 m/px).
    - output: Filename for the generated map image (default: "map.png").
    - output_yaml: Filename for the generated YAML metadata (default: "map.yaml").
    - water_color: Hex color code for water areas (default: "#2A7FFF").
    - land_color: Hex color code for land areas (default: "#3CB371").
    - snap_tolerance: Distance in meters to snap open coastline endpoints to the bounding box (default: 100.0 m).
    - dpi: Dots per inch for the output image (default: 100). Don't touch it.
    """
# 1. Coordinate & Projection Setup
    zone = int(math.floor((lon_tl + 180) / 6)) + 1
    utm_crs = f"EPSG:{32600 + zone if lat_tl >= 0 else 32700 + zone}"
    trans = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    x0, y1 = trans.transform(lon_tl, lat_tl)
    x1, y0 = trans.transform(lon_br, lat_br)
    minx, miny, maxx, maxy = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    bbox_geom = box(minx, miny, maxx, maxy)
    width_m, height_m = maxx - minx, maxy - miny

    # 2. Fetch OSM Data
    bbox_wgs = (lon_tl, lat_br, lon_br, lat_tl)

    try:
        coast = ox.features_from_bbox(bbox=bbox_wgs, tags={"natural": "coastline"})
        coast_utm = coast.to_crs(utm_crs)
    except Exception:
        coast_utm = None

    try:
        water = ox.features_from_bbox(bbox=bbox_wgs, tags={"natural": ["water", "bay"]})
        water_utm = water.to_crs(utm_crs)
    except Exception:
        water_utm = None

    # 3. Stitch, Snap, and Polygonize
    land_polys = []

    if coast_utm is not None and not coast_utm.empty:

        # Save closed coastlines (Islands) directly
        island_polys = coast_utm[coast_utm.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
        for geom in island_polys.geometry:
            if geom.geom_type == 'Polygon':
                land_polys.append(geom)
            else:
                land_polys.extend(list(geom.geoms))

        # Extract open coastline segments
        lines = [geom for geom in coast_utm.geometry if geom.geom_type in ['LineString', 'MultiLineString']]

        if lines:
            merged = linemerge(unary_union(lines))

            # Clip the coastline to the exact bounding box
            clipped = merged.intersection(bbox_geom)

            if clipped.geom_type == 'LineString':
                merged_lines = [clipped]
            elif clipped.geom_type == 'MultiLineString':
                merged_lines = list(clipped.geoms)
            else:
                # Handle GeometryCollections if intersection creates isolated points
                merged_lines = [g for g in getattr(clipped, 'geoms', []) if 'LineString' in g.geom_type]

            snapped_lines = []
            water_test_points = []

            for line in merged_lines:
                coords = list(line.coords)
                if len(coords) < 2: continue

                # Snap ends to bounding box using the user-defined tolerance
                if coords[0] != coords[-1]:
                    for idx in (0, -1):
                        pt = Point(coords[idx])
                        if pt.distance(bbox_geom.boundary) < snap_tolerance:
                            _, p_bound = nearest_points(pt, bbox_geom.boundary)
                            coords[idx] = (p_bound.x, p_bound.y)

                snapped_lines.append(LineString(coords))

                # Find the longest segment and cast test point 2 meters to the right
                longest_seg = 0
                best_p1, best_p2 = coords[0], coords[1]
                for i in range(1, len(coords)):
                    p1, p2 = coords[i-1], coords[i]
                    l = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    if l > longest_seg:
                        longest_seg, best_p1, best_p2 = l, p1, p2

                if longest_seg > 0:
                    dx, dy = best_p2[0] - best_p1[0], best_p2[1] - best_p1[1]
                    nx, ny = dy / longest_seg, -dx / longest_seg
                    mid_x, mid_y = (best_p1[0] + best_p2[0])/2, (best_p1[1] + best_p2[1])/2

                    test_pt = Point(mid_x + nx * 2.0, mid_y + ny * 2.0)
                    if bbox_geom.contains(test_pt):
                        water_test_points.append(test_pt)

            all_boundaries = unary_union(snapped_lines + [bbox_geom.boundary])
            candidate_polys = list(polygonize(all_boundaries))

            # If a polygon DOESN'T contain any water test point, it is land.
            for poly in candidate_polys:
                if not any(poly.contains(pt) for pt in water_test_points):
                    land_polys.append(poly)
    else:
        land_polys = [bbox_geom]

    # 4. Generate the Final Image
    dpi = 100
    fig_w, fig_h = (width_m / resolution) / dpi, (height_m / resolution) / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    gpd.GeoSeries([bbox_geom]).plot(ax=ax, color=water_color, edgecolor="none")

    if land_polys:
        unified_land = unary_union([p.buffer(0.001) for p in land_polys if p.is_valid]).buffer(-0.001)
        gpd.GeoSeries([unified_land]).plot(ax=ax, color=land_color, edgecolor=land_color, linewidth=0.5)

    if water_utm is not None and not water_utm.empty:
        water_polygons_only = water_utm[water_utm.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polygons_only.empty:
            water_polygons_only.plot(ax=ax, color=water_color, edgecolor="none")

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output, bbox_inches='tight', pad_inches=0)

    # 5. Generate YAML Metadata File
    map_yaml = {
        "image": f"./{Path(output).name}",
        "resolution": round(resolution, 6),
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    with open(output_yaml, "w") as f:
        yaml.dump(map_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"Map saved to {output} ({int(width_m/resolution)}x{int(height_m/resolution)} px)")
    print(f"Metadata saved to {output_yaml}")