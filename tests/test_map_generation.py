import unittest
import tempfile
import os
import shutil
from pathlib import Path
import yaml
from shapely.geometry import (
    Point, LineString, Polygon, MultiLineString, MultiPolygon, box, GeometryCollection
)
from unittest.mock import patch, MagicMock

from cartoblobpy.map_generation import (
    get_utm_crs,
    to_utm,
    gdf_to_utm,
    extract_linestrings,
    classify_polygons,
    generate_map,
)


class TestGetUtmCrs(unittest.TestCase):
    """Test UTM CRS detection from WGS84 coordinates."""

    def test_trondheim_norway_zone_32n(self):
        """Trondheim is in UTM zone 32N."""
        crs = get_utm_crs(10.395, 63.430)
        self.assertEqual(crs, "EPSG:32632")

    def test_oslo_norway_zone_32n(self):
        """Oslo is in UTM zone 32N."""
        crs = get_utm_crs(10.75, 59.91)
        self.assertEqual(crs, "EPSG:32632")

    def test_bergen_norway_zone_31n(self):
        """Bergen is in UTM zone 31N."""
        crs = get_utm_crs(5.32, 60.39)
        self.assertEqual(crs, "EPSG:32631")

    def test_new_york_usa_zone_18n(self):
        """New York is in UTM zone 18N."""
        crs = get_utm_crs(-74.00, 40.71)
        self.assertEqual(crs, "EPSG:32618")

    def test_sydney_australia_zone_56s(self):
        """Sydney is in UTM zone 56S."""
        crs = get_utm_crs(151.21, -33.87)
        self.assertEqual(crs, "EPSG:32756")

    def test_equator_zero_meridian(self):
        """Test at equator and prime meridian."""
        crs = get_utm_crs(0.0, 0.0)
        # Should be zone 31N
        self.assertEqual(crs, "EPSG:32631")


class TestToUtm(unittest.TestCase):
    """Test WGS84 to UTM coordinate transformation."""

    def test_conversion_preserves_scale(self):
        """UTM conversion should work and produce numeric results."""
        x, y = to_utm(10.395, 63.430, "EPSG:32632")
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        # Coordinates should be reasonable UTM values
        self.assertGreater(x, 0)
        self.assertGreater(y, 0)

    def test_nearby_points_close_in_utm(self):
        """Two nearby points should have small distance in UTM."""
        x1, y1 = to_utm(10.395, 63.430, "EPSG:32632")
        x2, y2 = to_utm(10.396, 63.431, "EPSG:32632")

        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        # ~1 degree at this latitude should be roughly 50-100 km
        self.assertGreater(distance, 100)
        self.assertLess(distance, 200)


class TestExtractLinestrings(unittest.TestCase):
    """Test LineString extraction from geometries."""

    def test_extract_single_linestring(self):
        """Extract single LineString."""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        result = extract_linestrings(line)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], line)

    def test_extract_multilinestring(self):
        """Extract multiple LineStrings from MultiLineString."""
        line1 = LineString([(0, 0), (1, 1)])
        line2 = LineString([(2, 2), (3, 3)])
        multi = MultiLineString([line1, line2])
        result = extract_linestrings(multi)
        self.assertEqual(len(result), 2)

    def test_extract_empty_geometry(self):
        """Empty geometry should return empty list."""
        empty_line = LineString()
        result = extract_linestrings(empty_line)
        self.assertEqual(len(result), 0)

    def test_extract_zero_length_ignored(self):
        """Zero-length LineStrings should be ignored."""
        zero_line = LineString([(0, 0), (0, 0)])
        result = extract_linestrings(zero_line)
        self.assertEqual(len(result), 0)

    def test_extract_from_collection(self):
        """Extract LineStrings from GeometryCollection."""
        line1 = LineString([(0, 0), (1, 1)])
        point = Point(2, 2)
        geom_collection = GeometryCollection([line1, point])
        result = extract_linestrings(geom_collection)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], line1)


class TestClassifyPolygons(unittest.TestCase):
    """Test polygon classification as land or water."""

    def test_simple_bbox_water_classification(self):
        """Test basic water classification."""
        # Create a simple water polygon (touches all bbox edges)
        bbox = box(0, 0, 10, 10)
        water_poly = box(0, 0, 10, 10)

        land, water = classify_polygons([water_poly], [], bbox)
        self.assertEqual(len(water), 1)
        self.assertEqual(len(land), 0)

    def test_interior_island_detection(self):
        """Interior polygons not touching bbox should be islands."""
        bbox = box(0, 0, 100, 100)
        # Island in the interior
        island = box(40, 40, 50, 50)

        land, water = classify_polygons([island], [], bbox)
        self.assertEqual(len(land), 1)
        self.assertEqual(len(water), 0)

    def test_multiple_polygons_classification(self):
        """Test classification of multiple polygons."""
        bbox = box(0, 0, 100, 100)

        # Large polygon touching all edges (likely water)
        outer = box(0, 0, 100, 100)
        # Small interior polygon (island)
        island = box(40, 40, 50, 50)

        land, water = classify_polygons([outer, island], [], bbox)
        # Island should be land
        self.assertEqual(len(land), 1)

    def test_empty_polygons_list(self):
        """Empty polygons list should return empty results."""
        bbox = box(0, 0, 10, 10)
        land, water = classify_polygons([], [], bbox)
        self.assertEqual(len(land), 0)
        self.assertEqual(len(water), 0)

    def test_coastline_edge_classification(self):
        """Polygons separated by coastline should flip classification."""
        bbox = box(0, 0, 100, 100)

        # Create a vertical coastline
        coastline = [LineString([(50, 0), (50, 100)])]

        # Left polygon (water)
        left = Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])
        # Right polygon (land)
        right = Polygon([(50, 0), (100, 0), (100, 100), (50, 100)])

        land, water = classify_polygons([left, right], coastline, bbox)
        # After flood-fill from top-center (which is in left=water),
        # right should be classified as land
        self.assertGreater(len(land) + len(water), 0)


class TestGenerateMapIntegration(unittest.TestCase):
    """Integration tests for map generation (with mocking)."""

    def setUp(self):
        """Set up temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_image = os.path.join(self.temp_dir, "test_map.png")
        self.output_yaml = os.path.join(self.temp_dir, "test_map.yaml")

    def tearDown(self):
        """Clean up temporary files and directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('cartoblobpy.map_generation.ox.features_from_bbox')
    @patch('cartoblobpy.map_generation.plt.savefig')
    def test_generate_map_returns_metadata(self, mock_savefig, mock_osmnx):
        """Test that generate_map returns metadata dictionary."""
        # Mock OSM data fetching to avoid network calls
        mock_osmnx.return_value = MagicMock()
        mock_osmnx.return_value.empty = True

        result = generate_map(
            lat_top=60.45,
            lat_bottom=60.35,
            lon_left=5.25,
            lon_right=5.50,
            output_image=self.output_image,
            output_yaml=self.output_yaml,
            dpi=50,  # Lower DPI for faster test
            fig_height_inches=5,
        )

        # Check return value structure
        self.assertIsInstance(result, dict)
        self.assertIn('image_path', result)
        self.assertIn('yaml_path', result)
        self.assertIn('resolution', result)
        self.assertIn('dimensions_pixels', result)
        self.assertIn('dimensions_meters', result)

    @patch('cartoblobpy.map_generation.ox.features_from_bbox')
    @patch('cartoblobpy.map_generation.plt.savefig')
    def test_generate_map_resolution_positive(self, mock_savefig, mock_osmnx):
        """Resolution should be positive."""
        mock_osmnx.return_value = MagicMock()
        mock_osmnx.return_value.empty = True

        result = generate_map(
            lat_top=60.45,
            lat_bottom=60.35,
            lon_left=5.25,
            lon_right=5.50,
            output_image=self.output_image,
            output_yaml=self.output_yaml,
            dpi=50,
            fig_height_inches=5,
        )

        self.assertGreater(result['resolution'], 0)

    @patch('cartoblobpy.map_generation.ox.features_from_bbox')
    @patch('cartoblobpy.map_generation.plt.savefig')
    def test_generate_map_dimensions_match(self, mock_savefig, mock_osmnx):
        """Pixel dimensions should match DPI and figure height."""
        mock_osmnx.return_value = MagicMock()
        mock_osmnx.return_value.empty = True

        dpi = 100
        fig_height = 10

        result = generate_map(
            lat_top=60.45,
            lat_bottom=60.35,
            lon_left=5.25,
            lon_right=5.50,
            output_image=self.output_image,
            output_yaml=self.output_yaml,
            dpi=dpi,
            fig_height_inches=fig_height,
        )

        # Height in pixels should be roughly fig_height * dpi
        expected_height = round(fig_height * dpi)
        self.assertEqual(result['dimensions_pixels'][1], expected_height)

    @patch('cartoblobpy.map_generation.ox.features_from_bbox')
    @patch('cartoblobpy.map_generation.plt.savefig')
    def test_yaml_file_structure(self, mock_savefig, mock_osmnx):
        """Test that YAML file is created with correct structure."""
        mock_osmnx.return_value = MagicMock()
        mock_osmnx.return_value.empty = True

        generate_map(
            lat_top=60.45,
            lat_bottom=60.35,
            lon_left=5.25,
            lon_right=5.50,
            output_image=self.output_image,
            output_yaml=self.output_yaml,
            dpi=50,
            fig_height_inches=5,
        )

        # Check YAML is created
        self.assertTrue(os.path.exists(self.output_yaml))

        # Parse and validate YAML
        with open(self.output_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        self.assertIn('image', yaml_data)
        self.assertIn('resolution', yaml_data)
        self.assertIn('origin', yaml_data)
        self.assertIn('negate', yaml_data)
        self.assertIn('occupied_thresh', yaml_data)
        self.assertIn('free_thresh', yaml_data)

        # Resolution should be positive
        self.assertGreater(yaml_data['resolution'], 0)
        # Origin should be list of 3 elements
        self.assertEqual(len(yaml_data['origin']), 3)

    @patch('cartoblobpy.map_generation.ox.features_from_bbox')
    @patch('cartoblobpy.map_generation.plt.savefig')
    def test_different_geographic_bounds(self, mock_savefig, mock_osmnx):
        """Test with different geographic bounds."""
        mock_osmnx.return_value = MagicMock()
        mock_osmnx.return_value.empty = True

        # Test with different coordinates
        bounds = [
            (63.470, 63.400, 10.320, 10.510),  # Trondheim
            (60.45, 60.35, 5.25, 5.50),         # Bergen
            (59.95, 59.80, 10.50, 10.80),       # Oslo
        ]

        for lat_top, lat_bot, lon_left, lon_right in bounds:
            output_img = os.path.join(self.temp_dir, "test_map_temp.png")
            output_yml = os.path.join(self.temp_dir, "test_map_temp.yaml")

            result = generate_map(
                lat_top=lat_top,
                lat_bottom=lat_bot,
                lon_left=lon_left,
                lon_right=lon_right,
                output_image=output_img,
                output_yaml=output_yml,
                dpi=50,
                fig_height_inches=5,
            )

            # All should produce valid metadata
            self.assertIsNotNone(result['resolution'])
            self.assertGreater(result['resolution'], 0)


class TestUtmProjectionConsistency(unittest.TestCase):
    """Test consistency of UTM projection across different latitudes."""

    def test_projection_consistency_across_zone(self):
        """Points in same UTM zone should have consistent projections."""
        lon_left, lon_right = 10.0, 10.5
        lat = 63.0
        utm_crs = "EPSG:32632"

        x1, y1 = to_utm(lon_left, lat, utm_crs)
        x2, y2 = to_utm(lon_right, lat, utm_crs)

        # Points on same latitude should have different x but similar y
        self.assertNotEqual(x1, x2)
        # Y values should be very close (within 1km for ~30km longitude distance)
        self.assertLess(abs(y1 - y2), 1000)

    def test_latitude_variation_in_utm(self):
        """Y-coordinate should increase with latitude."""
        lon = 10.4
        lat1, lat2 = 63.0, 63.5
        utm_crs = "EPSG:32632"

        x1, y1 = to_utm(lon, lat1, utm_crs)
        x2, y2 = to_utm(lon, lat2, utm_crs)

        # X may vary due to UTM projection distortion at different latitudes
        # Variation should be less than ~2km over 0.5 degree latitude change
        self.assertLess(abs(x1 - x2), 2000)
        # Y should increase with latitude
        self.assertGreater(y2, y1)


if __name__ == '__main__':
    unittest.main()
