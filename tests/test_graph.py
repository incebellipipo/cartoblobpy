import math
import os
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

from cartoblobpy.graph import Graph


class TestGraphCoordinates(unittest.TestCase):
    def test_world_grid_roundtrip_enu(self):
        g = Graph()
        g.origin = np.array([1.0, 2.0, 0.3])
        g.resolution = 0.5

        w = np.array([3.0, 4.0])
        gc = g.world_to_grid(w)
        w_back = g.grid_to_world(gc)

        np.testing.assert_allclose(w_back, w, rtol=0, atol=1e-9)

    def test_world_grid_roundtrip_ned(self):
        g = Graph()
        g.coordinate_frame = "NED"
        g.origin = np.array([0.5, -0.5, math.radians(30)])
        g.resolution = 0.2

        w = np.array([1.2, -0.7])  # [x_north, y_east]
        gc = g.world_to_grid(w)
        w_back = g.grid_to_world(gc)

        np.testing.assert_allclose(w_back, w, rtol=0, atol=1e-9)

    def test_coordinate_frame_validation(self):
        g = Graph()
        with self.assertRaises(ValueError):
            g.coordinate_frame = "ABC"


class TestGraphGridOps(unittest.TestCase):
    def _set_grid(self, g: Graph, grid: np.ndarray):
        # Access private for testing purposes only
        g._Graph__grid = grid

    def test_width_height_shape(self):
        g = Graph()
        grid = np.zeros((10, 20), dtype=float)
        self._set_grid(g, grid)
        g.resolution = 0.5

        self.assertEqual(g.width, 20 * 0.5)
        self.assertEqual(g.height, 10 * 0.5)
        self.assertEqual(g.shape, (g.width, g.height))

    def test_is_free_and_bounds(self):
        g = Graph()
        grid = np.zeros((5, 5), dtype=float)
        grid[2, 2] = 1.0  # obstacle > threshold
        self._set_grid(g, grid)
        g.resolution = 1.0
        g.origin = np.array([0.0, 0.0, 0.0])

        self.assertTrue(g.is_free(np.array([1.0, 1.0])))
        self.assertFalse(g.is_free(np.array([2.0, 2.0])))
        self.assertFalse(g.is_free(np.array([-1.0, 0.0])))  # out of bounds

    def test_is_free_path(self):
        g = Graph()
        grid = np.zeros((5, 5), dtype=float)
        grid[2, 2] = 1.0
        self._set_grid(g, grid)
        g.resolution = 1.0
        g.origin = np.array([0.0, 0.0, 0.0])

        # Free path from (0,0) to (4,0)
        self.assertTrue(g.is_free_path(np.array([0.0, 0.0]), np.array([4.0, 0.0])))
        # Blocked path crosses (2,2)
        self.assertFalse(g.is_free_path(np.array([0.0, 0.0]), np.array([4.0, 4.0])))

    @unittest.expectedFailure
    def test_build_graph_basic(self):
        g = Graph()
        grid = np.zeros((3, 3), dtype=float)
        grid[1, 1] = 1.0  # obstacle
        self._set_grid(g, grid)
        g._Graph__treshold = 0.6  # ensure matching threshold

        g.build_graph()
        # Center cell is an obstacle and should not be a node
        # Known issue: build_graph currently includes obstacle nodes
        self.assertNotIn((1, 1), g.nodes)
        # Check adjacency and weights between (0,0) and (0,1)
        self.assertTrue(g.nodes.has_edge((0, 0), (0, 1)))
        w01 = g.nodes.get_edge_data((0, 0), (0, 1))["weight"]
        self.assertAlmostEqual(w01, 1.0 * (1 + grid[0, 1]))
        # Diagonal neighbor (0,0) -> (1,1) should not exist due to obstacle at (1,1)
        self.assertFalse(g.nodes.has_edge((0, 0), (1, 1)))

    def test_inflate_obstacles(self):
        g = Graph()
        grid = np.zeros((7, 7), dtype=float)
        grid[3, 3] = 1.0
        self._set_grid(g, grid)
        g.resolution = 1.0

        inflated = g.inflate_obstacles(radius=2.0, use_world_units=True)
        self.assertEqual(inflated.shape, grid.shape)
        self.assertEqual(inflated[3, 3], 1.0)
        # Neighbor cells should have value between 0 and 1
        self.assertGreater(inflated[3, 4], 0.0)
        self.assertLess(inflated[3, 4], 1.0)

    def test_distance_to_closest_obstacle(self):
        g = Graph()
        grid = np.zeros((5, 5), dtype=float)
        grid[2, 2] = 1.0
        self._set_grid(g, grid)
        g.resolution = 1.0
        g.origin = np.array([0.0, 0.0, 0.0])

        # world point [x=2, y=3] -> grid [row=3, col=2], adjacent to obstacle at [2,2]
        d = g.distance_to_closest_obstacle(np.array([2.0, 3.0]))
        self.assertAlmostEqual(d, 1.0)


class TestGraphYamlImageLoading(unittest.TestCase):
    def test_load_from_yaml_assets(self):
        # Use one of the packaged YAML maps
        from cartoblobpy.assets import get_map_path
        yaml_path = get_map_path("map012.yaml")

        g = Graph()
        g.load_from_yaml(yaml_path)

        self.assertIsNotNone(g.grid)
        self.assertAlmostEqual(g.resolution, 0.05)
        np.testing.assert_allclose(g.origin, np.array([0.0, 0.0, 0.0]))

    def test_load_from_yaml_with_layers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            base_image = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
            base_image.save(tmp_path / "base.png")

            layer_image = Image.new("L", (4, 4), 255)
            layer_image.putpixel((1, 1), 0)
            layer_image.save(tmp_path / "layer_a.png")

            yaml_path = tmp_path / "map.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "image: ./base.png",
                        "resolution: 1.0",
                        "origin: [0.0, 0.0, 0.0]",
                        "layers:",
                        "  - name: layer_a",
                        "    file: ./layer_a.png",
                        "  - name: layer_b",
                        "    file: null",
                    ]
                ),
                encoding="utf-8",
            )

            g = Graph()
            g.load_from_yaml(str(yaml_path))
            g.build_graph()

            self.assertIn("layer_a", g.layers)
            self.assertIn("layer_b", g.layers)
            self.assertIsNone(g.layers["layer_b"])
            np.testing.assert_allclose(g.layers["layer_a"][2, 1], 1.0)
            np.testing.assert_allclose(g.layers["layer_a"][0, 0], 0.0)

            self.assertIn((0, 0), g.nodes)
            self.assertAlmostEqual(g.nodes.nodes[(0, 0)]["layer_a"], 0.0)
            self.assertIsNone(g.nodes.nodes[(0, 0)]["layer_b"])
            self.assertAlmostEqual(g.nodes.nodes[(2, 1)]["layer_a"], 1.0)

    def test_load_from_yaml_with_rgba_layer_respects_alpha(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            base_image = Image.new("RGBA", (2, 2), (255, 255, 255, 0))
            base_image.save(tmp_path / "base.png")

            # Build RGBA layer where darkness is encoded in RGB and alpha scales contribution.
            layer_image = Image.new("RGBA", (2, 2), (255, 255, 255, 0))
            layer_image.putpixel((0, 0), (0, 0, 0, 255))      # fully dark, fully visible -> 1.0
            layer_image.putpixel((1, 0), (0, 0, 0, 128))      # fully dark, 50% alpha -> ~0.502
            layer_image.putpixel((0, 1), (0, 0, 0, 0))        # fully dark, transparent -> 0.0
            layer_image.putpixel((1, 1), (255, 255, 255, 255))  # white, visible -> 0.0
            layer_image.save(tmp_path / "layer_rgba.png")

            yaml_path = tmp_path / "map.yaml"
            yaml_path.write_text(
                "\n".join(
                    [
                        "image: ./base.png",
                        "resolution: 1.0",
                        "origin: [0.0, 0.0, 0.0]",
                        "layers:",
                        "  - name: traffic",
                        "    file: ./layer_rgba.png",
                    ]
                ),
                encoding="utf-8",
            )

            g = Graph()
            g.load_from_yaml(str(yaml_path))

            # Images are flipped top-bottom on load, so row 0 corresponds to source y=1.
            np.testing.assert_allclose(g.layers["traffic"][1, 0], 1.0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(g.layers["traffic"][1, 1], 128.0 / 255.0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(g.layers["traffic"][0, 0], 0.0, rtol=0, atol=1e-6)
            np.testing.assert_allclose(g.layers["traffic"][0, 1], 0.0, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
